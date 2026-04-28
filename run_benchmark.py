"""Run fork-based agent on OSWorld benchmark tasks.

Usage:
    python run_benchmark.py --task-id 0 --provider-name aws --region us-east-1 --headless
    python run_benchmark.py --num-tasks 10 --provider-name aws --region us-east-1 --headless
"""

import argparse
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import requests

from agent_runtime import AgentRuntime, AgentStatus
from bedrock_client import BedrockClient
from fork_agent import run_fork_agent
from google_workspace_oauth import (
    create_sheet_from_template_oauth,
    create_doc_from_template_oauth,
    create_slide_from_template_oauth,
    get_sheet_id_from_url,
    reset_sheet_from_template,
    reset_doc_from_template,
    reset_slide_from_template,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def aggregate_token_usage(bedrock_clients: Dict[str, BedrockClient]) -> Dict[str, Any]:
    """Aggregate token usage from all bedrock clients."""
    total_step_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_uncached_input_tokens = 0
    total_cache_write_tokens = 0
    total_cache_read_tokens = 0
    total_cost_usd = 0.0
    total_input_cost_usd = 0.0
    total_output_cost_usd = 0.0
    total_latency_seconds = 0.0
    all_llm_calls = []

    for agent_id, client in bedrock_clients.items():
        usage = client.get_token_usage()
        total_step_count += usage["step_count"]
        total_input_tokens += usage["total_input_tokens"]
        total_output_tokens += usage["total_output_tokens"]
        total_uncached_input_tokens += usage["total_uncached_input_tokens"]
        total_cache_write_tokens += usage["total_cache_write_tokens"]
        total_cache_read_tokens += usage["total_cache_read_tokens"]
        total_cost_usd += usage["total_cost_usd"]
        total_input_cost_usd += usage["total_input_cost_usd"]
        total_output_cost_usd += usage["total_output_cost_usd"]
        total_latency_seconds += usage["total_latency_seconds"]

        # Add agent_id to each call for tracking
        for call in usage["llm_calls"]:
            call_with_agent = call.copy()
            call_with_agent["agent_id"] = agent_id
            all_llm_calls.append(call_with_agent)

    avg_latency = (
        total_latency_seconds / total_step_count if total_step_count > 0 else 0.0
    )

    return {
        "step_count": total_step_count,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "total_uncached_input_tokens": total_uncached_input_tokens,
        "total_cache_write_tokens": total_cache_write_tokens,
        "total_cache_read_tokens": total_cache_read_tokens,
        "total_cost_usd": round(total_cost_usd, 6),
        "total_input_cost_usd": round(total_input_cost_usd, 6),
        "total_output_cost_usd": round(total_output_cost_usd, 6),
        "total_latency_seconds": round(total_latency_seconds, 3),
        "average_latency_per_step_seconds": round(avg_latency, 3),
        "llm_calls": all_llm_calls,
        "num_llm_calls": total_step_count,
    }


def _process_google_workspace_config(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process google_sheet_from_template and google_doc_from_template config items.

    Simplified approach for collaborative tasks:
    1. Create/reset sheet (pre-boot)
    2. Keep non-Google config items (download, open, sleep)
    3. Add chrome_open_tabs with sheet URL
    4. Add final activate_window to bring non-Chrome window to front

    Returns modified task_data.
    """
    if "config" not in task_data:
        return task_data

    # Load pre-created Workspace URLs if available
    workspace_urls_file = "collaborative_workspace_urls.json"
    pre_created_urls = {}
    if os.path.exists(workspace_urls_file):
        with open(workspace_urls_file) as f:
            pre_created_urls = json.load(f)
        logger.info("[setup] Loaded %d pre-created Workspace URLs", len(pre_created_urls))

    # Check both new format (specific.google_account.config) and old format (config)
    config_items = task_data.get("specific", {}).get("google_account", {}).get("config", [])
    if not config_items:
        config_items = task_data.get("config", [])

    new_config = []
    replacements = {}  # {placeholder: url}
    task_id = task_data.get("id", "unknown")
    chrome_urls = []
    final_window = None  # Track which window should be in front at the end
    last_opened_file = None  # Track last non-Chrome file opened

    # Process each config item
    for item in config_items:
        item_type = item.get("type")

        if item_type == "google_sheet_from_template":
            params = item["parameters"]
            template_url = params["template_url"]
            placeholder = params.get("placeholder", "{SHEET_URL}")
            title = params.get("title", f"OSWorld Task {task_id}")
            client_secret_path = params.get("client_secret_path", "oauth_client_secret.json")
            token_path = params.get("token_path", "oauth_token.pickle")

            # Create/reset sheet
            if task_id in pre_created_urls:
                sheet_url = pre_created_urls[task_id]
                logger.info("[setup] Using pre-created sheet: %s", sheet_url)
                logger.info("[setup] Resetting sheet to template state...")
                success = reset_sheet_from_template(
                    sheet_url=sheet_url,
                    template_url=template_url,
                    client_secret_path=client_secret_path,
                    token_path=token_path,
                )
                if not success:
                    logger.warning("[setup] Reset failed, creating new sheet instead")
                    sheet_url = create_sheet_from_template_oauth(
                        template_url=template_url,
                        client_secret_path=client_secret_path,
                        token_path=token_path,
                        title=title
                    )
                    logger.info("[setup] Created new sheet: %s", sheet_url)
            else:
                logger.info("[setup] Creating Google Sheet from template: %s", template_url)
                sheet_url = create_sheet_from_template_oauth(
                    template_url=template_url,
                    client_secret_path=client_secret_path,
                    token_path=token_path,
                    title=title
                )
                logger.info("[setup] Created sheet: %s", sheet_url)

            replacements[placeholder] = sheet_url

            # Update evaluator
            if "evaluator" in task_data and "result" in task_data["evaluator"]:
                result_config = task_data["evaluator"]["result"]
                # Handle both single dict and list of dicts
                if isinstance(result_config, list):
                    for config in result_config:
                        if config.get("type") == "google_sheet":
                            sheet_id = get_sheet_id_from_url(sheet_url)
                            config["sheet_id"] = sheet_id
                elif result_config.get("type") == "google_sheet":
                    sheet_id = get_sheet_id_from_url(sheet_url)
                    result_config["sheet_id"] = sheet_id

            # Add to Chrome URLs if requested
            if params.get("open_in_chrome", False):
                chrome_urls.append(sheet_url)

        elif item_type == "google_doc_from_template":
            params = item["parameters"]
            template_url = params["template_url"]
            placeholder = params.get("placeholder", "{DOC_URL}")
            title = params.get("title", f"OSWorld Task Doc {task_id}")
            client_secret_path = params.get("client_secret_path", "oauth_client_secret.json")
            token_path = params.get("token_path", "oauth_token.pickle")

            # Create/reset doc
            if task_id in pre_created_urls:
                doc_url = pre_created_urls[task_id]
                logger.info("[setup] Using pre-created doc: %s", doc_url)
                logger.info("[setup] Resetting doc to template state...")
                success = reset_doc_from_template(
                    doc_url=doc_url,
                    template_url=template_url,
                    client_secret_path=client_secret_path,
                    token_path=token_path,
                )
                if not success:
                    logger.warning("[setup] Reset failed, creating new doc instead")
                    doc_url = create_doc_from_template_oauth(
                        template_url=template_url,
                        client_secret_path=client_secret_path,
                        token_path=token_path,
                        title=title
                    )
                    logger.info("[setup] Created new doc: %s", doc_url)
            else:
                logger.info("[setup] Creating Google Doc from template: %s", template_url)
                doc_url = create_doc_from_template_oauth(
                    template_url=template_url,
                    client_secret_path=client_secret_path,
                    token_path=token_path,
                    title=title
                )
                logger.info("[setup] Created doc: %s", doc_url)

            replacements[placeholder] = doc_url

            # Update evaluator
            if "evaluator" in task_data and "result" in task_data["evaluator"]:
                result_config = task_data["evaluator"]["result"]
                # Handle both single dict and list of dicts
                if isinstance(result_config, list):
                    for config in result_config:
                        if config.get("type") == "google_doc":
                            doc_id = get_sheet_id_from_url(doc_url)
                            config["doc_id"] = doc_id
                elif result_config.get("type") == "google_doc":
                    doc_id = get_sheet_id_from_url(doc_url)
                    result_config["doc_id"] = doc_id

            # Add to Chrome URLs if requested
            if params.get("open_in_chrome", False):
                chrome_urls.append(doc_url)

        elif item_type == "google_slide_from_template":
            params = item["parameters"]
            template_url = params["template_url"]
            placeholder = params.get("placeholder", "{SLIDE_URL}")
            title = params.get("title", f"OSWorld Task Slides {task_id}")
            client_secret_path = params.get("client_secret_path", "oauth_client_secret.json")
            token_path = params.get("token_path", "oauth_token.pickle")

            # Create/reset slides
            if task_id in pre_created_urls:
                slide_url = pre_created_urls[task_id]
                logger.info("[setup] Using pre-created slides: %s", slide_url)
                logger.info("[setup] Resetting slides to template state...")
                success = reset_slide_from_template(
                    slide_url=slide_url,
                    template_url=template_url,
                    client_secret_path=client_secret_path,
                    token_path=token_path,
                )
                if not success:
                    logger.warning("[setup] Reset failed, creating new slides instead")
                    slide_url = create_slide_from_template_oauth(
                        template_url=template_url,
                        client_secret_path=client_secret_path,
                        token_path=token_path,
                        title=title
                    )
                    logger.info("[setup] Created new slides: %s", slide_url)
            else:
                logger.info("[setup] Creating Google Slides from template: %s", template_url)
                slide_url = create_slide_from_template_oauth(
                    template_url=template_url,
                    client_secret_path=client_secret_path,
                    token_path=token_path,
                    title=title
                )
                logger.info("[setup] Created slides: %s", slide_url)

            replacements[placeholder] = slide_url

            # Update evaluator
            if "evaluator" in task_data and "result" in task_data["evaluator"]:
                result_config = task_data["evaluator"]["result"]
                # Handle both single dict and list of dicts
                if isinstance(result_config, list):
                    for config in result_config:
                        if config.get("type") == "google_slide":
                            slide_id = get_sheet_id_from_url(slide_url)
                            config["slide_id"] = slide_id
                elif result_config.get("type") == "google_slide":
                    slide_id = get_sheet_id_from_url(slide_url)
                    result_config["slide_id"] = slide_id

            # Add to Chrome URLs if requested
            if params.get("open_in_chrome", False):
                chrome_urls.append(slide_url)

        elif item_type == "open":
            # Track last opened file (in case we need to activate it at the end)
            path = item.get("parameters", {}).get("path", "")
            if path:
                # Extract filename for window matching
                import os as _os
                filename = _os.path.basename(path)
                # Common window title patterns: "filename - app" or just "filename"
                last_opened_file = filename
            new_config.append(item)

        elif item_type == "activate_window":
            # Remember the last window to activate (we'll do it at the end)
            final_window = item.get("parameters", {}).get("window_name")

        elif item_type == "launch":
            # Skip Chrome launches - chrome_open_tabs handles it
            command = item.get("parameters", {}).get("command", [])
            if isinstance(command, list) and command and "chrome" in command[0].lower():
                continue
            new_config.append(item)

        elif item_type == "chrome_open_tabs":
            # Don't add yet - we'll add at the end
            pass

        else:
            # Keep everything else (download, open, sleep, etc.)
            new_config.append(item)

    # Add chrome_open_tabs with all collected URLs
    if chrome_urls:
        new_config.append({
            "type": "chrome_open_tabs",
            "parameters": {"urls_to_open": chrome_urls}
        })

    # Add final window activation to control what's visible to the agent
    # Priority: explicit activate_window > last opened file > Chrome
    if final_window:
        new_config.append({
            "type": "activate_window",
            "parameters": {"window_name": final_window, "strict": True}
        })
    elif last_opened_file:
        new_config.append({
            "type": "activate_window",
            "parameters": {"window_name": last_opened_file, "strict": False}
        })
    else:
        new_config.append({
            "type": "activate_window",
            "parameters": {"window_name": "Chrome", "strict": False}
        })

    # Replace placeholders in instruction
    if "instruction" in task_data:
        for placeholder, url in replacements.items():
            task_data["instruction"] = task_data["instruction"].replace(placeholder, url)

    task_data["config"] = new_config
    return task_data


def agent_monitor_thread(
    runtime: AgentRuntime,
    vm_ip: str,
    server_port: int,
    bedrock: BedrockClient,
    model: str,
    max_steps: int,
    output_dir: str,
    region: str,
    bedrock_clients: Dict[str, BedrockClient],
):
    """Monitor for new agents and spawn threads to run them."""
    running_threads = {}

    while True:
        time.sleep(0.5)

        all_agents = runtime.get_all_agents()

        for agent_id, status_dict in all_agents.items():
            status = status_dict["status"]

            if agent_id in running_threads:
                continue
            if status in ("completed", "failed", "killed"):
                continue
            if status != "running":
                continue

            logger.info(f"[Monitor] Spawning thread for {agent_id}")

            agent_output = os.path.join(output_dir, agent_id)
            os.makedirs(agent_output, exist_ok=True)

            agent_info = runtime.get_agent_status(agent_id)
            task = agent_info.get("subtask", "")
            context_summary = agent_info.get("context_summary")

            # Create dedicated bedrock client for this agent with its own log dir
            agent_bedrock = BedrockClient(
                region=region,
                log_dir=agent_output,
                agent_id=agent_id
            )
            bedrock_clients[agent_id] = agent_bedrock  # Track for token aggregation

            thread = threading.Thread(
                target=run_agent_thread,
                args=(
                    agent_id,
                    runtime,
                    vm_ip,
                    server_port,
                    agent_bedrock,  # Use agent-specific client
                    model,
                    max_steps,
                    task,
                    context_summary,
                    agent_output,
                ),
                daemon=True,
            )
            thread.start()
            running_threads[agent_id] = thread

        completed = [
            aid for aid, thread in running_threads.items()
            if not thread.is_alive()
        ]
        for aid in completed:
            del running_threads[aid]

        root_status = runtime.get_agent_status("root")
        if root_status and root_status["status"] in ("completed", "failed"):
            break


def run_agent_thread(
    agent_id: str,
    runtime: AgentRuntime,
    vm_ip: str,
    server_port: int,
    bedrock: BedrockClient,
    model: str,
    max_steps: int,
    task: str,
    parent_context: Optional[str],
    output_dir: str,
):
    """Run an agent in a thread."""
    try:
        result = run_fork_agent(
            agent_id=agent_id,
            runtime=runtime,
            vm_ip=vm_ip,
            server_port=server_port,
            bedrock=bedrock,
            model=model,
            task=task,
            parent_context=parent_context,
            max_steps=max_steps,
            temperature=0.7,
            output_dir=output_dir,
        )
        logger.info(f"[{agent_id}] Finished: {result['status']}")
    except Exception as e:
        logger.error(f"[{agent_id}] Crashed: {e}", exc_info=True)
        runtime.fail_agent(agent_id, error=str(e))


def load_osworld_tasks(task_type="standard", test_file=None):
    """Load OSWorld benchmark tasks from evaluation_examples.

    Args:
        task_type: "standard" for regular tasks, "collaborative" for collaborative tasks
        test_file: Optional specific test file (e.g., "test_all.json", "test_small.json")
    """
    if task_type == "collaborative":
        return load_collaborative_tasks()

    # Load task IDs from test file
    if test_file:
        task_list_paths = [f"evaluation_examples/{test_file}"]
    else:
        task_list_paths = [
            "evaluation_examples/test_small.json",  # Start with small set
            "evaluation_examples/test_all.json",
        ]

    task_ids_by_category = None
    for path in task_list_paths:
        if os.path.exists(path):
            logger.info(f"Loading task IDs from {path}")
            with open(path) as f:
                task_ids_by_category = json.load(f)
            break

    if not task_ids_by_category:
        logger.error("No task list file found in evaluation_examples/")
        return []

    # Flatten task IDs
    all_task_ids = []
    for category, ids in task_ids_by_category.items():
        all_task_ids.extend([(category, task_id) for task_id in ids])

    logger.info(f"Found {len(all_task_ids)} tasks across {len(task_ids_by_category)} categories")

    # Load actual task details from examples/
    tasks = []
    for category, task_id in all_task_ids:
        task_file = f"evaluation_examples/examples/{category}/{task_id}.json"
        if os.path.exists(task_file):
            with open(task_file) as f:
                task_data = json.load(f)
                tasks.append(task_data)
        else:
            logger.warning(f"Task file not found: {task_file}")

    logger.info(f"Loaded {len(tasks)} task definitions")
    return tasks


def load_collaborative_tasks():
    """Load collaborative tasks from evaluation_examples."""
    config_file = "evaluation_examples/collaborative_task_configs.json"

    if not os.path.exists(config_file):
        logger.error(f"Collaborative task config not found: {config_file}")
        return []

    logger.info(f"Loading collaborative tasks from {config_file}")
    with open(config_file) as f:
        config_data = json.load(f)

    task_metadata = config_data.get("tasks", {})
    logger.info(f"Found {len(task_metadata)} collaborative tasks")

    # Load actual task files
    tasks = []
    for task_id, metadata in task_metadata.items():
        # Only load active tasks
        if metadata.get("status") != "active":
            continue

        task_file = f"evaluation_examples/examples/collaborative/{task_id}.json"
        if os.path.exists(task_file):
            with open(task_file) as f:
                task_data = json.load(f)
                tasks.append(task_data)
        else:
            logger.warning(f"Task file not found: {task_file}")

    logger.info(f"Loaded {len(tasks)} collaborative task definitions")
    return tasks


def run_single_task(task_data, args, output_base):
    """Run a single benchmark task."""
    # Process Google Workspace configs BEFORE booting VM
    task_data = _process_google_workspace_config(task_data)

    task_id = task_data.get("id", "unknown")
    instruction = task_data.get("instruction", "")
    config = task_data.get("config", [])

    logger.info("\n" + "=" * 80)
    logger.info(f"Task {task_id}: {instruction[:100]}")
    logger.info("=" * 80)

    output_dir = os.path.join(output_base, f"task_{task_id}")
    os.makedirs(output_dir, exist_ok=True)

    # Boot VM
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.providers.aws.manager import IMAGE_ID_MAP

    screen_size = (1920, 1080)
    region_map = IMAGE_ID_MAP[args.region]
    ami_id = region_map.get(screen_size, region_map.get((1920, 1080)))

    env = DesktopEnv(
        provider_name=args.provider_name,
        action_space="pyautogui",
        screen_size=screen_size,
        headless=args.headless,
        os_type="Ubuntu",
        client_password="osworld-public-evaluation",
        region=args.region,
        snapshot_name=ami_id,
    )
    env.reset(task_config=task_data)

    vm_ip = env.vm_ip
    port = env.server_port

    def vm_exec(cmd: str, timeout: int = 120) -> Optional[dict]:
        try:
            r = requests.post(
                f"http://{vm_ip}:{port}/setup/execute",
                json={"command": cmd, "shell": True},
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"vm_exec failed: {e}")
        return None

    # Wait for VM
    logger.info("Waiting for VM...")
    for _ in range(30):
        try:
            r = requests.post(
                f"http://{vm_ip}:{port}/setup/execute",
                json={"command": "echo ready", "shell": True},
                timeout=10,
            )
            if r.status_code == 200:
                break
        except:
            pass
        time.sleep(2)

    # Initialize runtime
    runtime = AgentRuntime(vm_exec=vm_exec, num_displays=8, password="osworld-public-evaluation")
    runtime.initialize()

    # Initialize Bedrock - shared dict to track all agent clients
    bedrock = BedrockClient(region=args.region, log_dir=output_dir, agent_id="root")
    bedrock_clients = {"root": bedrock}  # Track all bedrock clients for token aggregation

    # Setup already ran via env.reset(task_config=task_data) above
    # No need to execute config again

    # Spawn root agent
    root_id = runtime.spawn_root_agent(task=instruction, display_num=0)

    # Start monitor
    monitor = threading.Thread(
        target=agent_monitor_thread,
        args=(runtime, vm_ip, port, bedrock, args.model, args.max_steps, output_dir, args.region, bedrock_clients),
        daemon=True,
    )
    monitor.start()

    # Wait for completion
    start_time = time.time()
    while True:
        time.sleep(2)
        root_status = runtime.get_agent_status(root_id)
        if not root_status:
            break
        if root_status["status"] in ("completed", "failed", "killed"):
            break

        # Timeout after 20 minutes
        if time.time() - start_time > 1200:
            logger.warning("Task timeout (20 min)")
            runtime.fail_agent(root_id, error="Timeout")
            break

    time.sleep(2)

    # Collect results
    all_agents = runtime.get_all_agents()
    root_info = runtime.get_agent_status(root_id)

    # Aggregate token usage from all agents
    aggregated_token_usage = aggregate_token_usage(bedrock_clients)

    result = {
        "task_id": task_id,
        "instruction": instruction,
        "status": root_info.get("status", "unknown") if root_info else "error",
        "num_agents": len(all_agents),
        "forked": len(all_agents) > 1,
        "duration": root_info.get("duration", 0) if root_info else 0,
        "steps": root_info.get("result", {}).get("steps_used", 0) if root_info else 0,
        "token_usage": aggregated_token_usage,
        "agents": {
            agent_id: {
                "status": info["status"],
                "display": info["display_num"],
            }
            for agent_id, info in all_agents.items()
        }
    }

    # Write token_usage.json (separate file for backward compatibility)
    with open(os.path.join(output_dir, "token_usage.json"), "w") as f:
        json.dump(aggregated_token_usage, f, indent=2)

    # Evaluate task completion if evaluator is configured
    score = None
    if hasattr(env, 'evaluator') and env.evaluator:
        logger.info("Waiting 20s before evaluation...")
        time.sleep(20)
        score = env.evaluate()
        logger.info(f"Benchmark score: {score:.4f}")

        result["score"] = score

        # Save evaluation details if available
        if hasattr(env, 'last_eval_details') and env.last_eval_details:
            result["eval_details"] = env.last_eval_details
            with open(os.path.join(output_dir, "eval_details.json"), "w") as f:
                json.dump(env.last_eval_details, f, indent=2)

        with open(os.path.join(output_dir, "result.txt"), "w") as f:
            f.write(f"{score}\n")
    else:
        logger.info("No evaluator configured - skipping evaluation")

    # Save result
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Cleanup
    runtime.shutdown()

    logger.info(f"\nTask {task_id} completed: {result['status']}")
    logger.info(f"  Forked: {result['forked']} ({result['num_agents']} agents)")
    logger.info(f"  Duration: {result['duration']:.1f}s")
    logger.info(f"  Cost: ${result['token_usage']['total_cost_usd']:.4f}")
    if score is not None:
        logger.info(f"  Score: {score:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run OSWorld benchmark")

    # Task selection (mutually exclusive)
    task_group = parser.add_mutually_exclusive_group()
    task_group.add_argument("--task-id", type=int, help="Run specific task by index (0, 1, 2, ...)")
    task_group.add_argument("--task-name", type=str, help="Run specific task by UUID (e.g., bb5e4c0d-f964-439c-97b6-bdb9747de3f4)")

    parser.add_argument("--num-tasks", type=int, default=1, help="Number of tasks to run (when not using --task-id or --task-name)")
    parser.add_argument("--domain", type=str, default=None, help="Filter tasks by domain (e.g., chrome, gimp, multi_apps, all)")
    parser.add_argument("--task-type", default="standard", choices=["standard", "collaborative"],
                        help="Task type: standard or collaborative")
    parser.add_argument("--test-file", type=str, default=None,
                        help="Specific test file to use (e.g., test_all.json, test_small.json). Default: auto-detect")
    parser.add_argument("--provider-name", default="aws", help="Provider")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model name")
    parser.add_argument("--max-steps", type=int, default=30, help="Max steps per agent (default: 30)")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tasks
    tasks = load_osworld_tasks(task_type=args.task_type, test_file=args.test_file)
    if not tasks:
        logger.error("No tasks found!")
        return

    logger.info(f"Loaded {len(tasks)} tasks")

    # Filter by domain if specified
    if args.domain and args.domain != "all":
        original_count = len(tasks)
        tasks = [t for t in tasks if t.get("snapshot") == args.domain or args.domain in t.get("related_apps", [])]
        logger.info(f"Filtered to {len(tasks)} tasks in domain '{args.domain}' (from {original_count})")
        if not tasks:
            logger.error(f"No tasks found for domain '{args.domain}'")
            return

    # Select tasks to run
    if args.task_name:
        # Find task by UUID
        matching_tasks = [t for t in tasks if t.get("id") == args.task_name]
        if not matching_tasks:
            logger.error(f"Task '{args.task_name}' not found")
            logger.info("Available task IDs:")
            for i, t in enumerate(tasks[:10]):
                logger.info(f"  {i}: {t.get('id')}")
            if len(tasks) > 10:
                logger.info(f"  ... and {len(tasks) - 10} more")
            return
        tasks_to_run = matching_tasks
    elif args.task_id is not None:
        # Find task by index
        if args.task_id >= len(tasks):
            logger.error(f"Task index {args.task_id} out of range (0-{len(tasks)-1})")
            return
        tasks_to_run = [tasks[args.task_id]]
    else:
        # Run first N tasks
        tasks_to_run = tasks[:args.num_tasks]

    # Run tasks
    results = []
    for i, task in enumerate(tasks_to_run):
        logger.info(f"\n{'='*80}\nRunning task {i+1}/{len(tasks_to_run)}\n{'='*80}")
        result = run_single_task(task, args, args.output_dir)
        results.append(result)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("BENCHMARK SUMMARY")
    logger.info("=" * 80)

    total_cost = sum(r["token_usage"]["total_cost_usd"] for r in results)
    num_forked = sum(1 for r in results if r["forked"])

    logger.info(f"Tasks run: {len(results)}")
    logger.info(f"Tasks that forked: {num_forked}/{len(results)} ({100*num_forked/len(results):.1f}%)")
    logger.info(f"Total cost: ${total_cost:.4f}")
    logger.info(f"Avg cost per task: ${total_cost/len(results):.4f}")

    # Save summary
    summary = {
        "num_tasks": len(results),
        "num_forked": num_forked,
        "total_cost": total_cost,
        "results": results,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
