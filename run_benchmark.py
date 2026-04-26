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
from typing import Optional

import requests

from agent_runtime import AgentRuntime, AgentStatus
from bedrock_client import BedrockClient
from fork_agent import run_fork_agent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def agent_monitor_thread(
    runtime: AgentRuntime,
    vm_ip: str,
    server_port: int,
    bedrock: BedrockClient,
    model: str,
    output_dir: str,
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

            thread = threading.Thread(
                target=run_agent_thread,
                args=(
                    agent_id,
                    runtime,
                    vm_ip,
                    server_port,
                    bedrock,
                    model,
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
            max_steps=30,
            temperature=0.7,
            output_dir=output_dir,
        )
        logger.info(f"[{agent_id}] Finished: {result['status']}")
    except Exception as e:
        logger.error(f"[{agent_id}] Crashed: {e}", exc_info=True)
        runtime.fail_agent(agent_id, error=str(e))


def load_osworld_tasks():
    """Load OSWorld benchmark tasks from evaluation_examples."""
    # Load task IDs from test file
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


def run_single_task(task_data, args, output_base):
    """Run a single benchmark task."""
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
    env.reset()

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

    # Initialize Bedrock
    bedrock = BedrockClient(region=args.region, log_dir=output_dir)

    # Run initial setup config
    if config:
        from setup_executor import SetupExecutor
        setup_executor = SetupExecutor(display_num=0, vm_exec=vm_exec)
        logger.info(f"Running {len(config)} setup steps...")
        setup_executor.execute_config(config)
        time.sleep(2)

    # Spawn root agent
    root_id = runtime.spawn_root_agent(task=instruction, display_num=0)

    # Start monitor
    monitor = threading.Thread(
        target=agent_monitor_thread,
        args=(runtime, vm_ip, port, bedrock, args.model, output_dir),
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

        # Timeout after 10 minutes
        if time.time() - start_time > 600:
            logger.warning("Task timeout (10 min)")
            runtime.fail_agent(root_id, error="Timeout")
            break

    time.sleep(2)

    # Collect results
    all_agents = runtime.get_all_agents()
    root_info = runtime.get_agent_status(root_id)

    result = {
        "task_id": task_id,
        "instruction": instruction,
        "status": root_info.get("status", "unknown") if root_info else "error",
        "num_agents": len(all_agents),
        "forked": len(all_agents) > 1,
        "duration": root_info.get("duration", 0) if root_info else 0,
        "steps": root_info.get("result", {}).get("steps_used", 0) if root_info else 0,
        "token_usage": bedrock.get_token_usage(),
        "agents": {
            agent_id: {
                "status": info["status"],
                "display": info["display_num"],
            }
            for agent_id, info in all_agents.items()
        }
    }

    # Save result
    with open(os.path.join(output_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)

    # Cleanup
    runtime.shutdown()

    logger.info(f"\nTask {task_id} completed: {result['status']}")
    logger.info(f"  Forked: {result['forked']} ({result['num_agents']} agents)")
    logger.info(f"  Duration: {result['duration']:.1f}s")
    logger.info(f"  Cost: ${result['token_usage']['total_cost_usd']:.4f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Run OSWorld benchmark")
    parser.add_argument("--task-id", type=int, help="Run specific task by index")
    parser.add_argument("--num-tasks", type=int, default=1, help="Number of tasks to run")
    parser.add_argument("--provider-name", default="aws", help="Provider")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--model", default="claude-opus-4-6", help="Model name")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output dir")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tasks
    tasks = load_osworld_tasks()
    if not tasks:
        logger.error("No tasks found!")
        return

    logger.info(f"Loaded {len(tasks)} tasks")

    # Select tasks to run
    if args.task_id is not None:
        if args.task_id >= len(tasks):
            logger.error(f"Task {args.task_id} out of range (0-{len(tasks)-1})")
            return
        tasks_to_run = [tasks[args.task_id]]
    else:
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
