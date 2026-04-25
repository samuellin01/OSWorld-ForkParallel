"""Single-task baseline CUA agent for OSWorld.

Boots a DesktopEnv, sends a task to Claude
via AWS Bedrock with the computer-use tool, and loops until the agent outputs
DONE/FAIL or hits --max-steps.

Example usage::

    # Free-form task
    python run_baseline_task.py \\
        --task "Open the terminal and run 'echo hello world'" \\
        --provider-name aws --region us-east-1 --headless

    # Benchmark task by ID (domain auto-detected)
    python run_baseline_task.py \\
        --task-id bb5e4c0d-f964-439c-97b6-bdb9747de3f4 \\
        --provider-name aws --region us-east-1 --headless

    # Benchmark task with explicit domain
    python run_baseline_task.py \\
        --task-id bb5e4c0d-f964-439c-97b6-bdb9747de3f4 \\
        --domain chrome \\
        --provider-name aws --region us-east-1 --headless
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

from google_sheets_oauth import create_sheet_from_template_oauth, create_doc_from_template_oauth, get_sheet_id_from_url

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AWS credential loader
# ---------------------------------------------------------------------------

_DEFAULT_CREDENTIALS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aws_credentials.json")


def _load_aws_credentials(path: str) -> Tuple[str, str, str]:
    """Load AWS credentials from *path* (JSON file).

    Returns (aws_access_key_id, aws_secret_access_key, aws_session_token).
    Falls back to empty strings and logs a warning if the file is absent
    or malformed so that the script never crashes at startup.
    """
    logger = logging.getLogger(__name__)
    _empty: Tuple[str, str, str] = ("", "", "")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return (
            data.get("AWS_ACCESS_KEY_ID", ""),
            data.get("AWS_SECRET_ACCESS_KEY", ""),
            data.get("AWS_SESSION_TOKEN", ""),
        )
    except FileNotFoundError:
        logger.warning(
            "Credentials file not found at '%s'. Using empty credential values.",
            path,
        )
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "Could not read credentials file '%s': %s. Using empty credential values.",
            path,
            exc,
        )
    return _empty


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = (
    "You are a computer-use agent on Ubuntu 22.04 with GNOME. "
    "Complete the given task by interacting with the desktop. "
    "Password: '{client_password}'. Home directory: /home/user. "
    "If you launch Chrome from the terminal, add --remote-debugging-port=1337. "
    "Output DONE when finished, FAIL if impossible, or [INFEASIBLE] if the task "
    "cannot be completed due to missing features or system limitations."
)

def _build_system_prompt(client_password: str, **_kwargs) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(client_password=client_password)


# ---------------------------------------------------------------------------
# Benchmark task loader
# ---------------------------------------------------------------------------

def _find_domain_for_task_id(task_id: str, base_dir: str) -> Optional[str]:
    """Return the domain name for *task_id* by searching test_all.json, then
    scanning all example directories.  Returns None if not found."""
    logger = logging.getLogger(__name__)

    # 1. Try test_all.json first.
    test_all_path = os.path.join(base_dir, "test_all.json")
    if os.path.isfile(test_all_path):
        try:
            with open(test_all_path, "r", encoding="utf-8") as fh:
                index: Dict[str, List[str]] = json.load(fh)
            for domain, ids in index.items():
                if task_id in ids:
                    return domain
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read test_all.json: %s", exc)

    # 2. Fall back to scanning all domain directories.
    examples_dir = os.path.join(base_dir, "examples")
    if not os.path.isdir(examples_dir):
        return None
    for domain in os.listdir(examples_dir):
        domain_dir = os.path.join(examples_dir, domain)
        if not os.path.isdir(domain_dir):
            continue
        if os.path.isfile(os.path.join(domain_dir, f"{task_id}.json")):
            return domain
    return None


def _process_google_workspace_config(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Process google_sheet_from_template and google_doc_from_template config items.

    Creates fresh Google Sheets/Docs from templates and injects URLs
    into the task instruction and evaluator.

    Returns modified task_data.
    """
    if "config" not in task_data:
        return task_data

    config_items = task_data["config"]
    new_config = []
    replacements = {}  # {placeholder: url}

    # First pass: create sheets/docs and collect URLs
    for item in config_items:
        if item.get("type") == "google_sheet_from_template":
            params = item["parameters"]
            template_url = params["template_url"]
            placeholder = params.get("placeholder", "{SHEET_URL}")
            title = params.get("title", f"OSWorld Task {task_data.get('id', 'unknown')}")
            client_secret_path = params.get("client_secret_path", "oauth_client_secret.json")
            token_path = params.get("token_path", "oauth_token.pickle")

            logger.info("[setup] Creating Google Sheet from template: %s", template_url)
            sheet_url = create_sheet_from_template_oauth(
                template_url=template_url,
                client_secret_path=client_secret_path,
                token_path=token_path,
                title=title
            )
            logger.info("[setup] Created sheet: %s", sheet_url)
            replacements[placeholder] = sheet_url

            # Update evaluator if it references google_sheet type
            if "evaluator" in task_data and "result" in task_data["evaluator"]:
                result_config = task_data["evaluator"]["result"]
                if result_config.get("type") == "google_sheet":
                    sheet_id = get_sheet_id_from_url(sheet_url)
                    result_config["sheet_id"] = sheet_id

        elif item.get("type") == "google_doc_from_template":
            params = item["parameters"]
            template_url = params["template_url"]
            placeholder = params.get("placeholder", "{DOC_URL}")
            title = params.get("title", f"OSWorld Task Doc {task_data.get('id', 'unknown')}")
            client_secret_path = params.get("client_secret_path", "oauth_client_secret.json")
            token_path = params.get("token_path", "oauth_token.pickle")

            logger.info("[setup] Creating Google Doc from template: %s", template_url)
            doc_url = create_doc_from_template_oauth(
                template_url=template_url,
                client_secret_path=client_secret_path,
                token_path=token_path,
                title=title
            )
            logger.info("[setup] Created doc: %s", doc_url)
            replacements[placeholder] = doc_url

            # Update evaluator if it references google_doc type
            if "evaluator" in task_data and "result" in task_data["evaluator"]:
                result_config = task_data["evaluator"]["result"]
                if result_config.get("type") == "google_doc":
                    doc_id = get_sheet_id_from_url(doc_url)  # Same extraction logic
                    result_config["doc_id"] = doc_id
        else:
            # Keep other config items as-is
            new_config.append(item)

    # Replace all placeholders in instruction
    if "instruction" in task_data:
        for placeholder, url in replacements.items():
            task_data["instruction"] = task_data["instruction"].replace(placeholder, url)

    task_data["config"] = new_config
    return task_data


def _load_benchmark_task(
    task_id: str,
    base_dir: str,
    domain: Optional[str] = None,
) -> Tuple[Dict[str, Any], str]:
    """Load a benchmark task JSON and return (task_data, domain).

    The full task dict is returned so callers can pass it directly to
    ``env.reset(task_config=task_data)``, which handles snapshot revert,
    proxy setup, evaluator configuration, and all setup step types.

    Raises FileNotFoundError if the task JSON cannot be located.
    """
    if domain is None:
        domain = _find_domain_for_task_id(task_id, base_dir)
        if domain is None:
            raise FileNotFoundError(
                f"Task ID '{task_id}' not found in any domain under '{base_dir}'."
            )

    task_path = os.path.join(base_dir, "examples", domain, f"{task_id}.json")
    if not os.path.isfile(task_path):
        raise FileNotFoundError(
            f"Task JSON not found at expected path: {task_path}"
        )

    with open(task_path, "r", encoding="utf-8") as fh:
        data: Dict[str, Any] = json.load(fh)

    return data, domain


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single task in OSWorld using Claude computer-use (baseline).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task source — mutually exclusive: either a free-form string or a benchmark ID.
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--task",
        default=None,
        help="Free-form task description for the agent to execute.",
    )
    task_group.add_argument(
        "--task-id",
        default=None,
        metavar="UUID",
        help=(
            "UUID of a benchmark task under evaluation_examples/examples/. "
            "The domain is auto-detected unless --domain is also provided."
        ),
    )

    parser.add_argument(
        "--domain",
        default=None,
        help=(
            "Benchmark domain (e.g. chrome, gimp, os). "
            "Only used with --task-id. If omitted the domain is auto-detected."
        ),
    )
    parser.add_argument(
        "--test-config-base-dir",
        default="evaluation_examples",
        metavar="DIR",
        help="Base directory containing benchmark task JSONs (test_all.json + examples/).",
    )
    parser.add_argument(
        "--credentials-file",
        default=_DEFAULT_CREDENTIALS_PATH,
        metavar="PATH",
        help="Path to a JSON file with AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN.",
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-6",
        help="Friendly model name (resolved to a Bedrock model ID).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of agent steps before giving up.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the LLM.",
    )
    parser.add_argument(
        "--observation-type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree"],
        default="screenshot",
        help="Type of observation to pass to the agent.",
    )
    parser.add_argument(
        "--provider-name",
        default="aws",
        help="DesktopEnv provider: 'podman', 'docker', 'vmware', or 'aws'.",
    )
    parser.add_argument(
        "--path-to-vm",
        default=None,
        help="Path to the VM snapshot (required for VMware provider).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Run the desktop environment in headless mode.",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (used when --provider-name is 'aws').",
    )
    parser.add_argument(
        "--screen-width",
        type=int,
        default=1920,
        help="Desktop screen width in pixels.",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=1080,
        help="Desktop screen height in pixels.",
    )
    parser.add_argument(
        "--client-password",
        default=None,
        help="Password for the desktop client. Defaults to 'osworld-public-evaluation' for AWS, 'password' otherwise.",
    )
    parser.add_argument(
        "--output-dir",
        default="task_results",
        help="Directory to save per-step screenshots and action logs.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def run_task(
    task: str,
    env: Any,
    bedrock: Any,
    model: str,
    temperature: float,
    max_steps: int,
    observation_type: str,
    screen_width: int,
    screen_height: int,
    output_dir: str,
    task_config: Optional[Dict[str, Any]] = None,
    client_password: str = "osworld-public-evaluation",
    aws_access_key_id: str = "",
    aws_secret_access_key: str = "",
    aws_session_token: str = "",
) -> Tuple[bool, Optional[float]]:
    """Execute *task* in *env* using the Bedrock computer-use agent (baseline, no CC).

    Returns ``(success, score)`` where *success* is ``True`` if the agent
    output DONE, and *score* is the ``env.evaluate()`` result when
    *task_config* is provided (``None`` for free-form tasks).
    """
    from agent_utils import build_observation_message, COMPUTER_USE_TOOL, _resize_screenshot, filter_to_n_most_recent_images, parse_computer_use_actions

    logger = logging.getLogger(__name__)

    os.makedirs(output_dir, exist_ok=True)

    # Save task instruction for trajectory review.
    with open(os.path.join(output_dir, "task.txt"), "w", encoding="utf-8") as fh:
        fh.write(task)

    tools = [COMPUTER_USE_TOOL]
    resize_factor: Tuple[float, float] = (
        screen_width / 1280.0,
        screen_height / 720.0,
    )

    system_prompt = _build_system_prompt(
        client_password=client_password,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_session_token=aws_session_token,
    )

    # Build the initial user message with the task description.
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": [{"type": "text", "text": f"Task: {task}"}],
        }
    ]

    last_tool_use_id: Optional[str] = None
    action_log: List[Dict[str, Any]] = []

    # Reset environment and get first observation.
    if task_config is not None:
        env.reset(task_config=task_config)
    else:
        env.reset()

    # Wait for the VM server to become healthy before running setup commands.
    import requests as _req
    _setup_url = f"http://{env.vm_ip}:{env.server_port}/setup/execute"
    _health_ok = False
    for _wait in range(30):  # up to ~60s
        try:
            _hr = _req.post(
                _setup_url,
                json={"command": "echo ready", "shell": True},
                timeout=10,
            )
            if _hr.status_code == 200 and _hr.json().get("returncode") == 0:
                logger.info("[SETUP] VM server is ready (waited %ds).", _wait * 2)
                _health_ok = True
                break
        except Exception:
            pass
        time.sleep(2)
    if not _health_ok:
        logger.warning("[SETUP] VM server did not become healthy — setup commands may fail.")

    # Pre-install tools.
    _setup_commands = [
        ("echo '{pw}' | sudo -S apt-get update -qq",
         "apt-get update"),
        ("echo '{pw}' | sudo -S apt-get install -y xdotool curl",
         "xdotool+curl install"),
        ("echo '{pw}' | sudo -S apt-get install -y socat 2>/dev/null; "
         "DESKTOP_FILE=/usr/share/applications/google-chrome.desktop; "
         "if [ -f \"$DESKTOP_FILE\" ]; then "
         "  echo '{pw}' | sudo -S sed -i 's|^Exec=/usr/bin/google-chrome-stable|Exec=/usr/bin/google-chrome-stable --remote-debugging-port=1337|g' \"$DESKTOP_FILE\" 2>/dev/null; "
         "  echo '{pw}' | sudo -S sed -i 's|--remote-debugging-port=1337 --remote-debugging-port=1337|--remote-debugging-port=1337|g' \"$DESKTOP_FILE\" 2>/dev/null; "
         "fi; "
         "pgrep -f 'socat.*9222' || nohup socat tcp-listen:9222,fork,reuseaddr tcp:localhost:1337 &>/dev/null &",
         "Chrome debug port + socat relay"),
    ]
    for _cmd_template, _label in _setup_commands:
        try:
            _cmd = _cmd_template.replace("{pw}", client_password)
            _resp = _req.post(
                _setup_url,
                json={"command": _cmd, "shell": True},
                timeout=120,
            )
            if _resp.status_code == 200:
                _data = _resp.json()
                logger.info("[SETUP] %s: rc=%s output=%s", _label,
                            _data.get("returncode"), _data.get("output", "")[:200])
                if _data.get("error"):
                    logger.warning("[SETUP] %s stderr: %s", _label, _data["error"][:200])
        except Exception as e:
            logger.warning("[SETUP] %s failed: %s", _label, e)

    # Verify xdotool is available; retry installation if not.
    _xdotool_ok = False
    for _attempt in range(3):
        try:
            _check_resp = _req.post(
                _setup_url,
                json={"command": "which xdotool", "shell": True},
                timeout=15,
            )
            if _check_resp.status_code == 200 and _check_resp.json().get("returncode") == 0:
                logger.info("[SETUP] xdotool verified: %s", _check_resp.json().get("output", "").strip())
                _xdotool_ok = True
                break
        except Exception:
            pass
        logger.warning("[SETUP] xdotool not found (attempt %d/3), retrying install …", _attempt + 1)
        try:
            _retry_cmd = (
                f"echo '{client_password}' | sudo -S dpkg --configure -a 2>/dev/null; "
                f"echo '{client_password}' | sudo -S apt-get install -y xdotool"
            )
            _req.post(_setup_url, json={"command": _retry_cmd, "shell": True}, timeout=120)
        except Exception as e:
            logger.warning("[SETUP] xdotool retry install failed: %s", e)
        time.sleep(3)

    if not _xdotool_ok:
        logger.error("[SETUP] xdotool is NOT available after retries — typing actions will fail!")

    # Debug: verify background processes launched by config steps.
    if task_config is not None:
        try:
            _debug_url = f"http://{env.vm_ip}:{env.server_port}/setup/execute"
            for _dbg_cmd, _dbg_label in [
                ("ps aux | grep -E 'socat|chrome' | grep -v grep", "socat/chrome processes"),
                ("ss -tlnp | grep -E '9222|1337'", "listening ports 9222/1337"),
                ("curl -s -o /dev/null -w 'HTTP %{http_code}' http://localhost:1337/json/version/", "CDP localhost test"),
                (f"curl -s -o /dev/null -w 'HTTP %{{http_code}}' -H 'Host: {env.vm_ip}:9222' http://localhost:1337/json/version/", "CDP external-host test"),
            ]:
                _dbg_resp = _req.post(
                    _debug_url,
                    json={"command": _dbg_cmd, "shell": True},
                    timeout=10,
                )
                if _dbg_resp.status_code == 200:
                    _dbg_data = _dbg_resp.json()
                    logger.info("[DEBUG] %s:\n%s", _dbg_label, _dbg_data.get("output", "(empty)"))
                    if _dbg_data.get("error"):
                        logger.info("[DEBUG] %s stderr: %s", _dbg_label, _dbg_data["error"])
                else:
                    logger.warning("[DEBUG] %s check failed: HTTP %s", _dbg_label, _dbg_resp.status_code)
        except Exception as _dbg_exc:
            logger.warning("[DEBUG] Could not verify background processes: %s", _dbg_exc)

    logger.info("Waiting 60s for the environment to settle …")
    time.sleep(60)
    obs = env._get_obs()

    success = False
    wall_clock_start = time.monotonic()
    for step in range(1, max_steps + 1):
        logger.info("=== Step %d / %d ===", step, max_steps)

        # Resize screenshot for the computer-use tool (calibrated for 1280×720).
        if obs.get("screenshot"):
            obs = dict(obs)
            obs["screenshot"] = _resize_screenshot(obs["screenshot"])

        # Save screenshot artifact.
        step_dir = os.path.join(output_dir, f"step_{step:04d}")
        os.makedirs(step_dir, exist_ok=True)
        if obs.get("screenshot"):
            shot = obs["screenshot"]
            if hasattr(shot, "read"):
                shot = shot.read()
            with open(os.path.join(step_dir, "screenshot.png"), "wb") as fh:
                fh.write(shot)

        # Build observation content blocks.
        observation_content = build_observation_message(obs, observation_type, step)

        # On subsequent steps, prepend the tool_result for the previous tool_use.
        if last_tool_use_id is not None:
            observation_content.insert(0, {
                "type": "tool_result",
                "tool_use_id": last_tool_use_id,
                "content": "Action executed.",
            })
            last_tool_use_id = None

        messages.append({"role": "user", "content": observation_content})

        # Remove old screenshots to prevent context overflow.
        filter_to_n_most_recent_images(messages, images_to_keep=10, min_removal_threshold=10)

        # Call Bedrock.
        content_blocks, _ = bedrock.chat(
            messages=messages,
            system=system_prompt,
            model=model,
            temperature=temperature,
            tools=tools,
        )

        response_text = "".join(
            b.get("text", "")
            for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("Agent response (first 300 chars): %s", response_text[:300])

        # Save response artifact.
        with open(os.path.join(step_dir, "response.txt"), "w", encoding="utf-8") as fh:
            fh.write(response_text)

        messages.append({"role": "assistant", "content": content_blocks})

        # Parse computer-use actions.
        actions = parse_computer_use_actions(content_blocks, resize_factor)

        # Track tool_use_id for the next tool_result message.
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                last_tool_use_id = block.get("id")
                break

        # Determine executable action code (first non-terminal action).
        action_code: Optional[str] = None
        for act in actions:
            if act not in ("DONE", "FAIL", "WAIT", "CALL_USER"):
                action_code = act
                break

        # Save action artifact.
        if action_code:
            with open(os.path.join(step_dir, "action.py"), "w", encoding="utf-8") as fh:
                fh.write(action_code)

        action_log.append({
            "step": step,
            "actions": actions,
            "action_code": action_code,
            "response_text": response_text[:500],
        })

        # Handle terminal tokens.
        if "DONE" in actions:
            logger.info("Agent output DONE — task complete.")
            env.step("DONE")
            _save_action_log(output_dir, action_log)
            success = True
            break

        if "FAIL" in actions:
            logger.info("Agent output FAIL — task cannot be completed.")
            env.step("FAIL")
            _save_action_log(output_dir, action_log)
            break

        if action_code:
            logger.info("Executing action: %s", action_code[:200])
            try:
                obs, _reward, done, _info = env.step(action_code)
            except (RuntimeError, OSError, ValueError) as exc:
                logger.warning("env.step() raised: %s", exc)
                error_content: List[Dict[str, Any]] = []
                if last_tool_use_id is not None:
                    error_content.append({
                        "type": "tool_result",
                        "tool_use_id": last_tool_use_id,
                        "content": f"Action error: {exc}",
                        "is_error": True,
                    })
                    last_tool_use_id = None
                else:
                    error_content.append({
                        "type": "text",
                        "text": f"Action error: {exc}",
                    })
                messages.append({"role": "user", "content": error_content})
                continue
            if done:
                logger.info("Environment signalled done.")
                _save_action_log(output_dir, action_log)
                success = True
                break
        elif "WAIT" in actions:
            logger.info("Agent WAIT.")
            time.sleep(2)
        else:
            logger.warning("No action or token found — skipping step.")

    else:
        logger.warning("Reached max steps (%d) without DONE/FAIL.", max_steps)
        _save_action_log(output_dir, action_log)

    # Evaluate the benchmark result if a task config was provided.
    score: Optional[float] = None
    if task_config is not None:
        logger.info("Waiting 20s before evaluation …")
        time.sleep(20)
        score = env.evaluate()
        logger.info("Benchmark score: %.4f", score)
        with open(os.path.join(output_dir, "result.txt"), "w", encoding="utf-8") as fh:
            fh.write(f"{score}\n")
        # Write evaluation diagnostics
        if hasattr(env, 'last_eval_details') and env.last_eval_details:
            with open(os.path.join(output_dir, "eval_details.json"), "w", encoding="utf-8") as fh:
                json.dump(env.last_eval_details, fh, indent=2, default=str, ensure_ascii=False)

    # Log and save token usage summary.
    wall_clock_seconds = round(time.monotonic() - wall_clock_start, 3)
    logger.info("Wall-clock time: %.1fs", wall_clock_seconds)
    if hasattr(bedrock, "get_token_usage"):
        token_usage = bedrock.get_token_usage()
        token_usage["wall_clock_seconds"] = wall_clock_seconds
        logger.info(
            "Token usage | steps=%d total_input=%d output=%d uncached=%d "
            "cache_write=%d cache_read=%d cost=$%.4f latency=%.1fs wall=%.1fs",
            token_usage["step_count"],
            token_usage["total_input_tokens"],
            token_usage["total_output_tokens"],
            token_usage["total_uncached_input_tokens"],
            token_usage["total_cache_write_tokens"],
            token_usage["total_cache_read_tokens"],
            token_usage["total_cost_usd"],
            token_usage["total_latency_seconds"],
            wall_clock_seconds,
        )
        with open(os.path.join(output_dir, "token_usage.json"), "w") as fh:
            json.dump(token_usage, fh, indent=2)

    return success, score


def _save_action_log(output_dir: str, action_log: List[Dict[str, Any]]) -> None:
    log_path = os.path.join(output_dir, "action_log.json")
    with open(log_path, "w", encoding="utf-8") as fh:
        json.dump(action_log, fh, indent=2)
    logging.getLogger(__name__).info("Action log saved to %s", log_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger(__name__)

    args = _parse_args(argv)

    # Load AWS credentials fresh at launch.
    aws_access_key_id, aws_secret_access_key, aws_session_token = _load_aws_credentials(
        args.credentials_file
    )

    # Resolve task instruction and output directory.
    task_data: Optional[Dict[str, Any]] = None
    if args.task_id is not None:
        # Benchmark task mode.
        try:
            task_data, domain = _load_benchmark_task(
                task_id=args.task_id,
                base_dir=args.test_config_base_dir,
                domain=args.domain,
            )
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            sys.exit(1)

        # Process Google Workspace templates if present
        task_data = _process_google_workspace_config(task_data)

        task_instruction: str = task_data.get("instruction", "")
        logger.info("Benchmark task ID: %s (domain: %s)", args.task_id, domain)
        logger.info("Instruction: %s", task_instruction)
        output_dir = os.path.join(args.output_dir, domain, args.task_id)
    else:
        # Free-form task mode.
        task_instruction = args.task
        output_dir = args.output_dir
        domain = None

    # Resolve client_password default based on provider.
    client_password = args.client_password
    if client_password is None:
        client_password = "osworld-public-evaluation" if args.provider_name == "aws" else "password"

    # Import DesktopEnv lazily so the module can be imported without it installed.
    try:
        from desktop_env.desktop_env import DesktopEnv
    except ImportError as exc:
        logger.error(
            "Could not import DesktopEnv: %s\n"
            "Make sure you have installed the desktop_env package from this repo.",
            exc,
        )
        sys.exit(1)

    try:
        from bedrock_client import BedrockClient
    except ImportError as exc:
        logger.error("Could not import BedrockClient: %s", exc)
        sys.exit(1)

    screen_size = (args.screen_width, args.screen_height)
    env_kwargs: dict = {
        "provider_name": args.provider_name,
        "action_space": "pyautogui",
        "screen_size": screen_size,
        "headless": args.headless,
        "os_type": "Ubuntu",
        "require_a11y_tree": args.observation_type in ("a11y_tree", "screenshot_a11y_tree"),
        "enable_proxy": True,
        "client_password": client_password,
    }
    if args.path_to_vm:
        env_kwargs["path_to_vm"] = args.path_to_vm
    if args.provider_name == "aws":
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP
        if args.region not in IMAGE_ID_MAP:
            raise ValueError(
                f"AWS region '{args.region}' is not in IMAGE_ID_MAP. "
                f"Available regions: {list(IMAGE_ID_MAP.keys())}"
            )
        region_map = IMAGE_ID_MAP[args.region]
        ami_id = region_map.get(screen_size, region_map.get((1920, 1080)))
        if ami_id is None:
            raise ValueError(
                f"No AMI found for screen size {screen_size} or default (1920, 1080) "
                f"in region '{args.region}'."
            )
        env_kwargs["region"] = args.region
        env_kwargs["snapshot_name"] = ami_id

    logger.info("Task: %s", task_instruction)
    logger.info("Creating DesktopEnv with provider '%s' …", args.provider_name)
    env = DesktopEnv(**env_kwargs)

    bedrock = BedrockClient(region=args.region, log_dir=output_dir)

    try:
        success, score = run_task(
            task=task_instruction,
            env=env,
            bedrock=bedrock,
            model=args.model,
            temperature=args.temperature,
            max_steps=args.max_steps,
            observation_type=args.observation_type,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            output_dir=output_dir,
            task_config=task_data,
            client_password=client_password,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_session_token,
        )
    finally:
        logger.info("Closing environment …")
        env.close()

    status = "DONE" if success else "FAIL/INCOMPLETE"
    print(f"\nTask result: {status}")
    if score is not None:
        print(f"Benchmark score: {score:.4f}")
    if hasattr(bedrock, "get_token_usage"):
        tu = bedrock.get_token_usage()
        print(
            f"Token usage: steps={tu['step_count']} total_input={tu['total_input_tokens']} "
            f"output={tu['total_output_tokens']} cost=${tu['total_cost_usd']:.4f} "
            f"latency={tu['total_latency_seconds']:.1f}s"
        )
    print(f"Artifacts saved to: {output_dir}")
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
