"""Script to run a single OSWorld task with the Confucius agent (Claude via AWS Bedrock).

Example usage:

    # Run a single OS task:
    python scripts/python/run_single_confucius.py --domain os --task_id b6781586-6346-41cd-935a-a6b1487918fc --headless

    # Run a Chrome task with a specific AMI:
    python scripts/python/run_single_confucius.py --domain chrome --task_id 121ba48f-9e17-48ce-9bc6-a4fb17a7ebba --ami_id ami-xxx --headless

    # Run with custom model settings:
    python scripts/python/run_single_confucius.py --domain os --task_id b6781586-6346-41cd-935a-a6b1487918fc --model claude-opus-4 --max_steps 25
"""

import argparse
import datetime
import json
import logging
import os
import sys

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import lib_run_single
from desktop_env.desktop_env import DesktopEnv
from mm_agents.confucius_agent import ConfuciusAgent, SmartContextConfig

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

file_handler = logging.FileHandler(
    os.path.join("logs", "normal-{:}.log".format(datetime_str)), encoding="utf-8"
)
debug_handler = logging.FileHandler(
    os.path.join("logs", "debug-{:}.log".format(datetime_str)), encoding="utf-8"
)
stdout_handler = logging.StreamHandler(sys.stdout)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)

logger = logging.getLogger("desktopenv.experiment")

DEFAULT_SNAPSHOT_NAME = "init_state"


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single OSWorld task with the Confucius agent (Claude via AWS Bedrock)"
    )

    # Required task identification
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
        help="UUID of the task to run (e.g. b6781586-6346-41cd-935a-a6b1487918fc)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        required=True,
        help="Domain subdirectory under evaluation_examples/examples/ (e.g. os, chrome, gimp)",
    )

    # Environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--ami_id", type=str, default=None, help="AMI ID for AWS provider")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument(
        "--headless", action="store_true", help="Run in headless machine"
    )
    parser.add_argument(
        "--action_space", type=str, default="pyautogui", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=["screenshot", "a11y_tree", "screenshot_a11y_tree", "som"],
        default="screenshot",
        help="Observation type",
    )
    parser.add_argument(
        "--provider_name",
        type=str,
        default="aws",
        choices=["aws", "virtualbox", "vmware", "docker", "azure"],
        help="Provider name",
    )
    parser.add_argument("--client_password", type=str, default="")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)
    parser.add_argument("--sleep_after_execution", type=float, default=2.0)
    parser.add_argument("--max_steps", type=int, default=50)

    # Agent / model config
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-6",
        help="Claude model alias (e.g. 'claude-opus-4-6', 'claude-sonnet-4')",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--max_tokens", type=int, default=4096)

    # Smart context management config
    parser.add_argument(
        "--smart_context", action="store_true",
        help="Enable smart context edit for long-running tasks",
    )
    parser.add_argument(
        "--smart_context_compression_threshold", type=int, default=None,
        help="Input token count that triggers context_edit (requires --smart_context)",
    )
    parser.add_argument(
        "--smart_context_clear_at_least", type=int, default=None,
        help="Minimum tokens each context_edit must free (requires --smart_context)",
    )

    # Benchmark config
    parser.add_argument(
        "--test_config_base_dir", type=str, default="evaluation_examples"
    )

    # Logging / result config
    parser.add_argument("--result_dir", type=str, default="./results")

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()

    # Load task config
    config_file = os.path.join(
        args.test_config_base_dir, "examples", args.domain, f"{args.task_id}.json"
    )
    if not os.path.exists(config_file):
        print(f"Error: Task config not found: {config_file}", file=sys.stderr)
        sys.exit(1)

    with open(config_file, "r", encoding="utf-8") as f:
        example = json.load(f)

    instruction = example["instruction"]
    logger.info("Domain: %s", args.domain)
    logger.info("Task ID: %s", args.task_id)
    logger.info("Instruction: %s", instruction)

    smart_ctx = SmartContextConfig(
        enabled=args.smart_context,
        compression_threshold=args.smart_context_compression_threshold,
        clear_at_least=args.smart_context_clear_at_least,
        reminder_enabled=False,
    )

    # Create agent
    agent = ConfuciusAgent(
        model=args.model,
        observation_type=args.observation_type,
        action_space=args.action_space,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        smart_context_config=smart_ctx,
    )

    # Determine snapshot name
    snapshot_name = args.ami_id or DEFAULT_SNAPSHOT_NAME
    if args.provider_name == "aws" and not args.ami_id:
        from desktop_env.providers.aws.manager import IMAGE_ID_MAP
        screen_size = (args.screen_width, args.screen_height)
        region_map = IMAGE_ID_MAP.get(args.region) or IMAGE_ID_MAP.get("us-east-1", {})
        if screen_size in region_map:
            snapshot_name = region_map[screen_size]
        elif (1920, 1080) in region_map:
            snapshot_name = region_map[(1920, 1080)]
        else:
            snapshot_name = DEFAULT_SNAPSHOT_NAME

    # Create environment
    env = DesktopEnv(
        path_to_vm=args.path_to_vm,
        action_space=agent.action_space,
        screen_size=(args.screen_width, args.screen_height),
        headless=args.headless,
        os_type="Ubuntu",
        require_a11y_tree=args.observation_type in ["a11y_tree", "screenshot_a11y_tree", "som"],
        provider_name=args.provider_name,
        client_password=args.client_password,
        region=args.region,
        snapshot_name=snapshot_name,
    )

    # Prepare result directory
    example_result_dir = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        args.domain,
        args.task_id,
    )
    os.makedirs(example_result_dir, exist_ok=True)

    scores = []
    try:
        lib_run_single.run_single_example(
            agent,
            env,
            example,
            args.max_steps,
            instruction,
            args,
            example_result_dir,
            scores,
        )
        result = scores[0] if scores else 0.0
        if not scores:
            logger.warning("Task completed without producing a score result.")
        print(f"Task result: {result}")
        logger.info("Task result: %.4f", result)

        # Print episode stats for interactive visibility
        # (stats.json is already written by lib_run_single.run_single_example)
        stats = agent.get_stats()
        stats_file = os.path.join(example_result_dir, "stats.json")
        logger.info(
            "Episode stats: steps=%d tokens=%s cost_usd=%s latency=%.1fs — see %s",
            stats.get("step_count", 0),
            stats.get("total_tokens"),
            stats.get("total_cost_usd"),
            stats.get("total_latency_seconds", 0.0),
            stats_file,
        )
        print(f"Episode stats: {json.dumps(stats, indent=2)}")
    except Exception as e:
        logger.exception("Exception during task execution: %s", e)
        print(f"Error: {e}", file=sys.stderr)
    finally:
        env.close()
