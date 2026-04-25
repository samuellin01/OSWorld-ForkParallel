"""Script to run end-to-end evaluation on the OSWorld benchmark using the
Confucius agent (Claude via AWS Bedrock).

Utils and basic architecture credit to
https://github.com/web-arena-x/webarena/blob/main/run.py.
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
from multiprocessing import Manager, Process
from typing import Dict, List

from tqdm import tqdm

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
sdebug_handler = logging.FileHandler(
    os.path.join("logs", "sdebug-{:}.log".format(datetime_str)), encoding="utf-8"
)

file_handler.setLevel(logging.INFO)
debug_handler.setLevel(logging.DEBUG)
stdout_handler.setLevel(logging.INFO)
sdebug_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter(
    fmt="\x1b[1;33m[%(asctime)s \x1b[31m%(levelname)s \x1b[32m%(module)s/%(lineno)d-%(processName)s\x1b[1;33m] \x1b[0m%(message)s"
)
file_handler.setFormatter(formatter)
debug_handler.setFormatter(formatter)
stdout_handler.setFormatter(formatter)
sdebug_handler.setFormatter(formatter)

stdout_handler.addFilter(logging.Filter("desktopenv"))
sdebug_handler.addFilter(logging.Filter("desktopenv"))

logger.addHandler(file_handler)
logger.addHandler(debug_handler)
logger.addHandler(stdout_handler)
logger.addHandler(sdebug_handler)

logger = logging.getLogger("desktopenv.experiment")

DEFAULT_SNAPSHOT_NAME = "init_state"


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSWorld benchmark with the Confucius agent (Claude via AWS Bedrock)"
    )

    # Environment config
    parser.add_argument("--path_to_vm", type=str, default=None)
    parser.add_argument("--ami_id", type=str, default=None, help="AMI ID to use as the VM snapshot (AWS provider)")
    parser.add_argument("--region", type=str, default="us-east-1", help="AWS region for the VM (default: us-east-1)")
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
    parser.add_argument("--domain", type=str, default="all")
    parser.add_argument(
        "--test_all_meta_path",
        type=str,
        default="evaluation_examples/test_all.json",
    )

    # Logging / result config
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel environments",
    )

    args = parser.parse_args()
    return args


# ---------------------------------------------------------------------------
# Task distribution
# ---------------------------------------------------------------------------

def distribute_tasks(test_all_meta: dict, num_envs: int) -> List[Dict]:
    """Distribute tasks evenly across environments."""
    all_tasks = [
        (domain, example_id)
        for domain, examples in test_all_meta.items()
        for example_id in examples
    ]

    tasks_per_env = math.ceil(len(all_tasks) / num_envs)

    distributed: List[Dict] = []
    for i in range(num_envs):
        env_tasks: Dict[str, List[str]] = {}
        start_idx = i * tasks_per_env
        end_idx = min((i + 1) * tasks_per_env, len(all_tasks))
        for domain, example_id in all_tasks[start_idx:end_idx]:
            env_tasks.setdefault(domain, []).append(example_id)
        distributed.append(env_tasks)

    return distributed


# ---------------------------------------------------------------------------
# Per-environment worker
# ---------------------------------------------------------------------------

def run_env_tasks(
    env_idx: int,
    env: DesktopEnv,
    agent: ConfuciusAgent,
    env_tasks: Dict,
    args: argparse.Namespace,
    shared_scores: list,
) -> None:
    """Run all assigned tasks inside one environment."""
    logger.info(f"Executing tasks in environment {env_idx + 1}/{args.num_envs}")

    for domain in tqdm(env_tasks, desc=f"Env{env_idx + 1}-Domain"):
        for example_id in tqdm(env_tasks[domain], desc="Example", leave=False):
            config_file = os.path.join(
                args.test_config_base_dir, f"examples/{domain}/{example_id}.json"
            )
            with open(config_file, "r", encoding="utf-8") as f:
                example = json.load(f)

            logger.info(f"[Env {env_idx + 1}][Domain]: {domain}")
            logger.info(f"[Env {env_idx + 1}][Example ID]: {example_id}")
            logger.info(f"[Env {env_idx + 1}][Instruction]: {example['instruction']}")

            example_result_dir = os.path.join(
                args.result_dir,
                args.action_space,
                args.observation_type,
                args.model,
                domain,
                example_id,
            )
            os.makedirs(example_result_dir, exist_ok=True)

            try:
                lib_run_single.run_single_example(
                    agent,
                    env,
                    example,
                    args.max_steps,
                    example["instruction"],
                    args,
                    example_result_dir,
                    shared_scores,
                )
                # Append a per-episode stats summary entry to traj.jsonl
                episode_stats = agent.get_stats()
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(json.dumps({"episode_stats": episode_stats}))
                    f.write("\n")
            except Exception as e:
                logger.exception(
                    f"Exception in Env{env_idx + 1} {domain}/{example_id}: {e}"
                )
                env.controller.end_recording(
                    os.path.join(example_result_dir, "recording.mp4")
                )
                with open(os.path.join(example_result_dir, "traj.jsonl"), "a") as f:
                    f.write(
                        json.dumps(
                            {
                                "Error": f"Exception in {domain}/{example_id}: {e}"
                            }
                        )
                    )
                    f.write("\n")

    env.close()


# ---------------------------------------------------------------------------
# Main test function
# ---------------------------------------------------------------------------

def test(args: argparse.Namespace, test_all_meta: dict) -> None:
    logger.info("Args: %s", args)

    distributed_tasks = distribute_tasks(test_all_meta, args.num_envs)

    logger.info("Setting up all environments...")
    envs: List[DesktopEnv] = []
    agents: List[ConfuciusAgent] = []

    for env_idx in range(args.num_envs):
        logger.info(f"Setting up environment {env_idx + 1}/{args.num_envs}")

        smart_ctx = SmartContextConfig(
            enabled=args.smart_context,
            compression_threshold=args.smart_context_compression_threshold,
            clear_at_least=args.smart_context_clear_at_least,
            reminder_enabled=False,
        )

        agent = ConfuciusAgent(
            model=args.model,
            observation_type=args.observation_type,
            action_space=args.action_space,
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
            smart_context_config=smart_ctx,
        )
        agents.append(agent)

        snapshot_name = args.ami_id or DEFAULT_SNAPSHOT_NAME
        if args.provider_name == "aws" and not args.ami_id:
            from desktop_env.providers.aws.manager import IMAGE_ID_MAP
            screen_size = (args.screen_width, args.screen_height)
            region_map = IMAGE_ID_MAP.get(args.region) or IMAGE_ID_MAP.get("us-east-1", {})
            snapshot_name = region_map.get(screen_size) or region_map.get((1920, 1080), DEFAULT_SNAPSHOT_NAME)

        env = DesktopEnv(
            path_to_vm=args.path_to_vm,
            action_space=agent.action_space,
            screen_size=(args.screen_width, args.screen_height),
            headless=args.headless,
            os_type="Ubuntu",
            require_a11y_tree=args.observation_type
            in ["a11y_tree", "screenshot_a11y_tree", "som"],
            provider_name=args.provider_name,
            client_password=args.client_password,
            region=args.region,
            snapshot_name=snapshot_name,
        )
        envs.append(env)

    logger.info("All environments ready. Starting parallel task execution...")

    with Manager() as manager:
        shared_scores = manager.list()

        processes = []
        for env_idx, (env, agent, env_tasks) in enumerate(
            zip(envs, agents, distributed_tasks)
        ):
            p = Process(
                target=run_env_tasks,
                args=(env_idx, env, agent, env_tasks, args, shared_scores),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        scores = list(shared_scores)

    logger.info(
        "Average score: %.4f",
        sum(scores) / len(scores) if scores else 0.0,
    )

    # ------------------------------------------------------------------
    # Aggregate run summary from per-task stats.json files
    # ------------------------------------------------------------------
    all_task_stats = []
    for env_tasks in distributed_tasks:
        for domain, example_ids in env_tasks.items():
            for example_id in example_ids:
                stats_path = os.path.join(
                    args.result_dir,
                    args.action_space,
                    args.observation_type,
                    args.model,
                    domain,
                    example_id,
                    "stats.json",
                )
                if os.path.exists(stats_path):
                    try:
                        with open(stats_path, "r", encoding="utf-8") as f:
                            all_task_stats.append(json.load(f))
                    except Exception:
                        pass

    if all_task_stats:
        total_input = sum(s.get("total_input_tokens") or 0 for s in all_task_stats)
        total_output = sum(s.get("total_output_tokens") or 0 for s in all_task_stats)
        total_cost = sum(s.get("total_cost_usd") or 0.0 for s in all_task_stats)
        total_latency = sum(s.get("total_latency_seconds") or 0.0 for s in all_task_stats)
        total_steps = sum(s.get("step_count") or 0 for s in all_task_stats)
        num_tasks = len(all_task_stats)

        has_input_tokens = any(s.get("total_input_tokens") is not None for s in all_task_stats)
        has_output_tokens = any(s.get("total_output_tokens") is not None for s in all_task_stats)
        has_total_tokens = any(s.get("total_tokens") is not None for s in all_task_stats)

        run_summary = {
            "total_tasks": num_tasks,
            "total_input_tokens": total_input if has_input_tokens else None,
            "total_output_tokens": total_output if has_output_tokens else None,
            "total_tokens": (total_input + total_output) if has_total_tokens else None,
            "total_estimated_cost_usd": round(total_cost, 6),
            "total_steps": total_steps,
            "average_steps_per_task": round(total_steps / num_tasks, 2) if num_tasks > 0 else None,
            "average_latency_per_step_seconds": round(total_latency / total_steps, 3) if total_steps > 0 else None,
        }

        summary_path = os.path.join(
            args.result_dir,
            args.action_space,
            args.observation_type,
            args.model,
            "run_summary.json",
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(run_summary, f, indent=2)
        logger.info("Run summary written to %s", summary_path)


# ---------------------------------------------------------------------------
# Helpers for skipping already-finished tasks
# ---------------------------------------------------------------------------

def get_unfinished(
    action_space: str,
    use_model: str,
    observation_type: str,
    result_dir: str,
    total_file_json: dict,
) -> dict:
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)

    if not os.path.exists(target_dir):
        return total_file_json

    finished: Dict[str, List[str]] = {}
    for domain in os.listdir(target_dir):
        finished[domain] = []
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                if example_id == "onboard":
                    continue
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    if "result.txt" not in os.listdir(example_path):
                        for file in os.listdir(example_path):
                            file_path = os.path.join(example_path, file)
                            logger.debug("Removing incomplete result file: %s", file_path)
                            os.remove(file_path)
                    else:
                        finished[domain].append(example_id)

    if not finished:
        return total_file_json

    for domain, examples in finished.items():
        if domain in total_file_json:
            total_file_json[domain] = [
                x for x in total_file_json[domain] if x not in examples
            ]

    return total_file_json


def get_result(
    action_space: str,
    use_model: str,
    observation_type: str,
    result_dir: str,
    total_file_json: dict,
) -> None:
    target_dir = os.path.join(result_dir, action_space, observation_type, use_model)
    if not os.path.exists(target_dir):
        print("New experiment, no result yet.")
        return

    all_result: List[float] = []
    for domain in os.listdir(target_dir):
        domain_path = os.path.join(target_dir, domain)
        if os.path.isdir(domain_path):
            for example_id in os.listdir(domain_path):
                example_path = os.path.join(domain_path, example_id)
                if os.path.isdir(example_path):
                    result_file = os.path.join(example_path, "result.txt")
                    if os.path.exists(result_file):
                        try:
                            all_result.append(float(open(result_file).read()))
                        except Exception:
                            all_result.append(0.0)

    if not all_result:
        print("New experiment, no result yet.")
    else:
        print(
            f"Current Success Rate: {sum(all_result) / len(all_result) * 100:.2f}%"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    args = config()

    # Save args
    path_to_args = os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        "args.json",
    )
    os.makedirs(os.path.dirname(path_to_args), exist_ok=True)
    with open(path_to_args, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=4)

    with open(args.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)

    if args.domain != "all":
        test_all_meta = {args.domain: test_all_meta[args.domain]}

    test_file_list = get_unfinished(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )

    left_info = ""
    for domain in test_file_list:
        left_info += f"{domain}: {len(test_file_list[domain])}\n"
    logger.info(f"Left tasks:\n{left_info}")

    get_result(
        args.action_space,
        args.model,
        args.observation_type,
        args.result_dir,
        test_all_meta,
    )

    test(args, test_file_list)
