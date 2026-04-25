"""Batch evaluation script for running the Confucius agent across all multi_apps
tasks in the OSWorld benchmark and uploading results to GitHub.

Example usage:

    # Run all multi_apps tasks with both configurations:
    python scripts/python/run_batch_osworld_confucius.py

    # Run only specific task IDs:
    python scripts/python/run_batch_osworld_confucius.py \
        --task_ids 00fa164e-2612-4439-992e-157d019a8436 02ce9a50-7af2-47ed-8596-af0c230501f8

    # Run only the no_compression config:
    python scripts/python/run_batch_osworld_confucius.py --configs no_compression

    # Dry-run to preview commands without executing:
    python scripts/python/run_batch_osworld_confucius.py --dry_run

    # Custom thresholds:
    python scripts/python/run_batch_osworld_confucius.py \
        --smart_context_compression_threshold 100000 \
        --smart_context_clear_at_least 50000
"""

import argparse
import base64
import datetime
import glob
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

_file_handler = logging.FileHandler(
    os.path.join("logs", f"batch_osworld_confucius-{datetime_str}.log"), encoding="utf-8"
)
_stdout_handler = logging.StreamHandler(sys.stdout)

_file_handler.setLevel(logging.DEBUG)
_stdout_handler.setLevel(logging.INFO)

_formatter = logging.Formatter(
    fmt="[%(asctime)s %(levelname)s %(module)s/%(lineno)d] %(message)s"
)
_file_handler.setFormatter(_formatter)
_stdout_handler.setFormatter(_formatter)

logger.addHandler(_file_handler)
logger.addHandler(_stdout_handler)


# ---------------------------------------------------------------------------
# GitHub upload constants
# ---------------------------------------------------------------------------

_GITHUB_API_BASE = "https://api.github.com"
_GITHUB_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB — GitHub Contents API limit
_PROXY_URL = "http://127.0.0.1:18080"  # devvm proxy


# ---------------------------------------------------------------------------
# CLI config
# ---------------------------------------------------------------------------

CONFIG_NO_COMPRESSION = "no_compression"
CONFIG_SMART_CONTEXT = "smart_context"
ALL_CONFIGS = [CONFIG_NO_COMPRESSION, CONFIG_SMART_CONTEXT]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of Confucius agent on OSWorld multi_apps tasks."
    )

    # Task / domain selection
    parser.add_argument(
        "--domain",
        type=str,
        default="multi_apps",
        help="Domain subdirectory under evaluation_examples/examples/ (default: multi_apps)",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,
        help="Optional list of specific task IDs to run. If omitted, all task IDs in the "
             "domain directory are discovered automatically.",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=ALL_CONFIGS,
        default=ALL_CONFIGS,
        help="Experiment configurations to run (default: both no_compression and smart_context).",
    )

    # Agent / model config
    parser.add_argument("--model", type=str, default="claude-opus-4-6")
    parser.add_argument("--observation_type", type=str, default="screenshot_a11y_tree")
    parser.add_argument("--action_space", type=str, default="claude_computer_use")
    parser.add_argument("--max_steps", type=int, default=150)

    # Smart context thresholds
    parser.add_argument(
        "--smart_context_compression_threshold",
        type=int,
        default=50000,
        help="Input token count that triggers smart context compression (default: 50000).",
    )
    parser.add_argument(
        "--smart_context_clear_at_least",
        type=int,
        default=20000,
        help="Minimum tokens each smart context compression must free (default: 20000).",
    )

    # GitHub upload config
    parser.add_argument(
        "--github_results_repo",
        type=str,
        default="samuellin01/memory_experiments",
        help="GitHub repository to upload results to (default: samuellin01/memory_experiments).",
    )
    parser.add_argument(
        "--github_results_path",
        type=str,
        default="osworld",
        help="Path prefix within the GitHub repository for uploaded results (default: osworld).",
    )
    parser.add_argument(
        "--skip_github_upload",
        action="store_true",
        help="Skip uploading results to GitHub after each task.",
    )

    # Timeout config
    parser.add_argument(
        "--task_timeout",
        type=int,
        default=7200,
        help="Timeout in seconds for each task execution subprocess (default: 7200).",
    )

    # AWS credential refresh
    parser.add_argument(
        "--skip_credential_refresh",
        action="store_true",
        help="Skip automatic AWS credential refresh (useful when credentials are already "
             "sourced or an instance role is in use).",
    )
    parser.add_argument(
        "--credential_refresh_interval",
        type=int,
        default=0,
        help="Re-refresh AWS credentials before each config run if at least this many seconds "
             "have elapsed since the last refresh. 0 means refresh only at startup. "
             "Recommended: 10800 (3 hours) for long runs, since credentials are valid for "
             "4 hours (14400s).",
    )

    # Misc
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./results",
        help="Local results directory (default: ./results).",
    )
    parser.add_argument(
        "--test_config_base_dir",
        type=str,
        default="evaluation_examples",
        help="Base directory for evaluation examples (default: evaluation_examples).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands that would be run without actually executing them.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# AWS credential refresh
# ---------------------------------------------------------------------------

def refresh_aws_credentials() -> None:
    """Refresh AWS credentials by running `cloud aws get-creds` and injecting
    the exported variables into the current process environment so that all
    subsequent subprocesses inherit them.

    Raises SystemExit if the command fails or is not found.
    """
    cmd = [
        "cloud", "aws", "get-creds", "009160068926",
        "--role", "SSOAdmin",
        "--duration", "14400",
    ]
    logger.info("Refreshing AWS credentials: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        logger.error(
            "AWS credential refresh failed: 'cloud' CLI not found on PATH. "
            "Install it or pass --skip_credential_refresh if credentials are already set."
        )
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        logger.error(
            "AWS credential refresh command failed (returncode=%d).\n"
            "stdout: %s\nstderr: %s",
            e.returncode, e.stdout, e.stderr,
        )
        sys.exit(1)

    # Parse `export KEY=VALUE` lines and inject into os.environ.
    refreshed: list[str] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        rest = line[len("export "):]
        if "=" not in rest:
            continue
        key, _, value = rest.partition("=")
        key = key.strip()
        value = value.strip()
        if key:
            os.environ[key] = value
            refreshed.append(key)

    if not refreshed:
        logger.warning(
            "AWS credential refresh command succeeded but no 'export KEY=VALUE' lines "
            "were found in its output. Credentials may not have been updated."
        )
    else:
        logger.info(
            "AWS credentials refreshed successfully. Updated variables: %s",
            ", ".join(refreshed),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_task_ids(domain: str, test_config_base_dir: str) -> list[str]:
    """Discover all task IDs by globbing *.json files in the domain directory."""
    pattern = os.path.join(test_config_base_dir, "examples", domain, "*.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        logger.warning("No JSON task files found matching pattern: %s", pattern)
    task_ids = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
    logger.info("Discovered %d task IDs for domain '%s'.", len(task_ids), domain)
    return task_ids


def config_subfolder(config_name: str, threshold: int, clear_at_least: int) -> str:
    """Return the config subfolder name for a given config."""
    if config_name == CONFIG_NO_COMPRESSION:
        return "no_compression"
    return f"smart_context_{threshold}_{clear_at_least}"


def build_run_cmd(task_id: str, config_name: str, args: argparse.Namespace) -> list:
    """Build the subprocess command to run a single task."""
    cmd = [
        sys.executable,
        "scripts/python/run_single_confucius.py",
        "--domain", args.domain,
        "--task_id", task_id,
        "--headless",
        "--observation_type", args.observation_type,
        "--max_steps", str(args.max_steps),
        "--action_space", args.action_space,
        "--model", args.model,
        "--result_dir", args.result_dir,
        "--test_config_base_dir", args.test_config_base_dir,
    ]
    if config_name == CONFIG_SMART_CONTEXT:
        cmd += [
            "--smart_context",
            "--smart_context_compression_threshold",
            str(args.smart_context_compression_threshold),
            "--smart_context_clear_at_least",
            str(args.smart_context_clear_at_least),
        ]
    return cmd


def local_result_dir(task_id: str, args: argparse.Namespace) -> str:
    """Return the expected local result directory for a task."""
    return os.path.join(
        args.result_dir,
        args.action_space,
        args.observation_type,
        args.model,
        args.domain,
        task_id,
    )


def run_subprocess(cmd: list, timeout: int, dry_run: bool, description: str) -> bool:
    """Run a subprocess command. Returns True on success, False on failure."""
    logger.info("[CMD] %s: %s", description, " ".join(cmd))
    if dry_run:
        logger.info("[DRY RUN] Would execute: %s", " ".join(cmd))
        return True
    try:
        result = subprocess.run(
            cmd,
            timeout=timeout,
            check=True,
            capture_output=False,
        )
        logger.info("[SUCCESS] %s (returncode=%d)", description, result.returncode)
        return True
    except subprocess.CalledProcessError as e:
        logger.error("[FAILED] %s — returncode=%d", description, e.returncode)
        return False
    except subprocess.TimeoutExpired:
        logger.error("[TIMEOUT] %s — exceeded %ds", description, timeout)
        return False
    except Exception as e:  # noqa: BLE001
        logger.error("[ERROR] %s — %s: %s", description, type(e).__name__, e)
        return False


def upload_task_results_to_github(
    local_dir: str,
    task_id: str,
    config_name: str,
    args: argparse.Namespace,
) -> None:
    """Upload a single task's result directory to GitHub immediately after it completes.

    Files under ``local_dir`` are uploaded to
    ``{github_results_path}/{task_id}/{config_subfolder}/`` in the target repo,
    so that results are grouped by task first and then by config.

    Each file is uploaded as a separate commit via the GitHub Contents API PUT endpoint.
    For a typical task with a few result files this produces a small number of commits.

    Requires the ``GITHUB_TOKEN`` environment variable to be set with a fine-grained
    PAT that has Contents read/write permission on the target repository.

    Requests are routed through the proxy at ``_PROXY_URL`` as required on the devvm.
    Individual file upload failures are logged as errors but do not abort the remaining
    uploads or the overall batch run.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        logger.warning(
            "GITHUB_TOKEN is not set; skipping GitHub upload. "
            "Set GITHUB_TOKEN with Contents read/write permission on %s.",
            args.github_results_repo,
        )
        return

    if not args.dry_run and not os.path.isdir(local_dir):
        logger.warning(
            "[UPLOAD SKIP] Local result directory not found (task may have crashed "
            "before writing output): %s",
            local_dir,
        )
        return

    proxy_handler = urllib.request.ProxyHandler(
        {"http": _PROXY_URL, "https": _PROXY_URL}
    )
    opener = urllib.request.build_opener(proxy_handler)

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    config_dir = config_subfolder(
        config_name,
        args.smart_context_compression_threshold,
        args.smart_context_clear_at_least,
    )

    if args.dry_run:
        logger.info(
            "[dry-run] Would upload %s → %s/%s/%s/%s/",
            local_dir,
            args.github_results_repo,
            args.github_results_path,
            task_id,
            config_dir,
        )
        return

    local_path = pathlib.Path(local_dir)
    files = sorted(p for p in local_path.rglob("*") if p.is_file())
    logger.info(
        "Uploading %d file(s) for task %s [%s] to %s/%s/%s/%s/",
        len(files),
        task_id,
        config_name,
        args.github_results_repo,
        args.github_results_path,
        task_id,
        config_dir,
    )

    for file_path in files:
        rel_path = file_path.relative_to(local_path)
        github_path = f"{args.github_results_path}/{task_id}/{config_dir}/{rel_path}"

        file_size = file_path.stat().st_size
        if file_size > _GITHUB_MAX_FILE_BYTES:
            logger.warning(
                "Skipping %s — file too large (%d bytes > 50 MB limit)",
                rel_path,
                file_size,
            )
            continue

        content_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        body = json.dumps(
            {
                "message": f"Add OSWorld eval result: {task_id}/{config_dir}/{rel_path}",
                "content": content_b64,
            }
        ).encode("utf-8")

        url = f"{api_base}/{github_path}"
        req = urllib.request.Request(url, data=body, headers=headers, method="PUT")
        try:
            with opener.open(req) as resp:
                status = resp.status
            logger.info("Uploaded %s (HTTP %d)", github_path, status)
        except urllib.error.HTTPError as exc:
            logger.error("Failed to upload %s: HTTP %d %s", github_path, exc.code, exc.reason)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to upload %s: %s", github_path, exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # AWS credential refresh (at startup, unless skipped)
    # ------------------------------------------------------------------
    last_credential_refresh: float = 0.0
    if args.skip_credential_refresh:
        logger.info(
            "--skip_credential_refresh set; skipping automatic AWS credential refresh."
        )
    else:
        refresh_aws_credentials()
        last_credential_refresh = time.monotonic()

    # Discover or use provided task IDs.
    if args.task_ids:
        task_ids = args.task_ids
        logger.info("Using %d task IDs provided via --task_ids.", len(task_ids))
    else:
        task_ids = discover_task_ids(args.domain, args.test_config_base_dir)

    if not task_ids:
        logger.error("No task IDs to process. Exiting.")
        sys.exit(1)

    logger.info(
        "Starting batch evaluation: %d tasks × %d config(s) = %d runs.",
        len(task_ids),
        len(args.configs),
        len(task_ids) * len(args.configs),
    )
    if args.dry_run:
        logger.info("[DRY RUN] No commands will actually be executed.")

    # Track results: {(task_id, config_name): {"run": bool}}
    results: dict[tuple[str, str], dict[str, bool]] = {}

    for task_idx, task_id in enumerate(task_ids, start=1):
        logger.info(
            "=== Task %d/%d: %s ===", task_idx, len(task_ids), task_id
        )
        for config_name in args.configs:
            key = (task_id, config_name)
            results[key] = {"run": False}

            logger.info("--- Config: %s ---", config_name)

            # Refresh credentials before every task so long-running batches never
            # encounter expired session tokens (credentials are valid for ~4 hours).
            # An interval-based check is also supported as an alternative: if
            # --credential_refresh_interval is set to a positive value, only refresh
            # when that many seconds have elapsed since the last refresh.
            if not args.skip_credential_refresh:
                if (
                    args.credential_refresh_interval <= 0
                    or (time.monotonic() - last_credential_refresh) >= args.credential_refresh_interval
                ):
                    refresh_aws_credentials()
                    last_credential_refresh = time.monotonic()

            # Build and run the single-task command.
            run_cmd = build_run_cmd(task_id, config_name, args)
            run_ok = run_subprocess(
                run_cmd,
                timeout=args.task_timeout,
                dry_run=args.dry_run,
                description=f"run task {task_id} [{config_name}]",
            )
            results[key]["run"] = run_ok

            if not run_ok:
                logger.warning(
                    "Task %s [%s] run FAILED — skipping upload.", task_id, config_name
                )
                continue

            # Upload results to GitHub.
            if not args.skip_github_upload:
                upload_task_results_to_github(
                    local_dir=local_result_dir(task_id, args),
                    task_id=task_id,
                    config_name=config_name,
                    args=args,
                )

            logger.info("Task %s [%s] run SUCCESS.", task_id, config_name)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BATCH EVALUATION SUMMARY")
    logger.info("=" * 60)

    run_success = [(t, c) for (t, c), v in results.items() if v["run"]]
    run_failed = [(t, c) for (t, c), v in results.items() if not v["run"]]

    logger.info(
        "Tasks run:    %d succeeded, %d failed (out of %d total)",
        len(run_success),
        len(run_failed),
        len(results),
    )

    if run_failed:
        logger.info("FAILED RUNS:")
        for task_id, config_name in run_failed:
            logger.info("  - %s [%s]", task_id, config_name)

    if not run_failed:
        logger.info("All tasks completed successfully.")

    logger.info("=" * 60)

    # Exit with non-zero code if any task failed.
    if run_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
