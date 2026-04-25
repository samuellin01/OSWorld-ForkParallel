"""Batch evaluation script for running the baseline AnthropicAgent
(run_baseline_task.py) across all tasks in a domain and uploading results
to GitHub.

Example usage:

    # Run all multi_apps tasks (skips tasks already in results.json):
    python scripts/python/run_batch_osworld_baseline.py

    # Run chrome tasks:
    python scripts/python/run_batch_osworld_baseline.py --domain chrome

    # Force re-run tasks that already have results:
    python scripts/python/run_batch_osworld_baseline.py --force

    # Run only specific task IDs:
    python scripts/python/run_batch_osworld_baseline.py \
        --task_ids 00fa164e-2612-4439-992e-157d019a8436

    # Dry-run to preview commands without executing:
    python scripts/python/run_batch_osworld_baseline.py --dry_run

    # Skip GitHub upload:
    python scripts/python/run_batch_osworld_baseline.py --skip_github_upload
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
from typing import Any

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

_file_handler = logging.FileHandler(
    os.path.join("logs", f"batch_run_task-{datetime_str}.log"), encoding="utf-8"
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
_PROXY_URL = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", ""))  # use env proxy if set


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of CUA agent on OSWorld tasks."
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

    # Agent / model config
    parser.add_argument("--model", type=str, default="claude-opus-4-6")
    parser.add_argument("--observation_type", type=str, default="screenshot")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)

    # Provider config
    parser.add_argument("--provider_name", type=str, default="aws")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--screen_width", type=int, default=1920)
    parser.add_argument("--screen_height", type=int, default=1080)

    # GitHub upload config
    parser.add_argument(
        "--github_results_repo",
        type=str,
        default="samuellin01/memory_experiments_3",
        help="GitHub repository to upload results to (default: samuellin01/memory_experiments_3).",
    )
    parser.add_argument(
        "--github_results_path",
        type=str,
        default="osworld",
        help="Path prefix within the GitHub repository for uploaded results (default: osworld).",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="baseline",
        help="Config subfolder name for results in GitHub (default: baseline).",
    )
    parser.add_argument(
        "--skip_github_upload",
        action="store_true",
        help="Skip uploading results to GitHub after each task.",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials to run per task (default: 1).",
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
        help="Skip automatic AWS credential refresh.",
    )
    parser.add_argument(
        "--credential_refresh_interval",
        type=int,
        default=0,
        help="Re-refresh AWS credentials if at least this many seconds have elapsed "
             "since the last refresh. 0 means refresh before every task.",
    )

    # Skip / force
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-run of tasks that already have results in GitHub.",
    )

    # Misc
    parser.add_argument(
        "--result_dir",
        type=str,
        default="./task_results",
        help="Local results directory (default: ./task_results).",
    )
    parser.add_argument(
        "--test_config_base_dir",
        type=str,
        default="evaluation_examples",
        help="Base directory for evaluation examples (default: evaluation_examples).",
    )
    parser.add_argument(
        "--credentials_file",
        type=str,
        default=None,
        help="Path to AWS credentials JSON. Defaults to aws_credentials.json in repo root.",
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
    the exported variables into the current process environment."""
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

    # Write credentials to aws_credentials.json so run_task.py can read them.
    creds_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "aws_credentials.json")
    creds_path = os.path.normpath(creds_path)
    try:
        creds = {
            "AWS_ACCESS_KEY_ID": os.environ.get("AWS_ACCESS_KEY_ID", ""),
            "AWS_SECRET_ACCESS_KEY": os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
            "AWS_SESSION_TOKEN": os.environ.get("AWS_SESSION_TOKEN", ""),
        }
        with open(creds_path, "w", encoding="utf-8") as fh:
            json.dump(creds, fh, indent=2)
        os.chmod(creds_path, 0o600)
        logger.info("AWS credentials written to %s", creds_path)
    except OSError as exc:
        logger.error("Failed to write credentials file %s: %s", creds_path, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fetch_existing_results(args: argparse.Namespace) -> dict[str, float | None]:
    """Fetch the existing results.json from GitHub and return the scores for
    the current config_name.  Returns an empty dict on any failure."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return {}

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
    github_path = f"{args.github_results_path}/results.json"
    url = f"{api_base}/{github_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with opener.open(req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            content_b64 = resp_data.get("content", "")
            data = json.loads(base64.b64decode(content_b64).decode("utf-8"))
            config_data = data.get(args.config_name, {})
            # Support both old flat format {task_id: score} and new nested
            # format {domain: {task_id: score}}.
            domain_data = config_data.get(args.domain)
            if isinstance(domain_data, dict):
                return domain_data
            # Flat format — return as-is for backward compat.
            return config_data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not fetch existing results.json: %s", exc)
        return {}


def get_existing_trial_numbers(
    task_id: str, args: argparse.Namespace
) -> list[int]:
    """Get list of existing trial numbers for a task/config on GitHub.

    Returns a list of trial numbers (e.g., [1, 3, 5] if trial_1, trial_3, trial_5 exist).
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return []

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
    github_path = (
        f"{args.github_results_path}/{args.domain}/{task_id}/{args.config_name}"
    )
    url = f"{api_base}/{github_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with opener.open(req) as resp:
            items = json.loads(resp.read().decode("utf-8"))
            trial_numbers = []
            for item in items:
                if item.get("type") == "dir" and item.get("name", "").startswith("trial_"):
                    try:
                        trial_num = int(item.get("name", "").replace("trial_", ""))
                        trial_numbers.append(trial_num)
                    except ValueError:
                        pass
            return sorted(trial_numbers)
    except Exception:
        return []


def find_next_trial_slots(existing_trials: list[int], num_new_trials: int) -> list[int]:
    """Find the next available trial slots, filling gaps first.

    Args:
        existing_trials: List of existing trial numbers (e.g., [1, 3, 5])
        num_new_trials: Number of new trial slots needed

    Returns:
        List of trial numbers to use (e.g., [2, 4] if 2 new trials needed)

    Examples:
        >>> find_next_trial_slots([1, 3], 2)
        [2, 4]
        >>> find_next_trial_slots([2], 1)
        [1]
        >>> find_next_trial_slots([], 3)
        [1, 2, 3]
    """
    if not existing_trials:
        return list(range(1, num_new_trials + 1))

    slots = []
    candidate = 1
    existing_set = set(existing_trials)

    while len(slots) < num_new_trials:
        if candidate not in existing_set:
            slots.append(candidate)
        candidate += 1

    return slots


def discover_task_ids(domain: str, test_config_base_dir: str) -> list[str]:
    """Discover all task IDs by globbing *.json files in the domain directory."""
    pattern = os.path.join(test_config_base_dir, "examples", domain, "*.json")
    json_files = sorted(glob.glob(pattern))
    if not json_files:
        logger.warning("No JSON task files found matching pattern: %s", pattern)
    task_ids = [os.path.splitext(os.path.basename(f))[0] for f in json_files]
    logger.info("Discovered %d task IDs for domain '%s'.", len(task_ids), domain)
    return task_ids


def local_result_dir(task_id: str, args: argparse.Namespace, trial: int = 1) -> str:
    """Return the expected local result directory for a task trial."""
    return os.path.join(
        os.path.abspath(args.result_dir), f"trial_{trial}", args.domain, task_id
    )


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def build_run_cmd(task_id: str, args: argparse.Namespace) -> list:
    """Build the subprocess command to run a single task."""
    run_task_path = os.path.join(_REPO_ROOT, "run_baseline_task.py")
    result_dir = os.path.abspath(args.result_dir)
    config_base_dir = os.path.abspath(args.test_config_base_dir)
    cmd = [
        sys.executable,
        run_task_path,
        "--task-id", task_id,
        "--domain", args.domain,
        "--headless",
        "--observation-type", args.observation_type,
        "--max-steps", str(args.max_steps),
        "--model", args.model,
        "--temperature", str(args.temperature),
        "--provider-name", args.provider_name,
        "--region", args.region,
        "--screen-width", str(args.screen_width),
        "--screen-height", str(args.screen_height),
        "--output-dir", result_dir,
        "--test-config-base-dir", config_base_dir,
    ]
    if args.credentials_file:
        cmd += ["--credentials-file", args.credentials_file]
    return cmd


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
            check=False,
            capture_output=False,
        )
        # run_task.py exits 0 on DONE, 1 on FAIL/INCOMPLETE — both are valid completions
        if result.returncode in (0, 1):
            logger.info("[COMPLETED] %s (returncode=%d)", description, result.returncode)
            return True
        logger.error("[FAILED] %s — returncode=%d", description, result.returncode)
        return False
    except subprocess.TimeoutExpired:
        logger.error("[TIMEOUT] %s — exceeded %ds", description, timeout)
        return False
    except Exception as e:  # noqa: BLE001
        logger.error("[ERROR] %s — %s: %s", description, type(e).__name__, e)
        return False


# ---------------------------------------------------------------------------
# Trajectory markdown generator
# ---------------------------------------------------------------------------

def generate_trajectory_md(local_dir: str, task_id: str) -> None:
    """Generate trajectory.md in the result directory with all screenshots and responses."""
    local_path = pathlib.Path(local_dir)
    if not local_path.is_dir():
        return

    # Read task instruction.
    task_txt = local_path / "task.txt"
    instruction = ""
    if task_txt.is_file():
        instruction = task_txt.read_text(encoding="utf-8", errors="replace").strip()

    # Read action log for response texts.
    action_log_path = local_path / "action_log.json"
    action_log: list[dict] = []
    if action_log_path.is_file():
        try:
            action_log = json.loads(action_log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Read result score.
    result_path = local_path / "result.txt"
    score_str = "N/A"
    if result_path.is_file():
        try:
            score_str = result_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    # Read token usage for cost and wall clock.
    token_usage_path = local_path / "token_usage.json"
    token_usage: dict = {}
    if token_usage_path.is_file():
        try:
            token_usage = json.loads(token_usage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Find step directories sorted by number.
    step_dirs = sorted(
        (d for d in local_path.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: d.name,
    )

    # Format wall clock duration.
    def fmt_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"

    lines: list[str] = []
    lines.append(f"# Task {task_id}\n")
    if instruction:
        lines.append(f"**Instruction:** {instruction}\n")
    lines.append(f"**Score:** {score_str}\n")
    wall_clock = token_usage.get("wall_clock_seconds") or token_usage.get("total_latency_seconds")
    if wall_clock is not None:
        lines.append(f"**Wall clock:** {fmt_duration(wall_clock)}\n")
    lines.append(f"**Steps:** {len(step_dirs)}\n")
    cost = token_usage.get("total_cost_usd")
    if cost is not None:
        lines.append(f"**Cost:** ${cost:.2f}\n")
    lines.append("---\n")

    for step_dir in step_dirs:
        step_name = step_dir.name  # e.g. "step_0001"
        step_num = step_name.replace("step_", "").lstrip("0") or "0"
        lines.append(f"## Step {step_num}\n")

        # Screenshot.
        screenshot = step_dir / "screenshot.png"
        if screenshot.is_file():
            rel = screenshot.relative_to(local_path)
            lines.append(f"![Step {step_num}]({rel})\n")

        # Response text.
        response_file = step_dir / "response.txt"
        if response_file.is_file():
            resp = response_file.read_text(encoding="utf-8", errors="replace").strip()
            if resp:
                lines.append(f"**Response:**\n```\n{resp}\n```\n")

        # Action from action_log.
        step_idx = int(step_num) - 1
        if 0 <= step_idx < len(action_log):
            entry = action_log[step_idx]
            actions = entry.get("actions", [])
            action_code = entry.get("action_code")
            if actions:
                actions_str = ", ".join(str(a)[:200] for a in actions)
                lines.append(f"**Actions:** `{actions_str}`\n")

        lines.append("---\n")

    md_path = local_path / "trajectory.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Generated %s (%d steps)", md_path, len(step_dirs))


# ---------------------------------------------------------------------------
# GitHub upload
# ---------------------------------------------------------------------------

def _collect_eval_artifact_files(local_dir: str, task_id: str) -> list[pathlib.Path]:
    """Find eval artifact files (actual/expected values) referenced in eval_details.json.

    The evaluator stores fetched files under ``cache/{task_id}/``.  This function
    reads eval_details.json, extracts any ``actual_value`` or ``expected_value``
    that looks like a file path, and returns the ones that exist on disk.
    """
    eval_details_path = os.path.join(local_dir, "eval_details.json")
    if not os.path.isfile(eval_details_path):
        return []

    try:
        with open(eval_details_path, "r", encoding="utf-8") as fh:
            details = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []

    artifact_files: list[pathlib.Path] = []
    for metric in details.get("metric_details", []):
        for key in ("actual_value", "expected_value"):
            val = metric.get(key)
            if not isinstance(val, str):
                continue
            # Skip values that are clearly not file paths (too long, contain braces, etc.)
            if len(val) > 500 or "{" in val:
                continue
            # Values like "cache/00fa164e-.../file.docx" are relative paths
            p = pathlib.Path(val)
            try:
                if p.is_file():
                    artifact_files.append(p)
            except OSError:
                continue
    return artifact_files


def _github_api_request(
    opener: urllib.request.OpenerDirector,
    url: str,
    headers: dict,
    method: str = "GET",
    body: bytes | None = None,
) -> dict:
    """Make a GitHub API request and return the parsed JSON response."""
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with opener.open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def upload_task_results_to_github(
    local_dir: str,
    task_id: str,
    args: argparse.Namespace,
    trial: int = 1,
) -> None:
    """Upload a single task's result directory to GitHub.

    Files under ``local_dir`` are uploaded to
    ``{github_results_path}/{task_id}/{config_name}/`` in the target repo.
    Also uploads eval artifact files (actual/expected values from eval_details.json)
    found under the ``cache/`` directory.

    All files are batched into a single commit using the Git Trees API to avoid
    409 Conflict errors that occur when many files are uploaded via the Contents API.

    Requires the ``GITHUB_TOKEN`` environment variable to be set with a fine-grained
    PAT that has Contents read/write permission on the target repository.
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

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    config_dir = args.config_name
    domain = args.domain
    trial_dir = f"trial_{trial}"

    if args.dry_run:
        logger.info(
            "[dry-run] Would upload %s → %s/%s/%s/%s/%s/%s/",
            local_dir,
            args.github_results_repo,
            args.github_results_path,
            domain,
            task_id,
            config_dir,
            trial_dir,
        )
        return

    # Collect result directory files.
    local_path = pathlib.Path(local_dir)
    files = sorted(p for p in local_path.rglob("*") if p.is_file())

    # Collect eval artifact files (actual/expected values from eval_details.json).
    eval_artifacts = _collect_eval_artifact_files(local_dir, task_id)
    if eval_artifacts:
        logger.info(
            "Found %d eval artifact file(s) to upload: %s",
            len(eval_artifacts),
            [str(p) for p in eval_artifacts],
        )

    total_files = len(files) + len(eval_artifacts)
    if total_files == 0:
        logger.warning("No files to upload for task %s", task_id)
        return

    logger.info(
        "Uploading %d result file(s) + %d eval artifact(s) for task %s via Git Trees API",
        len(files),
        len(eval_artifacts),
        task_id,
    )

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    branch = "main"

    # Step 1: GET the current commit SHA of the default branch.
    try:
        ref_data = _github_api_request(
            opener, f"{api_base}/git/ref/heads/{branch}", headers,
        )
        head_commit_sha = ref_data["object"]["sha"]
    except Exception as exc:
        logger.error("Failed to get HEAD ref for branch '%s': %s", branch, exc)
        return

    # Step 2: GET the tree SHA from that commit.
    try:
        commit_data = _github_api_request(
            opener, f"{api_base}/git/commits/{head_commit_sha}", headers,
        )
        base_tree_sha = commit_data["tree"]["sha"]
    except Exception as exc:
        logger.error("Failed to get commit %s: %s", head_commit_sha, exc)
        return

    # Step 3: Create blobs for each file.
    tree_items: list[dict] = []

    def _create_blob(file_path: pathlib.Path, github_path: str) -> bool:
        file_size = file_path.stat().st_size
        if file_size > _GITHUB_MAX_FILE_BYTES:
            logger.warning(
                "Skipping %s — file too large (%d bytes > 50 MB limit)",
                github_path, file_size,
            )
            return False
        content_b64 = base64.b64encode(file_path.read_bytes()).decode("ascii")
        blob_body = json.dumps({
            "content": content_b64,
            "encoding": "base64",
        }).encode("utf-8")
        try:
            blob_data = _github_api_request(
                opener, f"{api_base}/git/blobs", headers,
                method="POST", body=blob_body,
            )
            tree_items.append({
                "path": github_path,
                "mode": "100644",
                "type": "blob",
                "sha": blob_data["sha"],
            })
            return True
        except urllib.error.HTTPError as exc:
            logger.error("Failed to create blob for %s: HTTP %d %s", github_path, exc.code, exc.reason)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to create blob for %s: %s", github_path, exc)
            return False

    for file_path in files:
        rel_path = file_path.relative_to(local_path)
        github_path = f"{args.github_results_path}/{domain}/{task_id}/{config_dir}/{trial_dir}/{rel_path}"
        _create_blob(file_path, github_path)

    for artifact_path in eval_artifacts:
        github_path = (
            f"{args.github_results_path}/{domain}/{task_id}/{config_dir}/{trial_dir}"
            f"/eval_artifacts/{artifact_path.name}"
        )
        _create_blob(artifact_path, github_path)

    if not tree_items:
        logger.warning("No blobs were created for task %s; skipping commit.", task_id)
        return

    # Step 4: Create a new tree with all blobs.
    try:
        tree_body = json.dumps({
            "base_tree": base_tree_sha,
            "tree": tree_items,
        }).encode("utf-8")
        tree_data = _github_api_request(
            opener, f"{api_base}/git/trees", headers,
            method="POST", body=tree_body,
        )
        new_tree_sha = tree_data["sha"]
        logger.info("Created tree with %d file(s) (SHA: %s)", len(tree_items), new_tree_sha[:12])
    except Exception as exc:
        logger.error("Failed to create tree for task %s: %s", task_id, exc)
        return

    # Steps 5-6: Create commit and update ref, with retry on race condition.
    commit_message = f"Add OSWorld eval results: {task_id}/{config_dir}/{trial_dir}"
    max_ref_retries = 5
    for attempt in range(max_ref_retries):
        # Re-fetch HEAD on retries (another terminal may have pushed).
        if attempt > 0:
            try:
                ref_data = _github_api_request(
                    opener, f"{api_base}/git/ref/heads/{branch}", headers,
                )
                head_commit_sha = ref_data["object"]["sha"]
                logger.info("Retry %d: re-fetched HEAD (SHA: %s)", attempt, head_commit_sha[:12])
            except Exception as exc:
                logger.error("Retry %d: failed to re-fetch HEAD: %s", attempt, exc)
                return

        # Create commit with current HEAD as parent.
        try:
            commit_body = json.dumps({
                "message": commit_message,
                "tree": new_tree_sha,
                "parents": [head_commit_sha],
            }).encode("utf-8")
            new_commit_data = _github_api_request(
                opener, f"{api_base}/git/commits", headers,
                method="POST", body=commit_body,
            )
            new_commit_sha = new_commit_data["sha"]
            logger.info("Created commit (SHA: %s): %s", new_commit_sha[:12], commit_message)
        except Exception as exc:
            logger.error("Failed to create commit for task %s: %s", task_id, exc)
            return

        # Update ref — may fail with 422 if another terminal pushed in between.
        try:
            ref_body = json.dumps({"sha": new_commit_sha}).encode("utf-8")
            _github_api_request(
                opener, f"{api_base}/git/refs/heads/{branch}", headers,
                method="PATCH", body=ref_body,
            )
            logger.info(
                "Committed %d file(s) for task %s to refs/heads/%s",
                len(tree_items), task_id, branch,
            )
            break  # Success
        except urllib.error.HTTPError as exc:
            if exc.code == 422 and attempt < max_ref_retries - 1:
                wait = 1.0 * (attempt + 1)
                logger.warning(
                    "Ref update race (attempt %d/%d), retrying in %.0fs …",
                    attempt + 1, max_ref_retries, wait,
                )
                time.sleep(wait)
            else:
                logger.error("Failed to update ref heads/%s: HTTP %d %s", branch, exc.code, exc.reason)
                return
        except Exception as exc:
            logger.error("Failed to update ref heads/%s: %s", branch, exc)
            return


def update_results_json_on_github(
    task_id: str,
    score: float | None,
    args: argparse.Namespace,
    trial: int = 1,
) -> None:
    """Update the central results.json on GitHub with this task's score."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
    github_path = f"{args.github_results_path}/results.json"
    url = f"{api_base}/{github_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    existing_data: dict = {}
    file_sha: str | None = None
    get_req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with opener.open(get_req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            file_sha = resp_data.get("sha")
            content_b64 = resp_data.get("content", "")
            existing_data = json.loads(base64.b64decode(content_b64).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.info("results.json does not exist yet, will create it.")
        else:
            logger.error("Failed to fetch results.json: HTTP %d", exc.code)
            return
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch results.json: %s", exc)
        return

    config_dir = args.config_name
    domain = args.domain
    if config_dir not in existing_data:
        existing_data[config_dir] = {}
    if domain not in existing_data[config_dir]:
        existing_data[config_dir][domain] = {}
    if task_id not in existing_data[config_dir][domain]:
        existing_data[config_dir][domain][task_id] = {}
    if not isinstance(existing_data[config_dir][domain][task_id], dict):
        old_score = existing_data[config_dir][domain][task_id]
        existing_data[config_dir][domain][task_id] = {"trial_1": old_score}
    existing_data[config_dir][domain][task_id][f"trial_{trial}"] = score

    new_content = json.dumps(existing_data, indent=2, sort_keys=True)
    body: dict[str, Any] = {
        "message": f"Update results.json: {task_id}/{config_dir} = {score}",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("ascii"),
    }
    if file_sha:
        body["sha"] = file_sha

    put_req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="PUT",
    )
    try:
        with opener.open(put_req) as resp:
            logger.info("Updated results.json (HTTP %d): %s/%s = %s", resp.status, config_dir, task_id, score)
    except urllib.error.HTTPError as exc:
        logger.error("Failed to update results.json: HTTP %d %s", exc.code, exc.reason)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to update results.json: %s", exc)


def update_latency_json_on_github(
    task_id: str,
    wall_clock_seconds: float | None,
    args: argparse.Namespace,
    trial: int = 1,
) -> None:
    """Update latency_results.json on GitHub with this task's wall-clock time."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}/contents"
    github_path = f"{args.github_results_path}/latency_results.json"
    url = f"{api_base}/{github_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    existing_data: dict = {}
    file_sha: str | None = None
    get_req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with opener.open(get_req) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            file_sha = resp_data.get("sha")
            content_b64 = resp_data.get("content", "")
            existing_data = json.loads(base64.b64decode(content_b64).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            logger.info("latency_results.json does not exist yet, will create it.")
        else:
            logger.error("Failed to fetch latency_results.json: HTTP %d", exc.code)
            return
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch latency_results.json: %s", exc)
        return

    config_dir = args.config_name
    domain = args.domain
    if config_dir not in existing_data:
        existing_data[config_dir] = {}
    if domain not in existing_data[config_dir]:
        existing_data[config_dir][domain] = {}
    if task_id not in existing_data[config_dir][domain]:
        existing_data[config_dir][domain][task_id] = {}
    if not isinstance(existing_data[config_dir][domain][task_id], dict):
        old_val = existing_data[config_dir][domain][task_id]
        existing_data[config_dir][domain][task_id] = {"trial_1": old_val}
    existing_data[config_dir][domain][task_id][f"trial_{trial}"] = wall_clock_seconds

    new_content = json.dumps(existing_data, indent=2, sort_keys=True)
    body: dict[str, Any] = {
        "message": f"Update latency_results.json: {task_id}/{config_dir} = {wall_clock_seconds}s",
        "content": base64.b64encode(new_content.encode("utf-8")).decode("ascii"),
    }
    if file_sha:
        body["sha"] = file_sha

    put_req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="PUT",
    )
    try:
        with opener.open(put_req) as resp:
            logger.info("Updated latency_results.json (HTTP %d): %s/%s = %ss", resp.status, config_dir, task_id, wall_clock_seconds)
    except urllib.error.HTTPError as exc:
        logger.error("Failed to update latency_results.json: HTTP %d %s", exc.code, exc.reason)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to update latency_results.json: %s", exc)


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

    # Fetch existing results to skip already-completed tasks.
    existing_results: dict[str, float | None] = {}
    if not args.force and not args.skip_github_upload:
        existing_results = fetch_existing_results(args)
        if existing_results:
            logger.info(
                "Found %d existing results in GitHub. Use --force to re-run them.",
                len(existing_results),
            )

    logger.info(
        "Starting batch evaluation: %d tasks.",
        len(task_ids),
    )
    if args.dry_run:
        logger.info("[DRY RUN] No commands will actually be executed.")

    # Track results: {task_id: {"run": bool, "score": float | None}}
    results: dict[str, dict] = {}
    skipped: list[str] = []

    for task_idx, task_id in enumerate(task_ids, start=1):
        logger.info(
            "=== Task %d/%d: %s ===", task_idx, len(task_ids), task_id
        )

        # Skip if already has a result (score is not None).
        if task_id in existing_results and existing_results[task_id] is not None:
            logger.info(
                "SKIP %s — already has result (score=%s). Use --force to re-run.",
                task_id, existing_results[task_id],
            )
            skipped.append(task_id)
            continue

        results[task_id] = {"trials": []}

        # Determine which trial slots to use, filling gaps in existing trials.
        existing_trials = []
        if not args.skip_github_upload:
            existing_trials = get_existing_trial_numbers(task_id, args)

        trial_slots = find_next_trial_slots(existing_trials, args.num_trials)
        logger.info(
            "Task %s: existing trials=%s, will run trials=%s",
            task_id, existing_trials, trial_slots,
        )

        for trial_num, trial_idx in enumerate(trial_slots, start=1):
            logger.info(
                "--- Trial %d/%d (trial_%d) for task %s ---",
                trial_num, args.num_trials, trial_idx, task_id,
            )

            # Refresh credentials if interval elapsed.
            if not args.skip_credential_refresh:
                if (
                    args.credential_refresh_interval <= 0
                    or (time.monotonic() - last_credential_refresh) >= args.credential_refresh_interval
                ):
                    refresh_aws_credentials()
                    last_credential_refresh = time.monotonic()

            # Build and run the single-task command.
            trial_base = os.path.join(
                os.path.abspath(args.result_dir), f"trial_{trial_idx}"
            )
            trial_result_dir = os.path.join(trial_base, args.domain, task_id)
            run_cmd = build_run_cmd(task_id, args)
            for i, arg in enumerate(run_cmd):
                if arg == "--output-dir":
                    run_cmd[i + 1] = trial_base
                    break
            run_ok = run_subprocess(
                run_cmd,
                timeout=args.task_timeout,
                dry_run=args.dry_run,
                description=f"run task {task_id} trial {trial_idx}",
            )

            trial_score = None
            result_txt = os.path.join(trial_result_dir, "result.txt")
            if os.path.isfile(result_txt):
                try:
                    with open(result_txt) as fh:
                        trial_score = float(fh.read().strip())
                except (ValueError, OSError):
                    pass

            results[task_id]["trials"].append({
                "trial": trial_idx, "run": run_ok, "score": trial_score,
            })

            if not run_ok:
                logger.warning(
                    "Task %s trial %d FAILED — skipping upload.", task_id, trial_idx
                )
                continue

            # Generate trajectory.md before upload.
            generate_trajectory_md(trial_result_dir, task_id)

            # Upload results to GitHub.
            if not args.skip_github_upload:
                upload_task_results_to_github(
                    local_dir=trial_result_dir,
                    task_id=task_id,
                    args=args,
                    trial=trial_idx,
                )
                update_results_json_on_github(
                    task_id=task_id,
                    score=trial_score,
                    args=args,
                    trial=trial_idx,
                )
                task_wall_clock = None
                tu_path = os.path.join(trial_result_dir, "token_usage.json")
                if os.path.isfile(tu_path):
                    try:
                        with open(tu_path) as fh:
                            task_wall_clock = json.loads(fh.read()).get("wall_clock_seconds")
                    except (json.JSONDecodeError, OSError):
                        pass
                update_latency_json_on_github(
                    task_id=task_id,
                    wall_clock_seconds=task_wall_clock,
                    args=args,
                    trial=trial_idx,
                )

            score_str = f" score={trial_score}" if trial_score is not None else ""
            logger.info("Task %s trial %d COMPLETED.%s", task_id, trial_idx, score_str)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("BATCH EVALUATION SUMMARY")
    logger.info("=" * 60)

    all_trials = []
    for v in results.values():
        all_trials.extend(v.get("trials", []))
    run_success = [t for t in all_trials if t["run"]]
    run_failed = [t for t in all_trials if not t["run"]]
    scores = [t["score"] for t in all_trials if t["score"] is not None]

    if skipped:
        logger.info("Tasks skipped: %d (already have results)", len(skipped))
    logger.info(
        "Trials run:   %d succeeded, %d failed (out of %d total across %d tasks)",
        len(run_success),
        len(run_failed),
        len(all_trials),
        len(results),
    )

    if scores:
        avg_score = sum(scores) / len(scores)
        pass_count = sum(1 for s in scores if s > 0)
        logger.info(
            "Scores:       %d evaluated, %d passed, avg=%.4f",
            len(scores),
            pass_count,
            avg_score,
        )

    if run_failed:
        logger.info("FAILED RUNS:")
        for t in run_failed:
            logger.info("  - trial_%d", t["trial"])

    # Write batch summary JSON.
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config_name": args.config_name,
        "model": args.model,
        "domain": args.domain,
        "total_tasks": len(results),
        "tasks_succeeded": sum(1 for v in results.values() if any(t["run"] for t in v["trials"])),
        "tasks_failed": sum(1 for v in results.values() if not any(t["run"] for t in v["trials"])),
        "tasks_evaluated": len(scores),
        "tasks_passed": sum(1 for s in scores if s > 0) if scores else 0,
        "average_score": round(sum(scores) / len(scores), 4) if scores else None,
        "results": results,
    }
    summary_path = os.path.join(os.path.abspath(args.result_dir), args.domain, "batch_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Batch summary saved to %s", summary_path)

    if not run_failed:
        logger.info("All tasks completed successfully.")

    logger.info("=" * 60)

    # Exit with non-zero code if any task failed.
    if run_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
