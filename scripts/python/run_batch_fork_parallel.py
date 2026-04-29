"""Batch evaluation script for running the fork-based parallel CUA agent
(run_benchmark.py) across all tasks in a domain and uploading results to GitHub.

Example usage:

    # Run all collaborative tasks (skips tasks already in results.json):
    python scripts/python/run_batch_fork_parallel.py

    # Run multi_apps tasks:
    python scripts/python/run_batch_fork_parallel.py --domain multi_apps

    # Force re-run tasks that already have results:
    python scripts/python/run_batch_fork_parallel.py --force

    # Run only specific task IDs:
    python scripts/python/run_batch_fork_parallel.py \
        --task_ids 01b269ae-collaborative

    # Dry-run to preview commands without executing:
    python scripts/python/run_batch_fork_parallel.py --dry_run

    # Skip GitHub upload:
    python scripts/python/run_batch_fork_parallel.py --skip_github_upload
"""

import argparse
import base64
import datetime
import glob
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Import trajectory generator
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)
from trajectory_generator import generate_trajectory_html

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

datetime_str: str = datetime.datetime.now().strftime("%Y%m%d@%H%M%S")

os.makedirs("logs", exist_ok=True)

_file_handler = logging.FileHandler(
    os.path.join("logs", f"batch_fork_parallel-{datetime_str}.log"), encoding="utf-8"
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
_GITHUB_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB
_PROXY_URL = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", ""))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluation of fork-based parallel CUA agent on OSWorld tasks."
    )

    # Task / domain selection
    parser.add_argument(
        "--task-type",
        type=str,
        default="collaborative",
        choices=["standard", "collaborative"],
        help="Task type: standard or collaborative (default: collaborative)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default=None,
        help="Domain filter (e.g., chrome, multi_apps, all). If omitted, all tasks are run.",
    )
    parser.add_argument(
        "--task_ids",
        nargs="+",
        default=None,
        help="Optional list of specific task IDs to run (UUIDs). If omitted, all task IDs are discovered.",
    )

    # Agent / model config
    parser.add_argument("--model", type=str, default="claude-opus-4-6")
    parser.add_argument("--max_steps", type=int, default=100,
                        help="Max steps per agent (default: 100).")
    parser.add_argument("--num_displays", type=int, default=8,
                        help="Number of virtual displays (default: 8).")
    parser.add_argument("--temperature", type=float, default=0.7)

    # Provider config
    parser.add_argument("--provider_name", type=str, default="aws")
    parser.add_argument("--region", type=str, default="us-east-1")
    parser.add_argument("--headless", action="store_true", help="Run headless")

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
        default="fork_parallel",
        help="Config subfolder name for results in GitHub (default: fork_parallel).",
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=1,
        help="Number of trials to run per task (default: 1).",
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
        default="./batch_results",
        help="Local results directory (default: ./batch_results).",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default=None,
        help="Specific test file to use (e.g., test_all.json, test_small.json). Default: auto-detect",
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
    """Refresh AWS credentials by running `cloud aws get-creds`."""
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

    # Parse `export KEY=VALUE` lines and inject into os.environ
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

    # Write credentials to aws_credentials.json
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
    """Fetch existing results.json from GitHub for the current config_name."""
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
            # Format: {config: {task_type: {task_id: {trial_N: score}}}}
            task_type_data = config_data.get(args.task_type, {})
            return task_type_data
    except Exception as exc:
        logger.warning("Could not fetch existing results.json: %s", exc)
        return {}


def get_existing_trial_numbers(task_id: str, args: argparse.Namespace) -> list[int]:
    """Get list of existing trial numbers for a task/config on GitHub."""
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
        f"{args.github_results_path}/{args.task_type}/{task_id}/{args.config_name}"
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
    """Find the next available trial slots, filling gaps first."""
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


def discover_task_ids_from_test_file(task_type: str, test_file: str | None) -> list[str]:
    """Discover all task IDs from test JSON files."""
    # Import from run_benchmark.py
    repo_root = pathlib.Path(__file__).parent.parent.parent
    sys.path.insert(0, str(repo_root))
    from run_benchmark import load_osworld_tasks

    tasks = load_osworld_tasks(task_type=task_type, test_file=test_file)
    task_ids = [t.get("id") for t in tasks if t.get("id")]
    logger.info("Discovered %d task IDs for task_type '%s'.", len(task_ids), task_type)
    return task_ids


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def build_run_cmd(task_id: str, trial_base_dir: str, args: argparse.Namespace) -> list:
    """Build the subprocess command to run a single task via run_benchmark.py.

    Note: run_benchmark.py will create a task_{task_id} subdirectory inside trial_base_dir.
    """
    run_script_path = os.path.join(_REPO_ROOT, "run_benchmark.py")
    cmd = [
        sys.executable,
        run_script_path,
        "--task-name", task_id,
        "--task-type", args.task_type,
        "--provider-name", args.provider_name,
        "--region", args.region,
        "--model", args.model,
        "--max-steps", str(args.max_steps),
        "--output-dir", trial_base_dir,
    ]
    if args.headless:
        cmd.append("--headless")
    if args.test_file:
        cmd.extend(["--test-file", args.test_file])
    if args.domain:
        cmd.extend(["--domain", args.domain])
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
        if result.returncode == 0:
            logger.info("[COMPLETED] %s (returncode=%d)", description, result.returncode)
            return True
        logger.error("[FAILED] %s — returncode=%d", description, result.returncode)
        return False
    except subprocess.TimeoutExpired:
        logger.error("[TIMEOUT] %s — exceeded %ds", description, timeout)
        return False
    except Exception as e:
        logger.error("[ERROR] %s — %s: %s", description, type(e).__name__, e)
        return False


# ---------------------------------------------------------------------------
# GitHub upload
# ---------------------------------------------------------------------------

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
    """Upload a single task's result directory to GitHub using Git Trees API."""
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
            "[UPLOAD SKIP] Local result directory not found: %s", local_dir
        )
        return

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    if args.dry_run:
        logger.info(
            "[dry-run] Would upload %s → %s/%s/%s/%s/%s/trial_%d/",
            local_dir,
            args.github_results_repo,
            args.github_results_path,
            args.task_type,
            task_id,
            args.config_name,
            trial,
        )
        return

    # Collect files to upload
    local_path = pathlib.Path(local_dir)
    files = sorted(p for p in local_path.rglob("*") if p.is_file())

    if not files:
        logger.warning("No files to upload for task %s", task_id)
        return

    logger.info("Uploading %d file(s) for task %s via Git Trees API", len(files), task_id)

    api_base = f"{_GITHUB_API_BASE}/repos/{args.github_results_repo}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }
    branch = "main"

    # Step 1: GET the current commit SHA
    try:
        ref_data = _github_api_request(
            opener, f"{api_base}/git/ref/heads/{branch}", headers,
        )
        head_commit_sha = ref_data["object"]["sha"]
    except Exception as exc:
        logger.error("Failed to get HEAD ref for branch '%s': %s", branch, exc)
        return

    # Step 2: GET the tree SHA from that commit
    try:
        commit_data = _github_api_request(
            opener, f"{api_base}/git/commits/{head_commit_sha}", headers,
        )
        base_tree_sha = commit_data["tree"]["sha"]
    except Exception as exc:
        logger.error("Failed to get commit %s: %s", head_commit_sha, exc)
        return

    # Step 3: Create blobs for each file
    tree_items: list[dict] = []

    for file_path in files:
        file_size = file_path.stat().st_size
        if file_size > _GITHUB_MAX_FILE_BYTES:
            logger.warning(
                "Skipping %s — file too large (%d bytes > 50 MB limit)",
                file_path.name, file_size,
            )
            continue

        rel_path = file_path.relative_to(local_path)
        github_path = (
            f"{args.github_results_path}/{args.task_type}/{task_id}/"
            f"{args.config_name}/trial_{trial}/{rel_path}"
        )

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
        except urllib.error.HTTPError as exc:
            logger.error("Failed to create blob for %s: HTTP %d %s",
                        github_path, exc.code, exc.reason)
        except Exception as exc:
            logger.error("Failed to create blob for %s: %s", github_path, exc)

    if not tree_items:
        logger.warning("No blobs were created for task %s; skipping commit.", task_id)
        return

    # Steps 4-6: Create tree, commit, and update ref with retry on race
    commit_message = f"Add OSWorld fork-parallel results: {task_id}/trial_{trial}"
    max_ref_retries = 5
    for attempt in range(max_ref_retries):
        # Re-fetch HEAD and base tree on retries (need fresh base_tree for new tree)
        if attempt > 0:
            try:
                ref_data = _github_api_request(
                    opener, f"{api_base}/git/ref/heads/{branch}", headers,
                )
                head_commit_sha = ref_data["object"]["sha"]
                commit_data = _github_api_request(
                    opener, f"{api_base}/git/commits/{head_commit_sha}", headers,
                )
                base_tree_sha = commit_data["tree"]["sha"]
                logger.info("Retry %d: re-fetched HEAD (SHA: %s)", attempt, head_commit_sha[:12])
            except Exception as exc:
                logger.error("Retry %d: failed to re-fetch HEAD: %s", attempt, exc)
                return

        # Create tree (must recreate on retry with updated base_tree)
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
            if attempt == 0:
                logger.info("Created tree with %d file(s) (SHA: %s)", len(tree_items), new_tree_sha[:12])
        except Exception as exc:
            logger.error("Failed to create tree for task %s: %s", task_id, exc)
            return

        # Create commit
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

        # Update ref
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
                logger.error("Failed to update ref heads/%s: HTTP %d %s",
                           branch, exc.code, exc.reason)
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
    """Update the central results.json on GitHub with this task's score.

    Format: {config: {task_type: {task_id: {trial_N: score}}}}
    """
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

    # GET existing file
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
    except Exception as exc:
        logger.error("Failed to fetch results.json: %s", exc)
        return

    # Update data
    config_name = args.config_name
    task_type = args.task_type
    if config_name not in existing_data:
        existing_data[config_name] = {}
    if task_type not in existing_data[config_name]:
        existing_data[config_name][task_type] = {}
    if task_id not in existing_data[config_name][task_type]:
        existing_data[config_name][task_type][task_id] = {}
    # Migrate old format if needed
    if not isinstance(existing_data[config_name][task_type][task_id], dict):
        old_score = existing_data[config_name][task_type][task_id]
        existing_data[config_name][task_type][task_id] = {"trial_1": old_score}
    existing_data[config_name][task_type][task_id][f"trial_{trial}"] = score

    # PUT updated file
    new_content = json.dumps(existing_data, indent=2, sort_keys=True)
    body: dict[str, Any] = {
        "message": f"Update results.json: {task_id}/{config_name}/trial_{trial} = {score}",
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
            logger.info("Updated results.json (HTTP %d): %s/%s/trial_%d = %s",
                       resp.status, config_name, task_id, trial, score)
    except urllib.error.HTTPError as exc:
        logger.error("Failed to update results.json: HTTP %d %s", exc.code, exc.reason)
    except Exception as exc:
        logger.error("Failed to update results.json: %s", exc)


def update_latency_json_on_github(
    task_id: str,
    duration_seconds: float | None,
    args: argparse.Namespace,
    trial: int = 1,
) -> None:
    """Update latency_results.json on GitHub with this task's duration."""
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
    except Exception as exc:
        logger.error("Failed to fetch latency_results.json: %s", exc)
        return

    config_name = args.config_name
    task_type = args.task_type
    if config_name not in existing_data:
        existing_data[config_name] = {}
    if task_type not in existing_data[config_name]:
        existing_data[config_name][task_type] = {}
    if task_id not in existing_data[config_name][task_type]:
        existing_data[config_name][task_type][task_id] = {}
    if not isinstance(existing_data[config_name][task_type][task_id], dict):
        old_val = existing_data[config_name][task_type][task_id]
        existing_data[config_name][task_type][task_id] = {"trial_1": old_val}
    existing_data[config_name][task_type][task_id][f"trial_{trial}"] = duration_seconds

    new_content = json.dumps(existing_data, indent=2, sort_keys=True)
    body: dict[str, Any] = {
        "message": f"Update latency_results.json: {task_id}/{config_name}/trial_{trial} = {duration_seconds}s",
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
            logger.info("Updated latency_results.json (HTTP %d): %s/%s/trial_%d = %ss",
                       resp.status, config_name, task_id, trial, duration_seconds)
    except urllib.error.HTTPError as exc:
        logger.error("Failed to update latency_results.json: HTTP %d %s", exc.code, exc.reason)
    except Exception as exc:
        logger.error("Failed to update latency_results.json: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # AWS credential refresh
    last_credential_refresh: float = 0.0
    if args.skip_credential_refresh:
        logger.info("--skip_credential_refresh set; skipping AWS credential refresh.")
    else:
        refresh_aws_credentials()
        last_credential_refresh = time.monotonic()

    # Discover or use provided task IDs
    if args.task_ids:
        task_ids = args.task_ids
        logger.info("Using %d task IDs provided via --task_ids.", len(task_ids))
    else:
        task_ids = discover_task_ids_from_test_file(args.task_type, args.test_file)
        if args.domain and args.domain != "all":
            # Filter by domain if needed
            # For now, just log a warning that domain filtering after discovery isn't implemented yet
            logger.warning("Domain filtering via --domain not yet implemented for task discovery. "
                          "Use run_benchmark.py's --domain flag directly or filter task_ids manually.")

    if not task_ids:
        logger.error("No task IDs to process. Exiting.")
        sys.exit(1)

    # Fetch existing results to skip already-completed tasks
    existing_results: dict[str, dict] = {}
    if not args.force and not args.skip_github_upload:
        existing_results = fetch_existing_results(args)
        if existing_results:
            logger.info(
                "Found %d existing results in GitHub. Use --force to re-run them.",
                len(existing_results),
            )

    logger.info("Starting batch evaluation: %d tasks.", len(task_ids))
    if args.dry_run:
        logger.info("[DRY RUN] No commands will actually be executed.")

    # Track results
    results: dict[str, dict] = {}
    skipped: list[str] = []

    for task_idx, task_id in enumerate(task_ids, start=1):
        logger.info("=== Task %d/%d: %s ===", task_idx, len(task_ids), task_id)

        # Skip if already has a result
        if task_id in existing_results:
            existing_trials = existing_results[task_id]
            if isinstance(existing_trials, dict) and any(v is not None for v in existing_trials.values()):
                logger.info(
                    "SKIP %s — already has result(s): %s. Use --force to re-run.",
                    task_id, existing_trials,
                )
                skipped.append(task_id)
                continue

        results[task_id] = {"trials": []}

        # Determine which trial slots to use
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

            # Refresh credentials if interval elapsed
            if not args.skip_credential_refresh:
                if (
                    args.credential_refresh_interval <= 0
                    or (time.monotonic() - last_credential_refresh) >= args.credential_refresh_interval
                ):
                    refresh_aws_credentials()
                    last_credential_refresh = time.monotonic()

            # Build output directory for this trial
            # run_benchmark.py creates a "task_{task_id}" subdirectory, so pass the parent
            trial_base_dir = os.path.join(os.path.abspath(args.result_dir), f"trial_{trial_idx}")
            os.makedirs(trial_base_dir, exist_ok=True)

            # Clear cache before each trial to prevent cross-trial contamination
            cache_dir = f"cache/{task_id}"
            if os.path.exists(cache_dir):
                if not args.dry_run:
                    shutil.rmtree(cache_dir)
                    logger.info("Cleared cache for task %s (trial %d)", task_id, trial_idx)
                else:
                    logger.info("[dry-run] Would clear cache for task %s", task_id)

            # Build and run the command (run_benchmark will create task_{task_id} subdir)
            run_cmd = build_run_cmd(task_id, trial_base_dir, args)
            run_ok = run_subprocess(
                run_cmd,
                timeout=args.task_timeout,
                dry_run=args.dry_run,
                description=f"run task {task_id} trial {trial_idx}",
            )

            # The actual output is in task_{task_id} subdirectory
            trial_output_dir = os.path.join(trial_base_dir, f"task_{task_id}")

            trial_score = None
            trial_duration = None
            trial_instruction = None
            trial_error = None

            if not args.dry_run and os.path.isdir(trial_output_dir):
                # Read result.json first to check for errors
                result_json = os.path.join(trial_output_dir, "result.json")
                if os.path.isfile(result_json):
                    try:
                        with open(result_json) as fh:
                            result_data = json.load(fh)
                            trial_duration = result_data.get("duration")
                            trial_instruction = result_data.get("instruction")

                            # Check if agent failed
                            if result_data.get("status") == "failed":
                                agent_result = result_data.get("result", {})
                                if isinstance(agent_result, dict) and "error" in agent_result:
                                    error_msg = agent_result["error"]

                                    # Detect conversation history corruption bug
                                    if "tool_use" in error_msg and "tool_result" in error_msg:
                                        trial_error = "tool_result_missing"
                                        logger.error(
                                            "Task %s trial %d: Conversation history corruption detected: %s",
                                            task_id, trial_idx, error_msg
                                        )
                                        # Write error.txt for visibility
                                        error_txt = os.path.join(trial_output_dir, "error.txt")
                                        with open(error_txt, "w") as fh:
                                            fh.write(f"ERROR: {trial_error}\n{error_msg}\n")
                                    else:
                                        trial_error = "agent_failed"
                                        logger.error(
                                            "Task %s trial %d: Agent failed: %s",
                                            task_id, trial_idx, error_msg
                                        )
                    except (json.JSONDecodeError, OSError):
                        pass

                # Read score from result.txt only if no error
                if trial_error is None:
                    result_txt = os.path.join(trial_output_dir, "result.txt")
                    if os.path.isfile(result_txt):
                        try:
                            with open(result_txt) as fh:
                                trial_score = float(fh.read().strip())
                        except (ValueError, OSError):
                            pass

                # Create task.txt if instruction available
                if trial_instruction:
                    task_txt = os.path.join(trial_output_dir, "task.txt")
                    with open(task_txt, "w") as fh:
                        fh.write(trial_instruction + "\n")

                # Create token_usage.json from result.json
                if os.path.isfile(result_json):
                    try:
                        with open(result_json) as fh:
                            result_data = json.load(fh)
                            token_usage = result_data.get("token_usage", {})
                            token_usage_json = os.path.join(trial_output_dir, "token_usage.json")
                            with open(token_usage_json, "w") as fh2:
                                json.dump(token_usage, fh2, indent=2)
                    except (json.JSONDecodeError, OSError):
                        pass

            results[task_id]["trials"].append({
                "trial": trial_idx, "run": run_ok, "score": trial_score,
                "duration": trial_duration, "error": trial_error,
            })

            if not run_ok:
                logger.warning("Task %s trial %d FAILED — skipping upload.", task_id, trial_idx)
                continue

            # Generate trajectory HTML visualization
            if not args.dry_run and os.path.isdir(trial_output_dir):
                try:
                    generate_trajectory_html(
                        local_dir=trial_output_dir,
                        task_id=task_id,
                        github_repo=args.github_results_repo,
                        github_path=args.github_results_path,
                        task_type=args.task_type,
                        config_name=args.config_name,
                        trial=trial_idx,
                    )
                except Exception as exc:
                    logger.warning("Failed to generate trajectory HTML: %s", exc)

            # Upload results to GitHub
            if not args.skip_github_upload:
                upload_task_results_to_github(
                    local_dir=trial_output_dir,
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
                update_latency_json_on_github(
                    task_id=task_id,
                    duration_seconds=trial_duration,
                    args=args,
                    trial=trial_idx,
                )

            if trial_error:
                logger.warning("Task %s trial %d ERROR: %s", task_id, trial_idx, trial_error)
            else:
                score_str = f" score={trial_score}" if trial_score is not None else ""
                logger.info("Task %s trial %d COMPLETED.%s", task_id, trial_idx, score_str)

    # Summary
    logger.info("=" * 60)
    logger.info("BATCH EVALUATION SUMMARY")
    logger.info("=" * 60)

    run_success = [t for t, v in results.items() if any(trial["run"] for trial in v["trials"])]
    run_failed = [t for t, v in results.items() if v["trials"] and not any(trial["run"] for trial in v["trials"])]
    all_scores = []
    error_trials = []
    for task_id, v in results.items():
        for trial in v["trials"]:
            if trial.get("error"):
                error_trials.append((task_id, trial["trial"], trial["error"]))
            if trial["score"] is not None:
                all_scores.append(trial["score"])

    if skipped:
        logger.info("Tasks skipped: %d (already have results)", len(skipped))
    logger.info(
        "Tasks run:    %d succeeded, %d failed (out of %d total)",
        len(run_success),
        len(run_failed),
        len(results),
    )

    if error_trials:
        logger.warning("Trials with errors: %d", len(error_trials))
        for task_id, trial_num, error_type in error_trials:
            logger.warning("  - %s trial_%d: %s", task_id, trial_num, error_type)

    if all_scores:
        avg_score = sum(all_scores) / len(all_scores)
        pass_count = sum(1 for s in all_scores if s > 0)
        logger.info(
            "Scores:       %d evaluated, %d passed, avg=%.4f",
            len(all_scores),
            pass_count,
            avg_score,
        )

    if run_failed:
        logger.info("FAILED RUNS:")
        for task_id in run_failed:
            logger.info("  - %s", task_id)

    # Write batch summary JSON
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "config_name": args.config_name,
        "task_type": args.task_type,
        "model": args.model,
        "total_tasks": len(results),
        "tasks_succeeded": len(run_success),
        "tasks_failed": len(run_failed),
        "tasks_evaluated": len(all_scores),
        "tasks_passed": pass_count if all_scores else 0,
        "average_score": round(avg_score, 4) if all_scores else None,
        "results": results,
    }
    summary_path = os.path.join(os.path.abspath(args.result_dir), "batch_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Batch summary saved to %s", summary_path)

    if not run_failed:
        logger.info("All tasks completed successfully.")

    logger.info("=" * 60)

    if run_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
