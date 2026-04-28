"""Regenerate trajectory.html for existing tasks in memory_experiments_3 repo.

Downloads task files from GitHub, generates trajectory.html, and uploads it back.
Supports both parallel (fork_parallel) and baseline configs.

Usage:
    # Regenerate for a specific task/trial (fork_parallel)
    python scripts/python/regenerate_trajectory.py \
        --task-id 01b269ae-collaborative \
        --trial 1

    # Regenerate baseline trajectory
    python scripts/python/regenerate_trajectory.py \
        --task-id 01b269ae-collaborative \
        --trial 1 \
        --config baseline

    # Regenerate for all trials of a task
    python scripts/python/regenerate_trajectory.py \
        --task-id 01b269ae-collaborative \
        --all-trials

    # Regenerate for all tasks in a task type
    python scripts/python/regenerate_trajectory.py \
        --task-type collaborative \
        --all-tasks
"""

import argparse
import base64
import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

# Add script dir to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from trajectory_generator import generate_trajectory_html
from trajectory_generator_baseline import generate_trajectory_html_baseline

_GITHUB_API_BASE = "https://api.github.com"
_PROXY_URL = os.environ.get("HTTPS_PROXY", os.environ.get("HTTP_PROXY", ""))


def _github_api_request(opener, url, headers, method="GET", body=None):
    """Make a GitHub API request and return the parsed JSON response."""
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with opener.open(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def download_task_from_github(
    task_id: str,
    trial: int,
    github_repo: str,
    github_path: str,
    task_type: str,
    config_name: str,
    temp_dir: str,
) -> str:
    """Download task files from GitHub to a temp directory.

    Returns:
        str: Path to the downloaded task directory
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    api_base = f"{_GITHUB_API_BASE}/repos/{github_repo}/contents"
    github_dir = f"{github_path}/{task_type}/{task_id}/{config_name}/trial_{trial}"
    url = f"{api_base}/{github_dir}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    print(f"Downloading from GitHub: {github_dir}")

    # Create local directory
    local_dir = os.path.join(temp_dir, f"trial_{trial}")
    os.makedirs(local_dir, exist_ok=True)

    # Recursively download all files
    def download_recursive(path, local_path):
        req_url = f"{api_base}/{path}"
        try:
            items = _github_api_request(opener, req_url, headers)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"Error: Task not found at {path}")
                sys.exit(1)
            raise

        for item in items:
            if item["type"] == "file":
                # Download file
                file_local_path = os.path.join(local_path, item["name"])
                print(f"  Downloading {item['name']}...")

                download_url = item["download_url"]
                req = urllib.request.Request(download_url)
                with opener.open(req) as resp:
                    content = resp.read()

                os.makedirs(os.path.dirname(file_local_path), exist_ok=True)
                with open(file_local_path, "wb") as f:
                    f.write(content)

            elif item["type"] == "dir":
                # Recurse into directory
                subdir_local_path = os.path.join(local_path, item["name"])
                os.makedirs(subdir_local_path, exist_ok=True)
                download_recursive(item["path"], subdir_local_path)

    download_recursive(github_dir, local_dir)
    print(f"✓ Downloaded to {local_dir}")

    # Debug: show directory structure
    print(f"\nDirectory structure:")
    for root, dirs, files in os.walk(local_dir):
        level = root.replace(local_dir, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files per dir
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")

    return local_dir


def upload_file_to_github(
    local_file_path: str,
    github_file_path: str,
    github_repo: str,
    commit_message: str,
):
    """Upload a single file to GitHub (creates or updates)."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("Error: GITHUB_TOKEN environment variable not set")
        sys.exit(1)

    if _PROXY_URL:
        proxy_handler = urllib.request.ProxyHandler(
            {"http": _PROXY_URL, "https": _PROXY_URL}
        )
        opener = urllib.request.build_opener(proxy_handler)
    else:
        opener = urllib.request.build_opener()

    api_base = f"{_GITHUB_API_BASE}/repos/{github_repo}/contents"
    url = f"{api_base}/{github_file_path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
        "Content-Type": "application/json",
    }

    # Read local file
    with open(local_file_path, "rb") as f:
        content = base64.b64encode(f.read()).decode("ascii")

    # Check if file already exists (to get SHA for update)
    file_sha = None
    try:
        existing = _github_api_request(opener, url, headers)
        file_sha = existing.get("sha")
        print(f"  File exists, will update (SHA: {file_sha[:12]})")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            print(f"  File does not exist, will create")
        else:
            raise

    # Upload file
    body = {
        "message": commit_message,
        "content": content,
    }
    if file_sha:
        body["sha"] = file_sha

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="PUT",
    )

    try:
        resp_data = _github_api_request(opener, url, headers, method="PUT", body=json.dumps(body).encode("utf-8"))
        print(f"✓ Uploaded {github_file_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to upload {github_file_path}: {e}")
        return False


def regenerate_single_task(
    task_id: str,
    trial: int,
    github_repo: str,
    github_path: str,
    task_type: str,
    config_name: str,
):
    """Regenerate trajectory.html for a single task/trial."""
    print(f"\n{'='*80}")
    print(f"Regenerating trajectory for {task_id} trial {trial}")
    print(f"{'='*80}\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Download task files from GitHub
        local_dir = download_task_from_github(
            task_id=task_id,
            trial=trial,
            github_repo=github_repo,
            github_path=github_path,
            task_type=task_type,
            config_name=config_name,
            temp_dir=temp_dir,
        )

        # Generate trajectory.html
        print(f"\nGenerating trajectory.html...")
        if config_name == "baseline":
            generate_trajectory_html_baseline(
                local_dir=local_dir,
                task_id=task_id,
                github_repo=github_repo,
                github_path=github_path,
                task_type=task_type,
                config_name=config_name,
                trial=trial,
            )
        else:
            generate_trajectory_html(
                local_dir=local_dir,
                task_id=task_id,
                github_repo=github_repo,
                github_path=github_path,
                task_type=task_type,
                config_name=config_name,
                trial=trial,
            )

        # Upload trajectory.html back to GitHub
        html_path = os.path.join(local_dir, "trajectory.html")
        if os.path.isfile(html_path):
            print(f"\nUploading trajectory.html to GitHub...")
            github_file_path = (
                f"{github_path}/{task_type}/{task_id}/{config_name}/trial_{trial}/trajectory.html"
            )
            upload_file_to_github(
                local_file_path=html_path,
                github_file_path=github_file_path,
                github_repo=github_repo,
                commit_message=f"Regenerate trajectory.html for {task_id}/trial_{trial}",
            )

            # Print GitHub URL
            view_url = (
                f"https://github.com/{github_repo}/blob/main/"
                f"{github_path}/{task_type}/{task_id}/{config_name}/trial_{trial}/trajectory.html"
            )
            print(f"\n✓ View on GitHub: {view_url}")
        else:
            print(f"✗ trajectory.html not generated")


def get_all_tasks_from_github(
    github_repo: str,
    github_path: str,
    task_type: str,
) -> list[str]:
    """Get list of all task IDs for a task type from GitHub."""
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

    api_base = f"{_GITHUB_API_BASE}/repos/{github_repo}/contents"
    url = f"{api_base}/{github_path}/{task_type}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        items = _github_api_request(opener, url, headers)
        task_ids = [item["name"] for item in items if item["type"] == "dir"]
        return sorted(task_ids)
    except Exception as e:
        print(f"Error fetching task list: {e}")
        return []


def get_trials_for_task(
    task_id: str,
    github_repo: str,
    github_path: str,
    task_type: str,
    config_name: str,
) -> list[int]:
    """Get list of trial numbers for a task from GitHub."""
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

    api_base = f"{_GITHUB_API_BASE}/repos/{github_repo}/contents"
    url = f"{api_base}/{github_path}/{task_type}/{task_id}/{config_name}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json",
    }

    try:
        items = _github_api_request(opener, url, headers)
        trials = []
        for item in items:
            if item["type"] == "dir" and item["name"].startswith("trial_"):
                try:
                    trial_num = int(item["name"].replace("trial_", ""))
                    trials.append(trial_num)
                except ValueError:
                    pass
        return sorted(trials)
    except Exception as e:
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate trajectory.html for existing tasks in memory_experiments_3"
    )

    parser.add_argument("--task-id", help="Task ID to regenerate")
    parser.add_argument("--trial", type=int, help="Trial number (default: 1)")
    parser.add_argument("--all-trials", action="store_true", help="Regenerate all trials for task")
    parser.add_argument("--all-tasks", action="store_true", help="Regenerate all tasks in task type")

    parser.add_argument(
        "--github-repo",
        default="samuellin01/memory_experiments_3",
        help="GitHub repo (default: samuellin01/memory_experiments_3)",
    )
    parser.add_argument(
        "--github-path",
        default="osworld",
        help="Path prefix in repo (default: osworld)",
    )
    parser.add_argument(
        "--task-type",
        default="collaborative",
        help="Task type (default: collaborative)",
    )
    parser.add_argument(
        "--config",
        default="fork_parallel",
        help="Config name (default: fork_parallel)",
    )

    args = parser.parse_args()

    # Validate GITHUB_TOKEN
    if not os.environ.get("GITHUB_TOKEN"):
        print("Error: GITHUB_TOKEN environment variable not set")
        print("Set it with: export GITHUB_TOKEN=ghp_...")
        sys.exit(1)

    if args.all_tasks:
        # Regenerate all tasks
        task_ids = get_all_tasks_from_github(
            github_repo=args.github_repo,
            github_path=args.github_path,
            task_type=args.task_type,
        )
        if not task_ids:
            print(f"No tasks found for {args.task_type}")
            sys.exit(1)

        print(f"Found {len(task_ids)} tasks for {args.task_type}")
        for task_id in task_ids:
            trials = get_trials_for_task(
                task_id=task_id,
                github_repo=args.github_repo,
                github_path=args.github_path,
                task_type=args.task_type,
                config_name=args.config,
            )
            for trial in trials:
                try:
                    regenerate_single_task(
                        task_id=task_id,
                        trial=trial,
                        github_repo=args.github_repo,
                        github_path=args.github_path,
                        task_type=args.task_type,
                        config_name=args.config,
                    )
                except Exception as e:
                    print(f"✗ Failed to regenerate {task_id}/trial_{trial}: {e}")

    elif args.all_trials:
        # Regenerate all trials for a task
        if not args.task_id:
            print("Error: --task-id required with --all-trials")
            sys.exit(1)

        trials = get_trials_for_task(
            task_id=args.task_id,
            github_repo=args.github_repo,
            github_path=args.github_path,
            task_type=args.task_type,
            config_name=args.config,
        )

        if not trials:
            print(f"No trials found for {args.task_id}")
            sys.exit(1)

        print(f"Found {len(trials)} trials for {args.task_id}: {trials}")
        for trial in trials:
            try:
                regenerate_single_task(
                    task_id=args.task_id,
                    trial=trial,
                    github_repo=args.github_repo,
                    github_path=args.github_path,
                    task_type=args.task_type,
                    config_name=args.config,
                )
            except Exception as e:
                print(f"✗ Failed to regenerate trial {trial}: {e}")

    else:
        # Regenerate single task/trial
        if not args.task_id:
            print("Error: --task-id required")
            sys.exit(1)

        trial = args.trial if args.trial else 1
        regenerate_single_task(
            task_id=args.task_id,
            trial=trial,
            github_repo=args.github_repo,
            github_path=args.github_path,
            task_type=args.task_type,
            config_name=args.config,
        )


if __name__ == "__main__":
    main()
