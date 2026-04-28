"""Test trajectory HTML generation on existing results.

Usage:
    # Test on local directory
    python scripts/python/test_trajectory_gen.py \
        --local-dir batch_results/trial_1/task_01b269ae-collaborative \
        --task-id 01b269ae-collaborative

    # Or specify all params manually
    python scripts/python/test_trajectory_gen.py \
        --local-dir batch_results/trial_1/task_01b269ae-collaborative \
        --task-id 01b269ae-collaborative \
        --task-type collaborative \
        --config fork_parallel \
        --trial 1
"""

import argparse
import os
import sys

# Add script dir to path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from trajectory_generator import generate_trajectory_html


def main():
    parser = argparse.ArgumentParser(description="Test trajectory HTML generation")
    parser.add_argument("--local-dir", required=True, help="Local directory with task results")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument(
        "--github-repo",
        default="samuellin01/memory_experiments_3",
        help="GitHub repo for image URLs (default: samuellin01/memory_experiments_3)",
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
    parser.add_argument(
        "--trial",
        type=int,
        default=1,
        help="Trial number (default: 1)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.local_dir):
        print(f"Error: Directory not found: {args.local_dir}")
        sys.exit(1)

    print(f"Generating trajectory HTML for {args.task_id}...")
    print(f"  Local dir: {args.local_dir}")
    print(f"  GitHub: {args.github_repo}/{args.github_path}")

    generate_trajectory_html(
        local_dir=args.local_dir,
        task_id=args.task_id,
        github_repo=args.github_repo,
        github_path=args.github_path,
        task_type=args.task_type,
        config_name=args.config,
        trial=args.trial,
    )

    html_path = os.path.join(args.local_dir, "trajectory.html")
    print(f"\n✓ Generated: {html_path}")
    print(f"\nOpen in browser:")
    print(f"  file://{os.path.abspath(html_path)}")


if __name__ == "__main__":
    main()
