"""One-time setup: Create Google Workspace files for all collaborative tasks.

This script creates Google Sheets/Docs/Slides for collaborative tasks and saves
their URLs to collaborative_workspace_urls.json. After running this once, you can
run collaborative tasks infinitely without creating new files.

Usage:
    python setup_collaborative_workspace.py

The script will:
1. Load collaborative task metadata
2. Create Google Sheets/Docs/Slides for each task from templates
3. Set permissions to "anyone with link can edit"
4. Save URLs to collaborative_workspace_urls.json

You'll need:
- oauth_client_secret.json (Google OAuth credentials)
- Internet connection
"""

import argparse
import json
import logging
import os
import time

from google_workspace_oauth import (
    create_sheet_from_template_oauth,
    create_doc_from_template_oauth,
    create_slide_from_template_oauth,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pre-create Google Workspace files for collaborative tasks")
    parser.add_argument(
        "--config-file",
        default="evaluation_examples/collaborative_task_configs.json",
        help="Path to collaborative task configs"
    )
    parser.add_argument(
        "--output-file",
        default="collaborative_workspace_urls.json",
        help="Where to save Workspace file URLs"
    )
    parser.add_argument(
        "--client-secret",
        default="oauth_client_secret.json",
        help="Path to OAuth client secret"
    )
    parser.add_argument(
        "--token",
        default="oauth_token.pickle",
        help="Path to OAuth token"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Delay between sheet creations (seconds) to avoid rate limits"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate all sheets even if they already exist"
    )
    args = parser.parse_args()

    # Check for OAuth credentials
    if not os.path.exists(args.client_secret):
        logger.error(f"OAuth client secret not found: {args.client_secret}")
        logger.error("Download from Google Cloud Console: APIs & Services > Credentials")
        return

    # Load collaborative task configs
    if not os.path.exists(args.config_file):
        logger.error(f"Config file not found: {args.config_file}")
        return

    logger.info(f"Loading task configs from {args.config_file}")
    with open(args.config_file) as f:
        config_data = json.load(f)

    task_metadata = config_data.get("tasks", {})
    active_tasks = {
        task_id: meta
        for task_id, meta in task_metadata.items()
        if meta.get("status") == "active"
    }

    logger.info(f"Found {len(active_tasks)} active collaborative tasks")

    # Load existing workspace URLs if they exist
    workspace_urls = {}
    if os.path.exists(args.output_file) and not args.force:
        logger.info(f"\nLoading existing Workspace URLs from {args.output_file}")
        with open(args.output_file) as f:
            workspace_urls = json.load(f)
        logger.info(f"Found {len(workspace_urls)} existing Workspace files")
        if len(workspace_urls) == len(active_tasks):
            logger.info("\n✓ All Workspace files already created!")
            logger.info(f"URLs saved in: {args.output_file}")
            logger.info(f"\nTo recreate all files, use: --force")
            logger.info(f"\nYou can now run collaborative tasks:")
            logger.info(f"  python run_benchmark.py --task-type collaborative --num-tasks 10 ...")
            return
        logger.info(f"Will create {len(active_tasks) - len(workspace_urls)} missing files")
    elif args.force:
        logger.info(f"\n--force flag set: will recreate all Workspace files")

    # Load task details to get templates
    total = len(active_tasks)
    created_count = 0
    skipped_count = 0

    for idx, (task_id, metadata) in enumerate(active_tasks.items(), 1):
        logger.info(f"\n{'='*70}")
        logger.info(f"Task {idx}/{total}: {task_id}")
        logger.info(f"{'='*70}")

        # Load full task config
        task_file = f"evaluation_examples/examples/collaborative/{task_id}.json"
        if not os.path.exists(task_file):
            logger.warning(f"Task file not found: {task_file}, skipping")
            continue

        with open(task_file) as f:
            task_data = json.load(f)

        # Find Google Workspace config step (sheet/doc/slide)
        # Check both new format (specific.google_account.config) and old format (config)
        config_steps = task_data.get("specific", {}).get("google_account", {}).get("config", [])
        if not config_steps:
            config_steps = task_data.get("config", [])

        workspace_config = None
        workspace_type = None

        for step in config_steps:
            step_type = step.get("type")
            if step_type in ("google_sheet_from_template", "google_doc_from_template", "google_slide_from_template"):
                workspace_config = step
                workspace_type = step_type
                break

        if not workspace_config:
            logger.warning(f"No Google Workspace config found for {task_id}, skipping")
            continue

        params = workspace_config["parameters"]
        template_url = params["template_url"]
        title = params.get("title", f"OSWorld Task {task_id}")

        logger.info(f"Type: {workspace_type}")
        logger.info(f"Template: {template_url[:80]}...")
        logger.info(f"Title: {title}")

        # Check if file already exists (unless --force)
        if task_id in workspace_urls and not args.force:
            logger.info(f"⊙ Already exists: {workspace_urls[task_id]}")
            logger.info(f"   Skipping (use --force to recreate)")
            skipped_count += 1
            continue

        # Create Workspace file based on type
        try:
            if workspace_type == "google_sheet_from_template":
                file_url = create_sheet_from_template_oauth(
                    template_url=template_url,
                    client_secret_path=args.client_secret,
                    token_path=args.token,
                    title=title
                )
            elif workspace_type == "google_doc_from_template":
                file_url = create_doc_from_template_oauth(
                    template_url=template_url,
                    client_secret_path=args.client_secret,
                    token_path=args.token,
                    title=title
                )
            elif workspace_type == "google_slide_from_template":
                file_url = create_slide_from_template_oauth(
                    template_url=template_url,
                    client_secret_path=args.client_secret,
                    token_path=args.token,
                    title=title
                )
            else:
                logger.error(f"Unknown workspace type: {workspace_type}")
                continue

            workspace_urls[task_id] = file_url
            logger.info(f"✓ Created: {file_url}")
            created_count += 1

            # Save progress incrementally
            with open(args.output_file, 'w') as f:
                json.dump(workspace_urls, f, indent=2)

            # Rate limiting
            if idx < total:
                logger.info(f"Waiting {args.delay}s before next file...")
                time.sleep(args.delay)

        except Exception as e:
            logger.error(f"✗ Failed to create file for {task_id}: {e}")
            continue

    # Final summary
    logger.info(f"\n{'='*70}")
    logger.info("SETUP COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Total Workspace files: {len(workspace_urls)}/{total}")
    if created_count > 0:
        logger.info(f"  Created: {created_count}")
    if skipped_count > 0:
        logger.info(f"  Skipped (already exist): {skipped_count}")
    logger.info(f"URLs saved to: {args.output_file}")
    logger.info(f"\nYou can now run collaborative tasks infinitely:")
    logger.info(f"  python run_benchmark.py --task-type collaborative --num-tasks 10 ...")


if __name__ == "__main__":
    main()
