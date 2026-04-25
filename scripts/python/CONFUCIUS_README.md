# Confucius Agent — OSWorld Batch Evaluation Guide

A practical guide to running the **Confucius agent** (Claude via AWS Bedrock) on the
[OSWorld](https://os-world.github.io/) benchmark.

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Script Comparison](#script-comparison)
4. [run\_batch\_osworld\_confucius.py — Primary Batch Evaluation](#run_batch_osworld_confuciuspy--primary-batch-evaluation)
   - [How Task Discovery Works](#how-task-discovery-works)
   - [Experiment Configurations](#experiment-configurations)
   - [Parameter Reference](#parameter-reference)
   - [Example Commands](#example-commands)
5. [run\_single\_confucius.py — Single-Task Runner](#run_single_confuciuspy--single-task-runner)
   - [Parameter Reference](#parameter-reference-1)
   - [Example Commands](#example-commands-1)
6. [run\_confucius.py — Full Benchmark Suite Runner](#run_confuciuspy--full-benchmark-suite-runner)
   - [Parameter Reference](#parameter-reference-2)
   - [Example Commands](#example-commands-2)
7. [Understanding Results](#understanding-results)
8. [Smart Context Compression](#smart-context-compression)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The **Confucius agent** is an AI agent backed by Anthropic's Claude models, accessed via
AWS Bedrock, and integrated into OSWorld through the `confucius/` library in this
repository. It drives a desktop virtual machine, observes screenshots and/or the
accessibility tree, and emits computer-use actions to complete tasks.

The three scripts in `scripts/python/` cover the three main evaluation workflows:

| Script | What it does |
|--------|-------------|
| `run_batch_osworld_confucius.py` | Runs every task in a domain (default: `multi_apps`), optionally with both `no_compression` and `smart_context` configs, then uploads results to GitHub |
| `run_single_confucius.py` | Runs exactly one task — ideal for debugging and quick iteration |
| `run_confucius.py` | End-to-end benchmark suite runner across all domains, with multi-environment parallelism |

---

## Prerequisites

1. **AWS credentials** — The agent uses AWS Bedrock to call Claude.
   `run_batch_osworld_confucius.py` **automatically refreshes credentials** at startup
   and **before every task** by running:
   ```
   cloud aws get-creds 009160068926 --role SSOAdmin --duration 14400
   ```
   and injecting the resulting `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
   `AWS_SESSION_TOKEN`, and `AWS_DEFAULT_REGION` variables into the process
   environment. No manual credential setup is required before running the batch script.
   Refreshing before every task ensures that 4-hour session tokens never expire
   mid-run, regardless of how many tasks the batch contains.

   If you are running in an environment that already has valid credentials (e.g., an EC2
   instance role) or have already sourced them yourself, pass
   `--skip_credential_refresh` to bypass this step entirely.

   If you want to limit how often credentials are refreshed (e.g., to reduce overhead),
   use `--credential_refresh_interval <seconds>` to only re-refresh when at least that
   many seconds have elapsed since the last refresh (e.g., `--credential_refresh_interval
   10800` for every 3 hours).

2. **Python environment** — Install all dependencies from the project root:
   ```bash
   pip install -e .
   # or
   pip install -r requirements.txt
   ```

3. **`confucius` library** — The `confucius/` package must be importable from the
   project root. If you cloned this repository it is already on the path.

4. **Evaluation examples** — The task JSON files must exist under
   `evaluation_examples/examples/<domain>/`. For batch evaluation the
   `evaluation_examples/test_all.json` index is also required.

5. **`GITHUB_TOKEN`** *(batch script only)* — A fine-grained GitHub Personal Access Token
   with **Contents read/write** permission on the target repository
   (`samuellin01/memory_experiments` by default) must be set as the `GITHUB_TOKEN`
   environment variable for results to be uploaded after each task.

   ```bash
   export GITHUB_TOKEN=<your-fine-grained-pat>
   ```

   If the variable is not set the script logs a warning and skips the upload; task runs
   are not affected. Pass `--skip_github_upload` to suppress the warning entirely.

---

## Script Comparison

| Feature | `run_batch_osworld_confucius.py` | `run_single_confucius.py` | `run_confucius.py` |
|---------|----------------------------------|---------------------------|-------------------|
| Scope | All tasks in one domain | One specific task | All tasks across all (or one) domain(s) |
| Best for | Full domain evals, CI runs | Debugging, iterating | Complete benchmark suite |
| A/B configs | ✓ (`no_compression` vs `smart_context`) | ✗ (manual flag) | ✗ (manual flag) |
| GitHub upload | ✓ automatic | ✗ | ✗ |
| Parallel envs | ✗ (sequential) | ✗ | ✓ (`--num_envs`) |
| Resume / skip finished | ✗ | ✗ | ✓ (checks `result.txt`) |
| Dry-run mode | ✓ | ✗ | ✗ |

---

## `run_batch_osworld_confucius.py` — Primary Batch Evaluation

### How Task Discovery Works

By default, the script discovers all task IDs by globbing:

```
evaluation_examples/examples/<domain>/*.json
```

Each JSON file's stem (filename without extension) is treated as a task UUID. You can
override this with `--task_ids` to run only a specific subset.

### Experiment Configurations

Two experiment configurations can be run for every task:

| Config name | `--smart_context` flag | Description |
|-------------|------------------------|-------------|
| `no_compression` | disabled | Runs the agent without any context management. The full conversation history is always in context. May hit context-window limits on long tasks. |
| `smart_context` | enabled | Activates smart context compression (see [Smart Context Compression](#smart-context-compression)). Prevents context-window overflow at the cost of some information loss. |

By default, the script runs **both** configs for every task, producing a side-by-side A/B
comparison. Pass `--configs no_compression` or `--configs smart_context` to run only one.

### Parameter Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--domain` | `multi_apps` | Domain subdirectory under `evaluation_examples/examples/` |
| `--task_ids` | *(auto-discover)* | Optional list of specific task UUIDs to run |
| `--configs` | `no_compression smart_context` | Which experiment configs to run |
| `--model` | `claude-opus-4-6` | Claude model alias |
| `--observation_type` | `screenshot_a11y_tree` | Observation type |
| `--action_space` | `claude_computer_use` | Action space |
| `--max_steps` | `150` | Maximum agent steps per task |
| `--smart_context_compression_threshold` | `50000` | Input token count that triggers compression |
| `--smart_context_clear_at_least` | `20000` | Minimum tokens freed per compression event |
| `--github_results_repo` | `samuellin01/memory_experiments` | GitHub repository to upload results to |
| `--github_results_path` | `osworld` | Path prefix in the GitHub repository for uploaded results |
| `--skip_github_upload` | `False` | Skip uploading results to GitHub after each task |
| `--task_timeout` | `7200` | Per-task subprocess timeout (seconds) |
| `--skip_credential_refresh` | `False` | Skip automatic AWS credential refresh entirely (use when credentials are already sourced or an instance role is in use) |
| `--credential_refresh_interval` | `0` | When `> 0`, only re-refresh credentials before a task if this many seconds have elapsed since the last refresh; `0` (default) = refresh before every task |
| `--result_dir` | `./results` | Local results directory |
| `--test_config_base_dir` | `evaluation_examples` | Base dir for evaluation examples |
| `--dry_run` | `False` | Print commands without executing |

### Example Commands

**Run all `multi_apps` tasks with both configs (default)**
```bash
python scripts/python/run_batch_osworld_confucius.py
```

**Run only specific task IDs**
```bash
python scripts/python/run_batch_osworld_confucius.py \
    --task_ids 00fa164e-2612-4439-992e-157d019a8436 02ce9a50-7af2-47ed-8596-af0c230501f8
```

**Run only the `no_compression` config**
```bash
python scripts/python/run_batch_osworld_confucius.py --configs no_compression
```

**Run only the `smart_context` config**
```bash
python scripts/python/run_batch_osworld_confucius.py --configs smart_context
```

**Custom smart context thresholds**
```bash
python scripts/python/run_batch_osworld_confucius.py \
    --smart_context_compression_threshold 100000 \
    --smart_context_clear_at_least 50000
```

**Dry-run — preview commands without executing**
```bash
python scripts/python/run_batch_osworld_confucius.py --dry_run
```

**Different domain (`os`)**
```bash
python scripts/python/run_batch_osworld_confucius.py --domain os
```

**Different domain (`chrome`)**
```bash
python scripts/python/run_batch_osworld_confucius.py --domain chrome
```

**Different model (`claude-sonnet-4`)**
```bash
python scripts/python/run_batch_osworld_confucius.py --model claude-sonnet-4
```

**Custom max steps**
```bash
python scripts/python/run_batch_osworld_confucius.py --max_steps 200
```

**Custom task timeout**
```bash
python scripts/python/run_batch_osworld_confucius.py --task_timeout 10800
```

**Custom GitHub upload repository and path**
```bash
python scripts/python/run_batch_osworld_confucius.py \
    --github_results_repo myorg/my-results-repo \
    --github_results_path osworld-evals-v2
```

**Skip GitHub upload (run tasks locally only)**
```bash
python scripts/python/run_batch_osworld_confucius.py --skip_github_upload
```

**Combining multiple flags — `chrome` domain, `smart_context` only, custom model and thresholds**
```bash
python scripts/python/run_batch_osworld_confucius.py \
    --domain chrome \
    --configs smart_context \
    --model claude-sonnet-4 \
    --max_steps 200 \
    --smart_context_compression_threshold 80000 \
    --smart_context_clear_at_least 40000 \
    --github_results_path osworld-chrome-evals
```

---

## `run_single_confucius.py` — Single-Task Runner

Use this script when you need to:
- Debug a specific failing task interactively
- Iterate quickly on agent or prompt changes
- Reproduce an exact result from a batch run

Both `--domain` and `--task_id` are **required**.

### Parameter Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--task_id` | **required** | UUID of the task (e.g. `b6781586-6346-41cd-935a-a6b1487918fc`) |
| `--domain` | **required** | Domain (e.g. `os`, `chrome`, `gimp`, `multi_apps`) |
| `--headless` | `False` | Run the VM in headless mode |
| `--model` | `claude-opus-4-6` | Claude model alias |
| `--observation_type` | `screenshot` | Observation type (`screenshot`, `a11y_tree`, `screenshot_a11y_tree`, `som`) |
| `--action_space` | `pyautogui` | Action space |
| `--max_steps` | `50` | Maximum agent steps |
| `--smart_context` | `False` | Enable smart context compression |
| `--smart_context_compression_threshold` | `None` | Token threshold for compression |
| `--smart_context_clear_at_least` | `None` | Minimum tokens freed per compression |
| `--provider_name` | `aws` | VM provider (`aws`, `virtualbox`, `vmware`, `docker`, `azure`) |
| `--ami_id` | `None` | Specific AMI ID (overrides auto-detection) |
| `--region` | `us-east-1` | AWS region |
| `--temperature` | `0.0` | Sampling temperature |
| `--top_p` | `0.9` | Top-p sampling |
| `--max_tokens` | `4096` | Max tokens per model response |
| `--result_dir` | `./results` | Local results directory |
| `--test_config_base_dir` | `evaluation_examples` | Base dir for evaluation examples |

### Example Commands

**Run a single OS task in headless mode**
```bash
python scripts/python/run_single_confucius.py \
    --domain os \
    --task_id b6781586-6346-41cd-935a-a6b1487918fc \
    --headless
```

**Run a Chrome task with a specific AMI**
```bash
python scripts/python/run_single_confucius.py \
    --domain chrome \
    --task_id 121ba48f-9e17-48ce-9bc6-a4fb17a7ebba \
    --ami_id ami-0123456789abcdef0 \
    --headless
```

**Run with smart context compression enabled**
```bash
python scripts/python/run_single_confucius.py \
    --domain multi_apps \
    --task_id 00fa164e-2612-4439-992e-157d019a8436 \
    --headless \
    --smart_context \
    --smart_context_compression_threshold 50000 \
    --smart_context_clear_at_least 20000
```

**Run with a different model and more steps**
```bash
python scripts/python/run_single_confucius.py \
    --domain os \
    --task_id b6781586-6346-41cd-935a-a6b1487918fc \
    --model claude-sonnet-4 \
    --max_steps 100 \
    --headless
```

**Run with screenshot + a11y tree observation**
```bash
python scripts/python/run_single_confucius.py \
    --domain gimp \
    --task_id <task-uuid> \
    --observation_type screenshot_a11y_tree \
    --action_space claude_computer_use \
    --headless
```

---

## `run_confucius.py` — Full Benchmark Suite Runner

Use this script when you need to:
- Run the complete OSWorld benchmark across all (or a filtered set of) domains
- Distribute tasks across multiple parallel VM environments
- Automatically resume an interrupted run (already-finished tasks are skipped)

Task metadata is read from `evaluation_examples/test_all.json`. After the run, a
`run_summary.json` is written with aggregate token counts, cost, and step statistics.

### Parameter Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--domain` | `all` | Domain to evaluate; `all` runs every domain in `test_all.json` |
| `--test_all_meta_path` | `evaluation_examples/test_all.json` | Path to the test metadata file |
| `--num_envs` | `1` | Number of parallel VM environments |
| `--max_steps` | `50` | Max agent steps per task |
| `--model` | `claude-opus-4-6` | Claude model alias |
| `--observation_type` | `screenshot` | Observation type |
| `--action_space` | `pyautogui` | Action space |
| `--smart_context` | `False` | Enable smart context compression |
| `--smart_context_compression_threshold` | `None` | Token threshold |
| `--smart_context_clear_at_least` | `None` | Minimum tokens freed |
| `--provider_name` | `aws` | VM provider |
| `--ami_id` | `None` | Specific AMI ID |
| `--region` | `us-east-1` | AWS region |
| `--headless` | `False` | Headless mode |
| `--result_dir` | `./results` | Local results directory |
| `--test_config_base_dir` | `evaluation_examples` | Base dir for evaluation examples |

### Example Commands

**Run the full benchmark (all domains, single environment)**
```bash
python scripts/python/run_confucius.py --headless
```

**Run only the `os` domain**
```bash
python scripts/python/run_confucius.py --domain os --headless
```

**Run with 4 parallel environments**
```bash
python scripts/python/run_confucius.py --num_envs 4 --headless
```

**Run with smart context compression on 2 parallel environments**
```bash
python scripts/python/run_confucius.py \
    --num_envs 2 \
    --smart_context \
    --smart_context_compression_threshold 50000 \
    --smart_context_clear_at_least 20000 \
    --headless
```

**Run a specific domain with a faster model**
```bash
python scripts/python/run_confucius.py \
    --domain chrome \
    --model claude-sonnet-4 \
    --max_steps 75 \
    --headless
```

---

## Understanding Results

### Directory Structure

All three scripts write results to the same directory layout:

```
<result_dir>/
└── <action_space>/
    └── <observation_type>/
        └── <model>/
            └── <domain>/
                └── <task_id>/
                    ├── result.txt       # Final task score (float, 0.0–1.0)
                    ├── traj.jsonl       # Step-by-step trajectory (one JSON per line)
                    └── stats.json       # Episode statistics (tokens, cost, latency, steps)
```

**Example path** (default settings, `multi_apps` domain):
```
./results/claude_computer_use/screenshot_a11y_tree/claude-opus-4-6/multi_apps/<task-uuid>/
```

### Output Files

| File | Contents |
|------|----------|
| `result.txt` | A single float (`0.0` to `1.0`) representing the task score |
| `traj.jsonl` | One JSON object per line; each line is an agent step or an `episode_stats` summary |
| `stats.json` | Aggregate episode stats: `step_count`, `total_input_tokens`, `total_output_tokens`, `total_cost_usd`, `total_latency_seconds` |

`run_confucius.py` additionally writes:
```
<result_dir>/<action_space>/<observation_type>/<model>/run_summary.json
```
with totals across all tasks in the run.

### Viewing Aggregate Results

To compute the overall success rate across all completed tasks, you can use the
`get_result` helper embedded in `run_confucius.py`, or read `result.txt` files directly:

```bash
# Quick success rate from the command line
python - <<'EOF'
import os, glob
result_files = glob.glob("./results/**/**/result.txt", recursive=True)
scores = [float(open(f).read().strip()) for f in result_files]
print(f"{len(scores)} tasks — avg score: {sum(scores)/len(scores)*100:.2f}%")
EOF
```

### GitHub Upload Structure (batch script)

`run_batch_osworld_confucius.py` uploads each task's result files to the GitHub
repository specified by `--github_results_repo` (default: `samuellin01/memory_experiments`).
Each file is placed at:

```
{github_results_path}/{task_id}/{config_subfolder}/{relative_file_path}
```

Where `{config_subfolder}` is:
- `no_compression` for the `no_compression` config
- `smart_context_<threshold>_<clear_at_least>` for the `smart_context` config

**Example** (default settings, `samuellin01/memory_experiments` repo):
```
osworld/<task-uuid>/no_compression/result.txt
osworld/<task-uuid>/no_compression/traj.jsonl
osworld/<task-uuid>/smart_context_50000_20000/result.txt
osworld/<task-uuid>/smart_context_50000_20000/traj.jsonl
```

Results are uploaded immediately after each task completes (not at the end of the batch),
so partial results are preserved if the run is interrupted.

Uploads require the `GITHUB_TOKEN` environment variable to be set with a fine-grained PAT
that has Contents read/write permission on the target repository. If the token is missing,
a warning is logged and the upload is skipped; the task run itself is unaffected.

---

## Smart Context Compression

### What It Is

Claude has a finite context window. On long tasks (many steps, large screenshots,
verbose a11y trees), the conversation history can grow to hundreds of thousands of
tokens and eventually exceed the model's context limit, causing the run to fail.

**Smart context compression** (`--smart_context`) monitors the running token count and,
when it exceeds a threshold, selectively removes earlier messages from the context to
free space — while preserving the most recent, most relevant context.

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--smart_context_compression_threshold` | The input token count at which compression is triggered. Lower values compress more aggressively and more often. |
| `--smart_context_clear_at_least` | Minimum number of tokens that must be freed in each compression event. This ensures compression is worthwhile each time it fires. |

**Recommended starting values** (used by `run_batch_osworld_confucius.py`):
```
compression_threshold = 50000
clear_at_least        = 20000
```

### The A/B Comparison

`run_batch_osworld_confucius.py` runs both configs by default so you can compare:

- `no_compression` — higher fidelity context, but may fail on very long tasks
- `smart_context_50000_20000` — more robust on long tasks, slight risk of losing relevant earlier context

This produces a clean A/B dataset per task, uploaded to separate GitHub subdirectories.

---

## Troubleshooting

### AWS Credentials

```
botocore.exceptions.NoCredentialsError: Unable to locate credentials
```

`run_batch_osworld_confucius.py` automatically refreshes credentials at startup and
before every task via `cloud aws get-creds`. If you see this error:

- Check that the `cloud` CLI is installed and on your `PATH`.
- Verify that the account ID (`009160068926`) and role (`SSOAdmin`) are correct for your
  environment.
- If you are running on an EC2 instance with an IAM role, pass `--skip_credential_refresh`
  to bypass the `cloud aws get-creds` step and rely on the instance role instead.
- Alternatively, set `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and
  `AWS_DEFAULT_REGION` environment variables manually and pass `--skip_credential_refresh`.

### AMI Not Found

```
AMI ami-xxx not found in region us-east-1
```

- Check that the AMI exists in the target region.
- Pass `--ami_id` explicitly to override auto-detection.
- Or pass `--region` to switch regions.

### Task Timeout

```
[TIMEOUT] run task <id> [no_compression] — exceeded 7200s
```

The default per-task timeout is 2 hours. For particularly long tasks:
```bash
python scripts/python/run_batch_osworld_confucius.py --task_timeout 14400
```

### GitHub Upload Failures

If GitHub upload warnings or errors appear in the log:

1. Check that `GITHUB_TOKEN` is set: `echo $GITHUB_TOKEN` (should print a non-empty string).
2. Verify the token has **Contents read/write** permission on the target repository
   (`samuellin01/memory_experiments` by default, or the value of `--github_results_repo`).
3. If you do not want uploads at all, pass `--skip_github_upload` to suppress both the
   upload and the warning.
4. To retry uploads for specific tasks, re-run the batch script with `--task_ids` pointing
   at those tasks — the script will re-run and re-upload them.

### Import Errors

```
ModuleNotFoundError: No module named 'confucius'
```

Run the scripts from the **project root** (the directory containing `confucius/`), not
from inside `scripts/python/`:

```bash
# Correct — run from the repo root
cd /path/to/OSWorld
python scripts/python/run_batch_osworld_confucius.py
```

### Task Config Not Found

```
Error: Task config not found: evaluation_examples/examples/os/<task-uuid>.json
```

Make sure:
- You are running from the project root.
- The `evaluation_examples/examples/<domain>/` directory contains the expected JSON files.
- `--test_config_base_dir` points to the correct directory if you use a non-default layout.
