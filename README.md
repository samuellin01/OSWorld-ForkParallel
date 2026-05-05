# osworld-dag-scheduler

Research fork of [OSWorld](https://github.com/xlang-ai/OSWorld) exploring **fork-parallel multi-agent execution** for computer-use tasks.

## Overview

This repository implements and evaluates a novel approach where a parent agent can **dynamically fork child agents** to complete subtasks in parallel on a single desktop environment, comparing performance against traditional single-agent baselines.

**Key differences from original OSWorld:**
- **Fork-parallel execution**: Parent agent spawns child agents that work concurrently within one VM
- **Latency optimization**: Focus on completing individual tasks faster through agent collaboration
- **Interactive trajectory visualization**: HTML-based playback of multi-agent execution timelines
- **AWS Bedrock integration**: Uses Claude Opus/Sonnet models via AWS Bedrock API

## Architecture

### Baseline (Single-Agent)
```
User Task → Single Agent → Sequential Actions → Result
```
- Traditional computer-use agent approach
- Completes task step-by-step
- Reference implementation for comparison

### Fork-Parallel (Multi-Agent)
```
User Task → Parent Agent → Fork Child Agents → Parallel Execution → Merge Results
                ├─ Child Agent 1 (subtask A)
                ├─ Child Agent 2 (subtask B)
                └─ Child Agent 3 (subtask C)
```
- Parent agent analyzes task and decides when to fork
- Child agents work on independent subtasks concurrently
- Parent merges results and completes task
- Designed for tasks with parallelizable subtasks

## Installation

### Prerequisites
- Python 3.10+
- AWS account with Bedrock access (Claude models enabled)
- AWS CLI configured with credentials

### Setup
```bash
# Clone repository
git clone https://github.com/samuellin01/osworld-dag-scheduler.git
cd osworld-dag-scheduler

# Install dependencies
pip install -e .

# Configure AWS credentials (for Bedrock API)
# Method 1: Using cloud CLI tool
cloud aws get-creds <account-id> --role <role-name> --duration 14400

# Method 2: Manual configuration
export AWS_ACCESS_KEY_ID=<your-key>
export AWS_SECRET_ACCESS_KEY=<your-secret>
export AWS_SESSION_TOKEN=<your-token>  # if using temporary credentials

# For Google Workspace tasks (optional)
# See COLLABORATIVE_SETUP.md for OAuth configuration
```

## Quick Start

### Run Single Task (Baseline)
```bash
python run_baseline_task.py \
  --task-id 030eeff7-collaborative \
  --provider-name aws \
  --region us-east-1 \
  --headless \
  --max-steps 300 \
  --model claude-opus-4-6 \
  --output-dir ./results/baseline
```

### Run Single Task (Fork-Parallel)
```bash
python run_benchmark.py \
  --task-id 030eeff7-collaborative \
  --provider-name aws \
  --region us-east-1 \
  --headless \
  --max-steps 300 \
  --model claude-opus-4-6 \
  --output-dir ./results/fork_parallel
```

### Run Batch Evaluation

**Baseline:**
```bash
python scripts/python/run_batch_osworld_baseline.py \
  --task-ids 030eeff7-collaborative 05dd4c1d-collaborative \
  --num-trials 3 \
  --provider-name aws \
  --region us-east-1
```

**Fork-Parallel:**
```bash
python scripts/python/run_batch_fork_parallel.py \
  --task-ids 030eeff7-collaborative 05dd4c1d-collaborative \
  --num-trials 3 \
  --provider-name aws \
  --region us-east-1
```

## Task Types

This fork focuses on **collaborative tasks** that benefit from parallelization:

- **Google Workspace**: Multi-cell spreadsheet operations, document editing, slides formatting
- **Web Research**: Gathering data from multiple sources simultaneously
- **Multi-file Operations**: Processing multiple documents/images in parallel
- **Chrome Settings**: Independent configuration changes

Available tasks: See `evaluation_examples/collaborative_task_configs.json`

## Results & Visualization

### Trajectory Visualization

After running a task, view the interactive execution timeline:

```bash
# Baseline trajectories
open batch_results/trial_1/task_<task-id>/trajectory.html

# Fork-parallel trajectories (shows multi-agent timeline)
open batch_results/<task-id>/fork_parallel/trial_1/trajectory.html
```

**Features:**
- ⏱️ Timeline scrubber with autoplay
- 📺 Side-by-side screenshot + agent thinking display
- 📋 Timestamped action log
- 🌈 Color-coded agent visualization (parent + up to 6 children)
- 🎬 Adjustable playback speed (1x, 2x, 4x, 8x, 16x)

### Upload Results to GitHub

Trajectories are automatically uploaded to the configured GitHub repository for easy sharing:

```bash
# Results appear at:
# https://github.com/<user>/<repo>/tree/main/osworld/<task-type>/<task-id>/<config>/trial_<N>
```

### Regenerate Trajectories

Regenerate trajectory HTML from existing results without re-running tasks:

```bash
# Regenerate specific task/trial
python scripts/python/regenerate_trajectory.py \
  --task-id 030eeff7-collaborative \
  --trial 1 \
  --config fork_parallel

# Regenerate all trials for a task
python scripts/python/regenerate_trajectory.py \
  --task-id 030eeff7-collaborative \
  --all-trials \
  --config baseline
```

## Research Metrics

Both configurations track:
- **Latency**: Agent execution time (excludes evaluation overhead)
- **Cost**: Total API usage cost (input/output/cache tokens)
- **Success Rate**: Task completion accuracy
- **Steps**: Number of actions taken
- **Token Usage**: Detailed breakdown of cache hits/writes

**Key Research Question:** Does fork-parallel execution reduce latency for tasks with parallelizable subtasks?

## Project Structure

```
osworld-dag-scheduler/
├── run_baseline_task.py              # Single-agent task runner
├── run_benchmark.py                  # Fork-parallel task runner
├── fork_agent.py                     # Fork-parallel agent implementation
├── agent_runtime.py                  # Agent process management
├── bedrock_client.py                 # AWS Bedrock API client
├── google_workspace_oauth.py         # Google Workspace integration
├── scripts/python/
│   ├── run_batch_osworld_baseline.py # Baseline batch evaluation
│   ├── run_batch_fork_parallel.py    # Fork-parallel batch evaluation
│   ├── trajectory_generator.py       # Fork-parallel HTML generator
│   ├── trajectory_generator_baseline.py # Baseline HTML generator
│   └── regenerate_trajectory.py      # Trajectory regeneration tool
├── evaluation_examples/
│   ├── examples/collaborative/       # Task definitions
│   └── collaborative_task_configs.json
├── desktop_env/                      # OSWorld environment framework
└── test_*.py                         # Unit tests
```

## Configuration

### Model Selection
- `claude-opus-4-6`: Most capable, higher cost
- `claude-sonnet-4-6`: Balanced performance/cost
- `claude-haiku-4-5`: Fastest, lowest cost

### Task Configuration
Tasks are defined in JSON format with:
- Instructions
- Evaluators (scoring functions)
- Google Workspace setup (for collaborative tasks)
- Expected outputs

See `evaluation_examples/examples/collaborative/` for examples.

## Google Workspace Setup

For tasks using Google Sheets/Docs/Slides:

1. Create OAuth 2.0 credentials (see `COLLABORATIVE_SETUP.md`)
2. Download `oauth_client_secret.json`
3. Run workspace setup tool:
   ```bash
   python setup_collaborative_workspace.py --all-tasks
   ```

## Development

### Running Tests
```bash
# Test agent runtime
python test_agent_runtime.py

# Test fork mechanism
python test_fork_agent.py

# Test autonomous forking
python test_autonomous_fork.py
```

### Adding New Tasks

1. Create task JSON in `evaluation_examples/examples/collaborative/`
2. Add entry to `collaborative_task_configs.json`
3. Test with single trial:
   ```bash
   python run_baseline_task.py --task-id <new-task-id> --headless
   ```

## Research Context

This work explores **agent parallelism** for computer-use tasks:
- **Goal**: Reduce task completion latency through concurrent agent execution
- **Approach**: Dynamic forking based on task decomposition
- **Evaluation**: Compare against single-agent baseline on identical tasks
- **Fairness**: Both configs use same prompts (efficiency goals, Google Workspace tips)

**Not included in this fork:**
- Data parallelism (running multiple tasks simultaneously)
- Original OSWorld baseline agents (AutoGLM, GPT-4V, etc.)
- Non-collaborative task domains

## Citation

If you use this work, please cite both this repository and the original OSWorld:

```bibtex
@article{xie2024osworld,
  title={OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments},
  author={Xie, Tianbao and Fan, Danyang and others},
  journal={arXiv preprint arXiv:2404.07972},
  year={2024}
}
```

## License

Apache 2.0 (inherited from original OSWorld)

## Acknowledgments

- Original [OSWorld](https://github.com/xlang-ai/OSWorld) framework by Xie et al.
- AWS Bedrock team for Claude API access
- Anthropic for Claude computer-use capabilities
