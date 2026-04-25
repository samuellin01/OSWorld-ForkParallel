# Collaborative Tasks - Google Workspace Variants

This directory contains **27 collaborative variants** of OSWorld tasks that use Google Workspace (Docs, Sheets, Slides) instead of single-user desktop applications (LibreOffice).

## Purpose

Test whether multi-agent systems perform better with collaborative tools (Google Sheets/Docs) vs isolated tools (LibreOffice Calc/Writer) on the same underlying tasks.

## Directory Structure

```
evaluation_examples/
├── examples/
│   └── collaborative/              # Individual task configurations (27 JSON files)
│       ├── 67890eb6-collaborative.json
│       ├── d1acdb87-collaborative.json
│       └── ...
├── collaborative_task_configs.json # Master list of all collaborative tasks
└── COLLABORATIVE_TASKS.md          # This file
```

## Task ID Convention

Collaborative task IDs follow the pattern: `{original_osworld_id}-collaborative`

Example:
- Original OSWorld task: `67890eb6-6ce5-4c00-9e3d-fb4972699b06`
- Collaborative variant: `67890eb6-collaborative`

## Running Collaborative Tasks

### Setup (One-time)

1. **OAuth credentials are required** (already in repo as `oauth_client_secret.json`)
2. **First run will prompt for login** - a browser will open asking you to authorize the app
3. **Token is saved** to `oauth_token.pickle` for future runs (no login needed again)

### Single Task

```bash
# Run a single collaborative task
python run_parallel_task.py \
  --collaborative-task-id 67890eb6-collaborative \
  --provider-name aws --region us-east-1 --headless
```

### Batch Evaluation

```bash
# Baseline (sequential) - all 27 tasks
python scripts/python/run_batch_osworld_baseline.py \
  --domain collaborative \
  --config_name baseline_collaborative \
  --skip_github_upload

# Parallel orchestrator - all 27 tasks
python scripts/python/run_batch_osworld_parallel.py \
  --domain collaborative \
  --config_name parallel_collaborative \
  --skip_github_upload

# Run specific tasks only
python scripts/python/run_batch_osworld_parallel.py \
  --domain collaborative \
  --task_ids 67890eb6-collaborative d1acdb87-collaborative \
  --skip_github_upload
```

**Each run creates a fresh Google Sheet/Doc** from the template, so no manual cleanup needed!

## Evaluation

After trial completes:
1. Check Google Sheet/Doc manually or download as .xlsx/.docx
2. Compare to OSWorld gold file
3. Calculate accuracy score
4. Compare timing to baseline

## Task Categories

**Google Sheets tasks (17):**
- ACL papers research
- Conference cities lookup
- Professor contact extraction
- Hong Kong restaurants (Google Maps)
- ArXiv LLM papers
- IMDB unseen movies
- Books reading pace
- Papers/books collection
- Researchers database
- Employee performance
- Demographic profile
- Sales analysis
- Weekly sales profit
- Invoice pivot table
- Copy revenue column
- Gross profit calculation
- Fill blank cells

**Google Docs tasks (4):**
- Excel to Doc table insertion
- Image editing from Doc requirements
- Grammar test answers
- Bubble sort coding tutorial

**Multi-app tasks (6):**
- Download course PDFs
- Huggingface papers scraping
- Debug Snake game
- Invoice reconciliation
- Video to GIF
- Image compression
