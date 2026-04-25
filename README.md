# OSWorld-Parallel

Parallel Computer Use Agents for OSWorld. This project explores speeding up desktop automation tasks by running multiple CUA agents simultaneously on isolated virtual displays.

Built on [OSWorld](https://os-world.github.io/).

## Quick Start

```bash
# Single task execution
python run_task.py \
  --task "Open the terminal and run 'echo hello world'" \
  --provider-name aws \
  --region us-east-1 \
  --headless \
  --max-steps 30 \
  --model claude-opus-4-6
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--task` | *(required)* | Task description for the agent |
| `--task-id` | *(alternative)* | UUID of a benchmark task |
| `--domain` | *(auto)* | Benchmark domain (chrome, gimp, os, etc.) |
| `--model` | `claude-opus-4-6` | Bedrock model name |
| `--max-steps` | `30` | Maximum agent steps |
| `--temperature` | `0.7` | LLM sampling temperature |
| `--observation-type` | `screenshot` | `screenshot`, `a11y_tree`, or `screenshot_a11y_tree` |
| `--provider-name` | `aws` | `aws`, `vmware`, `docker`, `podman` |
| `--headless` | `False` | Run headless |
| `--region` | `us-east-1` | AWS region |
| `--screen-width` | `1920` | Desktop width in pixels |
| `--screen-height` | `1080` | Desktop height in pixels |
| `--output-dir` | `task_results` | Directory for screenshots and logs |

## Installation

```bash
pip install -r requirements.txt
```

See [SETUP_GUIDELINE.md](SETUP_GUIDELINE.md) for provider-specific setup (AWS, Docker, VMware).

## Citation

```
@misc{OSWorld,
      title={OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments}, 
      author={Tianbao Xie and Danyang Zhang and Jixuan Chen and Xiaochuan Li and Siheng Zhao and Ruisheng Cao and Toh Jing Hua and Zhoujun Cheng and Dongchan Shin and Fangyu Lei and Yitao Liu and Yiheng Xu and Shuyan Zhou and Silvio Savarese and Caiming Xiong and Victor Zhong and Tao Yu},
      year={2024},
      eprint={2404.07972},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
