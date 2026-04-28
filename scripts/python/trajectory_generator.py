"""Generate interactive trajectory HTML visualization for fork-based parallel agent execution."""

import html as html_mod
import json
import pathlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def parse_bedrock_api_calls(local_path: pathlib.Path, agent_dirs: List[Tuple[str, pathlib.Path]]) -> List[Dict]:
    """Parse bedrock_api_calls.jsonl from root and all agent directories."""
    calls = []

    # Read from root directory
    api_calls_path = local_path / "bedrock_api_calls.jsonl"
    if api_calls_path.is_file():
        try:
            with open(api_calls_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        calls.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass

    # Also read from each agent directory
    for agent_id, agent_dir in agent_dirs:
        agent_api_calls_path = agent_dir / "bedrock_api_calls.jsonl"
        if agent_api_calls_path.is_file():
            try:
                with open(agent_api_calls_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            calls.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                pass

    return calls


def build_timeline_from_api_calls(api_calls: List[Dict], agent_dirs: List[Tuple[str, pathlib.Path]]) -> Dict:
    """Build timeline data from bedrock API calls with actual timestamps."""

    # Track steps per agent
    agent_steps_map = {}  # agent_id -> {step_num -> {'timestamp': ..., 'action': ...}}
    agent_step_counters = {}  # agent_id -> current step number

    start_time = None
    end_time = None

    for call in api_calls:
        # Get timestamps
        request_ts_str = call.get('request_timestamp')
        response_ts_str = call.get('response_timestamp')

        if not response_ts_str:
            continue

        # Parse timestamps (use request time for step timing, response time for duration)
        try:
            response_ts = datetime.fromisoformat(response_ts_str.replace('Z', '+00:00'))
            request_ts = datetime.fromisoformat(request_ts_str.replace('Z', '+00:00')) if request_ts_str else response_ts
        except:
            continue

        if start_time is None:
            start_time = request_ts
        end_time = response_ts

        # Use request timestamp for step timing (when action was invoked)
        relative_time = (request_ts - start_time).total_seconds() if start_time else 0

        # Get agent ID directly from log entry (if available)
        agent_id = call.get('agent_id')

        # Fallback: Infer from system prompt if agent_id not in log
        if not agent_id:
            request = call.get('request', {})
            system_prompt = request.get('system_prompt', '')

            # Worker agents have "You are a worker agent" in system prompt
            if 'You are a worker agent' in system_prompt:
                # Look for display number pattern
                display_match = re.search(r'chrome_display_(\d+)', system_prompt)
                if display_match:
                    display_num = int(display_match.group(1))
                    if display_num == 0:
                        agent_id = 'root'
                    else:
                        # Display 2 = root_child_0, display 3 = root_child_1, etc.
                        agent_id = f'root_child_{display_num - 2}'
                else:
                    agent_id = 'root_child_0'
            else:
                agent_id = 'root'

        # Initialize agent entry
        if agent_id not in agent_step_counters:
            agent_step_counters[agent_id] = 0
        if agent_id not in agent_steps_map:
            agent_steps_map[agent_id] = {}

        # Extract all actions from response content blocks
        response = call.get('response', {})
        content_blocks = response.get('content_blocks', [])

        # Collect all tool uses and text blocks
        actions_in_response = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue

            # Text block
            if block.get('type') == 'text':
                text = block.get('text', '').strip()
                if text:
                    actions_in_response.append({
                        'action': text,
                        'tool': '',
                    })

            # Tool use block
            elif block.get('type') == 'tool_use':
                tool_name = block.get('name', '')
                tool_input = block.get('input', {})
                action_detail = ""

                if tool_name == 'computer':
                    action = tool_input.get('action', '')
                    if action == 'key':
                        action_detail = f"Key: {tool_input.get('text', '')}"
                    elif action == 'type':
                        text = tool_input.get('text', '')
                        action_detail = f"Type: {text}"
                    elif action in ['left_click', 'right_click', 'middle_click', 'double_click']:
                        coord = tool_input.get('coordinate', [])
                        action_detail = f"{action.replace('_', ' ').title()} at {coord}"
                    elif action == 'mouse_move':
                        coord = tool_input.get('coordinate', [])
                        action_detail = f"Move to {coord}"
                    elif action == 'screenshot':
                        action_detail = "Screenshot"
                    elif action == 'zoom':
                        region = tool_input.get('region', [])
                        action_detail = f"Zoom to {region}"
                    else:
                        action_detail = f"Computer: {action}"

                elif tool_name == 'bash':
                    command = tool_input.get('command', '')
                    action_detail = f"Bash: {command}"

                elif tool_name == 'fork_subtask':
                    subtask = tool_input.get('subtask', '')
                    action_detail = f"Fork worker: {subtask}"

                elif tool_name == 'peek_child':
                    child_id = tool_input.get('child_id', '')
                    action_detail = f"Peek at {child_id}"

                elif tool_name == 'message_child':
                    child_id = tool_input.get('child_id', '')
                    message = tool_input.get('message', '')
                    action_detail = f"Message to {child_id}: {message}"

                else:
                    action_detail = f"Tool: {tool_name}"

                if action_detail:
                    actions_in_response.append({
                        'action': action_detail,
                        'tool': tool_name,
                    })

        # Create a step entry for each action (or one for the whole response if no actions)
        if not actions_in_response:
            actions_in_response = [{'action': 'Step executed', 'tool': ''}]

        for action_info in actions_in_response:
            agent_step_counters[agent_id] += 1
            step_num = agent_step_counters[agent_id]

            agent_steps_map[agent_id][step_num] = {
                'timestamp': relative_time,
                'action': action_info['action'],
                'tool': action_info['tool'],
            }

    # Supplement with direct file reading for agents not in API calls
    for agent_id, agent_dir in agent_dirs:
        if agent_id not in agent_steps_map or not agent_steps_map[agent_id]:
            # Try to read step files directly
            step_files = sorted(agent_dir.glob('step_*_response.txt'))
            if step_files:
                if agent_id not in agent_steps_map:
                    agent_steps_map[agent_id] = {}

                # Get first step file's mtime as baseline
                first_mtime = step_files[0].stat().st_mtime if step_files else 0

                for step_file in step_files:
                    # Extract step number from filename: step_001_response.txt -> 1
                    match = re.search(r'step_(\d+)_response\.txt', step_file.name)
                    if not match:
                        continue
                    step_num = int(match.group(1))

                    # Read action from response file
                    try:
                        content = step_file.read_text(encoding='utf-8', errors='replace').strip()
                        action = content
                    except:
                        action = "Step executed"

                    # Use file modification time relative to start
                    file_mtime = step_file.stat().st_mtime
                    relative_time = file_mtime - first_mtime if start_time else 0

                    agent_steps_map[agent_id][step_num] = {
                        'timestamp': relative_time,
                        'action': action,
                        'tool': '',
                    }

    # Build agent timeline spans (start/end times)
    agent_timeline = {}
    for agent_id, steps_dict in agent_steps_map.items():
        if steps_dict:
            step_nums = sorted(steps_dict.keys())
            start = steps_dict[step_nums[0]]['timestamp']
            end = steps_dict[step_nums[-1]]['timestamp']

            agent_timeline[agent_id] = {
                'start': start,
                'end': end,
                'steps': steps_dict,
            }

    # Calculate total duration as max end time across all agents
    if agent_timeline:
        total_duration = max(agent_data['end'] for agent_data in agent_timeline.values())
    else:
        total_duration = (end_time - start_time).total_seconds() if start_time and end_time else 0

    return {
        'agent_timeline': agent_timeline,
        'agent_steps_map': agent_steps_map,
        'total_duration': total_duration,
    }


def calculate_cost_from_tokens(local_path: pathlib.Path, api_calls: List[Dict]) -> float:
    """Calculate cost from token_usage.json or bedrock_api_calls.jsonl."""
    token_usage_path = local_path / "token_usage.json"

    # Try token_usage.json first
    if token_usage_path.is_file():
        try:
            with open(token_usage_path, 'r', encoding='utf-8') as f:
                usage = json.load(f)
            cost = usage.get('total_cost_usd', 0.0)
            if cost > 0:
                return cost
        except (json.JSONDecodeError, OSError):
            pass

    # Fallback: calculate from bedrock_api_calls.jsonl
    # Opus 4.6 pricing: $15/M input, $75/M output, $1.50/M cache write, $0.15/M cache read
    total_cost = 0.0
    for call in api_calls:
        if call.get('event') != 'api_call':
            continue

        usage = call.get('response', {}).get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)
        cache_creation_tokens = usage.get('cache_creation_input_tokens', 0)
        cache_read_tokens = usage.get('cache_read_input_tokens', 0)

        # Calculate cost
        # Note: cache_read_tokens and cache_creation_tokens are PART of input_tokens, not separate
        # input_tokens = uncached_input + cache_read_tokens + cache_creation_tokens
        uncached_input = input_tokens - cache_read_tokens - cache_creation_tokens
        total_cost += (uncached_input / 1_000_000) * 15.0  # $15/M uncached input
        total_cost += (output_tokens / 1_000_000) * 75.0   # $75/M output
        total_cost += (cache_creation_tokens / 1_000_000) * 1.50  # $1.50/M cache write
        total_cost += (cache_read_tokens / 1_000_000) * 0.15  # $0.15/M cache read

    return total_cost


def generate_trajectory_html(
    local_dir: str,
    task_id: str,
    github_repo: str,
    github_path: str,
    task_type: str,
    config_name: str,
    trial: int,
) -> None:
    """Generate interactive trajectory.html for fork-based agent runs.

    Args:
        local_dir: Local directory containing task results
        task_id: Task ID
        github_repo: GitHub repo for raw image URLs
        github_path: Path prefix in repo
        task_type: Task type (e.g., collaborative)
        config_name: Config name (e.g., fork_parallel)
        trial: Trial number
    """
    local_path = pathlib.Path(local_dir)
    if not local_path.is_dir():
        return

    img_base = (
        f"https://raw.githubusercontent.com/{github_repo}/main"
        f"/{github_path}/{task_type}/{task_id}/{config_name}/trial_{trial}"
    )

    # -- Gather data --------------------------------------------------------

    task_txt = local_path / "task.txt"
    instruction = ""
    if task_txt.is_file():
        instruction = task_txt.read_text(encoding="utf-8", errors="replace").strip()

    result_path = local_path / "result.txt"
    score_str = "N/A"
    if result_path.is_file():
        try:
            score_str = result_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass

    result_json_path = local_path / "result.json"
    result_data: dict = {}
    if result_json_path.is_file():
        try:
            result_data = json.loads(result_json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Discover agent directories
    agent_dirs: List[Tuple[str, pathlib.Path]] = []
    for d in sorted(local_path.iterdir()):
        if not d.is_dir():
            continue
        if d.name == "root" or d.name.startswith("root_child_"):
            agent_dirs.append((d.name, d))

    agent_dirs.sort()

    total_agents = len(agent_dirs)
    duration = result_data.get("duration", 0)
    num_agents = result_data.get("num_agents", total_agents)
    forked = result_data.get("forked", num_agents > 1)

    # Parse API calls for timing data
    api_calls = parse_bedrock_api_calls(local_path, agent_dirs)
    timeline_data = build_timeline_from_api_calls(api_calls, agent_dirs)

    agent_timeline = timeline_data['agent_timeline']
    agent_steps_map = timeline_data['agent_steps_map']
    total_duration = timeline_data['total_duration'] or duration

    # Calculate cost from token usage
    cost = calculate_cost_from_tokens(local_path, api_calls)

    # -- Helper -------------------------------------------------------------

    def esc(text: str) -> str:
        """HTML-escape text."""
        return html_mod.escape(text)

    def fmt_duration(secs: float) -> str:
        """Format duration in seconds to 'Xm Ys'."""
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"

    # -- Build agent step data with screenshots -----------------------------

    # Find earliest timestamp across all agents for relative timing
    first_timestamp = None
    for agent_id, agent_dir in agent_dirs:
        timestamp_file = agent_dir / "step_001_timestamp.txt"
        if timestamp_file.is_file():
            try:
                ts = float(timestamp_file.read_text().strip())
                if first_timestamp is None or ts < first_timestamp:
                    first_timestamp = ts
            except (ValueError, OSError):
                pass

    agent_data = []
    for agent_id, agent_dir in agent_dirs:
        step_files = sorted([f for f in agent_dir.glob("step_*.png")])
        steps = []

        for step_file in step_files:
            m = re.match(r"step_(\d+)\.png$", step_file.name)
            if not m:
                continue
            step_num = int(m.group(1))

            # Read timestamp from step_NNN_timestamp.txt (absolute time when screenshot was taken)
            timestamp = 0
            timestamp_file = agent_dir / f"step_{step_num:03d}_timestamp.txt"
            if timestamp_file.is_file():
                try:
                    absolute_ts = float(timestamp_file.read_text().strip())
                    timestamp = absolute_ts - first_timestamp if first_timestamp else 0
                except (ValueError, OSError):
                    pass

            # Read thinking/reasoning from step response file
            thinking = ""
            response_file = agent_dir / f"step_{step_num:03d}_response.txt"
            if response_file.is_file():
                try:
                    thinking = response_file.read_text(encoding='utf-8', errors='replace').strip()
                except OSError:
                    pass

            # Screenshot URL
            screenshot_url = f"{img_base}/{agent_id}/step_{step_num:03d}.png"

            steps.append({
                'num': step_num,
                'timestamp': timestamp,
                'thinking': thinking,
                'screenshot': screenshot_url,
            })

        # Get agent timeline span
        timeline = agent_timeline.get(agent_id, {})
        start_time = timeline.get('start', 0)

        # Use completion timestamp if available (when agent finishes)
        # Otherwise use last step timestamp
        completion_file = agent_dir / "completion_timestamp.txt"
        if completion_file.is_file():
            try:
                completion_ts = float(completion_file.read_text().strip())
                end_time = completion_ts - first_timestamp if first_timestamp else 0
            except (ValueError, OSError):
                end_time = steps[-1]['timestamp'] if steps else timeline.get('end', 0)
        else:
            end_time = steps[-1]['timestamp'] if steps else timeline.get('end', 0)

        agent_info = result_data.get("agents", {}).get(agent_id, {})
        status = agent_info.get("status", "unknown")
        display_num = agent_info.get("display", "?")

        agent_data.append({
            'id': agent_id,
            'display': display_num,
            'status': status,
            'start': start_time,
            'end': end_time,
            'steps': steps,
        })

    # Recalculate total duration from actual agent end times
    # This ensures timeline duration matches the last actual step, not API call processing time
    if agent_data:
        total_duration = max(agent['end'] for agent in agent_data)

    # -- Build action log ---------------------------------------------------

    def classify_action_type(action_text: str, tool_name: str) -> str:
        """Classify action into communication or regular types."""
        if tool_name == 'fork_subtask' or action_text.startswith('Fork worker'):
            return 'fork'
        elif tool_name == 'message_child' or action_text.startswith('Message to'):
            return 'message'
        elif tool_name == 'peek_child' or action_text.startswith('Peek at'):
            return 'peek'
        elif 'SUBTASK COMPLETE' in action_text or action_text.startswith('DONE') or action_text.startswith('TASK COMPLETED'):
            return 'report'
        else:
            return 'action'

    all_actions = []
    for agent_id, steps_dict in agent_steps_map.items():
        for step_num in sorted(steps_dict.keys()):
            step_info = steps_dict[step_num]
            action_type = classify_action_type(step_info['action'], step_info.get('tool', ''))
            all_actions.append({
                'timestamp': step_info['timestamp'],
                'agent': agent_id,
                'step': step_num,
                'action': step_info['action'],
                'type': action_type,
            })

    all_actions.sort(key=lambda x: x['timestamp'])

    # -- Build HTML ---------------------------------------------------------

    h: List[str] = []
    h.append("<!DOCTYPE html>")
    h.append("<html lang='en'>")
    h.append("<head>")
    h.append("<meta charset='utf-8'>")
    h.append(f"<title>Trajectory — {esc(task_id)}</title>")
    # Calculate timeline height based on number of agents
    timeline_height = 40 + max(0, (total_agents - 1)) * 24

    h.append("<style>")
    h.append(f"""
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", sans-serif;
    background: #0d1117;
    color: #e6edf3;
    padding: 24px;
    line-height: 1.6;
    max-width: 100%;
    overflow-x: hidden;
}}
h1 {{
    font-size: 1.8em;
    margin-bottom: 8px;
    background: linear-gradient(90deg, #f0f6fc 0%, #c9d1d9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    letter-spacing: -0.03em;
}}
h2 {{
    font-size: 1.1em;
    margin: 32px 0 16px 0;
    color: #f0f6fc;
    border-bottom: 2px solid #21262d;
    padding-bottom: 8px;
    font-weight: 600;
    letter-spacing: -0.01em;
}}
.meta {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 20px 0 32px 0;
    font-size: 0.9em;
}}
.meta span {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    padding: 10px 16px;
    border-radius: 10px;
    border: 1px solid #30363d;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.03);
    transition: transform 0.15s, box-shadow 0.15s;
}}
.meta span:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.05);
}}
.meta strong {{
    color: #e6edf3;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}
.score-pass {{
    color: #3fb950;
    font-weight: 800;
    text-shadow: 0 0 12px rgba(63, 185, 80, 0.3);
}}
.score-fail {{
    color: #f85149;
    font-weight: 800;
    text-shadow: 0 0 12px rgba(248, 81, 73, 0.3);
}}
.cost-highlight {{
    font-size: 1.15em;
    color: #58a6ff;
    text-shadow: 0 0 8px rgba(88, 166, 255, 0.2);
}}
.agent-status {{
    font-size: 0.75em;
    padding: 2px 6px;
    border-radius: 4px;
    margin-left: 6px;
    font-weight: 600;
}}
.agent-status-completed {{
    background: rgba(63, 185, 80, 0.15);
    color: #3fb950;
}}
.agent-status-completed::after {{
    content: ' ✓';
}}
.agent-status-killed {{
    background: rgba(248, 81, 73, 0.15);
    color: #f85149;
}}
.agent-status-killed::after {{
    content: ' ⚠';
}}
.agent-status-failed {{
    background: rgba(248, 81, 73, 0.15);
    color: #f85149;
}}
.agent-status-failed::after {{
    content: ' ✗';
}}
.agent-status-running {{
    background: rgba(88, 166, 255, 0.15);
    color: #58a6ff;
}}
.agent-status-running::after {{
    content: ' ⟳';
}}

/* Timeline Scrubber */
.timeline-scrubber {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 28px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    max-width: 100%;
    overflow: hidden;
}}
.timeline-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 16px;
}}
.timeline-controls {{
    display: flex;
    align-items: center;
    gap: 12px;
}}
.timeline-play-button {{
    background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
    border: none;
    color: white;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2em;
    transition: all 0.2s ease;
    box-shadow: 0 2px 8px rgba(35, 134, 54, 0.3);
}}
.timeline-play-button:hover {{
    background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(35, 134, 54, 0.5);
}}
.timeline-play-button:active {{
    transform: scale(0.95);
}}
.timeline-speed-selector {{
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9em;
    font-weight: 600;
    transition: all 0.2s ease;
}}
.timeline-speed-selector:hover {{
    background: #21262d;
    border-color: #58a6ff;
}}
.timeline-time {{
    font-size: 1.4em;
    background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}}
.timeline-track-container {{
    margin-bottom: 20px;
}}
.timeline-slider {{
    width: 100%;
    height: 20px;
    background: #0d1117;
    border-radius: 10px;
    position: relative;
    cursor: pointer;
    margin-bottom: 16px;
    border: 1px solid #21262d;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.5);
}}
.timeline-progress {{
    height: 100%;
    background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
    border-radius: 10px;
    position: absolute;
    transition: width 0.1s ease-out;
    box-shadow: 0 0 8px rgba(88, 166, 255, 0.4);
}}
.timeline-knob {{
    position: absolute;
    width: 28px;
    height: 28px;
    background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
    border: 3px solid #0d1117;
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    cursor: grab;
    box-shadow: 0 2px 8px rgba(88, 166, 255, 0.6), 0 0 0 4px rgba(88, 166, 255, 0.1);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.timeline-knob:hover {{
    transform: translate(-50%, -50%) scale(1.1);
    box-shadow: 0 4px 12px rgba(88, 166, 255, 0.8), 0 0 0 6px rgba(88, 166, 255, 0.15);
}}
.timeline-knob:active {{
    cursor: grabbing;
    transform: translate(-50%, -50%) scale(0.95);
}}
.timeline-bars {{
    position: relative;
    height: {timeline_height}px;
    margin-top: 12px;
    overflow: hidden;
}}
.timeline-bar {{
    position: absolute;
    height: 20px;
    border-radius: 6px;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    border: 1px solid rgba(255, 255, 255, 0.1);
}}
.timeline-bar:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}}
.timeline-bar.agent-root {{
    background: linear-gradient(90deg, #3fb950 0%, #2ea043 100%);
    box-shadow: 0 2px 8px rgba(63, 185, 80, 0.3);
    top: 0;
}}
.timeline-bar.agent-child-0 {{
    background: linear-gradient(90deg, #58a6ff 0%, #1f6feb 100%);
    box-shadow: 0 2px 8px rgba(88, 166, 255, 0.3);
}}
.timeline-bar.agent-child-1 {{
    background: linear-gradient(90deg, #d29922 0%, #bb8009 100%);
    box-shadow: 0 2px 8px rgba(210, 153, 34, 0.3);
}}
.timeline-bar.agent-child-2 {{
    background: linear-gradient(90deg, #a371f7 0%, #8957e5 100%);
    box-shadow: 0 2px 8px rgba(163, 113, 247, 0.3);
}}
.timeline-bar.agent-child-3 {{
    background: linear-gradient(90deg, #f85149 0%, #da3633 100%);
    box-shadow: 0 2px 8px rgba(248, 81, 73, 0.3);
}}
.timeline-bar.agent-child-4 {{
    background: linear-gradient(90deg, #56d364 0%, #3fb950 100%);
    box-shadow: 0 2px 8px rgba(86, 211, 100, 0.3);
}}
.timeline-bar.agent-child-5 {{
    background: linear-gradient(90deg, #ff7b72 0%, #f85149 100%);
    box-shadow: 0 2px 8px rgba(255, 123, 114, 0.3);
}}
.timeline-bar.agent-child-6 {{
    background: linear-gradient(90deg, #ffa657 0%, #f0883e 100%);
    box-shadow: 0 2px 8px rgba(255, 166, 87, 0.3);
}}
.timeline-bar.agent-child {{
    background: linear-gradient(90deg, #8b949e 0%, #6e7681 100%);
    box-shadow: 0 2px 8px rgba(139, 148, 158, 0.3);
}}
.timeline-bar-label {{
    position: absolute;
    font-size: 0.7em;
    color: #fff;
    padding: 2px 8px;
    white-space: nowrap;
    pointer-events: none;
    font-weight: 600;
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}}
.timeline-playhead {{
    position: absolute;
    width: 2px;
    height: 100%;
    background: #f85149;
    top: 0;
    pointer-events: none;
    box-shadow: 0 0 8px rgba(248, 81, 73, 0.8);
    transition: left 0.1s ease-out;
}}
.timeline-playhead::before {{
    content: '';
    position: absolute;
    top: -4px;
    left: -3px;
    width: 8px;
    height: 8px;
    background: #f85149;
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(248, 81, 73, 1);
}}

/* Display Panels */
.display-panels {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(450px, 1fr));
    gap: 20px;
    margin-bottom: 28px;
}}
.display-panel {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.display-panel:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0, 0, 0, 0.5);
}}
.display-panel-header {{
    font-size: 0.9em;
    color: #b1bac4;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.display-panel-header strong {{
    color: #58a6ff;
    font-weight: 600;
}}
.display-panel-step {{
    font-size: 0.75em;
    color: #8b949e;
    font-variant-numeric: tabular-nums;
}}
.display-panel img {{
    width: 100%;
    height: auto;
    border-radius: 8px;
    border: 1px solid #30363d;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}}
.display-panel-thinking {{
    font-size: 0.85em;
    color: #c9d1d9;
    margin-top: 12px;
    padding: 12px;
    background: linear-gradient(135deg, #1c2128 0%, #22272e 100%);
    border-radius: 8px;
    border-left: 3px solid #58a6ff;
    line-height: 1.6;
    font-style: italic;
}}
.display-panel-thinking::before {{
    content: '\U0001F4AD ';
    opacity: 0.6;
}}
/* Single-agent layout: side-by-side */
.display-panel.single-agent {{
    padding: 20px;
}}
.display-content {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
}}
.display-screenshot {{
    width: 100%;
}}
.display-screenshot img {{
    width: 100%;
    border-radius: 8px;
    border: 1px solid #30363d;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
}}
.display-thinking-container {{
    min-height: 100px;
}}
.single-agent .display-panel-thinking {{
    margin-top: 0;
}}
/* Action Log */
.action-log {{
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
    max-height: 600px;
    overflow-y: auto;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}}
.action-log::-webkit-scrollbar {{
    width: 12px;
}}
.action-log::-webkit-scrollbar-track {{
    background: #0d1117;
    border-radius: 6px;
}}
.action-log::-webkit-scrollbar-thumb {{
    background: #30363d;
    border-radius: 6px;
    border: 2px solid #0d1117;
}}
.action-log::-webkit-scrollbar-thumb:hover {{
    background: #484f58;
}}
.action-item {{
    display: grid;
    grid-template-columns: 90px 140px 1fr;
    gap: 16px;
    padding: 12px 10px;
    border-bottom: 1px solid #21262d;
    font-size: 0.85em;
    transition: all 0.15s;
    border-radius: 4px;
}}
.action-item:hover {{
    background: linear-gradient(90deg, rgba(56, 139, 253, 0.08) 0%, rgba(56, 139, 253, 0.03) 100%);
    border-left: 2px solid #58a6ff;
    padding-left: 8px;
}}
.action-item:last-child {{
    border-bottom: none;
}}

/* Communication action types */
.action-item.fork {{
    background: linear-gradient(90deg, rgba(163, 113, 247, 0.08) 0%, rgba(163, 113, 247, 0.02) 100%);
    border-left: 3px solid #a371f7;
}}
.action-item.fork:hover {{
    background: linear-gradient(90deg, rgba(163, 113, 247, 0.15) 0%, rgba(163, 113, 247, 0.05) 100%);
    border-left: 3px solid #a371f7;
}}
.action-item.message {{
    background: linear-gradient(90deg, rgba(247, 208, 88, 0.08) 0%, rgba(247, 208, 88, 0.02) 100%);
    border-left: 3px solid #f7d058;
}}
.action-item.message:hover {{
    background: linear-gradient(90deg, rgba(247, 208, 88, 0.15) 0%, rgba(247, 208, 88, 0.05) 100%);
    border-left: 3px solid #f7d058;
}}
.action-item.peek {{
    background: linear-gradient(90deg, rgba(88, 166, 255, 0.08) 0%, rgba(88, 166, 255, 0.02) 100%);
    border-left: 3px solid #58a6ff;
}}
.action-item.peek:hover {{
    background: linear-gradient(90deg, rgba(88, 166, 255, 0.15) 0%, rgba(88, 166, 255, 0.05) 100%);
    border-left: 3px solid #58a6ff;
}}
.action-item.report {{
    background: linear-gradient(90deg, rgba(63, 185, 80, 0.08) 0%, rgba(63, 185, 80, 0.02) 100%);
    border-left: 3px solid #3fb950;
}}
.action-item.report:hover {{
    background: linear-gradient(90deg, rgba(63, 185, 80, 0.15) 0%, rgba(63, 185, 80, 0.05) 100%);
    border-left: 3px solid #3fb950;
}}

.action-time {{
    color: #8b949e;
    font-family: 'SF Mono', Monaco, monospace;
    font-variant-numeric: tabular-nums;
    font-weight: 600;
    font-size: 0.9em;
}}
.action-agent {{
    color: #79c0ff;
    font-weight: 700;
    font-size: 0.9em;
}}
.action-detail {{
    color: #c9d1d9;
    word-wrap: break-word;
    line-height: 1.6;
}}
.action-detail::before {{
    margin-right: 8px;
    font-weight: 700;
}}
.action-item.fork .action-detail::before {{
    content: '🔀 ';
    color: #a371f7;
}}
.action-item.message .action-detail::before {{
    content: '💬 ';
    color: #f7d058;
}}
.action-item.peek .action-detail::before {{
    content: '👀 ';
    color: #58a6ff;
}}
.action-item.report .action-detail::before {{
    content: '✅ ';
    color: #3fb950;
}}

/* Tabs */
.tabs {{
    display: flex;
    gap: 4px;
    margin-bottom: 16px;
    border-bottom: 2px solid #21262d;
}}
.tab {{
    padding: 10px 18px;
    cursor: pointer;
    color: #8b949e;
    border-bottom: 3px solid transparent;
    transition: all 0.2s;
    font-weight: 500;
    border-radius: 6px 6px 0 0;
}}
.tab:hover {{
    color: #e6edf3;
    background: rgba(56, 139, 253, 0.05);
}}
.tab.active {{
    color: #58a6ff;
    border-bottom-color: #58a6ff;
    background: rgba(56, 139, 253, 0.08);
}}
.tab-content {{
    display: none;
}}
.tab-content.active {{
    display: block;
}}

/* Communication Legend */
.comm-legend {{
    display: flex;
    gap: 20px;
    margin-bottom: 16px;
    padding: 12px 16px;
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 8px;
    font-size: 0.85em;
    flex-wrap: wrap;
}}
.comm-legend-item {{
    display: flex;
    align-items: center;
    gap: 6px;
    color: #8b949e;
}}
.comm-legend-icon {{
    font-size: 1.1em;
}}
.comm-legend-label {{
    font-weight: 500;
}}
.comm-legend-item.fork .comm-legend-label {{ color: #a371f7; }}
.comm-legend-item.message .comm-legend-label {{ color: #f7d058; }}
.comm-legend-item.peek .comm-legend-label {{ color: #58a6ff; }}
.comm-legend-item.report .comm-legend-label {{ color: #3fb950; }}
""")
    h.append("</style>")
    h.append("</head>")
    h.append("<body>")

    # Header
    h.append(f"<h1>Task {esc(task_id)}</h1>")
    h.append(f"<p style='margin:16px 0 8px 0;color:#8b949e;font-size:0.95em;line-height:1.6;max-width:900px;font-weight:400'>{esc(instruction)}</p>")

    # Meta info
    score_class = "score-pass" if score_str not in ["N/A", "0", "0.0"] else "score-fail"
    h.append("<div class='meta'>")
    h.append(f"  <span>Score: <strong class='{score_class}'>{esc(score_str)}</strong></span>")
    h.append(f"  <span>Duration: <strong>{fmt_duration(duration)}</strong></span>")
    h.append(f"  <span>Agents: <strong>{num_agents}</strong> {'(parallelized)' if forked else ''}</span>")
    h.append(f"  <span>Cost: <strong class='cost-highlight'>${cost:.2f}</strong></span>")
    h.append("</div>")

    # Timeline Scrubber
    h.append("<h2>⏱️ Execution Timeline</h2>")
    h.append("<div class='timeline-scrubber'>")
    h.append("  <div class='timeline-header'>")
    h.append("    <div class='timeline-controls'>")
    h.append("      <button class='timeline-play-button' id='timeline-play-button' title='Play/Pause'>▶️</button>")
    h.append("      <select class='timeline-speed-selector' id='timeline-speed-selector' title='Playback Speed'>")
    h.append("        <option value='1'>1x</option>")
    h.append("        <option value='2'>2x</option>")
    h.append("        <option value='4' selected>4x</option>")
    h.append("        <option value='8'>8x</option>")
    h.append("        <option value='16'>16x</option>")
    h.append("      </select>")
    h.append("      <div class='timeline-time' id='timeline-time'>0:00</div>")
    h.append("    </div>")
    h.append(f"    <div style='color:#8b949e;font-size:0.85em'>Total: {fmt_duration(total_duration)}</div>")
    h.append("  </div>")
    h.append("  <div class='timeline-track-container'>")
    h.append("    <div class='timeline-slider' id='timeline-slider'>")
    h.append("      <div class='timeline-progress' id='timeline-progress' style='width: 0%'></div>")
    h.append("      <div class='timeline-knob' id='timeline-knob' style='left: 0%'></div>")
    h.append("    </div>")
    h.append("    <div class='timeline-bars' id='timeline-bars'>")

    # Timeline bars for each agent
    for idx, agent in enumerate(agent_data):
        agent_id = agent['id']
        start_pct = (agent['start'] / total_duration * 100) if total_duration > 0 else 0
        duration_pct = ((agent['end'] - agent['start']) / total_duration * 100) if total_duration > 0 else 0

        # Ensure minimum width for visibility (2% or actual duration, whichever is larger)
        min_width_pct = 2.0
        display_width_pct = max(duration_pct, min_width_pct)

        is_root = (agent_id == 'root')

        # Determine bar class with specific child colors
        if is_root:
            bar_class = 'agent-root'
        else:
            # Extract child number for specific colors (root_child_0 -> 0)
            if 'child_' in agent_id:
                child_num = agent_id.split('_')[-1]
                bar_class = f'agent-child-{child_num}'
            else:
                bar_class = 'agent-child'

        top_offset = 0 if is_root else (idx * 24)

        # Status icon
        status = agent.get('status', 'unknown')
        status_icon = {'completed': '✓', 'killed': '⚠', 'failed': '✗', 'running': '⟳'}.get(status, '')

        # Build label with step count and status
        step_count = len(agent.get('steps', []))
        label = f"{agent_id} ({step_count} steps) {status_icon}".strip()

        # Build detailed tooltip
        start_time = fmt_duration(agent['start'])
        end_time = fmt_duration(agent['end'])
        tooltip = f"{agent_id}: {start_time} → {end_time} ({step_count} steps, {status})"

        h.append(f"    <div class='timeline-bar {bar_class}' style='left: {start_pct:.1f}%; width: {display_width_pct:.1f}%; top: {top_offset}px' title='{esc(tooltip)}'>")
        h.append(f"      <div class='timeline-bar-label'>{esc(label)}</div>")
        h.append("    </div>")

    h.append("      <div class='timeline-playhead' id='timeline-playhead' style='left: 0%'></div>")
    h.append("    </div>")
    h.append("  </div>")
    h.append("</div>")

    # Display Panels
    h.append("<h2>📺 Displays</h2>")
    h.append("<div class='display-panels'>")

    # Use side-by-side layout for single agent
    single_agent = len(agent_data) == 1

    for agent in agent_data:
        agent_id = agent['id']
        display = agent['display']
        status = agent.get('status', 'unknown')
        status_class = f'agent-status agent-status-{status}'

        panel_class = 'display-panel single-agent' if single_agent else 'display-panel'
        h.append(f"  <div class='{panel_class}' id='panel-{esc(agent_id)}'>")
        h.append(f"    <div class='display-panel-header'>")
        h.append(f"      <span><strong>{esc(agent_id)}</strong> (Display {esc(str(display))}) <span class='{status_class}'>{esc(status)}</span></span>")
        h.append(f"      <span class='display-panel-step' id='panel-step-{esc(agent_id)}'>Step —</span>")
        h.append(f"    </div>")

        if single_agent:
            # Side-by-side layout
            h.append(f"    <div class='display-content'>")
            h.append(f"      <div class='display-screenshot'>")
            h.append(f"        <img id='panel-img-{esc(agent_id)}' src='' alt='No screenshot' style='display:none'>")
            h.append(f"      </div>")
            h.append(f"      <div class='display-thinking-container'>")
            h.append(f"        <div id='panel-thinking-{esc(agent_id)}' class='display-panel-thinking' style='display:none'></div>")
            h.append(f"      </div>")
            h.append(f"    </div>")
        else:
            # Stacked layout for multi-agent
            h.append(f"    <img id='panel-img-{esc(agent_id)}' src='' alt='No screenshot' style='display:none'>")
            h.append(f"    <div id='panel-thinking-{esc(agent_id)}' class='display-panel-thinking' style='display:none'></div>")

        h.append(f"  </div>")

    h.append("</div>")

    # Tabs
    h.append("<div class='tabs'>")
    h.append("  <div class='tab active' onclick='showTab(\"log\")'>📋 Action Log</div>")
    h.append("</div>")

    # Action Log
    h.append("<div id='tab-log' class='tab-content active'>")

    # Communication Legend
    h.append("  <div class='comm-legend'>")
    h.append("    <div class='comm-legend-item fork'>")
    h.append("      <span class='comm-legend-icon'>🔀</span>")
    h.append("      <span class='comm-legend-label'>Fork Worker</span>")
    h.append("    </div>")
    h.append("    <div class='comm-legend-item message'>")
    h.append("      <span class='comm-legend-icon'>💬</span>")
    h.append("      <span class='comm-legend-label'>Message</span>")
    h.append("    </div>")
    h.append("    <div class='comm-legend-item peek'>")
    h.append("      <span class='comm-legend-icon'>👀</span>")
    h.append("      <span class='comm-legend-label'>Peek</span>")
    h.append("    </div>")
    h.append("    <div class='comm-legend-item report'>")
    h.append("      <span class='comm-legend-icon'>✅</span>")
    h.append("      <span class='comm-legend-label'>Report/Done</span>")
    h.append("    </div>")
    h.append("  </div>")

    h.append("  <div class='action-log'>")

    for action in all_actions:
        time_str = fmt_duration(action['timestamp'])
        action_type = action.get('type', 'action')
        h.append(f"    <div class='action-item {esc(action_type)}'>")
        h.append(f"      <div class='action-time'>{esc(time_str)}</div>")
        h.append(f"      <div class='action-agent'>{esc(action['agent'])} step {action['step']}</div>")
        h.append(f"      <div class='action-detail'>{esc(action['action'])}</div>")
        h.append("    </div>")

    h.append("  </div>")
    h.append("</div>")

    # JavaScript for interactivity
    h.append("<script>")

    # Embed agent data as JSON
    h.append("const agentData = " + json.dumps(agent_data) + ";")
    h.append(f"const totalDuration = {total_duration};")

    h.append("""
let currentTime = 0;
let isDragging = false;
let isPlaying = false;
let animationId = null;
let lastFrameTime = null;
let playbackSpeed = 4; // Default 4x speed

function updateDisplays(time) {
    currentTime = time;

    // Update time display
    const mins = Math.floor(time / 60);
    const secs = Math.floor(time % 60);
    document.getElementById('timeline-time').textContent = mins + ':' + (secs < 10 ? '0' : '') + secs;

    // Update progress bar and knob
    const progress = (time / totalDuration) * 100;
    document.getElementById('timeline-progress').style.width = progress + '%';
    document.getElementById('timeline-knob').style.left = progress + '%';
    document.getElementById('timeline-playhead').style.left = progress + '%';

    // Update each agent display
    agentData.forEach(agent => {
        const agentId = agent.id;
        const imgEl = document.getElementById('panel-img-' + agentId);
        const stepEl = document.getElementById('panel-step-' + agentId);
        const thinkingEl = document.getElementById('panel-thinking-' + agentId);

        // Hide display if before agent starts or after agent finishes
        if (time < agent.start || time > agent.end) {
            imgEl.style.display = 'none';
            stepEl.textContent = 'Step —';
            thinkingEl.style.display = 'none';
            return;
        }

        // Find the latest step at or before current time
        let currentStep = null;
        for (const step of agent.steps) {
            if (step.timestamp <= time) {
                currentStep = step;
            } else {
                break;
            }
        }

        if (currentStep) {
            imgEl.src = currentStep.screenshot;
            imgEl.style.display = 'block';
            stepEl.textContent = 'Step ' + currentStep.num;

            if (currentStep.thinking) {
                thinkingEl.textContent = currentStep.thinking;
                thinkingEl.style.display = 'block';
            } else {
                thinkingEl.style.display = 'none';
            }
        } else {
            imgEl.style.display = 'none';
            stepEl.textContent = 'Step —';
            thinkingEl.style.display = 'none';
        }
    });
}

// Timeline slider interaction
const slider = document.getElementById('timeline-slider');
const knob = document.getElementById('timeline-knob');

function seekToPosition(clientX) {
    const rect = slider.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = x / rect.width;
    const time = percent * totalDuration;
    updateDisplays(time);
}

// Play/Pause functionality
function animate(timestamp) {
    if (!isPlaying) return;

    if (lastFrameTime === null) {
        lastFrameTime = timestamp;
    }

    const deltaTime = (timestamp - lastFrameTime) / 1000; // Convert to seconds
    lastFrameTime = timestamp;

    const newTime = currentTime + (deltaTime * playbackSpeed);

    if (newTime >= totalDuration) {
        // Reached end, stop playing
        isPlaying = false;
        updateDisplays(totalDuration);
        document.getElementById('timeline-play-button').textContent = '▶️';
        lastFrameTime = null;
    } else {
        updateDisplays(newTime);
        animationId = requestAnimationFrame(animate);
    }
}

function togglePlay() {
    isPlaying = !isPlaying;
    const button = document.getElementById('timeline-play-button');

    if (isPlaying) {
        button.textContent = '⏸️';
        // If at end, restart from beginning
        if (currentTime >= totalDuration) {
            updateDisplays(0);
        }
        lastFrameTime = null;
        animationId = requestAnimationFrame(animate);
    } else {
        button.textContent = '▶️';
        if (animationId) {
            cancelAnimationFrame(animationId);
            animationId = null;
        }
        lastFrameTime = null;
    }
}

document.getElementById('timeline-play-button').addEventListener('click', togglePlay);

// Speed selector
document.getElementById('timeline-speed-selector').addEventListener('change', (e) => {
    playbackSpeed = parseFloat(e.target.value);
});

// Click on slider
slider.addEventListener('click', (e) => {
    if (e.target === slider || e.target === document.getElementById('timeline-progress')) {
        // Stop playing when manually seeking
        if (isPlaying) {
            togglePlay();
        }
        seekToPosition(e.clientX);
    }
});

// Drag knob
knob.addEventListener('mousedown', (e) => {
    isDragging = true;
    // Stop playing when dragging
    if (isPlaying) {
        togglePlay();
    }
    e.preventDefault();
});

document.addEventListener('mousemove', (e) => {
    if (isDragging) {
        seekToPosition(e.clientX);
    }
});

document.addEventListener('mouseup', () => {
    isDragging = false;
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
    } else if (e.key === 'ArrowLeft') {
        updateDisplays(Math.max(0, currentTime - 5));
    } else if (e.key === 'ArrowRight') {
        updateDisplays(Math.min(totalDuration, currentTime + 5));
    } else if (e.key === 'Home') {
        updateDisplays(0);
    } else if (e.key === 'End') {
        updateDisplays(totalDuration);
    }
});

// Initialize at first step of first agent
if (agentData.length > 0 && agentData[0].steps.length > 0) {
    updateDisplays(agentData[0].steps[0].timestamp);
}

function showTab(tabName) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    event.target.classList.add('active');
}
""")
    h.append("</script>")

    h.append("</body></html>")

    # Write HTML
    output_path = local_path / "trajectory.html"
    output_path.write_text("\n".join(h), encoding="utf-8")
