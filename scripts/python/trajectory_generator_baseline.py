"""Baseline trajectory HTML generator.

Generates interactive HTML visualization for single-agent baseline execution.
"""

import json
import logging
import pathlib

logger = logging.getLogger(__name__)


def generate_trajectory_html_baseline(
    local_dir: str,
    task_id: str,
    github_repo: str,
    github_path: str,
    task_type: str,
    config_name: str,
    trial: int = 1,
) -> None:
    """Generate interactive trajectory.html for baseline (single-agent) execution.

    Args:
        local_dir: Local directory containing task results
        task_id: Task ID
        github_repo: GitHub repository (e.g., "user/repo")
        github_path: Path prefix in GitHub (e.g., "osworld")
        task_type: Task type (e.g., "collaborative", "standard")
        config_name: Config name (e.g., "baseline")
        trial: Trial number
    """
    local_path = pathlib.Path(local_dir)
    if not local_path.is_dir():
        return

    # Read task instruction
    task_txt = local_path / "task.txt"
    instruction = ""
    if task_txt.is_file():
        instruction = task_txt.read_text(encoding="utf-8", errors="replace").strip()

    # Read action log
    action_log_path = local_path / "action_log.json"
    action_log: list[dict] = []
    if action_log_path.is_file():
        try:
            action_log = json.loads(action_log_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Read result score
    result_path = local_path / "result.txt"
    score = None
    if result_path.is_file():
        try:
            score = float(result_path.read_text(encoding="utf-8").strip())
        except (ValueError, OSError):
            pass

    # Read token usage
    token_usage_path = local_path / "token_usage.json"
    token_usage: dict = {}
    if token_usage_path.is_file():
        try:
            token_usage = json.loads(token_usage_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    wall_clock = token_usage.get("wall_clock_seconds") or token_usage.get("total_latency_seconds") or 0
    cost = token_usage.get("total_cost_usd") or 0

    # Find step directories
    step_dirs = sorted(
        (d for d in local_path.iterdir() if d.is_dir() and d.name.startswith("step_")),
        key=lambda d: d.name,
    )

    # Format duration helper
    def fmt_duration(secs: float) -> str:
        if secs < 60:
            return f"{secs:.0f}s"
        m, s = divmod(int(secs), 60)
        return f"{m}m {s}s"

    # Build step data
    steps = []
    for step_dir in step_dirs:
        step_name = step_dir.name
        step_num = int(step_name.replace("step_", "").lstrip("0") or "0")

        # Read thinking/response
        thinking = ""
        response_file = step_dir / "response.txt"
        if response_file.is_file():
            thinking = response_file.read_text(encoding="utf-8", errors="replace").strip()

        # Screenshot
        screenshot_file = step_dir / "screenshot.png"
        screenshot_url = f"{step_name}/screenshot.png" if screenshot_file.is_file() else ""

        # Action and timestamp from action log
        action = ""
        timestamp = 0.0
        if step_num - 1 < len(action_log):
            entry = action_log[step_num - 1]
            actions = entry.get("actions", [])
            if actions:
                action = ", ".join(str(a) for a in actions)
            timestamp = entry.get("timestamp", 0.0)

        steps.append({
            'num': step_num,
            'thinking': thinking,
            'screenshot': screenshot_url,
            'action': action,
            'timestamp': timestamp,
        })

    # Generate HTML
    img_base = f"https://raw.githubusercontent.com/{github_repo}/main/{github_path}/{task_type}/{task_id}/{config_name}/trial_{trial}"

    h = []
    h.append("<!DOCTYPE html>")
    h.append("<html lang='en'>")
    h.append("<head>")
    h.append("  <meta charset='UTF-8'>")
    h.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
    h.append(f"  <title>Baseline Trajectory: {task_id}</title>")
    h.append("  <style>")
    h.append("""
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0d1117;
    color: #e6edf3;
    line-height: 1.6;
    padding: 20px;
}
.container {
    max-width: 1400px;
    margin: 0 auto;
}
h1 {
    font-size: 2em;
    margin-bottom: 20px;
    color: #58a6ff;
}
.summary {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 30px;
}
.summary-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin-top: 15px;
}
.summary-item {
    background: #0d1117;
    padding: 12px;
    border-radius: 8px;
    border: 1px solid #21262d;
}
.summary-label {
    font-size: 0.85em;
    color: #8b949e;
    margin-bottom: 5px;
}
.summary-value {
    font-size: 1.3em;
    color: #e6edf3;
    font-weight: 600;
}
.timeline-scrubber {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 30px;
}
.timeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}
.timeline-controls {
    display: flex;
    align-items: center;
    gap: 12px;
}
.timeline-play-button {
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
}
.timeline-play-button:hover {
    background: linear-gradient(135deg, #2ea043 0%, #238636 100%);
    transform: scale(1.05);
    box-shadow: 0 4px 12px rgba(35, 134, 54, 0.5);
}
.timeline-play-button:active {
    transform: scale(0.95);
}
.timeline-speed-selector {
    background: #161b22;
    border: 1px solid #30363d;
    color: #e6edf3;
    padding: 6px 10px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 0.9em;
    font-weight: 600;
    transition: all 0.2s ease;
}
.timeline-speed-selector:hover {
    background: #21262d;
    border-color: #58a6ff;
}
.timeline-time {
    font-size: 1.4em;
    background: linear-gradient(90deg, #58a6ff 0%, #79c0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
.timeline-slider {
    width: 100%;
    height: 20px;
    background: #0d1117;
    border-radius: 10px;
    position: relative;
    cursor: pointer;
    border: 1px solid #21262d;
}
.timeline-progress {
    position: absolute;
    height: 100%;
    background: linear-gradient(90deg, #238636 0%, #2ea043 100%);
    border-radius: 10px;
    transition: width 0.1s ease-out;
}
.timeline-knob {
    position: absolute;
    width: 16px;
    height: 16px;
    background: #ffffff;
    border-radius: 50%;
    top: 50%;
    transform: translate(-50%, -50%);
    cursor: grab;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.5);
    transition: left 0.1s ease-out;
}
.timeline-knob:active {
    cursor: grabbing;
    transform: translate(-50%, -50%) scale(1.2);
}
.display-panel {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 30px;
}
.display-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 15px;
}
.display-step {
    font-size: 0.9em;
    color: #8b949e;
}
.display-content {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    align-items: start;
}
.display-screenshot {
    width: 100%;
}
.display-screenshot img {
    width: 100%;
    border-radius: 8px;
    border: 1px solid #30363d;
}
.display-thinking-container {
    min-height: 100px;
}
.display-panel-thinking {
    font-size: 0.85em;
    color: #c9d1d9;
    background: linear-gradient(135deg, #1c2128 0%, #22272e 100%);
    padding: 12px;
    border-radius: 8px;
    border-left: 3px solid #58a6ff;
    white-space: pre-wrap;
    line-height: 1.6;
    font-style: italic;
}
.display-panel-thinking::before {
    content: '💭 ';
    opacity: 0.6;
}
.action-log {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px;
}
.action-log h2 {
    margin-bottom: 15px;
    font-size: 1.3em;
}
.action-item {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 10px 12px;
    margin-bottom: 8px;
    display: grid;
    grid-template-columns: 140px 1fr;
    gap: 12px;
    font-size: 0.9em;
}
.action-step {
    color: #8b949e;
    font-weight: 600;
}
.action-detail {
    color: #c9d1d9;
    word-wrap: break-word;
    line-height: 1.6;
}
""")
    h.append("  </style>")
    h.append("</head>")
    h.append("<body>")
    h.append("  <div class='container'>")
    h.append(f"    <h1>Baseline Trajectory: {task_id}</h1>")

    # Summary
    h.append("    <div class='summary'>")
    h.append("      <div class='summary-grid'>")
    h.append("        <div class='summary-item'>")
    h.append("          <div class='summary-label'>Score</div>")
    h.append(f"          <div class='summary-value'>{score if score is not None else 'N/A'}</div>")
    h.append("        </div>")
    h.append("        <div class='summary-item'>")
    h.append("          <div class='summary-label'>Duration</div>")
    h.append(f"          <div class='summary-value'>{fmt_duration(wall_clock)}</div>")
    h.append("        </div>")
    h.append("        <div class='summary-item'>")
    h.append("          <div class='summary-label'>Steps</div>")
    h.append(f"          <div class='summary-value'>{len(steps)}</div>")
    h.append("        </div>")
    h.append("        <div class='summary-item'>")
    h.append("          <div class='summary-label'>Cost</div>")
    h.append(f"          <div class='summary-value'>${cost:.2f}</div>")
    h.append("        </div>")
    h.append("      </div>")
    h.append("    </div>")

    # Timeline scrubber
    h.append("    <div class='timeline-scrubber'>")
    h.append("      <h2>⏱️ Execution Timeline</h2>")
    h.append("      <div class='timeline-header'>")
    h.append("        <div class='timeline-controls'>")
    h.append("          <button class='timeline-play-button' id='timeline-play-button' title='Play/Pause'>▶️</button>")
    h.append("          <select class='timeline-speed-selector' id='timeline-speed-selector'>")
    h.append("            <option value='1'>1x</option>")
    h.append("            <option value='2'>2x</option>")
    h.append("            <option value='4' selected>4x</option>")
    h.append("            <option value='8'>8x</option>")
    h.append("            <option value='16'>16x</option>")
    h.append("          </select>")
    h.append("          <div class='timeline-time' id='timeline-time'>Step 0</div>")
    h.append("        </div>")
    h.append(f"        <div style='color:#8b949e;font-size:0.85em'>Total: {len(steps)} steps</div>")
    h.append("      </div>")
    h.append("      <div class='timeline-slider' id='timeline-slider'>")
    h.append("        <div class='timeline-progress' id='timeline-progress' style='width: 0%'></div>")
    h.append("        <div class='timeline-knob' id='timeline-knob' style='left: 0%'></div>")
    h.append("      </div>")
    h.append("    </div>")

    # Display panel
    h.append("    <div class='display-panel'>")
    h.append("      <div class='display-header'>")
    h.append("        <h2>📺 Display</h2>")
    h.append("        <div class='display-step' id='display-step'>Step —</div>")
    h.append("      </div>")
    h.append("      <div class='display-content'>")
    h.append("        <div class='display-screenshot'>")
    h.append("          <img id='display-img' src='' alt='No screenshot' style='display:none'>")
    h.append("        </div>")
    h.append("        <div class='display-thinking-container'>")
    h.append("          <div id='display-thinking' class='display-panel-thinking' style='display:none'></div>")
    h.append("        </div>")
    h.append("      </div>")
    h.append("    </div>")

    # Action log
    h.append("    <div class='action-log'>")
    h.append("      <h2>📋 Action Log</h2>")
    for i, step in enumerate(steps):
        h.append("      <div class='action-item'>")
        time_str = fmt_duration(step['timestamp'])
        h.append(f"        <div class='action-step'>Step {step['num']} • {time_str}</div>")
        h.append(f"        <div class='action-detail'>{step['action'] if step['action'] else '—'}</div>")
        h.append("      </div>")
    h.append("    </div>")

    h.append("  </div>")

    # JavaScript
    h.append("  <script>")
    h.append(f"const steps = {json.dumps(steps)};")
    h.append(f"const imgBase = '{img_base}';")
    h.append("""
let currentStepIndex = 0;
let currentPlaybackPosition = 0;  // For smooth playback
let isPlaying = false;
let animationId = null;
let lastFrameTime = null;
let playbackSpeed = 4;

function formatDuration(secs) {
    if (secs < 60) {
        return Math.floor(secs) + 's';
    }
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return m + 'm ' + s + 's';
}

function updateDisplay(stepIndex) {
    if (steps.length === 0) return;

    currentStepIndex = Math.max(0, Math.min(Math.floor(stepIndex), steps.length - 1));
    const step = steps[currentStepIndex];

    const timeStr = formatDuration(step.timestamp);
    document.getElementById('timeline-time').textContent = 'Step ' + step.num + ' • ' + timeStr;

    const progress = steps.length > 1 ? (currentStepIndex / (steps.length - 1)) * 100 : 0;
    document.getElementById('timeline-progress').style.width = progress + '%';
    document.getElementById('timeline-knob').style.left = progress + '%';

    const imgEl = document.getElementById('display-img');
    const stepEl = document.getElementById('display-step');
    const thinkingEl = document.getElementById('display-thinking');

    stepEl.textContent = 'Step ' + step.num + ' • ' + timeStr;

    if (step.screenshot) {
        imgEl.src = imgBase + '/' + step.screenshot;
        imgEl.style.display = 'block';
    } else {
        imgEl.style.display = 'none';
    }

    if (step.thinking) {
        thinkingEl.textContent = step.thinking;
        thinkingEl.style.display = 'block';
    } else {
        thinkingEl.style.display = 'none';
    }
}

function animate(timestamp) {
    if (!isPlaying) return;

    if (lastFrameTime === null) {
        lastFrameTime = timestamp;
    }

    const deltaTime = (timestamp - lastFrameTime) / 1000;
    lastFrameTime = timestamp;

    const stepsPerSecond = playbackSpeed;
    currentPlaybackPosition += deltaTime * stepsPerSecond;

    if (currentPlaybackPosition >= steps.length - 1) {
        isPlaying = false;
        currentPlaybackPosition = steps.length - 1;
        updateDisplay(currentPlaybackPosition);
        document.getElementById('timeline-play-button').textContent = '▶️';
        lastFrameTime = null;
    } else {
        updateDisplay(currentPlaybackPosition);
        animationId = requestAnimationFrame(animate);
    }
}

function togglePlay() {
    isPlaying = !isPlaying;
    const button = document.getElementById('timeline-play-button');

    if (isPlaying) {
        button.textContent = '⏸️';
        if (currentStepIndex >= steps.length - 1) {
            currentPlaybackPosition = 0;
            updateDisplay(0);
        } else {
            currentPlaybackPosition = currentStepIndex;
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

document.getElementById('timeline-speed-selector').addEventListener('change', (e) => {
    playbackSpeed = parseFloat(e.target.value);
});

const slider = document.getElementById('timeline-slider');
const knob = document.getElementById('timeline-knob');

function seekToPosition(clientX) {
    const rect = slider.getBoundingClientRect();
    const x = Math.max(0, Math.min(clientX - rect.left, rect.width));
    const percent = x / rect.width;
    const stepIndex = percent * (steps.length - 1);
    currentPlaybackPosition = stepIndex;
    updateDisplay(stepIndex);
}

slider.addEventListener('click', (e) => {
    if (e.target === slider || e.target === document.getElementById('timeline-progress')) {
        if (isPlaying) togglePlay();
        seekToPosition(e.clientX);
    }
});

let isDragging = false;
knob.addEventListener('mousedown', (e) => {
    isDragging = true;
    if (isPlaying) togglePlay();
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

document.addEventListener('keydown', (e) => {
    if (e.key === ' ') {
        e.preventDefault();
        togglePlay();
    } else if (e.key === 'ArrowLeft') {
        currentPlaybackPosition = Math.max(0, currentStepIndex - 1);
        updateDisplay(currentPlaybackPosition);
    } else if (e.key === 'ArrowRight') {
        currentPlaybackPosition = Math.min(steps.length - 1, currentStepIndex + 1);
        updateDisplay(currentPlaybackPosition);
    } else if (e.key === 'Home') {
        currentPlaybackPosition = 0;
        updateDisplay(0);
    } else if (e.key === 'End') {
        currentPlaybackPosition = steps.length - 1;
        updateDisplay(steps.length - 1);
    }
});

// Initialize
updateDisplay(0);
""")
    h.append("  </script>")
    h.append("</body>")
    h.append("</html>")

    html_path = local_path / "trajectory.html"
    html_path.write_text("\n".join(h), encoding="utf-8")
    logger.info("Generated %s (%d steps)", html_path, len(steps))
