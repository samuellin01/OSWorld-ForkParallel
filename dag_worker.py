"""Per-phase CUA worker execution.

The worker is a pure screen executor. It has two tools:
  - computer: interact with the screen
  - await_signal: block mid-loop until another agent's data arrives

The worker does NOT make parallelism decisions. Its manager (running
in a separate thread) watches the step history and handles that.

No hard step limit. The worker runs until it says COMPLETE or FAILED.
The task-level timeout is the hard wall. The manager can nudge the
worker to wrap up via messages.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import anthropic

from agent_utils import COMPUTER_USE_TOOL, _resize_screenshot, parse_computer_use_actions
from dag_core import AgentPlan, Phase, StepRecord, AWAIT_SIGNAL_TOOL
from fork_agent import XvfbDisplay

if TYPE_CHECKING:
    from dag_core import Orchestrator

logger = logging.getLogger(__name__)

_COMPLETION = re.compile(r'\b(PHASE\s+COMPLETE|SUBTASK\s+COMPLETE)\b', re.IGNORECASE)
_FAILURE = re.compile(r'\b(PHASE\s+FAILED|SUBTASK\s+FAILED)\b', re.IGNORECASE)

# Safety cap so a single phase can't loop forever if COMPLETE is never emitted
_ABSOLUTE_MAX_STEPS = 200


def run_phase(
    agent: AgentPlan,
    phase: Phase,
    phase_index: int,
    vm_ip: str,
    server_port: int,
    bedrock: Any,
    model: str,
    output_dir: str,
    password: str = "osworld-public-evaluation",
    signal_data: Optional[Dict[str, Any]] = None,
    orchestrator: Optional["Orchestrator"] = None,
) -> Dict[str, Any]:
    """Run a CUA agent loop for one phase. Returns result dict."""
    tag = f"[{agent.id}/{phase.id}]"
    display_num = agent.display_num or 0
    display = XvfbDisplay(vm_ip, server_port, display_num)
    system_prompt = _build_system_prompt(agent, phase, phase_index, password, signal_data,
                                          has_orchestrator=orchestrator is not None)
    tools: List[Any] = [COMPUTER_USE_TOOL]
    if orchestrator:
        tools.append(AWAIT_SIGNAL_TOOL)

    resize_factor = (1920.0 / 1280.0, 1080.0 / 720.0)
    is_continuation = phase_index > 0

    initial_text = (
        f"Continue on the same display. Your current phase:\n{phase.task}"
        if is_continuation else
        f"Your task:\n{phase.task}"
    )

    if signal_data:
        initial_text += "\n\nData from other agents:\n"
        for sig_name, data in signal_data.items():
            summary = data.get("summary", str(data))[:4000] if isinstance(data, dict) else str(data)[:4000]
            initial_text += f"  [{sig_name}]: {summary}\n"
        initial_text += "\nUse this data to complete your phase."

    phase_output = os.path.join(output_dir, phase.id)
    os.makedirs(phase_output, exist_ok=True)

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": initial_text}]}
    ]

    pending_tool_results: List[Dict[str, Any]] = []
    final_response_text = ""
    start_time = time.time()
    step = 0

    while True:
        step += 1

        # Safety cap
        if step > _ABSOLUTE_MAX_STEPS:
            logger.warning("%s Absolute step cap (%d) reached", tag, _ABSOLUTE_MAX_STEPS)
            break

        # Task timeout check
        if orchestrator and orchestrator._start_time:
            if (time.time() - orchestrator._start_time) > orchestrator.task_timeout:
                logger.warning("%s Task timeout reached", tag)
                break

        logger.info("%s Step %d", tag, step)

        shot = display.screenshot()
        step_timestamp = time.time()

        if shot:
            _save_screenshot(phase_output, step, shot, step_timestamp)
            obs_content = _build_screenshot_observation(step, shot)
        else:
            logger.warning("%s Screenshot failed at step %d", tag, step)
            obs_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"Step {step}: screenshot unavailable."},
            ]

        # Prepend pending tool results
        for tr in pending_tool_results:
            obs_content.insert(0, tr)
        pending_tool_results.clear()

        # Inject manager/orchestrator messages
        if orchestrator:
            pending = orchestrator.get_pending_messages(agent.id)
            if pending:
                msg_text = "\n".join(f"[MANAGER]: {m}" for m in pending)
                obs_content.append({"type": "text", "text": msg_text})

        messages.append({"role": "user", "content": obs_content})

        try:
            content_blocks, _ = bedrock.chat(
                messages=messages,
                system=system_prompt,
                model=model,
                temperature=0.7,
                tools=tools,
            )
        except anthropic.BadRequestError as e:
            error_msg = str(e)
            if "tool_use" in error_msg and "tool_result" in error_msg:
                logger.error("%s Conversation corruption at step %d", tag, step)
                return {
                    "status": "ERROR",
                    "summary": f"Conversation corruption: {error_msg}",
                    "steps_used": step,
                    "duration": time.time() - start_time,
                }
            raise

        messages.append({"role": "assistant", "content": content_blocks})

        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("%s Response: %s", tag, response_text[:200])
        final_response_text = response_text

        # Report progress (manager watches this)
        phase.current_step = step
        phase.latest_response = response_text

        with open(os.path.join(phase_output, f"step_{step:03d}_response.txt"), "w") as f:
            f.write(response_text)

        # Process tool calls
        step_action_summary = ""
        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_name = block.get("name")
            tool_id = block.get("id")
            tool_input = block.get("input", {})

            if tool_name == "computer":
                actions = parse_computer_use_actions([block], resize_factor)
                action_code = next(
                    (a for a in actions if a not in ("DONE", "FAIL", "WAIT")),
                    None,
                )
                if action_code:
                    logger.info("%s Action: %s", tag, action_code[:120])
                    display.run_action(action_code)
                    time.sleep(1)
                    step_action_summary = _action_summary(tool_input)
                    _save_action(phase_output, step, tool_input)

                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": "Action executed.",
                })

            elif tool_name == "await_signal" and orchestrator:
                signal_name = tool_input.get("signal_name", "")
                logger.info("%s await_signal('%s') — blocking", tag, signal_name)
                step_action_summary = f"await_signal({signal_name})"

                remaining_timeout = None
                if orchestrator._start_time:
                    remaining_timeout = max(1.0, orchestrator.task_timeout - (time.time() - orchestrator._start_time))

                data = orchestrator.wait_signal(signal_name, timeout=remaining_timeout)

                if data is None or orchestrator.is_signal_failed(signal_name):
                    result_content = json.dumps({"error": f"Signal '{signal_name}' failed or timed out"})
                    logger.warning("%s await_signal('%s') — failed", tag, signal_name)
                else:
                    summary = data.get("summary", str(data))[:4000] if isinstance(data, dict) else str(data)[:4000]
                    result_content = summary
                    logger.info("%s await_signal('%s') — received %d chars", tag, signal_name, len(result_content))

                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": result_content,
                })

            else:
                pending_tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": f"Unknown tool: {tool_name}",
                })

        # Record step in history (manager reads this)
        if not step_action_summary:
            step_action_summary = response_text[:80].replace("\n", " ")
        phase.step_history.append(StepRecord(
            step_num=step,
            timestamp=step_timestamp,
            elapsed=step_timestamp - start_time,
            action_summary=step_action_summary,
        ))

        # Check for completion
        for line in final_response_text.strip().split("\n"):
            line = line.strip()
            if _COMPLETION.search(line):
                logger.info("%s Complete at step %d", tag, step)
                return {
                    "status": "DONE",
                    "summary": final_response_text,
                    "steps_used": step,
                    "duration": time.time() - start_time,
                }
            if _FAILURE.search(line):
                logger.info("%s Failed at step %d", tag, step)
                return {
                    "status": "FAIL",
                    "summary": final_response_text,
                    "steps_used": step,
                    "duration": time.time() - start_time,
                }

    logger.warning("%s Exited loop at step %d (timeout or cap)", tag, step)
    return {
        "status": "FAIL",
        "summary": f"Worker stopped at step {step}. Last: {final_response_text}",
        "steps_used": step,
        "duration": time.time() - start_time,
    }


def _build_system_prompt(
    agent: AgentPlan,
    phase: Phase,
    phase_index: int,
    password: str,
    signal_data: Optional[Dict[str, Any]],
    has_orchestrator: bool = False,
) -> str:
    display_num = agent.display_num or 0
    chrome_port = 1337 + display_num
    is_last_phase = phase_index == len(agent.phases) - 1
    is_continuation = phase_index > 0

    prompt = (
        "You are a computer-use agent on Ubuntu 22.04 with openbox window manager. "
        f"Password: '{password}'. Home directory: /home/user. "
        "\n\n"
        "**Recovery**: Press Ctrl+Alt+T to open a terminal.\n\n"
        f"If you launch Chrome from terminal: google-chrome --remote-debugging-port={chrome_port} "
        f"--user-data-dir=/tmp/chrome_display_{display_num} --no-first-run "
        f"--no-default-browser-check --disable-default-apps URL\n\n"
    )

    if is_continuation:
        prompt += (
            "**DISPLAY CONTINUITY**: Your display carries over from the previous phase. "
            "Windows and tabs are already open. Do NOT re-launch applications. "
            "Review the current screen and continue.\n\n"
        )
    else:
        prompt += "Your display has been prepared with the necessary applications.\n\n"

    if has_orchestrator:
        prompt += (
            "**COLLABORATION TOOL**:\n"
            "- `await_signal(signal_name)`: Block until data from another agent is ready. "
            "Do your independent setup work first (open apps, navigate), then call this "
            "only when you actually need the data. Returns the data as the tool result.\n\n"
            "Your manager may send you messages during execution (shown as [MANAGER]: ...). "
            "Follow their instructions — they have visibility into the overall task and may "
            "tell you to skip certain work because a helper agent is handling it.\n\n"
        )

    if signal_data:
        prompt += "**Data from other agents** (received via signals):\n"
        for sig_name, data in signal_data.items():
            summary = data.get("summary", str(data))[:4000] if isinstance(data, dict) else str(data)[:4000]
            prompt += f"  [{sig_name}]: {summary}\n"
        prompt += "\nUse this data. Do NOT redo work that other agents already completed.\n\n"

    if phase.signals:
        prompt += (
            "**IMPORTANT**: When you complete this phase, include a detailed summary of "
            "your results and any key data (values, findings, URLs) because another agent "
            "is waiting for this information.\n\n"
        )

    completion_kw = "SUBTASK COMPLETE" if is_last_phase else "PHASE COMPLETE"
    failure_kw = "SUBTASK FAILED" if is_last_phase else "PHASE FAILED"

    prompt += (
        f"When done, output {completion_kw} followed by a summary.\n"
        f"If you cannot complete this, output {failure_kw} with explanation.\n\n"
        "**If setup failed** — empty desktop or wrong app on first screenshot:\n"
        f"{failure_kw}: Setup did not work. Display shows [describe what you see].\n\n"
        "Google Docs/Sheets/Slides: multiple agents can open the same URL simultaneously.\n"
        "Google Workspace: Do NOT use Apps Script — complete tasks through the UI.\n"
        "Google Sheets: Use the Name Box (top-left) to jump to cells. Batch with Tab/Enter.\n\n"
        "Verify you've completed the work before declaring completion."
    )

    return prompt


def _action_summary(tool_input: Dict[str, Any]) -> str:
    action_type = tool_input.get("action", "")
    if action_type == "type":
        text = tool_input.get("text", "")
        return f"type: {text[:40]}" if len(text) <= 40 else f"type: {text[:37]}..."
    elif action_type == "key":
        return f"key: {tool_input.get('text', '')}"
    elif action_type in ("left_click", "right_click", "double_click", "middle_click"):
        return f"{action_type} at {tool_input.get('coordinate', [])}"
    elif action_type == "mouse_move":
        return f"move to {tool_input.get('coordinate', [])}"
    elif action_type == "scroll":
        return f"scroll {tool_input.get('scroll_direction', '')} {tool_input.get('scroll_amount', 3)}"
    elif action_type == "screenshot":
        return "screenshot"
    return f"computer: {action_type}"


def _save_screenshot(phase_output: str, step: int, shot: bytes, timestamp: float):
    with open(os.path.join(phase_output, f"step_{step:03d}.png"), "wb") as f:
        f.write(shot)
    with open(os.path.join(phase_output, f"step_{step:03d}_timestamp.txt"), "w") as f:
        f.write(f"{timestamp:.6f}\n")


def _build_screenshot_observation(step: int, shot: bytes) -> List[Dict[str, Any]]:
    resized = _resize_screenshot(shot)
    return [
        {"type": "text", "text": f"Step {step}: current desktop state."},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(resized).decode(),
            },
        },
    ]


def _save_action(phase_output: str, step: int, tool_input: Dict[str, Any]):
    text = _action_summary(tool_input)
    with open(os.path.join(phase_output, f"step_{step:03d}_action.txt"), "w") as f:
        f.write(text)
