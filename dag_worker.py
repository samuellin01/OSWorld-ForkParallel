"""Per-phase CUA agent execution for the signal/await orchestrator.

Each agent runs its phases sequentially on the same display. A phase
is a CUA loop (screenshot → LLM → action → repeat) that ends when
the agent says PHASE COMPLETE or SUBTASK COMPLETE.

Signal coordination is handled by the Orchestrator. This module only
handles the CUA execution within a single phase.
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional

import anthropic

from agent_utils import COMPUTER_USE_TOOL, _resize_screenshot, parse_computer_use_actions
from dag_core import AgentPlan, Phase
from fork_agent import XvfbDisplay

logger = logging.getLogger(__name__)

_COMPLETION = re.compile(r'\b(PHASE\s+COMPLETE|SUBTASK\s+COMPLETE)\b', re.IGNORECASE)
_FAILURE = re.compile(r'\b(PHASE\s+FAILED|SUBTASK\s+FAILED)\b', re.IGNORECASE)


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
) -> Dict[str, Any]:
    """Run a CUA agent loop for one phase. Returns result dict."""
    tag = f"[{agent.id}/{phase.id}]"
    display_num = agent.display_num or 0
    display = XvfbDisplay(vm_ip, server_port, display_num)
    system_prompt = _build_system_prompt(agent, phase, phase_index, password, signal_data)
    tools = [COMPUTER_USE_TOOL]
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
            summary = data.get("summary", str(data))[:500] if isinstance(data, dict) else str(data)[:500]
            initial_text += f"  [{sig_name}]: {summary}\n"
        initial_text += "\nUse this data to complete your phase."

    phase_output = os.path.join(output_dir, phase.id)
    os.makedirs(phase_output, exist_ok=True)

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": initial_text}]}
    ]

    last_tool_use_id: Optional[str] = None
    final_response_text = ""
    start_time = time.time()

    for step in range(1, phase.max_steps + 1):
        logger.info("%s Step %d/%d", tag, step, phase.max_steps)

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

        if last_tool_use_id:
            obs_content.insert(0, {
                "type": "tool_result",
                "tool_use_id": last_tool_use_id,
                "content": "Action executed.",
            })
            last_tool_use_id = None

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

        with open(os.path.join(phase_output, f"step_{step:03d}_response.txt"), "w") as f:
            f.write(response_text)

        # Execute computer-use actions
        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue
            if block.get("name") != "computer":
                continue

            last_tool_use_id = block.get("id")
            tool_input = block.get("input", {})
            actions = parse_computer_use_actions([block], resize_factor)
            action_code = next(
                (a for a in actions if a not in ("DONE", "FAIL", "WAIT")),
                None,
            )

            if action_code:
                logger.info("%s Action: %s", tag, action_code[:120])
                display.run_action(action_code)
                time.sleep(1)
                _save_action(phase_output, step, tool_input)

        # Check for phase/subtask completion
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

    logger.warning("%s Max steps (%d) reached", tag, phase.max_steps)
    return {
        "status": "MAX_STEPS",
        "summary": f"Reached max steps. Last: {final_response_text}",
        "steps_used": phase.max_steps,
        "duration": time.time() - start_time,
    }


def _build_system_prompt(
    agent: AgentPlan,
    phase: Phase,
    phase_index: int,
    password: str,
    signal_data: Optional[Dict[str, Any]],
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

    if signal_data:
        prompt += "**Data from other agents** (received via signals):\n"
        for sig_name, data in signal_data.items():
            summary = data.get("summary", str(data))[:500] if isinstance(data, dict) else str(data)[:500]
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
    action_type = tool_input.get("action", "")
    labels = {
        "type": f"Type: {tool_input.get('text', '')}",
        "key": f"Key: {tool_input.get('text', '')}",
        "mouse_move": f"Move to {tool_input.get('coordinate', [])}",
        "screenshot": "Screenshot",
    }
    click_types = ("left_click", "right_click", "double_click", "middle_click")

    if action_type in labels:
        text = labels[action_type]
    elif action_type in click_types:
        text = f"{action_type.replace('_', ' ').title()} at {tool_input.get('coordinate', [])}"
    else:
        text = f"Computer: {action_type}"

    with open(os.path.join(phase_output, f"step_{step:03d}_action.txt"), "w") as f:
        f.write(text)
