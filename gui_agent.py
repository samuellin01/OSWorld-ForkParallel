"""GUI agent for subtask execution on a single display.

A fresh GUI agent is created per subtask. It receives a text instruction
(with context from the orchestrator), executes CUA actions on its assigned
display, and returns a text summary + final screenshot.

This module is used by the orchestrator in run_parallel_task.py.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import requests

from agent_utils import (
    COMPUTER_USE_TOOL,
    _resize_screenshot,
    build_observation_message,
    filter_to_n_most_recent_images,
    parse_computer_use_actions,
)
from bedrock_client import BedrockClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Display wrappers
# ---------------------------------------------------------------------------

class NativeDisplay:
    """Uses the VM server's built-in /screenshot and /execute endpoints."""

    def __init__(self, vm_ip: str, server_port: int):
        self.vm_ip = vm_ip
        self.port = server_port
        self.display = ":0"
        self.base_url = f"http://{vm_ip}:{server_port}"

    def screenshot(self) -> Optional[bytes]:
        try:
            r = requests.get(f"{self.base_url}/screenshot", timeout=15)
            if r.status_code == 200 and len(r.content) > 100:
                return r.content
        except Exception as e:
            logger.warning("[%s] screenshot failed: %s", self.display, e)
        return None

    def run_action(self, pyautogui_code: str) -> Optional[dict]:
        prefix = "import pyautogui; import time; pyautogui.FAILSAFE = False\n"
        full_code = prefix + pyautogui_code
        try:
            r = requests.post(
                f"{self.base_url}/execute",
                json={"command": ["python3", "-c", full_code], "shell": False},
                timeout=30,
            )
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.warning("[%s] action failed: %s", self.display, e)
            return None


class XvfbDisplay:
    """Targets an Xvfb display via scrot + DISPLAY=:N python3."""

    def __init__(self, vm_ip: str, server_port: int, display_num: int):
        self.vm_ip = vm_ip
        self.port = server_port
        self.display = f":{display_num}"
        self.exec_url = f"http://{vm_ip}:{server_port}/setup/execute"

    def _shell(self, cmd: str, timeout: int = 60) -> Optional[dict]:
        try:
            r = requests.post(
                self.exec_url,
                json={"command": cmd, "shell": True},
                timeout=timeout,
            )
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.warning("[%s] shell failed: %s", self.display, e)
            return None

    def screenshot(self) -> Optional[bytes]:
        tmp = f"/tmp/par_shot_{self.display.replace(':', '')}.png"
        # Try multiple screenshot methods in order of reliability.
        shot_result = self._shell(
            f"DISPLAY={self.display} python3 -c \""
            f"import pyautogui; pyautogui.screenshot('{tmp}')\" 2>&1"
        )
        # Check if pyautogui worked
        check = self._shell(f"test -s {tmp} && echo OK || echo FAIL")
        if not check or "OK" not in check.get("output", ""):
            # pyautogui failed — try xdotool + import or scrot as fallbacks
            err = shot_result.get("output", "") if shot_result else "no response"
            logger.warning("[%s] pyautogui screenshot failed: %s — trying fallbacks", self.display, err[:150])
            self._shell(
                f"DISPLAY={self.display} scrot -o {tmp} 2>/dev/null || "
                f"DISPLAY={self.display} import -window root {tmp} 2>/dev/null || "
                f"DISPLAY={self.display} xwd -root -silent 2>/dev/null | "
                f"convert xwd:- {tmp} 2>/dev/null"
            )
        result = self._shell(f"base64 -w0 {tmp}")
        if result and result.get("output"):
            try:
                return base64.b64decode(result["output"].strip())
            except Exception:
                pass
        logger.warning("[%s] screenshot failed — all methods exhausted", self.display)
        return None

    def run_action(self, pyautogui_code: str) -> Optional[dict]:
        tmp = f"/tmp/par_action_{self.display.replace(':', '')}.py"
        full_code = (
            "import pyautogui; import time; pyautogui.FAILSAFE = False\n"
            + pyautogui_code
        )
        write_cmd = f"cat > {tmp} << 'ACTIONEOF'\n{full_code}\nACTIONEOF"
        self._shell(write_cmd)
        return self._shell(f"DISPLAY={self.display} python3 {tmp}", timeout=30)


# ---------------------------------------------------------------------------
# Subtask result
# ---------------------------------------------------------------------------

@dataclass
class SubtaskResult:
    """Result returned by a GUI agent after completing a subtask."""
    subtask: str
    status: str  # "DONE", "FAIL", "MAX_STEPS"
    summary: str
    final_screenshot: Optional[bytes] = None
    steps_used: int = 0
    token_usage: Optional[Dict[str, Any]] = None
    duration_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# System prompt for subtask agents
# ---------------------------------------------------------------------------

_SUBTASK_SYSTEM_PROMPT_NATIVE = (
    "You are a computer-use agent on Ubuntu 22.04 with GNOME. "
    "You are executing a specific subtask as part of a larger task. "
    "Password: '{password}'. Home directory: /home/user. "
    "If you launch Chrome from the terminal, add --remote-debugging-port={chrome_port}. "
    "When you have completed your subtask, output DONE followed by a brief "
    "summary of what you accomplished and where any output files are saved. "
    "Output FAIL if the subtask is impossible."
)

_SUBTASK_SYSTEM_PROMPT_XVFB = (
    "You are a computer-use agent on Ubuntu 22.04 with a minimal openbox "
    "desktop. An xterm terminal window is already open and ready for use. "
    "Use the terminal to run commands. Right-click the desktop background "
    "for the openbox application menu. "
    "Password: '{password}'. Home directory: /home/user. "
    "If you launch Chrome, run: google-chrome --no-sandbox --disable-gpu "
    "--remote-debugging-port={chrome_port} --user-data-dir=/tmp/chrome-profile-{display_num} URL "
    "When you have completed your subtask, output DONE followed by a brief "
    "summary of what you accomplished and where any output files are saved. "
    "Output FAIL if the subtask is impossible."
)


# ---------------------------------------------------------------------------
# GUI agent loop
# ---------------------------------------------------------------------------

def run_gui_agent(
    display: NativeDisplay | XvfbDisplay,
    subtask: str,
    context: str,
    bedrock: BedrockClient,
    model: str,
    max_steps: int = 25,
    temperature: float = 0.7,
    output_dir: Optional[str] = None,
    password: str = "osworld-public-evaluation",
    resize_factor: Tuple[float, float] = (1920.0 / 1280.0, 1080.0 / 720.0),
    cancel_event: Optional[Any] = None,
) -> SubtaskResult:
    """Run a CUA agent loop on one display for a single subtask.

    Parameters
    ----------
    display : NativeDisplay or XvfbDisplay
        The display to interact with.
    subtask : str
        The specific subtask instruction.
    context : str
        Context from the orchestrator (original task + progress so far).
    bedrock : BedrockClient
        Bedrock client for LLM calls.
    model : str
        Model name.
    max_steps : int
        Maximum steps for this subtask.
    temperature : float
        Sampling temperature.
    output_dir : str or None
        Directory to save screenshots and logs.
    password : str
        VM sudo password.
    resize_factor : tuple
        Scale factor from model coords to screen coords.

    Returns
    -------
    SubtaskResult
        The result of executing the subtask.
    """
    tag = f"gui-agent[{display.display}]"

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "subtask.txt"), "w") as f:
            f.write(f"Context:\n{context}\n\nSubtask:\n{subtask}\n")

    # Each display gets a unique Chrome debugging port to avoid
    # one agent's Chrome opening tabs in another agent's window.
    # Native display :0 uses port 1337, Xvfb displays use 1337 + display_num.
    if isinstance(display, XvfbDisplay):
        display_num = int(display.display.replace(":", ""))
        chrome_port = 1337 + display_num
        system_prompt = _SUBTASK_SYSTEM_PROMPT_XVFB.format(
            password=password, chrome_port=chrome_port, display_num=display_num,
        )
    else:
        system_prompt = _SUBTASK_SYSTEM_PROMPT_NATIVE.format(
            password=password, chrome_port=1337,
        )
    tools = [COMPUTER_USE_TOOL]

    # Build initial message with context and subtask
    initial_text = f"Overall context:\n{context}\n\nYour subtask:\n{subtask}"
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": initial_text}]},
    ]

    last_tool_use_id: Optional[str] = None
    last_screenshot: Optional[bytes] = None
    final_response_text = ""

    for step in range(1, max_steps + 1):
        if cancel_event and cancel_event.is_set():
            logger.info("[%s] Cancelled at step %d", tag, step)
            return SubtaskResult(
                subtask=subtask,
                status="CANCELLED",
                summary="Task was cancelled by orchestrator.",
                final_screenshot=last_screenshot,
                steps_used=step - 1,
                token_usage=bedrock.get_token_usage(),
            )
        logger.info("[%s] Step %d/%d", tag, step, max_steps)

        # Take screenshot
        shot = display.screenshot()
        if shot:
            shot = _resize_screenshot(shot)
            last_screenshot = shot
            if output_dir:
                with open(os.path.join(output_dir, f"step_{step:03d}.png"), "wb") as f:
                    f.write(shot)

            obs_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"Step {step}: current desktop state."},
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(shot).decode(),
                    },
                },
            ]
        else:
            logger.warning("[%s] Screenshot returned None", tag)
            obs_content = [
                {"type": "text", "text": f"Step {step}: screenshot unavailable."},
            ]

        # Prepend tool_result if needed
        if last_tool_use_id:
            obs_content.insert(0, {
                "type": "tool_result",
                "tool_use_id": last_tool_use_id,
                "content": "Action executed.",
            })
            last_tool_use_id = None

        messages.append({"role": "user", "content": obs_content})


        # Call LLM
        content_blocks, _ = bedrock.chat(
            messages=messages,
            system=system_prompt,
            model=model,
            temperature=temperature,
            tools=tools,
        )
        messages.append({"role": "assistant", "content": content_blocks})

        # Extract response text
        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("[%s] Response: %s", tag, response_text[:200])
        final_response_text = response_text

        if output_dir:
            with open(os.path.join(output_dir, f"step_{step:03d}_response.txt"), "w") as f:
                f.write(response_text)

        # Parse actions
        actions = parse_computer_use_actions(content_blocks, resize_factor)

        # Track tool_use_id
        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                last_tool_use_id = block.get("id")
                break

        # Check for terminal actions
        if "DONE" in actions:
            logger.info("[%s] DONE at step %d", tag, step)
            return SubtaskResult(
                subtask=subtask,
                status="DONE",
                summary=response_text,
                final_screenshot=last_screenshot,
                steps_used=step,
                token_usage=bedrock.get_token_usage(),
            )
        if "FAIL" in actions:
            logger.info("[%s] FAIL at step %d", tag, step)
            return SubtaskResult(
                subtask=subtask,
                status="FAIL",
                summary=response_text,
                final_screenshot=last_screenshot,
                steps_used=step,
                token_usage=bedrock.get_token_usage(),
            )

        # Execute action
        action_code = next(
            (a for a in actions if a not in ("DONE", "FAIL", "WAIT", "CALL_USER")),
            None,
        )
        if action_code:
            logger.info("[%s] Action: %s", tag, action_code[:120])
            display.run_action(action_code)
            time.sleep(1)
        else:
            time.sleep(2)

    logger.warning("[%s] Max steps (%d) reached, requesting summary", tag, max_steps)

    # Ask the agent to summarize what it accomplished before we cut it off.
    summary_text = f"Reached max steps ({max_steps}). Last response: {final_response_text}"
    try:
        # The last assistant message may contain a tool_use block without a
        # corresponding tool_result.  Append a dummy tool_result so the API
        # accepts the follow-up user message.
        if last_tool_use_id:
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": last_tool_use_id,
                    "content": "Max steps reached. Action was not executed.",
                }],
            })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": (
                "You have run out of steps. Do NOT take any more actions. "
                "Provide a summary covering: "
                "1) What you successfully completed (file paths, values found). "
                "2) What approaches you tried and what worked vs didn't work. "
                "3) What is still incomplete. "
                "4) What approach you would recommend for a future agent "
                "continuing this task — what should they do differently?"
            )}],
        })
        summary_blocks, _ = bedrock.chat(
            messages=messages,
            system=system_prompt,
            model=model,
            temperature=temperature,
            tools=None,
        )
        summary_text = "".join(
            b.get("text", "") for b in summary_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("[%s] MAX_STEPS summary: %s", tag, summary_text[:300])
    except Exception as e:
        logger.warning("[%s] Failed to get MAX_STEPS summary: %s", tag, e)

    return SubtaskResult(
        subtask=subtask,
        status="MAX_STEPS",
        summary=summary_text,
        final_screenshot=last_screenshot,
        steps_used=max_steps,
        token_usage=bedrock.get_token_usage(),
    )
