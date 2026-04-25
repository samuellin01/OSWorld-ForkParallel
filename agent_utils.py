"""Shared utilities for run_task.py and run_bootstrap_experiment.py.

Contains the computer-use tool definition, screenshot resizer,
action parser, observation-message builder, and screenshot memory management.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Screenshot memory management (adapted from mm_agents/anthropic/utils.py)
# ---------------------------------------------------------------------------

def filter_to_n_most_recent_images(
    messages: List[Dict[str, Any]],
    images_to_keep: int,
    min_removal_threshold: int = 10,
) -> None:
    """Remove old screenshot images from messages, keeping only the most recent N.

    Adapted from Anthropic's reference CUA implementation.  Screenshots are of
    diminishing value as the conversation progresses — old ones waste context.

    Operates **in-place** on the messages list.  Removes images in chunks of
    ``min_removal_threshold`` to preserve prompt-cache efficiency.
    """
    if images_to_keep is None:
        return

    # Collect (message_index, content_index) of every image block.
    image_locations: List[tuple[int, int]] = []
    for msg_idx, message in enumerate(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block_idx, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image":
                image_locations.append((msg_idx, block_idx))

    total_images = len(image_locations)
    images_to_remove = total_images - images_to_keep
    if images_to_remove <= 0:
        return

    # Round down to chunks for better cache behavior.
    images_to_remove -= images_to_remove % min_removal_threshold

    if images_to_remove <= 0:
        return

    # Remove oldest images (front of the list).
    removed = 0
    for msg_idx, block_idx in image_locations:
        if removed >= images_to_remove:
            break
        content = messages[msg_idx]["content"]
        block = content[block_idx]
        if isinstance(block, dict) and block.get("type") == "image":
            # Replace image with a placeholder text so message structure stays valid.
            content[block_idx] = {
                "type": "text",
                "text": "[Screenshot removed to save context]",
            }
            removed += 1

    logger.info(
        "Image filter: kept %d, removed %d (total was %d)",
        total_images - removed, removed, total_images,
    )


# ---------------------------------------------------------------------------
# Computer-use tool definition
# ---------------------------------------------------------------------------

COMPUTER_USE_TOOL: Dict[str, Any] = {
    "type": "computer_20251124",
    "name": "computer",
    "display_width_px": 1280,
    "display_height_px": 720,
    "display_number": 1,
}


# ---------------------------------------------------------------------------
# Helper: resize screenshot bytes to 1280×720 for computer-use tool
# ---------------------------------------------------------------------------

def _resize_screenshot(screenshot_bytes: bytes) -> bytes:
    """Resize raw screenshot bytes to 1280×720 using PIL (LANCZOS).

    Claude's computer-use tool is calibrated for 1280×720.  Downscaling
    reduces token usage and may improve coordinate accuracy.
    """
    from PIL import Image  # noqa: PLC0415

    img = Image.open(io.BytesIO(screenshot_bytes))
    resized = img.resize((1280, 720), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    resized.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Helper: parse computer-use tool_use content blocks into action strings
# ---------------------------------------------------------------------------

def parse_computer_use_actions(
    content_blocks: List[Dict[str, Any]],
    resize_factor: Tuple[float, float],
) -> List[str]:
    """Convert computer-use ``tool_use`` content blocks into pyautogui action strings.

    Adapted from ``confucius/analects/osworld/agent.py``.  Operates on the
    list of content-block dicts returned by the Bedrock API.

    Parameters
    ----------
    content_blocks:
        List of content-block dicts from the AI message.  Each dict has at
        least a ``"type"`` key; ``tool_use`` blocks also carry ``"name"`` and
        ``"input"``.
    resize_factor:
        ``(x_factor, y_factor)`` used to scale model-space coordinates
        (1280×720) back to screen-space coordinates (e.g. 1920×1080).

    Returns
    -------
    A list of action strings — either special tokens (``DONE`` / ``FAIL`` /
    ``WAIT``) or snippets of pyautogui Python code — ready for OSWorld.
    """
    # Check for [INFEASIBLE] in any text block first.
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and "[INFEASIBLE]" in block.get("text", ""):
            return ["FAIL"]

    # Detect DONE/FAIL in text-only responses (no tool_use blocks).
    has_tool_use = any(
        isinstance(b, dict) and b.get("type") == "tool_use" for b in content_blocks
    )
    if not has_tool_use:
        combined_text = " ".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        if re.search(r"\bDONE\b", combined_text, re.IGNORECASE):
            return ["DONE"]
        if re.search(r"\bFAIL\b", combined_text, re.IGNORECASE):
            return ["FAIL"]

    actions: List[str] = []

    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "tool_use":
            continue

        tool_input: Dict[str, Any] = block.get("input", {})
        action: Optional[str] = tool_input.get("action")
        if not action:
            continue

        # Normalise legacy action name variants.
        action_conversion = {
            "left click": "click",
            "right click": "right_click",
        }
        action = action_conversion.get(action, action)

        text: Optional[str] = tool_input.get("text")
        coordinate: Optional[List[int]] = tool_input.get("coordinate")
        start_coordinate: Optional[List[int]] = tool_input.get("start_coordinate")
        scroll_direction: Optional[str] = tool_input.get("scroll_direction")
        scroll_amount = tool_input.get("scroll_amount", 3)
        duration = tool_input.get("duration")

        # Scale coordinates from model space (1280×720) to screen space.
        if coordinate:
            coordinate = [
                int(coordinate[0] * resize_factor[0]),
                int(coordinate[1] * resize_factor[1]),
            ]
        if start_coordinate:
            start_coordinate = [
                int(start_coordinate[0] * resize_factor[0]),
                int(start_coordinate[1] * resize_factor[1]),
            ]

        result = ""

        if action == "left_mouse_down":
            result = "pyautogui.mouseDown()\n"
        elif action == "left_mouse_up":
            result = "pyautogui.mouseUp()\n"
        elif action == "hold_key":
            if text:
                for key in text.split("+"):
                    result += f"pyautogui.keyDown('{key.strip().lower()}')\n"
        elif action in ("mouse_move", "left_click_drag"):
            if coordinate:
                x, y = coordinate[0], coordinate[1]
                if action == "mouse_move":
                    result = f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"
                else:  # left_click_drag
                    if start_coordinate:
                        sx, sy = start_coordinate[0], start_coordinate[1]
                        result += f"pyautogui.moveTo({sx}, {sy}, duration={duration or 0.5})\n"
                    result += f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"
        elif action in ("key", "type"):
            if text:
                if action == "key":
                    key_conversion = {
                        "page_down": "pagedown",
                        "page_up": "pageup",
                        "super_l": "win",
                        "super": "command",
                        "escape": "esc",
                    }
                    keys = text.split("+")
                    for key in keys:
                        k = key_conversion.get(key.strip().lower(), key.strip().lower())
                        result += f"pyautogui.keyDown('{k}')\n"
                    for key in reversed(keys):
                        k = key_conversion.get(key.strip().lower(), key.strip().lower())
                        result += f"pyautogui.keyUp('{k}')\n"
                else:  # type
                    # Use xdotool for text input with inter-chunk sleeps
                    # to prevent character dropping in terminals.
                    for i in range(0, len(text), 50):
                        chunk = text[i:i + 50]
                        result += f"import subprocess, shlex; subprocess.run(f'xdotool type --delay 12 -- ' + shlex.quote({repr(chunk)}), shell=True, check=True)\n"
                        result += "import time; time.sleep(0.05)\n"
        elif action == "scroll":
            if text:
                result += f"pyautogui.keyDown('{text.lower()}')\n"
            if coordinate:
                x, y = coordinate[0], coordinate[1]
                if scroll_direction in ("up", "down"):
                    amt = scroll_amount if scroll_direction == "up" else -scroll_amount
                    result += f"pyautogui.scroll({amt}, {x}, {y})\n"
                elif scroll_direction in ("left", "right"):
                    amt = scroll_amount if scroll_direction == "right" else -scroll_amount
                    result += f"pyautogui.hscroll({amt}, {x}, {y})\n"
            else:
                if scroll_direction in ("up", "down"):
                    amt = scroll_amount if scroll_direction == "up" else -scroll_amount
                    result += f"pyautogui.scroll({amt})\n"
                elif scroll_direction in ("left", "right"):
                    amt = scroll_amount if scroll_direction == "right" else -scroll_amount
                    result += f"pyautogui.hscroll({amt})\n"
            if text:
                result += f"pyautogui.keyUp('{text.lower()}')\n"
        elif action in (
            "left_click",
            "right_click",
            "double_click",
            "middle_click",
            "left_press",
            "triple_click",
        ):
            if text:
                for key in text.split("+"):
                    result += f"pyautogui.keyDown('{key.strip().lower()}')\n"
            if coordinate:
                x, y = coordinate[0], coordinate[1]
                if action == "left_click":
                    result += f"pyautogui.click({x}, {y})\n"
                elif action == "right_click":
                    result += f"pyautogui.rightClick({x}, {y})\n"
                elif action == "double_click":
                    result += f"pyautogui.doubleClick({x}, {y})\n"
                elif action == "middle_click":
                    result += f"pyautogui.middleClick({x}, {y})\n"
                elif action == "left_press":
                    result += (
                        f"pyautogui.mouseDown({x}, {y})\n"
                        "time.sleep(1)\n"
                        f"pyautogui.mouseUp({x}, {y})\n"
                    )
                elif action == "triple_click":
                    result += f"pyautogui.tripleClick({x}, {y})\n"
            else:
                if action == "left_click":
                    result += "pyautogui.click()\n"
                elif action == "right_click":
                    result += "pyautogui.rightClick()\n"
                elif action == "double_click":
                    result += "pyautogui.doubleClick()\n"
                elif action == "middle_click":
                    result += "pyautogui.middleClick()\n"
                elif action == "left_press":
                    result += "pyautogui.mouseDown()\ntime.sleep(1)\npyautogui.mouseUp()\n"
                elif action == "triple_click":
                    result += "pyautogui.tripleClick()\n"
            if text:
                for key in reversed(text.split("+")):
                    result += f"pyautogui.keyUp('{key.strip().lower()}')\n"
        elif action == "wait":
            result = "time.sleep(0.5)\n"
        elif action == "fail":
            result = "FAIL"
        elif action == "done":
            result = "DONE"
        elif action == "call_user":
            result = "CALL_USER"
        elif action == "screenshot":
            result = "time.sleep(0.1)\n"

        if result.strip():
            actions.append(result.strip())

    return actions if actions else ["WAIT"]


# ---------------------------------------------------------------------------
# Helper: build observation message content blocks
# ---------------------------------------------------------------------------

def build_observation_message(
    obs: Dict[str, Any],
    observation_type: str,
    step_num: int,
) -> List[Dict[str, Any]]:
    """Build a list of Anthropic message content blocks from a DesktopEnv observation."""
    content: List[Dict[str, Any]] = []

    content.append({
        "type": "text",
        "text": f"Step {step_num}: Here is the current desktop state.",
    })

    include_screenshot = observation_type in ("screenshot", "screenshot_a11y_tree")
    include_a11y = observation_type in ("a11y_tree", "screenshot_a11y_tree")

    if include_screenshot and obs.get("screenshot"):
        screenshot_bytes = obs["screenshot"]
        if hasattr(screenshot_bytes, "read"):
            screenshot_bytes = screenshot_bytes.read()
        b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": b64,
            },
        })

    if include_a11y:
        a11y = obs.get("accessibility_tree") or ""
        if a11y:
            content.append({
                "type": "text",
                "text": f"Accessibility tree:\n{a11y}",
            })

    return content
