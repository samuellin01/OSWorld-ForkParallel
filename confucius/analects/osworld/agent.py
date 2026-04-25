# pyre-strict
"""OSWorld-specific orchestrator infrastructure.

This module contains the sync/async bridge (``_StepCoordinator`` and
``_StepCoordinatorExtension``), helper utilities, and the ``ConfuciusAgent``
class that implements the OSWorld agent protocol.  Everything needed to run
Confucius on an OSWorld task lives here so that
``mm_agents/confucius_agent.py`` can be a thin re-export.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from pydantic import PrivateAttr

from ...core import types as cf
from ...core.analect import AnalectRunContext
from ...core.llm_manager.auto import AutoLLMManager
from ...core.llm_manager.llm_params import LLMParams
from ...core.memory import CfMessage, CfMemoryManager
from ...lib.confucius import BaseConfucius
from ...orchestrator.anthropic import AnthropicLLMOrchestrator
from ...orchestrator.exceptions import OrchestratorInterruption, OrchestratorTermination
from ...core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ...orchestrator.extensions.base import Extension
from ...orchestrator.extensions.caching.anthropic import AnthropicPromptCaching
from ...orchestrator.extensions.context import SmartContextManagementExtension
from ...orchestrator.extensions.plain_text import PlainTextExtension
from ...orchestrator.extensions.thinking import ThinkingExtension
from ...orchestrator.extensions.token.estimator import TokenEstimatorExtension
from ...orchestrator.extensions.tool_use import ToolUseExtension
from ...orchestrator.types import OrchestratorInput
from .llm_params import CLAUDE_4_6_OPUS
from .tasks import get_task_definition


# ---------------------------------------------------------------------------
# ComputerToolExtension — exposes ComputerTool through the extension system
# ---------------------------------------------------------------------------

def _is_retryable_computer_tool_exception(exc: Exception) -> bool:
    """Retryable predicate for ``ComputerToolExtension``.

    Orchestrator control-flow exceptions (``OrchestratorTermination``,
    ``OrchestratorInterruption``) must not be converted to error tool-result
    messages by the base ``ToolUseExtension`` error handler — they need to
    propagate up the call stack so that ``impl()`` can handle them correctly.
    All other exceptions are treated as retryable (converted to error results).
    """
    return not isinstance(exc, (OrchestratorTermination, OrchestratorInterruption))


class ComputerToolExtension(ToolUseExtension):
    """Extension that provides the Anthropic ``computer_20251124`` tool.

    By registering ``ComputerTool`` through the standard ``ToolUseExtension``
    mechanism, it is collected by ``AnthropicLLMOrchestrator.tools`` alongside
    other tool-use extensions (e.g. ``ThinkingExtension``) and is therefore
    not overwritten when the orchestrator assembles ``additional_kwargs["tools"]``.

    When a ``_StepCoordinator`` is provided the ``on_tool_use`` handler
    pauses the orchestrator, hands the serialised tool-use content to
    ``predict()``, and waits for the next real observation (screenshot)
    before returning the tool result.  This ensures each ``predict()`` call
    corresponds to exactly one action executed on the VM.
    """

    name: str = "computer_tool_extension"
    included_in_system_prompt: bool = False

    _computer_tool: ant.ComputerTool = PrivateAttr()
    _coordinator: Optional[_StepCoordinator] = PrivateAttr(default=None)

    def __init__(
        self,
        coordinator: Optional[_StepCoordinator] = None,
        display_width_px: int = 1280,
        display_height_px: int = 720,
        display_number: int = 1,
        **data: Any,
    ) -> None:
        # Ensure orchestrator control-flow exceptions propagate rather than
        # being swallowed by the base class's generic exception handler.
        data.setdefault("is_retryable_ex", _is_retryable_computer_tool_exception)
        super().__init__(**data)
        self._computer_tool = ant.ComputerTool(
            display_width_px=display_width_px,
            display_height_px=display_height_px,
            display_number=display_number,
        )
        self._coordinator = coordinator

    @property
    async def tools(self) -> list[ant.ToolLike]:
        return [self._computer_tool]

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """Handle a computer tool call.

        When a coordinator is present this method:
        1. Serialises the tool_use block as JSON and signals ``predict()``
           so the harness can execute the action on the real VM.
        2. Waits for the next observation (real screenshot) from ``predict()``.
        3. Returns a ``MessageContentToolResult`` containing the screenshot.

        Without a coordinator (e.g. in unit tests) it falls back to a no-op
        acknowledgement.
        """
        if self._coordinator is None:
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content="Action received. Awaiting next observation.",
            )

        # Serialize the tool_use block as a JSON array.  predict() calls
        # json.loads() on raw_reply and passes the resulting list to
        # parse_computer_use_actions(), which expects a list of content-block
        # dicts (matching the format stored in AI messages after serialisation).
        raw_reply = json.dumps([tool_use.dict()])

        # Signal predict() that an action is ready.
        self._coordinator.signal_response_ready(raw_reply)

        # Pause and wait for the next real observation from the harness.
        obs = await self._coordinator.get_next_observation()

        if obs is None:
            # Episode is over; propagate termination up through the orchestrator.
            raise OrchestratorTermination()

        # Build the tool result from the real observation (screenshot + a11y tree).
        result_content: List[ant.MessageContentText | ant.MessageContentImage] = []
        if isinstance(obs.content, list):
            for item in obs.content:
                if isinstance(item, dict):
                    if item.get("type") == "image":
                        source = item.get("source", {})
                        result_content.append(
                            ant.MessageContentImage(
                                source=ant.MessageContentImageSource(
                                    type="base64",
                                    media_type=source.get(
                                        "media_type",
                                        ant.MessageContentSourceMediaType.IMAGE_PNG,
                                    ),
                                    data=source.get("data", ""),
                                )
                            )
                        )
                    # Task instruction strings (e.g. "Task instruction:\n…") and
                    # any other non-image dict items in the observation are
                    # intentionally skipped here: only screenshot images carry
                    # visual feedback that the model needs in the tool result.
                    # Text accessibility-tree content is included via the `str`
                    # branch below.
                elif isinstance(item, str) and item.strip():
                    result_content.append(ant.MessageContentText(text=item))

        if result_content:
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=result_content,
            )
        # Observation contained no extractable content (no screenshot, no text).
        # This is unexpected; log a warning to aid debugging.
        logging.getLogger("desktopenv.confucius_agent").warning(
            "ComputerToolExtension: observation had no processable content "
            "(no screenshot or text).  Returning fallback tool result."
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use.id,
            content="Observation received.",
        )


# ---------------------------------------------------------------------------
# Step coordination — single long-lived orchestrator invocation per episode
# ---------------------------------------------------------------------------

class _StepCoordinator:
    """Thread-safe bridge between synchronous ``predict()`` calls and the
    asynchronous orchestrator coroutine running in a dedicated background
    event loop.

    Lifecycle
    ~~~~~~~~~
    1. ``reset()`` creates a fresh ``_StepCoordinator`` and starts a
       background event loop thread.
    2. On the **first** ``predict()`` the orchestrator coroutine is
       launched in that loop.  The coordinator's ``set_loop()`` method is
       called from inside the coroutine so the asyncio primitives are bound
       to the correct loop.
    3. Each ``predict()`` call either starts the orchestrator (step 0) or
       calls ``send_observation()`` to wake up the paused orchestrator.
    4. ``predict()`` then blocks on ``wait_for_response()`` until the
       orchestrator signals the AI reply.
    5. ``reset()`` calls ``finish_episode()`` which causes the orchestrator
       to raise ``OrchestratorTermination``, triggering
       ``on_session_complete()`` exactly once.
    """

    def __init__(self) -> None:
        # Thread → predict(): orchestrator signals when a reply is ready.
        self._response_event: threading.Event = threading.Event()
        self._last_ai_reply: str = ""
        self._error: Optional[Exception] = None

        # predict() → orchestrator (asyncio side, bound inside set_loop()).
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._obs_event: Optional[asyncio.Event] = None
        self._pending_obs: Optional[CfMessage] = None
        self._episode_done: bool = False

        # Signalled by _run_orchestrator_async() when the orchestrator
        # coroutine completes (whether normally or via exception).  At that
        # point child_context_and_callbacks has consolidated all child
        # messages into the root memory manager, so dump_trajectory() can
        # read the full conversation from self._cf.memory_manager.
        self._orchestrator_done: threading.Event = threading.Event()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Bind coordinator to *loop* and create the asyncio observation
        event.  Must be called from **within** the background event loop."""
        self._loop = loop
        self._obs_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Orchestrator → predict() direction
    # ------------------------------------------------------------------

    def signal_response_ready(self, reply: str) -> None:
        """Called from the background coroutine to unblock ``predict()``."""
        self._last_ai_reply = reply
        self._response_event.set()

    def signal_error(self, exc: Exception) -> None:
        """Called from the background coroutine when an unhandled error
        occurs so that ``predict()`` does not block indefinitely."""
        self._error = exc
        self._response_event.set()

    # ------------------------------------------------------------------
    # predict() → orchestrator direction
    # ------------------------------------------------------------------

    def wait_for_response(self, timeout: float = 600.0) -> str:
        """Block the calling thread until the orchestrator produces an AI
        reply (or the episode ends with an error).

        Returns the raw AI reply text.
        """
        signaled = self._response_event.wait(timeout=timeout)
        if not signaled:
            raise TimeoutError(
                "Orchestrator did not produce a response within "
                f"{timeout:.0f} s."
            )
        if self._error is not None:
            raise self._error
        return self._last_ai_reply

    def send_observation(self, obs: CfMessage) -> None:
        """Send the next observation from ``predict()`` to the waiting
        orchestrator coroutine.

        Clears the response-ready flag (a new response is now pending) and
        schedules ``obs_event.set()`` in the background loop in a
        thread-safe manner.
        """
        assert self._loop is not None, "Coordinator not bound to an event loop"
        assert self._obs_event is not None, "Observation event not initialized"
        self._response_event.clear()
        self._error = None
        self._pending_obs = obs
        # call_soon_threadsafe schedules event.set() inside the loop's thread,
        # which is required because asyncio.Event is not thread-safe.
        self._loop.call_soon_threadsafe(self._obs_event.set)

    async def get_next_observation(self) -> Optional[CfMessage]:
        """Await the next observation from ``predict()``.

        Returns ``None`` when the episode is over (``finish_episode()`` was
        called), which signals the extension to raise
        ``OrchestratorTermination``.

        Must be called from **within** the background event loop.
        """
        assert self._obs_event is not None
        await self._obs_event.wait()
        self._obs_event.clear()
        if self._episode_done:
            return None
        obs = self._pending_obs
        self._pending_obs = None
        return obs

    def finish_episode(self) -> None:
        """Signal the orchestrator to stop after the current (or next)
        step.  Safe to call from any thread.

        Thread-safety note: ``_episode_done`` is set before scheduling
        ``obs_event.set()`` via ``call_soon_threadsafe``.  The scheduling
        call acts as a happens-before boundary: the background loop thread
        will see ``_episode_done = True`` when it wakes up from
        ``obs_event.wait()``.
        """
        self._episode_done = True
        if self._loop is not None and self._obs_event is not None:
            # Wake up get_next_observation() so it can return None.
            self._loop.call_soon_threadsafe(self._obs_event.set)
        # Also unblock any predict() that may be waiting.
        self._response_event.set()


class _StepCoordinatorExtension(Extension):
    """Orchestrator extension that pauses after each AI turn.

    After the orchestrator produces an AI response
    (``on_process_messages_complete`` is called), this extension:

    1. Signals ``predict()`` that the response is ready.
    2. Awaits the next observation from ``predict()``.
    3. Injects it via ``OrchestratorInterruption`` so the orchestrator
       loops and calls the LLM again with the updated context.

    When the episode ends ``get_next_observation()`` returns ``None`` and
    the extension raises ``OrchestratorTermination``, which causes
    ``impl()`` to call ``on_session_complete()`` exactly once.
    """

    _coordinator: Any = PrivateAttr(default=None)
    _action_space: str = PrivateAttr(default="pyautogui")

    def __init__(
        self,
        coordinator: _StepCoordinator,
        action_space: str = "pyautogui",
        **data: Any,
    ) -> None:
        super().__init__(**data)
        self._coordinator = coordinator
        self._action_space = action_space

    async def on_process_messages_complete(
        self, context: AnalectRunContext
    ) -> None:
        """Pause the orchestrator, hand the AI reply to ``predict()``, and
        wait for the next observation before resuming.

        In ``claude_computer_use`` mode, if the last non-empty AI message
        contains ``tool_use`` blocks it means ``ComputerToolExtension``
        already handled that turn (signalled ``predict()`` and received the
        real observation).  Re-entering the pause-and-wait logic here would
        send ``predict()`` a stale duplicate of the tool-use content, so we
        skip and let the orchestrator loop to call the LLM again.
        """
        # Extract the last AI message from memory.
        session_memory = context.memory_manager.get_session_memory()
        reply = ""
        last_content: Any = None
        for msg in reversed(session_memory.messages):
            if msg.type == cf.MessageType.AI:
                content = msg.content
                if self._action_space == "claude_computer_use":
                    # Serialize the full content list as JSON so that
                    # parse_computer_use_actions() can work with structured data.
                    if isinstance(content, list) and content:
                        reply = json.dumps(content)
                        last_content = content
                    elif isinstance(content, str) and content.strip():
                        reply = json.dumps([{"type": "text", "text": content}])
                        last_content = content
                else:
                    if isinstance(content, str):
                        reply = content
                        last_content = content
                    elif isinstance(content, list):
                        reply = " ".join(
                            c if isinstance(c, str) else c.get("text", "")
                            for c in content
                            if isinstance(c, (str, dict))
                        )
                        last_content = content
                # Skip empty messages (e.g. pure thinking blocks) and keep
                # looking for the last non-empty AI message.
                if reply.strip():
                    break
                reply = ""
                last_content = None

        # In claude_computer_use mode, if the last non-empty AI message is a
        # computer tool_use block, ComputerToolExtension has already signalled predict()
        # for that turn and obtained the real observation.  Signalling again
        # here would send a stale duplicate, so we raise OrchestratorInterruption
        # with no new messages to let the orchestrator call the LLM once more.
        # Note: we only skip signalling for computer tool_use blocks (name starts
        # with "computer"), not for other tools like context_edit.
        if self._action_space == "claude_computer_use" and isinstance(
            last_content, list
        ):
            if any(
                isinstance(item, dict)
                and item.get("type") == "tool_use"
                and (item.get("name", "").startswith("computer"))
                for item in last_content
            ):
                raise OrchestratorInterruption(messages=[])

        # Signal predict() that the AI response is ready.
        self._coordinator.signal_response_ready(reply)

        # Pause here until predict() provides the next observation or
        # signals that the episode is over.
        obs = await self._coordinator.get_next_observation()

        if obs is None:
            # Episode over — let impl() call on_session_complete() once.
            raise OrchestratorTermination()

        # Feed the new observation into the orchestrator and loop.
        raise OrchestratorInterruption(messages=[obs])


# ---------------------------------------------------------------------------
# Cost estimation: approximate pricing per 1K tokens (USD) via AWS Bedrock
# ---------------------------------------------------------------------------

MODEL_PRICING: Dict[str, Dict[str, float]] = {
    "claude-sonnet-4":   {"input": 0.003,  "cache_write": 0.00375,  "cache_read": 0.0003,  "output": 0.015},
    "claude-sonnet-4-5": {"input": 0.003,  "cache_write": 0.00375,  "cache_read": 0.0003,  "output": 0.015},
    "claude-opus-4":     {"input": 0.015,  "cache_write": 0.01875,  "cache_read": 0.0015,  "output": 0.075},
    "claude-opus-4-1":   {"input": 0.015,  "cache_write": 0.01875,  "cache_read": 0.0015,  "output": 0.075},
    "claude-opus-4-5":   {"input": 0.015,  "cache_write": 0.01875,  "cache_read": 0.0015,  "output": 0.075},
    "claude-opus-4-6":   {"input": 0.015,  "cache_write": 0.01875,  "cache_read": 0.0015,  "output": 0.075},
    "claude-haiku-3.5":  {"input": 0.0008, "cache_write": 0.001,    "cache_read": 0.00008, "output": 0.004},
}


# ---------------------------------------------------------------------------
# Helper: extract pyautogui actions from model output
# ---------------------------------------------------------------------------

def extract_actions(text: str) -> List[str]:
    """Return a list of pyautogui action strings from the model's reply.

    Handles:
    - bare ``DONE`` / ``FAIL`` / ``WAIT``
    - ```python …``` blocks
    - ``` … ``` blocks (no language tag)
    """
    text = text.strip()

    if text in ("DONE", "FAIL", "WAIT"):
        return [text]

    # Check for a terminal command at the end of a code block
    actions: List[str] = []

    pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        for match in matches:
            match = match.strip()
            terminal_cmds = ("DONE", "FAIL", "WAIT")
            lines = match.split("\n")
            last_line = lines[-1].strip() if lines else ""
            if last_line in terminal_cmds:
                if len(lines) > 1:
                    code = "\n".join(lines[:-1]).strip()
                    if code:
                        actions.append(code)
                actions.append(last_line)
            else:
                if match:
                    actions.append(match)
        return actions if actions else ["WAIT"]

    # No code block — try to treat the whole reply as an action
    return [text] if text else ["WAIT"]


# ---------------------------------------------------------------------------
# Helper: resize screenshot bytes to 1280x720 for computer-use tool
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

    Mirrors ``AnthropicAgent.parse_actions_from_tool_call()`` but operates on
    a list of content-block dicts (as stored in the AI message after JSON
    serialisation by ``_StepCoordinatorExtension``).

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
                    result = (
                        f"pyautogui.moveTo({x}, {y}, duration={duration or 0.5})\n"
                    )
                else:  # left_click_drag
                    if start_coordinate:
                        sx, sy = start_coordinate[0], start_coordinate[1]
                        result += (
                            f"pyautogui.moveTo({sx}, {sy},"
                            f" duration={duration or 0.5})\n"
                        )
                    result += (
                        f"pyautogui.dragTo({x}, {y}, duration={duration or 0.5})\n"
                    )
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
                        k = key_conversion.get(
                            key.strip().lower(), key.strip().lower()
                        )
                        result += f"pyautogui.keyDown('{k}')\n"
                    for key in reversed(keys):
                        k = key_conversion.get(
                            key.strip().lower(), key.strip().lower()
                        )
                        result += f"pyautogui.keyUp('{k}')\n"
                else:  # type
                    for char in text:
                        if char == "\n":
                            result += "pyautogui.press('enter')\n"
                        elif char == "'":
                            result += 'pyautogui.press("\'")\n'
                        elif char == "\\":
                            result += "pyautogui.press('\\\\')\n"
                        elif char == '"':
                            result += 'pyautogui.press(\'"\')\n'
                        else:
                            result += f"pyautogui.press('{char}')\n"
        elif action == "scroll":
            if text:
                result += f"pyautogui.keyDown('{text.lower()}')\n"
            if coordinate:
                x, y = coordinate[0], coordinate[1]
                if scroll_direction in ("up", "down"):
                    amt = scroll_amount if scroll_direction == "up" else -scroll_amount
                    result += f"pyautogui.scroll({amt}, {x}, {y})\n"
                elif scroll_direction in ("left", "right"):
                    amt = (
                        scroll_amount if scroll_direction == "right" else -scroll_amount
                    )
                    result += f"pyautogui.hscroll({amt}, {x}, {y})\n"
            else:
                if scroll_direction in ("up", "down"):
                    amt = scroll_amount if scroll_direction == "up" else -scroll_amount
                    result += f"pyautogui.scroll({amt})\n"
                elif scroll_direction in ("left", "right"):
                    amt = (
                        scroll_amount if scroll_direction == "right" else -scroll_amount
                    )
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
# Helper: build an observation CfMessage from an OSWorld obs dict
# ---------------------------------------------------------------------------

def build_observation_message(
    observation_type: str,
    instruction: str,
    obs: Dict,
    is_first_step: bool,
) -> CfMessage:
    """Build a HUMAN ``CfMessage`` from an OSWorld observation dict.

    On the first step of an episode the task instruction is prepended so
    that the model knows what it is asked to accomplish.  Subsequent steps
    carry only the new observation (screenshot / a11y tree).
    """
    content: List[str | Dict] = []

    if is_first_step:
        content.append(f"Task instruction:\n{instruction}")

    if observation_type in ("screenshot", "screenshot_a11y_tree"):
        screenshot_bytes: bytes = obs.get("screenshot", b"")
        if screenshot_bytes:
            b64 = base64.b64encode(screenshot_bytes).decode("utf-8")
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                }
            )

    if observation_type in ("a11y_tree", "screenshot_a11y_tree"):
        a11y: str = obs.get("a11y_tree", "")
        if a11y:
            content.append(f"\nAccessibility tree:\n{a11y}")

    return CfMessage(type=cf.MessageType.HUMAN, content=content)


# ---------------------------------------------------------------------------
# SmartContextConfig — mirrors CodeAssistEntry's configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class SmartContextConfig:
    """Configuration for SmartContextManagementExtension in ConfuciusAgent."""

    enabled: bool = False
    compression_threshold: Optional[int] = None  # maps to input_tokens_trigger
    clear_at_least: Optional[int] = None
    clear_at_least_tolerance: Optional[float] = None  # maps to enforce_clear_at_least_tolerance
    reminder_enabled: Optional[bool] = None
    reminder_ratio: Optional[float] = None
    cache_min_prompt_length: Optional[int] = None
    cache_max_num_checkpoints: Optional[int] = None


# ---------------------------------------------------------------------------
# ConfuciusAgent — OSWorld agent adapter backed by Confucius / Claude Bedrock
# ---------------------------------------------------------------------------

_agent_logger = logging.getLogger("desktopenv.confucius_agent")


class ConfuciusAgent:
    """OSWorld agent adapter backed by the Confucius / Claude Bedrock stack.

    Parameters
    ----------
    model:
        Claude model alias understood by ``AutoLLMManager``, e.g.
        ``"claude-opus-4-6"`` (default).
    observation_type:
        One of ``"screenshot"``, ``"a11y_tree"``, ``"screenshot_a11y_tree"``.
    action_space:
        Action space string.  Use ``"pyautogui"`` (default) for raw code
        generation or ``"claude_computer_use"`` to leverage Claude's native
        computer-use tool (``computer_20251124``).
    temperature:
        Sampling temperature (0–1).
    top_p:
        Nucleus-sampling top-p (0–1).
    max_tokens:
        Maximum tokens to generate.
    smart_context_config:
        Configuration for optional ``SmartContextManagementExtension``.
        When ``smart_context_config.enabled`` is ``True``, both
        ``SmartContextManagementExtension`` and ``AnthropicPromptCaching``
        are wired into the orchestrator.  When disabled (default) only
        ``AnthropicPromptCaching`` is used.
    screen_size:
        Physical screen resolution ``(width, height)`` of the OSWorld VM.
        Used in ``claude_computer_use`` mode to scale tool-use coordinates
        (which are in 1280×720 model space) back to screen space.
    """

    def __init__(
        self,
        model: str = CLAUDE_4_6_OPUS.model,
        observation_type: str = "screenshot",
        action_space: str = "pyautogui",
        temperature: float = 0.0,
        top_p: Optional[float] = None,
        max_tokens: int = 4096,
        smart_context_config: Optional[SmartContextConfig] = None,
        screen_size: Tuple[int, int] = (1920, 1080),
    ) -> None:
        self.model = model
        self.observation_type = observation_type
        self.action_space = action_space

        self._llm_params = LLMParams(
            model=model,
            temperature=temperature,
            top_p=top_p,
            initial_max_tokens=max_tokens,
        )
        self._smart_context_config: SmartContextConfig = (
            smart_context_config if smart_context_config is not None else SmartContextConfig()
        )
        # Resize factor: scale model-space coords (1280×720) → screen-space.
        self._resize_factor: Tuple[float, float] = (
            screen_size[0] / 1280.0,
            screen_size[1] / 720.0,
        )

        self._session: Optional[str] = None
        self._cf: Optional[BaseConfucius] = None
        self._orchestrator: Optional[AnthropicLLMOrchestrator] = None
        # Generic reference to whichever TokenEstimatorExtension is active
        # (either AnthropicPromptCaching or SmartContextManagementExtension —
        # both inherit from TokenEstimatorExtension and populate the same
        # _*_history lists).
        self._token_ext: Optional[TokenEstimatorExtension] = None
        self._system_prompt_text: str = ""
        self._logger: Optional[logging.Logger] = None
        self._log_dir: Optional[str] = None

        # Background event loop + coordination for single long-lived invoke
        self._bg_loop: Optional[asyncio.AbstractEventLoop] = None
        self._bg_thread: Optional[threading.Thread] = None
        self._coordinator: Optional[_StepCoordinator] = None

        # Cumulative per-episode stats (reset on each reset() call)
        self._total_input_tokens: Optional[int] = None
        self._total_output_tokens: Optional[int] = None
        self._total_uncached_input_tokens: Optional[int] = None
        self._total_cache_write_tokens: Optional[int] = None
        self._total_cache_read_tokens: Optional[int] = None
        self._total_cost: float = 0.0
        self._total_input_cost: float = 0.0
        self._total_output_cost: float = 0.0
        self._total_latency: float = 0.0
        self._step_count: int = 0
        self._last_call_history_len: int = 0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_background_loop(self) -> None:
        """Create and start a fresh background event loop in a daemon thread.

        The loop runs :py:meth:`asyncio.AbstractEventLoop.run_forever` and is
        used exclusively by the long-lived orchestrator coroutine.
        """
        self._bg_loop = asyncio.new_event_loop()
        self._bg_thread = threading.Thread(
            target=self._bg_loop.run_forever,
            daemon=True,
            name="confucius-bg-loop",
        )
        self._bg_thread.start()

    async def _run_orchestrator_async(self) -> None:
        """Coroutine that drives the orchestrator for an entire episode.

        Runs in the dedicated background event loop.  Binds the coordinator
        to that loop, then invokes the orchestrator once.  The
        ``_StepCoordinatorExtension`` inside the orchestrator handles
        the step-by-step pausing.

        Any unhandled exception is reported back to ``predict()`` via
        :py:meth:`_StepCoordinator.signal_error`.
        """
        assert self._coordinator is not None
        assert self._cf is not None
        assert self._orchestrator is not None

        # Bind asyncio primitives to this running event loop.
        self._coordinator.set_loop(asyncio.get_running_loop())

        try:
            await self._cf.invoke(
                self._orchestrator,
                OrchestratorInput(task=self._system_prompt_text, messages=[]),
            )
        except Exception as exc:
            # Propagate unexpected errors to the waiting predict() call.
            self._coordinator.signal_error(exc)
        finally:
            # Signal that the orchestrator coroutine has completed.  At this
            # point child_context_and_callbacks (inside BaseConfucius.invoke)
            # has already consolidated child messages into the root memory
            # manager.  dump_conversation_history() waits on this event
            # before reading from self._cf.memory_manager.
            self._coordinator._orchestrator_done.set()

    def _build_extensions(self) -> Tuple[List[Extension], TokenEstimatorExtension]:
        """Build the orchestrator extension list and return the token-tracking
        extension that should be stored as ``_token_ext``.

        When ``smart_context_config.enabled`` is ``True`` the extension list
        mirrors the ``CodeAssistEntry`` pattern:
        ``SmartContextManagementExtension`` + ``AnthropicPromptCaching``.
        The ``SmartContextManagementExtension`` instance is returned as the
        ``TokenEstimatorExtension`` because it populates the same history
        lists and reflects compression events.

        When disabled (default) only ``AnthropicPromptCaching`` is added and
        it serves as the ``TokenEstimatorExtension``.
        """
        cfg = self._smart_context_config

        # Build cache kwargs
        cache_kwargs: Dict[str, Any] = {"min_prompt_length": 0}
        if cfg.cache_min_prompt_length is not None:
            cache_kwargs["min_prompt_length"] = cfg.cache_min_prompt_length
        if cfg.cache_max_num_checkpoints is not None:
            cache_kwargs["max_num_checkpoints"] = cfg.cache_max_num_checkpoints

        base_extensions: List[Extension] = [
            PlainTextExtension(),
            ThinkingExtension(enable_tool_use=True),
        ]

        if self.action_space == "claude_computer_use":
            base_extensions.append(ComputerToolExtension(coordinator=self._coordinator))

        if cfg.enabled:
            smart_context_kwargs: Dict[str, Any] = {}
            if cfg.compression_threshold is not None:
                smart_context_kwargs["input_tokens_trigger"] = cfg.compression_threshold
            if cfg.clear_at_least is not None:
                smart_context_kwargs["clear_at_least"] = cfg.clear_at_least
            if cfg.clear_at_least_tolerance is not None:
                smart_context_kwargs["enforce_clear_at_least_tolerance"] = (
                    cfg.clear_at_least_tolerance
                )
            if cfg.reminder_enabled is not None:
                smart_context_kwargs["reminder_enabled"] = cfg.reminder_enabled
            if cfg.reminder_ratio is not None:
                smart_context_kwargs["reminder_ratio"] = cfg.reminder_ratio

            smart_ext = SmartContextManagementExtension(**smart_context_kwargs, log_dir=self._log_dir)
            caching_ext = AnthropicPromptCaching(**cache_kwargs)
            extensions: List[Extension] = base_extensions + [smart_ext, caching_ext]
            token_ext: TokenEstimatorExtension = smart_ext
        else:
            caching_ext = AnthropicPromptCaching(**cache_kwargs)
            extensions = base_extensions + [caching_ext]
            token_ext = caching_ext

        return extensions, token_ext

    # ------------------------------------------------------------------
    # OSWorld interface
    # ------------------------------------------------------------------

    def reset(self, logger: Optional[logging.Logger] = None, log_dir: Optional[str] = None, **kwargs: Any) -> None:
        """Reset agent state for a new episode.

        Signals any running orchestrator from the previous episode to
        terminate (triggering ``on_session_complete()`` once), then creates
        fresh per-episode objects and starts a new background event loop
        ready for the first ``predict()`` call.
        """
        self._logger = logger or _agent_logger
        self._log_dir = log_dir

        # -- Tear down previous episode (if any) --
        if self._coordinator is not None:
            # Cause the paused orchestrator to raise OrchestratorTermination,
            # which triggers on_session_complete() exactly once.
            self._coordinator.finish_episode()

        # Stop the previous background loop.  We do not join the thread
        # because it is a daemon and will be cleaned up automatically when the
        # process exits; any in-flight work belongs to the old episode whose
        # objects are about to be replaced.
        if self._bg_loop is not None and not self._bg_loop.is_closed():
            self._bg_loop.call_soon_threadsafe(self._bg_loop.stop)

        # Clear cumulative episode stats
        self._total_input_tokens = None
        self._total_output_tokens = None
        self._total_uncached_input_tokens = None
        self._total_cache_write_tokens = None
        self._total_cache_read_tokens = None
        self._total_cost = 0.0
        self._total_input_cost = 0.0
        self._total_output_cost = 0.0
        self._total_latency = 0.0
        self._step_count = 0
        self._last_call_history_len = 0

        # Fresh session + memory for the new episode
        self._session = str(uuid.uuid4())
        llm_manager = AutoLLMManager()
        self._cf = BaseConfucius(
            session=self._session,
            llm_manager=llm_manager,
            memory_manager=CfMemoryManager(),
        )

        # Build the formatted system prompt for this episode using the
        # canonical prompt from the OSWorld analect.  For computer-use mode
        # set the required beta tag so the Anthropic API enables the
        # computer-use tool.  The ComputerTool itself is registered via
        # ComputerToolExtension (added by _build_extensions()) so that it is
        # collected by AnthropicLLMOrchestrator.tools alongside other
        # tool-use extensions and is not overwritten.
        if self.action_space == "claude_computer_use":
            self._llm_params.additional_kwargs = {
                "beta": ["computer-use-2025-11-24"],
            }
        self._system_prompt_text = get_task_definition(
            self.observation_type, action_space=self.action_space
        )

        # The Confucius LLMOrchestrator protocol requires the `prompt` field to
        # be a ChatPromptTemplate with a `{task}` placeholder.  At each invoke
        # the orchestrator calls format_messages(task=inp.task) to produce the
        # system message.  We therefore pass the fully-formatted system prompt
        # text as `task` in OrchestratorInput so it lands in the system message.
        orchestrator_prompt = ChatPromptTemplate.from_messages(
            [("system", "{task}")]
        )

        # Create a fresh coordinator for this episode.
        self._coordinator = _StepCoordinator()

        extensions, token_ext = self._build_extensions()
        self._token_ext = token_ext
        self._orchestrator = AnthropicLLMOrchestrator(
            llm_params=[self._llm_params],
            prompt=orchestrator_prompt,
            raw_output_parser=None,
            # max_iterations=None: the _StepCoordinatorExtension raises
            # OrchestratorTermination when finish_episode() is called, which
            # bounds execution to the episode lifetime.  The episode runner
            # (OSWorld) also limits steps, and wait_for_response() has its
            # own timeout (default 600 s) as a final safety net.
            max_iterations=None,
            extensions=extensions + [_StepCoordinatorExtension(coordinator=self._coordinator, action_space=self.action_space)],
        )

        # Start the background event loop that will host the single
        # long-lived orchestrator invocation.
        self._start_background_loop()

    def get_stats(self) -> Dict:
        """Return a summary dict of cumulative stats for the current episode.

        Token counts are ``None`` when the LLM backend did not return usage
        metadata; timing is always populated.

        The ``"llm_calls"`` list contains one entry per individual LLM call
        made during the episode (across all steps), so every call that
        contributed to the episode totals is represented individually.
        """
        total_tokens: Optional[int] = None
        if self._total_input_tokens is not None or self._total_output_tokens is not None:
            total_tokens = (self._total_input_tokens or 0) + (self._total_output_tokens or 0)

        # Build per-call history from the TokenEstimatorExtension.  The five
        # parallel history lists each contain exactly one entry per LLM call
        # in episode order, so zipping them gives a complete per-call record.
        llm_calls: List[Dict[str, int]] = (
            self._token_ext.get_call_history() if self._token_ext is not None else []
        )

        return {
            "step_count": self._step_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "total_tokens": total_tokens,
            "total_uncached_input_tokens": self._total_uncached_input_tokens,
            "total_cache_write_tokens": self._total_cache_write_tokens,
            "total_cache_read_tokens": self._total_cache_read_tokens,
            "total_cost_usd": round(self._total_cost, 6),
            "total_input_cost_usd": round(self._total_input_cost, 6),
            "total_output_cost_usd": round(self._total_output_cost, 6),
            "total_latency_seconds": round(self._total_latency, 3),
            "average_latency_per_step_seconds": (
                round(self._total_latency / self._step_count, 3)
                if self._step_count > 0 else None
            ),
            "llm_calls": llm_calls,
            "num_llm_calls": len(llm_calls),
        }

    def dump_conversation_history(self, output_dir: str) -> None:
        """Dump the full conversation history for the current episode to *output_dir*.

        The history is written to ``{output_dir}/conversation_history.json``.
        This method is safe to call unconditionally — any failure is caught and
        logged as a warning so it never interrupts the calling code.

        During an active episode the orchestrator runs in a background event
        loop inside a child ``AnalectRunContext`` whose ``CfMemoryManager``
        accumulates all messages (AI responses, thinking, tool use/results,
        observations).  Those child messages are only consolidated into the
        root ``self._cf.memory_manager`` when the orchestrator coroutine
        completes (via the ``child_context_and_callbacks`` context-manager
        ``finally`` block).

        This method therefore:

        1. Calls ``finish_episode()`` to trigger ``OrchestratorTermination``,
           which causes the orchestrator's ``impl()`` → ``on_session_complete``
           → context-manager consolidation chain to run.
        2. Waits (with a timeout) for the background coroutine to complete so
           that consolidation finishes before we read.
        3. Dumps from ``self._cf.memory_manager`` which now contains the full
           consolidated conversation history.

        Parameters
        ----------
        output_dir:
            Directory where ``conversation_history.json`` should be written
            (e.g. the per-task results directory).
        """
        try:
            if self._coordinator is not None:
                # Trigger orchestrator termination.  If the orchestrator is
                # paused in get_next_observation() (the normal case after the
                # last predict() returned), this causes it to raise
                # OrchestratorTermination, run on_session_complete(), and
                # then exit the child_context_and_callbacks scope — which
                # consolidates all child messages into the root memory.
                self._coordinator.finish_episode()
                # Wait for the background coroutine to complete.  The
                # timeout is generous because no LLM calls happen during
                # shutdown — only exception handling and memory operations.
                # If the orchestrator hasn't finished in time, proceed
                # anyway and dump whatever is available in root memory.
                if not self._coordinator._orchestrator_done.wait(timeout=30):
                    _agent_logger.warning(
                        "Timed out waiting for orchestrator to complete; "
                        "conversation history may be incomplete."
                    )
            self._cf.dump_trajectory(output_dir=output_dir)
        except Exception as e:
            _agent_logger.warning("dump_conversation_history failed: %s", e)

    def predict(
        self, instruction: str, obs: Dict
    ) -> Tuple[Dict, List[str]]:
        """Generate the next action(s) given the current observation.

        Parameters
        ----------
        instruction:
            Natural-language task instruction.
        obs:
            Observation dict with keys such as ``"screenshot"`` (bytes) and/or
            ``"a11y_tree"`` (str).

        Returns
        -------
        response:
            A dict containing ``"model_response"``, ``"model"``,
            ``"input_tokens"``, ``"output_tokens"``, ``"total_tokens"``,
            ``"latency_seconds"``, ``"estimated_cost_usd"``, and
            ``"step_number"``.
        actions:
            A list of pyautogui action strings (or special tokens ``DONE`` /
            ``FAIL`` / ``WAIT``).
        """
        if self._cf is None or self._orchestrator is None:
            # Auto-initialise if reset() was not called
            self.reset()

        assert self._cf is not None
        assert self._orchestrator is not None
        assert self._token_ext is not None
        assert self._coordinator is not None
        assert self._bg_loop is not None

        is_first_step = (self._step_count == 0)

        # In computer-use mode, resize the screenshot to 1280×720 so that the
        # model can use its native coordinate space, then scale coordinates
        # back via _resize_factor when parsing actions.
        if self.action_space == "claude_computer_use" and obs.get("screenshot"):
            obs = dict(obs)  # shallow copy — don't mutate the caller's dict
            obs["screenshot"] = _resize_screenshot(obs["screenshot"])

        obs_message = build_observation_message(
            self.observation_type, instruction, obs, is_first_step
        )

        t0 = time.time()

        if is_first_step:
            # Add the first observation to the shared memory so that
            # get_root_tag() sees it when the orchestrator starts.
            self._cf.memory_manager.add_messages([obs_message])
            # Launch the single long-lived orchestrator coroutine.
            asyncio.run_coroutine_threadsafe(
                self._run_orchestrator_async(), self._bg_loop
            )
        else:
            # Inject the new observation into the running orchestrator via
            # OrchestratorInterruption (handled by _StepCoordinatorExtension).
            self._coordinator.send_observation(obs_message)

        # Block until the orchestrator produces the AI reply for this step.
        raw_reply = self._coordinator.wait_for_response()
        latency = time.time() - t0

        if self._logger:
            self._logger.debug("ConfuciusAgent raw reply: %s", raw_reply)

        # --- Update cumulative episode stats ------------------------------
        self._step_count += 1
        self._total_latency += latency

        # Accumulate token counts from new LLM calls since the last step.
        if self._token_ext is not None:
            call_history = self._token_ext.get_call_history()
            new_calls = call_history[self._last_call_history_len :]
            self._last_call_history_len = len(call_history)
            if new_calls:
                if self._total_input_tokens is None:
                    self._total_input_tokens = 0
                    self._total_output_tokens = 0
                    self._total_uncached_input_tokens = 0
                    self._total_cache_write_tokens = 0
                    self._total_cache_read_tokens = 0
                pricing = MODEL_PRICING.get(self.model)
                for call in new_calls:
                    self._total_input_tokens += call["input_tokens"]
                    self._total_output_tokens += call["output_tokens"]
                    self._total_uncached_input_tokens += call["uncached_input_tokens"]
                    self._total_cache_write_tokens += call["cache_write_tokens"]
                    self._total_cache_read_tokens += call["cache_read_tokens"]
                    if pricing is not None:
                        call_input_cost = (
                            call["uncached_input_tokens"] * pricing["input"]
                            + call["cache_write_tokens"] * pricing["cache_write"]
                            + call["cache_read_tokens"] * pricing["cache_read"]
                        ) / 1000.0
                        call_output_cost = (
                            call["output_tokens"] * pricing["output"]
                        ) / 1000.0
                        self._total_input_cost += call_input_cost
                        self._total_output_cost += call_output_cost
                        self._total_cost += call_input_cost + call_output_cost

        if self.action_space == "claude_computer_use":
            try:
                content_blocks: List[Dict[str, Any]] = json.loads(raw_reply)
                actions = parse_computer_use_actions(
                    content_blocks, self._resize_factor
                )
            except Exception:
                if self._logger:
                    self._logger.exception(
                        "Failed to parse computer-use actions from reply: %s",
                        raw_reply,
                    )
                actions = ["WAIT"]
        else:
            actions = extract_actions(raw_reply)

        response = {
            "model_response": raw_reply,
            "model": self.model,
            "latency_seconds": round(latency, 3),
            "step_number": self._step_count,
        }
        return response, actions
