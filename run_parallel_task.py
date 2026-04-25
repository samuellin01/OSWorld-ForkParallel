"""Orchestrator agent for parallel CUA task execution.

The orchestrator is a text-only LLM agent that:
1. Reads a task instruction
2. Decomposes it into subtasks (serial or parallel)
3. Dispatches subtasks to GUI agents
4. Collects results (text summary + final screenshot)
5. Re-plans and dispatches more subtasks until done

The orchestrator never sees live screenshots — it only sees the final
screenshot from each completed subtask as a sanity check.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bedrock_client import BedrockClient
from gui_agent import SubtaskResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestrator system prompt
# ---------------------------------------------------------------------------

_ORCHESTRATOR_SYSTEM_PROMPT = """\
You are an orchestrator that manages GUI agents to complete desktop automation \
tasks on Ubuntu 22.04. Home directory: /home/user. The default browser is Google Chrome. \
You do NOT interact with the desktop yourself — you strategically \
plan and delegate subtasks to GUI agents, then review their results.

You have access to multiple displays. When you respond, output a JSON object \
with one of these actions:

1. **Dispatch subtasks** — each subtask MUST specify which display to run on:
```json
{
  "action": "dispatch",
  "subtasks": [
    {"instruction": "description of subtask 1", "display": 0},
    {"instruction": "description of subtask 2", "display": 2}
  ]
}
```
Subtasks assigned to different displays run concurrently. \
Only assign subtasks to displays that are currently FREE.

2. **Declare completion**:
```json
{"action": "done"}
```

3. **Declare failure**:
```json
{"action": "fail", "reason": "explanation"}
```

## Display architecture

There is one primary display (the native GNOME desktop) and additional \
secondary displays (minimal window managers). The task will be **evaluated \
on the primary display only** — any GUI state (open windows, visible apps) \
that needs to be present for the task to be considered complete must end up \
on the primary display.

Secondary displays are useful as workspaces for tasks that produce \
**file artifacts**. Any files saved to the shared filesystem are \
visible from all displays.

When you dispatch subtasks:
- Display 0 is the primary display. The task will be evaluated here, so \
any visible GUI state (open windows, apps) must end up on display 0.
- You can dispatch to multiple displays simultaneously \
to complete subtasks efficiently in parallel.
- You can dispatch to any available display while others are busy.
You are judged for task completion as well as efficiency.

## Guidelines

- Delegate goals, not procedures. Tell the agent WHAT to accomplish, not \
HOW to do it step by step. The agent is capable of figuring out the method.
- Include relevant details: file paths, specific values, app names.
- Parallel subtasks on separate displays share the same filesystem.
- Look for data parallelism: when the same operation applies to multiple \
independent items (files, emails, records), split them across parallel agents. \
For research/lookup tasks, agents should report findings in their final response. \
Only write files when they're actual deliverables or needed for pipeline parallelism \
where another agent will consume the output file.
- Look for pipeline parallelism: if one agent produces results that another \
agent consumes, start the consumer as soon as partial results are available.
- **Collaborative document parallelism (Google Workspace)**: When the task \
involves a Google Sheet, Doc, or Slide (identifiable by docs.google.com URLs), \
you should dispatch multiple agents to work on it concurrently. Each agent opens \
the same URL on their own display and edits different sections simultaneously — \
changes sync in real-time. This is a situation where a file \
can be written to by multiple agents concurrently.
- After reviewing results, dispatch follow-up subtasks that depend on \
earlier results. Include relevant information from earlier results in the \
new subtask instructions.
- You will be notified as each agent finishes, even while others are still \
running. You should think about the next steps of the task and \
dispatch new work to free displays without waiting for all agents to complete.
"""


# ---------------------------------------------------------------------------
# Orchestrator planning step result
# ---------------------------------------------------------------------------

@dataclass
class OrchestratorPlan:
    """Output from one orchestrator planning step."""
    action: str  # "dispatch", "done", "fail", "wait"
    subtasks: List[str] = field(default_factory=list)
    display_assignments: List[int] = field(default_factory=list)  # display index per subtask
    parallel: bool = False
    fail_reason: str = ""
    raw_response: str = ""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class Orchestrator:
    """Text-only planning agent that decomposes tasks and dispatches to GUI agents."""

    def __init__(
        self,
        bedrock: BedrockClient,
        model: str = "claude-opus-4-6",
        temperature: float = 0.7,
        max_rounds: int = 15,
        output_dir: Optional[str] = None,
        num_displays: int = 3,
    ):
        self.bedrock = bedrock
        self.model = model
        self.temperature = temperature
        self.max_rounds = max_rounds
        self.output_dir = output_dir
        self.num_displays = num_displays

        # Display indices: 0 is primary, then 2, 3, 4, ... (1 is skipped by Xvfb convention)
        self._display_indices = [0] + list(range(2, num_displays + 1))
        # Track state per display: "free" or last subtask summary
        self._display_state: Dict[int, str] = {d: "idle — never used" for d in self._display_indices}
        self._display_busy: Dict[int, bool] = {d: False for d in self._display_indices}

        self._system_prompt = _ORCHESTRATOR_SYSTEM_PROMPT + (
            f"\n\nYou have {num_displays} displays: {self._display_indices}. "
            f"Display 0 is the primary display (GNOME desktop). "
            f"Displays {self._display_indices[1:]} are secondary (minimal openbox + xterm). "
            f"You MUST assign each subtask to a specific display number."
        )
        self._messages: List[Dict[str, Any]] = []
        self._round = 0
        self._progress: List[Dict[str, Any]] = []  # completed subtask records

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    def plan(
        self,
        task: str,
        completed_results: Optional[List[SubtaskResult]] = None,
        initial_screenshot: Optional[bytes] = None,
    ) -> OrchestratorPlan:
        """Run one planning step.

        On the first call, provide the task instruction and optionally an
        initial screenshot of the desktop. On subsequent calls, provide the
        results from the previously dispatched subtasks.

        Returns an OrchestratorPlan indicating what to do next.
        """
        self._round += 1
        logger.info("[orchestrator] Round %d/%d", self._round, self.max_rounds)

        if self._round > self.max_rounds:
            logger.warning("[orchestrator] Max rounds reached")
            return OrchestratorPlan(
                action="fail",
                fail_reason=f"Reached maximum planning rounds ({self.max_rounds})",
            )

        # Build the user message for this round
        if self._round == 1:
            # First round: provide the task + initial screenshot
            user_content: List[Dict[str, Any]] = [
                {
                    "type": "text",
                    "text": (
                        f"Task to complete:\n{task}\n\n"
                        f"Current display states:\n{self._format_display_state()}\n\n"
                        f"Here is the current state of the primary desktop (display 0). "
                        f"Plan how to accomplish this task. Assign each subtask to a display."
                    ),
                }
            ]
            if initial_screenshot:
                user_content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(initial_screenshot).decode(),
                    },
                })
        else:
            # Subsequent rounds: provide results from previous subtasks
            result_text = self._format_results(completed_results or [])
            user_parts: List[Dict[str, Any]] = [
                {"type": "text", "text": result_text},
            ]
            # Attach final screenshots from completed subtasks
            if completed_results:
                for i, result in enumerate(completed_results):
                    if result.final_screenshot:
                        user_parts.append({
                            "type": "text",
                            "text": f"Final screenshot from subtask {i+1} ({result.status}):",
                        })
                        user_parts.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(result.final_screenshot).decode(),
                            },
                        })
            user_content = user_parts

        self._messages.append({"role": "user", "content": user_content})

        # Call LLM (no tools — orchestrator is text-only), retry on parse failure.
        max_parse_retries = 3
        for attempt in range(max_parse_retries):
            content_blocks, _ = self.bedrock.chat(
                messages=self._messages,
                system=self._system_prompt,
                model=self.model,
                temperature=self.temperature,
                tools=None,
            )
            self._messages.append({"role": "assistant", "content": content_blocks})

            response_text = "".join(
                b.get("text", "") for b in content_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
            logger.info("[orchestrator] Response: %s", response_text[:500])

            if self.output_dir:
                suffix = "" if attempt == 0 else f"_retry{attempt}"
                with open(
                    os.path.join(self.output_dir, f"round_{self._round:03d}{suffix}.txt"), "w"
                ) as f:
                    f.write(response_text)

            plan = self._parse_response(response_text)
            plan.raw_response = response_text

            if plan.action != "fail" or "JSON" not in plan.fail_reason:
                break  # Valid parse or a real failure, not a formatting issue.

            if attempt < max_parse_retries - 1:
                logger.warning(
                    "[orchestrator] Parse failed (attempt %d/%d), retrying",
                    attempt + 1, max_parse_retries,
                )
                self._messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        "Your response was not valid JSON. Please respond with "
                        "a JSON object containing an \"action\" field. Example:\n"
                        '{\"action\": \"dispatch\", \"subtasks\": [{\"instruction\": \"...\"}], \"parallel\": false}'
                    )}],
                })

        # Track progress
        if completed_results:
            for r in completed_results:
                self._progress.append({
                    "subtask": r.subtask,
                    "status": r.status,
                    "summary": r.summary,
                    "steps_used": r.steps_used,
                })

        return plan

    def mark_display_busy(self, display_idx: int, subtask: str):
        """Mark a display as busy with a subtask."""
        self._display_busy[display_idx] = True
        self._display_state[display_idx] = f"BUSY — running: {subtask[:80]}"

    def mark_display_free(self, display_idx: int, summary: str):
        """Mark a display as free and record what it last did."""
        self._display_busy[display_idx] = False
        self._display_state[display_idx] = f"FREE — last did: {summary[:120]}"

    def _format_display_state(self) -> str:
        """Format current display states for the orchestrator prompt."""
        lines = []
        for d in self._display_indices:
            prefix = "PRIMARY" if d == 0 else "secondary"
            lines.append(f"  Display {d} ({prefix}): {self._display_state[d]}")
        return "\n".join(lines)

    def on_subtask_complete(
        self,
        task: str,
        result: SubtaskResult,
        display_idx: int,
        free_display_screenshots: Optional[Dict[int, bytes]] = None,
    ) -> OrchestratorPlan:
        """Handle a single subtask completion incrementally.

        Called as soon as one subtask finishes, even if others are still running.
        The orchestrator can dispatch new work to free displays immediately.
        free_display_screenshots maps display index → current screenshot bytes
        for each free display, so the orchestrator can see what it's dispatching into.

        Returns an OrchestratorPlan — subtask(s) to dispatch, or "done"/"fail".
        """
        self._round += 1

        # Update display state
        self.mark_display_free(display_idx, result.summary[:120] if result.summary else result.status)

        logger.info(
            "[orchestrator] Incremental round %d — display %d completed (%s)",
            self._round, display_idx, result.status,
        )

        if self._round > self.max_rounds:
            return OrchestratorPlan(
                action="fail",
                fail_reason=f"Reached maximum planning rounds ({self.max_rounds})",
            )

        # Track progress
        self._progress.append({
            "subtask": result.subtask,
            "status": result.status,
            "summary": result.summary,
            "steps_used": result.steps_used,
            "display": display_idx,
        })

        result_text = (
            f"A subtask just completed on display {display_idx}:\n"
            f"  Subtask: {result.subtask}\n"
            f"  Status: {result.status}\n"
            f"  Steps used: {result.steps_used}\n"
            f"  Summary: {result.summary}\n\n"
            f"Current display states:\n{self._format_display_state()}\n\n"
            f"Overall progress:\n{self.get_progress_summary()}\n\n"
            f"Dispatch new subtask(s) to FREE displays, or declare done/fail.\n"
            f"Each subtask MUST include a \"display\" field with the display number.\n"
            f"Respond with a JSON object."
        )

        user_parts: List[Dict[str, Any]] = [
            {"type": "text", "text": result_text},
        ]
        if result.final_screenshot:
            user_parts.append({
                "type": "text",
                "text": f"Final screenshot from completed subtask on display {display_idx} ({result.status}):",
            })
            user_parts.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(result.final_screenshot).decode(),
                },
            })

        # Include current screenshots of free displays so the orchestrator
        # can see what state each display is in before dispatching.
        if free_display_screenshots:
            for disp_idx, screenshot in sorted(free_display_screenshots.items()):
                if screenshot and disp_idx != display_idx:
                    prefix = "PRIMARY" if disp_idx == 0 else "secondary"
                    user_parts.append({
                        "type": "text",
                        "text": f"Current state of free display {disp_idx} ({prefix}):",
                    })
                    user_parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64.b64encode(screenshot).decode(),
                        },
                    })

        self._messages.append({"role": "user", "content": user_parts})

        # Call LLM with retry on parse failure
        max_parse_retries = 3
        for attempt in range(max_parse_retries):
            content_blocks, _ = self.bedrock.chat(
                messages=self._messages,
                system=self._system_prompt,
                model=self.model,
                temperature=self.temperature,
                tools=None,
            )
            self._messages.append({"role": "assistant", "content": content_blocks})

            response_text = "".join(
                b.get("text", "") for b in content_blocks
                if isinstance(b, dict) and b.get("type") == "text"
            )
            logger.info("[orchestrator] Incremental response: %s", response_text[:500])

            if self.output_dir:
                suffix = "" if attempt == 0 else f"_retry{attempt}"
                with open(
                    os.path.join(self.output_dir, f"round_{self._round:03d}{suffix}.txt"), "w"
                ) as f:
                    f.write(response_text)

            plan = self._parse_response(response_text)
            plan.raw_response = response_text

            # Handle "wait" action
            if plan.action == "fail" and "wait" in plan.fail_reason.lower():
                return OrchestratorPlan(action="wait")

            if plan.action != "fail" or "JSON" not in plan.fail_reason:
                break

            if attempt < max_parse_retries - 1:
                self._messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        "Your response was not valid JSON. Please respond with "
                        "a JSON object. Use {\"action\": \"wait\"} if you want "
                        "to wait for running agents to finish."
                    )}],
                })

        return plan

    def get_progress_summary(self) -> str:
        """Return a text summary of all completed subtasks so far."""
        if not self._progress:
            return "No subtasks completed yet."
        lines = []
        for i, p in enumerate(self._progress, 1):
            lines.append(
                f"{i}. [{p['status']}] {p['subtask']}\n"
                f"   Result: {p['summary']}"
            )
        return "\n".join(lines)

    def _format_results(self, results: List[SubtaskResult]) -> str:
        """Format subtask results for the orchestrator's next planning step."""
        parts = [
            f"Round {self._round}: Results from the previous subtask(s):\n"
        ]
        for i, r in enumerate(results, 1):
            parts.append(
                f"Subtask {i}: {r.subtask}\n"
                f"  Status: {r.status}\n"
                f"  Steps used: {r.steps_used}\n"
                f"  Summary: {r.summary}\n"
            )
        parts.append(
            "\nOverall progress so far:\n" + self.get_progress_summary()
        )
        parts.append(
            "\nDecide what to do next: dispatch more subtasks, or declare "
            "done/fail. Respond with a JSON object."
        )
        return "\n".join(parts)

    def _parse_response(self, text: str) -> OrchestratorPlan:
        """Extract a JSON action from the orchestrator's response text."""
        # Try to find JSON in the response (may be wrapped in markdown code blocks)
        json_str = self._extract_json(text)
        if not json_str:
            logger.warning("[orchestrator] No JSON found in response, treating as failure")
            return OrchestratorPlan(
                action="fail",
                fail_reason="Orchestrator did not produce a valid JSON action.",
                raw_response=text,
            )

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning("[orchestrator] JSON parse error: %s", e)
            return OrchestratorPlan(
                action="fail",
                fail_reason=f"JSON parse error: {e}",
                raw_response=text,
            )

        action = data.get("action", "").lower()

        if action == "dispatch":
            subtasks_raw = data.get("subtasks", [])
            subtasks = []
            display_assignments = []
            for s in subtasks_raw:
                if isinstance(s, str):
                    subtasks.append(s)
                    display_assignments.append(0)  # default to primary
                elif isinstance(s, dict):
                    subtasks.append(s.get("instruction", str(s)))
                    display_assignments.append(int(s.get("display", 0)))
            if not subtasks:
                # Empty dispatch = orchestrator wants to wait.
                # Treat as no-op so the event loop continues.
                return OrchestratorPlan(action="wait", raw_response=text)
            # Determine if parallel based on whether multiple displays are used
            unique_displays = set(display_assignments)
            parallel = len(unique_displays) > 1
            return OrchestratorPlan(
                action="dispatch",
                subtasks=subtasks,
                display_assignments=display_assignments,
                parallel=parallel,
            )

        if action == "done":
            return OrchestratorPlan(action="done")

        if action == "wait":
            return OrchestratorPlan(action="wait")

        if action == "fail":
            return OrchestratorPlan(
                action="fail",
                fail_reason=data.get("reason", "Unknown failure"),
            )

        logger.warning("[orchestrator] Unknown action: %s", action)
        return OrchestratorPlan(
            action="fail",
            fail_reason=f"Unknown action: {action}",
            raw_response=text,
        )

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """Extract the first JSON object from text, handling code blocks."""
        import re

        # Try markdown code block first
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        if match:
            return match.group(1).strip()

        # Try to find a raw JSON object
        # Find the first { and match to its closing }
        start = text.find("{")
        if start == -1:
            return None

        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\":
                escape = True
                continue
            if c == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]

        return None
