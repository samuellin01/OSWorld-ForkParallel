"""LLM-based DAG planner for task decomposition.

Two-level planning:
1. Coarse planner: breaks a task into parallel subtasks (each gets a display)
2. Fine planner: breaks a subtask into sequential computer-use actions
   (click, type, key press — executed on the same display)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from dag_core import DAGNode, DAGState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Coarse planner: task → parallel subtasks (one display each)
# ---------------------------------------------------------------------------

_COARSE_PLANNER_PROMPT = """\
You are a task decomposition planner for a multi-agent computer-use system.

The system has multiple virtual displays. Each subtask you create will be \
assigned to a separate display and executed by an independent agent IN PARALLEL. \
Agents on different displays cannot see each other's screens.

Break the given task into subtasks that can run on separate displays simultaneously. \
Each subtask should be a coherent unit of work that one agent can complete on its \
own display (open apps, navigate, click, type, etc.).

Output a JSON list. Each element has:
- "id": short identifier (e.g., "step_0")
- "task": clear description of what the agent should accomplish on its display
- "depends_on": list of step IDs that must complete first (empty = can start immediately)
- "setup": list of setup actions to prepare the display BEFORE the agent starts \
(see types below). This is critical — the agent spawns on an EMPTY display.
- "max_steps": estimated number of computer-use actions needed (default 30)

Setup types (prepare the display before the agent starts):
- {"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://..."]}}
- {"type": "launch", "parameters": {"command": ["app", "arg1", ...]}}
- {"type": "command", "parameters": {"command": "shell command string"}}
- {"type": "download", "parameters": {"files": [{"path": "/tmp/f.csv", "url": "https://..."}]}}
- {"type": "sleep", "parameters": {"seconds": 3}}

Guidelines:
- Google Workspace (Sheets/Docs/Slides): multiple agents CAN open the same URL \
on different displays and edit the same document collaboratively in real-time. \
Specify which section/cells each agent should edit to avoid conflicts.
- Maximize parallelism: independent research, independent cells, independent files.
- Don't over-split: adjacent cells fillable with Tab, formula-based work, or \
tasks with fewer than 3 actions should NOT be separate steps.
- If the task is simple enough for one agent, return a single step.
- Every step MUST have setup — the display starts empty. At minimum, open Chrome \
or launch the required application.

Output ONLY the JSON list, no other text. Example:
[
  {"id": "step_0", "task": "Research topic X and note the answer", "depends_on": [], \
"setup": [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://google.com"]}}], \
"max_steps": 25},
  {"id": "step_1", "task": "Research topic Y and note the answer", "depends_on": [], \
"setup": [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://google.com"]}}], \
"max_steps": 25},
  {"id": "step_2", "task": "Fill both answers into the spreadsheet", \
"depends_on": ["step_0", "step_1"], \
"setup": [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://docs.google.com/spreadsheets/d/..."]}}], \
"max_steps": 15}
]"""


# ---------------------------------------------------------------------------
# Fine planner: subtask → sequential computer-use actions
# ---------------------------------------------------------------------------

_FINE_PLANNER_PROMPT = """\
You are planning the exact sequence of computer-use actions to complete a subtask.

The agent is on a desktop with the environment already set up (apps open, URLs loaded). \
Break the subtask into the individual actions the agent needs to perform, in order.

Each action is one of:
- click(x, y): left-click at screen coordinates
- type(text): type text using keyboard
- key(keys): press key(s), e.g., "Return", "ctrl+a", "Tab"
- scroll(direction, amount): scroll up/down/left/right
- wait(): pause briefly for page to load

Output a JSON list of actions. Each element has:
- "id": short identifier (e.g., "a0", "a1")
- "action": one of "click", "type", "key", "scroll", "wait"
- "parameters": action-specific parameters
- "description": brief human-readable description of what this action does
- "depends_on": list of action IDs that must complete first (usually just the previous action)

You do NOT know exact coordinates yet — describe clicks by what UI element to target. \
The executing agent will see the actual screen and determine coordinates.

Example for "Type 'hello' into cell B2 of a Google Sheet":
[
  {"id": "a0", "action": "click", "parameters": {"target": "Name Box (top-left, showing current cell)"}, \
"description": "Click the Name Box to select it", "depends_on": []},
  {"id": "a1", "action": "type", "parameters": {"text": "B2"}, \
"description": "Type cell address B2", "depends_on": ["a0"]},
  {"id": "a2", "action": "key", "parameters": {"keys": "Return"}, \
"description": "Press Enter to navigate to B2", "depends_on": ["a1"]},
  {"id": "a3", "action": "type", "parameters": {"text": "hello"}, \
"description": "Type the value", "depends_on": ["a2"]},
  {"id": "a4", "action": "key", "parameters": {"keys": "Return"}, \
"description": "Press Enter to confirm", "depends_on": ["a3"]}
]

Output ONLY the JSON list."""


# ---------------------------------------------------------------------------
# Decomposition check
# ---------------------------------------------------------------------------

_DECOMPOSE_CHECK_PROMPT = """\
You are deciding whether a subtask should be further decomposed into a plan \
of individual computer-use actions, or whether a CUA agent should handle it \
as a single unit (the agent sees the screen and decides what to do each step).

Subtask: {task_description}
Context from completed dependencies: {context}

A CUA agent can take screenshots, reason about what it sees, and issue actions \
(click, type, key press). It's good at multi-step UI interaction but has a \
budget of {max_steps} actions.

Should this subtask be:
- Executed by a CUA agent directly (it will figure out the steps by looking at the screen)
- Decomposed into an explicit action plan first (useful when you know the exact steps)

Answer with ONLY one of:
- "AGENT" — let a CUA agent handle it (it sees the screen and adapts)
- "PLAN" — decompose into explicit actions first"""


def plan_dag(
    task_description: str,
    bedrock: Any,
    model: str,
    context: Optional[str] = None,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Produce a coarse DAG plan: task → parallel subtasks."""
    user_msg = f"Task: {task_description}"
    if context:
        user_msg += f"\n\nAdditional context:\n{context}"

    messages = [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]

    content_blocks, _ = bedrock.chat(
        messages=messages,
        system=_COARSE_PLANNER_PROMPT,
        model=model,
        temperature=temperature,
        max_tokens=4096,
    )

    response_text = "".join(
        b.get("text", "") for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )

    plan = _parse_plan_json(response_text)
    if not plan:
        logger.warning("Planner returned empty/invalid plan, creating single-step fallback")
        plan = [{
            "id": "step_0",
            "task": task_description,
            "depends_on": [],
            "setup": [],
            "max_steps": 30,
        }]

    _validate_plan(plan)
    logger.info("DAG plan: %d steps", len(plan))
    for step in plan:
        logger.info(
            "  %s: %s (deps=%s, max_steps=%d)",
            step["id"], step["task"][:80], step["depends_on"], step.get("max_steps", 30),
        )
    return plan


def plan_fine_actions(
    task_description: str,
    bedrock: Any,
    model: str,
    context: Optional[str] = None,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Produce a fine action plan: subtask → sequential actions."""
    user_msg = f"Subtask: {task_description}"
    if context:
        user_msg += f"\n\nContext from prior steps:\n{context}"

    messages = [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]

    content_blocks, _ = bedrock.chat(
        messages=messages,
        system=_FINE_PLANNER_PROMPT,
        model=model,
        temperature=temperature,
        max_tokens=4096,
    )

    response_text = "".join(
        b.get("text", "") for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    )

    actions = _parse_plan_json(response_text)
    if not actions:
        logger.warning("Fine planner returned empty plan")
        return []

    logger.info("Fine action plan: %d actions", len(actions))
    for a in actions:
        logger.info("  %s: %s — %s", a.get("id"), a.get("action"), a.get("description", "")[:60])
    return actions


def should_decompose(
    task_description: str,
    bedrock: Any,
    model: str,
    context: str = "",
    max_steps: int = 30,
    temperature: float = 0.3,
) -> bool:
    """Ask the LLM whether a subtask should be decomposed into an action plan
    or handled directly by a CUA agent."""
    prompt = _DECOMPOSE_CHECK_PROMPT.format(
        task_description=task_description,
        context=context or "(none)",
        max_steps=max_steps,
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    content_blocks, _ = bedrock.chat(
        messages=messages,
        system="You are a task analysis assistant. Be concise.",
        model=model,
        temperature=temperature,
        max_tokens=50,
    )

    response_text = "".join(
        b.get("text", "") for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ).strip().upper()

    return "PLAN" in response_text


def convert_plan_to_dag_state(
    plan: List[Dict[str, Any]],
    root_task: str,
    max_depth: int = 3,
) -> DAGState:
    """Convert a coarse planner output into a DAGState."""
    nodes = {}
    for step in plan:
        node_id = step["id"]
        node = DAGNode(
            id=node_id,
            task_description=step["task"],
            depends_on=step.get("depends_on", []),
            setup_config=step.get("setup", []),
            max_steps=step.get("max_steps", 30),
            depth=0,
        )
        nodes[node_id] = node

    return DAGState(
        nodes=nodes,
        root_task=root_task,
        max_depth=max_depth,
    )


def _parse_plan_json(text: str) -> List[Dict[str, Any]]:
    """Extract JSON list from LLM response text."""
    text = text.strip()

    match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', text, re.DOTALL)
    if match:
        text = match.group(1)
    else:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1:
            text = text[start:end + 1]

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError as e:
        logger.error("Failed to parse plan JSON: %s", e)
        logger.debug("Raw text: %s", text[:500])

    return []


def _validate_plan(plan: List[Dict[str, Any]]):
    """Validate and fix common issues in the plan."""
    ids = {step.get("id") for step in plan}

    for step in plan:
        if "id" not in step:
            step["id"] = f"step_{plan.index(step)}"

        if "task" not in step:
            step["task"] = "Unknown task"

        if "depends_on" not in step:
            step["depends_on"] = []

        step["depends_on"] = [d for d in step["depends_on"] if d in ids and d != step["id"]]

        if "setup" not in step:
            step["setup"] = []

        if "max_steps" not in step:
            step["max_steps"] = 30
