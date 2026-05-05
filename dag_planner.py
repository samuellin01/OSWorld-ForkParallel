"""LLM-based DAG planner for task decomposition.

Takes a task description and produces a structured DAG plan (list of steps
with dependency edges) via an LLM call.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from dag_core import DAGNode, DAGState

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """\
You are a task decomposition planner. Given a computer-use task, break it into \
independent subtasks that can be executed in parallel where possible.

Each subtask will be executed by a separate agent on its own virtual display. \
Agents can work concurrently on independent subtasks.

Output a JSON list of steps. Each step has:
- "id": short identifier (e.g., "step_0", "step_1")
- "task": clear description of what the agent should accomplish
- "depends_on": list of step IDs that must complete before this step can start \
(empty list if this step can start immediately)
- "setup": list of setup actions to prepare the agent's display environment \
(see setup types below)
- "is_atomic": true if this step can be done by a single agent in ~15-30 actions, \
false if it should be further decomposed
- "max_steps": estimated max actions needed (default 30)

Setup action types:
- {"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://..."]}}
- {"type": "launch", "parameters": {"command": ["app", "arg1", ...]}}
- {"type": "command", "parameters": {"command": "shell command"}}
- {"type": "download", "parameters": {"files": [{"path": "/tmp/f.csv", "url": "https://..."}]}}
- {"type": "sleep", "parameters": {"seconds": 3}}

Key guidelines:
- Google Workspace (Sheets/Docs/Slides): Multiple agents CAN open the same URL \
simultaneously and edit collaboratively in real-time. Use chrome_open_tabs with \
the same URL for parallel editing. Specify which section/cells each agent should edit.
- Maximize parallelism: identify truly independent subtasks that can run concurrently.
- Avoid over-decomposition: don't split tasks that are faster done sequentially \
(e.g., filling adjacent cells with Tab, using formulas).
- Each subtask should involve 3+ meaningful actions to justify the setup overhead.
- If the entire task is simple enough for one agent, return a single step.
- For spreadsheet tasks: consider formulas, fill-down, batch operations before parallelizing.
- For web research: each independent lookup is a good parallelization candidate.

Output ONLY the JSON list, no other text. Example:
[
  {"id": "step_0", "task": "Look up X on website A", "depends_on": [], \
"setup": [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://a.com"]}}], \
"is_atomic": true, "max_steps": 20},
  {"id": "step_1", "task": "Look up Y on website B", "depends_on": [], \
"setup": [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://b.com"]}}], \
"is_atomic": true, "max_steps": 20},
  {"id": "step_2", "task": "Fill results into spreadsheet", "depends_on": ["step_0", "step_1"], \
"setup": [{"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://docs.google.com/..."]}}], \
"is_atomic": true, "max_steps": 15}
]"""


_DECOMPOSE_CHECK_PROMPT = """\
You are deciding whether a subtask needs further decomposition or can be executed directly.

The subtask will be executed by a computer-use agent that can click, type, and interact \
with a desktop environment. The agent has at most {max_steps} actions.

Subtask: {task_description}
Context from completed dependencies: {context}

Can this subtask be completed by a single agent in {max_steps} or fewer actions? \
Or does it need to be broken into multiple parallel subtasks?

Answer with ONLY one of:
- "ATOMIC" if the agent can do this directly
- "DECOMPOSE" if this should be split into parallel subtasks"""


def plan_dag(
    task_description: str,
    bedrock: Any,
    model: str,
    context: Optional[str] = None,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Produce a DAG plan for a task via LLM.

    Returns a list of step dicts with id, task, depends_on, setup, is_atomic, max_steps.
    """
    user_msg = f"Task: {task_description}"
    if context:
        user_msg += f"\n\nAdditional context:\n{context}"

    messages = [{"role": "user", "content": [{"type": "text", "text": user_msg}]}]

    content_blocks, _ = bedrock.chat(
        messages=messages,
        system=_PLANNER_SYSTEM_PROMPT,
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
            "is_atomic": True,
            "max_steps": 30,
        }]

    _validate_plan(plan)
    logger.info("DAG plan: %d steps", len(plan))
    for step in plan:
        logger.info(
            "  %s: %s (deps=%s, atomic=%s)",
            step["id"], step["task"][:80], step["depends_on"], step.get("is_atomic", True),
        )
    return plan


def should_decompose(
    task_description: str,
    bedrock: Any,
    model: str,
    context: str = "",
    max_steps: int = 30,
    temperature: float = 0.3,
) -> bool:
    """Ask the LLM whether a subtask should be further decomposed."""
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

    return "DECOMPOSE" in response_text


def convert_plan_to_dag_state(
    plan: List[Dict[str, Any]],
    root_task: str,
    max_depth: int = 2,
    prefix: str = "",
) -> DAGState:
    """Convert a planner output into a DAGState."""
    nodes = {}
    for step in plan:
        node_id = f"{prefix}{step['id']}" if prefix else step["id"]
        dep_ids = [
            f"{prefix}{d}" if prefix else d
            for d in step.get("depends_on", [])
        ]
        node = DAGNode(
            id=node_id,
            task_description=step["task"],
            depends_on=dep_ids,
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

        if "is_atomic" not in step:
            step["is_atomic"] = True

        if "max_steps" not in step:
            step["max_steps"] = 30
