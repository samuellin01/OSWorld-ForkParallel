"""LLM-based DAG planner for hierarchical task decomposition (spec §1).

Called at every level of the hierarchy:
- Root: task -> coarse parallel/sequential subtasks
- Worker: subtask -> finer subtasks (if decomposable)
- Recursion until leaf nodes are small enough for a single CUA agent

The planner outputs both parallel AND sequential subtasks. Sequential
subtasks share a display (the scheduler detects chains from depends_on).
Parallel subtasks get separate displays.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from dag_core import DAGNode, DAGState

logger = logging.getLogger(__name__)

_PLANNER_SYSTEM_PROMPT = """\
You are a task decomposition planner for a multi-agent computer-use system.

The system has multiple virtual displays (slots). Break the task into subtasks. \
The scheduler will automatically detect which subtasks are parallel vs sequential:

- **Parallel subtasks** (no depends_on between them): run simultaneously on \
SEPARATE displays. Agents cannot see each other's screens.
- **Sequential subtasks** (connected by depends_on): run one after another on \
the SAME display. The display state (open windows, tabs, cursor position) \
carries over. The later step sees exactly what the earlier step left on screen.

Output a JSON list. Each element has:
- "id": short identifier (e.g., "step_0")
- "task": clear, self-contained description. Include ALL information the agent \
needs. If data from a prior step is needed, say so — the system injects prior results.
- "depends_on": list of step IDs that must complete first (empty = can start immediately)
- "setup": list of setup actions for the display BEFORE the agent starts. \
Only needed for steps that START on a fresh display (first step in a parallel \
branch). Sequential continuations (depends_on a prior step on the same display) \
should have empty setup since the display carries over.
- "max_steps": estimated number of computer-use actions (default 30)

Setup types (only for steps starting on a fresh display):
- {{"type": "chrome_open_tabs", "parameters": {{"urls_to_open": ["https://..."]}}}}
- {{"type": "launch", "parameters": {{"command": ["app", "arg1", ...]}}}}
- {{"type": "sleep", "parameters": {{"seconds": 3}}}}

Guidelines:
- Google Workspace: multiple agents CAN open the same Doc/Sheet/Slides URL \
on different displays and edit collaboratively in real-time.
- Maximize parallelism where work is truly independent.
- Use sequential dependencies when step B needs to see step A's screen state \
or needs step A's output data.
- Don't over-split: if a single CUA agent can handle the whole thing in \
~15-30 actions, return a single step.
- Each parallel branch needs setup. Sequential continuations do NOT.

Output ONLY the JSON list, no other text."""


_DECOMPOSE_CHECK_PROMPT = """\
You are deciding whether a subtask should be further decomposed or executed \
directly by a single CUA (computer-use) agent.

Subtask: {task_description}
Context from completed dependencies: {context}

The CUA agent can take screenshots, reason about what it sees, and perform \
actions (click, type, key press, scroll). It has a budget of {max_steps} actions.

Should this subtask be:
- EXECUTE: a single CUA agent handles it directly (coherent unit of work, \
  one screen, ~{max_steps} or fewer actions)
- DECOMPOSE: break into subtasks — either parallel parts on separate displays, \
  or sequential phases where intermediate results matter, or a mix

Answer DECOMPOSE if there are genuinely independent sub-parts that benefit \
from parallelism, or distinct phases with different setup needs.
Answer EXECUTE if one agent can see the screen and handle it.

Answer with ONLY one word: EXECUTE or DECOMPOSE"""


def plan_dag(
    task_description: str,
    bedrock: Any,
    model: str,
    context: Optional[str] = None,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Produce a DAG plan for a task. Used at every decomposition level."""
    user_msg = f"Task: {task_description}"
    if context:
        user_msg += f"\n\nContext from completed prior steps:\n{context}"

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
        logger.warning("Planner returned empty/invalid plan, single-step fallback")
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


def should_decompose(
    task_description: str,
    bedrock: Any,
    model: str,
    context: str = "",
    max_steps: int = 30,
    temperature: float = 0.3,
) -> bool:
    """Ask the LLM whether a subtask should be decomposed further."""
    prompt = _DECOMPOSE_CHECK_PROMPT.format(
        task_description=task_description,
        context=context or "(none)",
        max_steps=max_steps,
    )

    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

    content_blocks, _ = bedrock.chat(
        messages=messages,
        system="You are a task analysis assistant. Answer with one word.",
        model=model,
        temperature=temperature,
        max_tokens=20,
    )

    response_text = "".join(
        b.get("text", "") for b in content_blocks
        if isinstance(b, dict) and b.get("type") == "text"
    ).strip().upper()

    result = "DECOMPOSE" in response_text
    logger.info("Decompose check: %s -> %s", task_description[:60], "DECOMPOSE" if result else "EXECUTE")
    return result


def convert_plan_to_dag_state(
    plan: List[Dict[str, Any]],
    root_task: str,
    max_depth: int = 3,
) -> DAGState:
    """Convert a planner output into a DAGState."""
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

    return DAGState(nodes=nodes, root_task=root_task, max_depth=max_depth)


def _parse_plan_json(text: str) -> List[Dict[str, Any]]:
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
