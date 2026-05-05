"""LLM-based DAG planner for hierarchical task decomposition (spec §1).

Called at two points:
1. Once at the root to produce the initial coarse DAG
2. By each worker before executing, to decide decompose-or-execute and
   produce a finer sub-DAG if decomposing

The planner always outputs the same format: a list of steps with dependency
edges. The scheduler merges sub-DAGs into the global DAG (spec §4).
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

from dag_core import DAGNode, DAGState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Planner prompt — used at every level of decomposition
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM_PROMPT = """\
You are a task decomposition planner for a multi-agent computer-use system.

The system has multiple virtual displays (slots). Each subtask you create will \
be assigned to a separate display and executed by an independent CUA agent \
IN PARALLEL with other ready subtasks. Agents on different displays cannot \
see each other's screens.

Break the given task into subtasks that maximize parallelism. Each subtask \
should be a coherent unit of work that one CUA agent can complete on its \
own display (open apps, navigate, click, type, etc.).

Output a JSON list. Each element has:
- "id": short identifier (e.g., "step_0")
- "task": clear, self-contained description of what the agent should accomplish. \
Include ALL information the agent needs — it cannot see other agents' screens. \
If the agent needs data from a prior step, say "using the result from step X" \
and the system will inject that result.
- "depends_on": list of step IDs that must complete first (empty = can start immediately). \
Use this to express true data dependencies. Steps with no dependencies run in parallel.
- "setup": list of setup actions to prepare the display BEFORE the agent starts \
(see types below). CRITICAL: agents spawn on EMPTY displays.
- "max_steps": estimated number of computer-use actions needed (default 30)

Setup types:
- {{"type": "chrome_open_tabs", "parameters": {{"urls_to_open": ["https://..."]}}}}
- {{"type": "launch", "parameters": {{"command": ["app", "arg1", ...]}}}}
- {{"type": "command", "parameters": {{"command": "shell command string"}}}}
- {{"type": "sleep", "parameters": {{"seconds": 3}}}}

Guidelines:
- MAXIMIZE PARALLEL WORK: if two subtasks don't need each other's output, \
make them independent (no depends_on between them).
- Google Workspace: multiple agents CAN open the same Google Doc/Sheet/Slides \
URL simultaneously and edit collaboratively in real-time. Use this for parallel \
editing — assign each agent a specific section/cells to avoid conflicts.
- Don't over-split: tasks with < 3 actions aren't worth a separate slot.
- If the task is simple enough for one agent, return a single step.
- Every step MUST have setup — displays start empty.

Output ONLY the JSON list, no other text."""


# ---------------------------------------------------------------------------
# Decomposition check (spec §1 "atomic detection")
# ---------------------------------------------------------------------------

_DECOMPOSE_CHECK_PROMPT = """\
You are deciding whether a subtask should be further decomposed into \
multiple parallel subtasks, or executed directly by a single CUA agent.

Subtask: {task_description}
Context from completed dependencies: {context}

The CUA agent can take screenshots, reason about what it sees, and perform \
actions (click, type, key press, scroll). It has a budget of {max_steps} actions.

Should this subtask be:
- EXECUTE directly by one CUA agent (it's a coherent unit of work)
- DECOMPOSE into multiple parallel subtasks (it contains independent parts \
  that different agents could work on simultaneously on separate displays)

Only answer DECOMPOSE if there are genuinely independent sub-parts that \
benefit from parallelism. Sequential steps on the same display don't benefit.

Answer with ONLY one word: "EXECUTE" or "DECOMPOSE"."""


def plan_dag(
    task_description: str,
    bedrock: Any,
    model: str,
    context: Optional[str] = None,
    temperature: float = 0.3,
) -> List[Dict[str, Any]]:
    """Produce a DAG plan for a task via LLM.

    Used both for initial root planning and for sub-DAG planning at any depth.
    Returns a list of step dicts with id, task, depends_on, setup, max_steps.
    """
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
    """Ask the LLM whether a subtask should be decomposed into parallel parts
    or executed directly by a single CUA agent (spec §1 atomic detection)."""
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
    logger.info("Decompose check: %s → %s", task_description[:60], "DECOMPOSE" if result else "EXECUTE")
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
