"""Worker execution for DAG scheduler nodes.

Each worker owns one display. It receives a coarse subtask and either:
1. Runs a CUA agent loop directly (screenshot → LLM → action → repeat)
2. Decomposes the subtask into a fine-grained action plan, then executes
   each action sequentially on its display

Both paths use the same display — sub-actions don't get separate displays.
Parallelism comes from multiple workers on different displays.
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
from dag_core import DAGNode
from fork_agent import XvfbDisplay

logger = logging.getLogger(__name__)


def run_dag_worker(
    node: DAGNode,
    vm_ip: str,
    server_port: int,
    bedrock: Any,
    model: str,
    output_dir: str,
    password: str = "osworld-public-evaluation",
    dependency_results: Optional[Dict[str, Any]] = None,
    max_depth: int = 3,
) -> Dict[str, Any]:
    """Execute a coarse DAG node on its assigned display.

    The worker decides whether to:
    - Run a CUA agent loop (for tasks requiring visual feedback / adaptation)
    - Decompose into a fine action plan and execute sequentially (when steps are known)

    In practice, most tasks benefit from the CUA agent loop since the agent
    needs to see the screen to determine coordinates and adapt to UI state.
    Decomposition into a fine plan is useful when the exact action sequence
    is predictable (e.g., fill specific cells with known values).
    """
    tag = f"[{node.id}]"
    logger.info("%s Starting worker (depth=%d, display=:%s)", tag, node.depth, node.display_num)

    # For now, always use CUA agent loop — the agent sees the screen
    # and decides what to do. Fine-grained action plans require knowing
    # exact coordinates which we don't have until we see the screenshot.
    #
    # Future: if should_decompose() returns True AND we have a way to
    # resolve "click the Name Box" → actual coordinates, we can use
    # plan_fine_actions() here. For now, the CUA loop handles it.

    return _run_cua_agent(
        node=node,
        vm_ip=vm_ip,
        server_port=server_port,
        bedrock=bedrock,
        model=model,
        output_dir=output_dir,
        password=password,
        dependency_results=dependency_results,
    )


def _format_dep_results(dependency_results: Optional[Dict[str, Any]]) -> str:
    if not dependency_results:
        return ""
    parts = []
    for dep_id, result in dependency_results.items():
        summary = result.get("summary", str(result))[:300]
        parts.append(f"[{dep_id}] {summary}")
    return "\n".join(parts)


def _build_worker_system_prompt(
    node: DAGNode,
    password: str,
    dependency_results: Optional[Dict[str, Any]],
) -> str:
    chrome_port = 1337 + (node.display_num or 0)
    prompt = (
        "You are a computer-use agent on Ubuntu 22.04 with openbox window manager. "
        f"Password: '{password}'. Home directory: /home/user. "
        "\n\n"
        "**Recovery**: Press Ctrl+Alt+T to open a terminal if you accidentally close windows.\n"
        "\n"
        f"If you launch Chrome from terminal: google-chrome --remote-debugging-port={chrome_port} "
        f"--user-data-dir=/tmp/chrome_display_{node.display_num or 0} --no-first-run "
        f"--no-default-browser-check --disable-default-apps URL "
        "\n\n"
        "You are a worker agent assigned a specific subtask. Your display has been prepared "
        "with the necessary applications. Focus on completing your subtask efficiently.\n"
        "\n"
        "When done, output SUBTASK COMPLETE followed by a summary of what you accomplished.\n"
        "If you cannot complete the task, output SUBTASK FAILED with explanation.\n"
        "\n"
        "**If setup failed** - If your first screenshot shows an empty desktop or wrong app:\n"
        "Do NOT try to fix it yourself. Immediately report:\n"
        "SUBTASK FAILED: Setup did not work. Display shows [describe what you see].\n"
        "\n"
        "Google Docs/Sheets/Slides are collaborative real-time editing environments. "
        "Multiple agents can open the same URL simultaneously and see each other's changes.\n"
        "\n"
        "Google Workspace: Do NOT use Apps Script - complete tasks through the UI directly.\n"
        "\n"
        "Google Sheets: Arrow keys work for navigation. Use the Name Box (top-left) to jump to cells. "
        "Batch actions together with Tab/Enter navigation.\n"
        "\n"
    )

    if dependency_results:
        prompt += "Results from previously completed steps:\n"
        for dep_id, result in dependency_results.items():
            summary = result.get("summary", str(result))[:500]
            prompt += f"  [{dep_id}]: {summary}\n"
        prompt += "\nUse these results to inform your work. Do NOT redo completed steps.\n\n"

    prompt += (
        "Before outputting SUBTASK COMPLETE, verify you've actually completed the subtask.\n"
        "You are judged on both task completion and efficiency."
    )
    return prompt


def _run_cua_agent(
    node: DAGNode,
    vm_ip: str,
    server_port: int,
    bedrock: Any,
    model: str,
    output_dir: str,
    password: str,
    dependency_results: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Run a CUA agent loop: screenshot → LLM → action → repeat.

    The agent sees the screen each step and decides what to do.
    This is the standard computer-use agent pattern.
    """
    tag = f"[{node.id}]"
    display_num = node.display_num or 0
    display = XvfbDisplay(vm_ip, server_port, display_num)
    system_prompt = _build_worker_system_prompt(node, password, dependency_results)
    tools = [COMPUTER_USE_TOOL]
    resize_factor = (1920.0 / 1280.0, 1080.0 / 720.0)

    initial_text = f"Your subtask:\n{node.task_description}"
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": initial_text}]}
    ]

    last_tool_use_id: Optional[str] = None
    final_response_text = ""
    start_time = time.time()

    for step in range(1, node.max_steps + 1):
        logger.info("%s Step %d/%d", tag, step, node.max_steps)

        shot = display.screenshot()
        step_timestamp = time.time()

        if shot:
            if output_dir:
                shot_path = os.path.join(output_dir, f"step_{step:03d}.png")
                with open(shot_path, "wb") as f:
                    f.write(shot)
                ts_path = os.path.join(output_dir, f"step_{step:03d}_timestamp.txt")
                with open(ts_path, "w") as f:
                    f.write(f"{step_timestamp:.6f}\n")

            obs_content: List[Dict[str, Any]] = [
                {"type": "text", "text": f"Step {step}: current desktop state."},
            ]
            resized = _resize_screenshot(shot)
            obs_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(resized).decode(),
                },
            })
        else:
            logger.warning("%s Screenshot failed at step %d", tag, step)
            obs_content = [
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
                    "error_type": "tool_result_missing",
                    "summary": f"Conversation history corruption: {error_msg}",
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

        if output_dir:
            with open(os.path.join(output_dir, f"step_{step:03d}_response.txt"), "w") as f:
                f.write(response_text)

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

                if output_dir:
                    action_file = os.path.join(output_dir, f"step_{step:03d}_action.txt")
                    with open(action_file, "w") as f:
                        action_type = tool_input.get("action", "")
                        if action_type == "type":
                            f.write(f"Type: {tool_input.get('text', '')}")
                        elif action_type == "key":
                            f.write(f"Key: {tool_input.get('text', '')}")
                        elif action_type in ("left_click", "right_click", "double_click", "middle_click"):
                            coord = tool_input.get("coordinate", [])
                            f.write(f"{action_type.replace('_', ' ').title()} at {coord}")
                        elif action_type == "mouse_move":
                            f.write(f"Move to {tool_input.get('coordinate', [])}")
                        elif action_type == "screenshot":
                            f.write("Screenshot")
                        else:
                            f.write(f"Computer: {action_type}")

        lines = final_response_text.strip().split("\n")
        for line in lines:
            line = line.strip()
            if re.search(r'\bSUBTASK\s+COMPLETE\b', line, re.IGNORECASE):
                logger.info("%s SUBTASK COMPLETE at step %d", tag, step)
                completion_time = time.time()
                duration = completion_time - start_time
                result = {
                    "status": "DONE",
                    "summary": final_response_text,
                    "steps_used": step,
                    "duration": duration,
                }
                if output_dir:
                    with open(os.path.join(output_dir, "completion_timestamp.txt"), "w") as f:
                        f.write(f"{completion_time:.6f}\n")
                return result

            if re.search(r'\bSUBTASK\s+FAILED\b', line, re.IGNORECASE):
                logger.info("%s SUBTASK FAILED at step %d", tag, step)
                return {
                    "status": "FAIL",
                    "summary": final_response_text,
                    "steps_used": step,
                    "duration": time.time() - start_time,
                }

    logger.warning("%s Max steps (%d) reached", tag, node.max_steps)
    return {
        "status": "MAX_STEPS",
        "summary": f"Reached max steps. Last: {final_response_text}",
        "steps_used": node.max_steps,
        "duration": time.time() - start_time,
    }
