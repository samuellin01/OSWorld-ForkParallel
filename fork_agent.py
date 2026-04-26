"""Fork-based CUA agent with real LLM integration.

This integrates:
- Real CUA agent loop (screenshots → LLM → actions)
- Fork tool (agents can spawn children)
- Message passing (parent ↔ child communication)
- Context compression (text-only history for children)
"""

import base64
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

from agent_runtime import AgentRuntime
from agent_utils import (
    COMPUTER_USE_TOOL,
    _resize_screenshot,
    filter_to_n_most_recent_images,
    parse_computer_use_actions,
)
from bedrock_client import BedrockClient
from gui_agent import XvfbDisplay

logger = logging.getLogger(__name__)


# Fork tool definition
FORK_TOOL: Dict[str, Any] = {
    "name": "fork_subtask",
    "description": (
        "Spawn a new agent to work on a subtask in parallel. The child agent "
        "will run on a separate display with its own environment. Use this when "
        "you have independent work that can be done in parallel (e.g., searching "
        "for multiple items, filling multiple forms, processing multiple files). "
        "The subtask should be substantial (>15 seconds of work) to justify fork overhead. "
        "You will receive the child's result via a message when it completes.\n\n"
        "Setup config examples:\n"
        "- Open Chrome to URL: {\"type\": \"chrome_open_tabs\", \"parameters\": {\"urls_to_open\": [\"https://google.com/search?q=Python\"]}}\n"
        "- Launch app: {\"type\": \"launch\", \"parameters\": {\"command\": [\"gedit\", \"/tmp/file.txt\"]}}\n"
        "- Run command: {\"type\": \"command\", \"parameters\": {\"command\": \"mkdir -p /tmp/workspace\"}}\n"
        "- Download file: {\"type\": \"download\", \"parameters\": {\"files\": [{\"path\": \"/tmp/data.csv\", \"url\": \"https://...\"}]}}\n"
        "- Wait: {\"type\": \"sleep\", \"parameters\": {\"seconds\": 2}}"
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "subtask": {
                "type": "string",
                "description": "Clear description of what the child agent should accomplish",
            },
            "setup": {
                "type": "array",
                "description": "Setup steps to prepare the child's environment before it starts",
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": ["chrome_open_tabs", "launch", "open", "command", "download", "sleep"],
                        },
                        "parameters": {"type": "object"},
                    },
                    "required": ["type", "parameters"],
                },
            },
        },
        "required": ["subtask", "setup"],
    },
}


KILL_CHILD_TOOL: Dict[str, Any] = {
    "name": "kill_child",
    "description": (
        "Terminate a child agent. Use this if a child is stuck, working on the wrong "
        "thing, or no longer needed. The child will be stopped immediately and its "
        "display will be released back to the pool. Only parents can kill their children."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "child_id": {
                "type": "string",
                "description": "ID of the child agent to terminate",
            },
        },
        "required": ["child_id"],
    },
}

PEEK_CHILD_TOOL: Dict[str, Any] = {
    "name": "peek_child",
    "description": (
        "Check on a child agent's progress without interrupting them. Returns the child's "
        "current status, screenshot, and full conversation history. Use this to monitor "
        "progress, detect if a child is stuck in a loop or going down the wrong path. "
        "The child will not be aware of the peek - they continue working normally."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "child_id": {
                "type": "string",
                "description": "ID of the child agent to peek at",
            },
        },
        "required": ["child_id"],
    },
}


def compress_context(messages: List[Dict[str, Any]]) -> str:
    """Compress agent conversation history to text-only summary.

    Removes all images, keeps only text content for passing to forked children.
    This drastically reduces token usage for child agents.

    Args:
        messages: Full conversation history with images

    Returns:
        Text summary of conversation
    """
    summary_parts = []

    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", [])

        if isinstance(content, str):
            summary_parts.append(f"[{role}] {content}")
            continue

        # Extract text from content blocks
        text_parts = []
        for block in content:
            if not isinstance(block, dict):
                continue

            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                name = block.get("name", "unknown")
                text_parts.append(f"[Used tool: {name}]")
            elif block.get("type") == "tool_result":
                # Skip tool results, too verbose
                pass
            elif block.get("type") == "image":
                text_parts.append("[Screenshot]")

        if text_parts:
            summary_parts.append(f"[{role}] {' '.join(text_parts)}")

    return "\n".join(summary_parts)


def run_fork_agent(
    agent_id: str,
    runtime: AgentRuntime,
    vm_ip: str,
    server_port: int,
    bedrock: BedrockClient,
    model: str,
    task: str,
    parent_context: Optional[str] = None,
    max_steps: int = 30,
    temperature: float = 0.7,
    output_dir: Optional[str] = None,
    password: str = "osworld-public-evaluation",
) -> Dict[str, Any]:
    """Run a fork-based CUA agent with real LLM calls.

    Args:
        agent_id: This agent's ID
        runtime: AgentRuntime instance
        vm_ip: VM IP address
        server_port: VM server port
        bedrock: BedrockClient for LLM calls
        model: Model name
        task: Task/subtask for this agent
        parent_context: Compressed context from parent (if child)
        max_steps: Maximum steps
        temperature: LLM temperature
        output_dir: Output directory for logs/screenshots
        password: VM sudo password

    Returns:
        Result dict with status, summary, etc.
    """
    # Get agent info from runtime
    agent_status = runtime.get_agent_status(agent_id)
    if not agent_status:
        logger.error(f"Agent {agent_id} not found in runtime")
        return {"status": "error", "summary": "Agent not found"}

    display_num = agent_status["display_num"]
    parent_id = agent_status["parent_id"]
    is_root = parent_id is None

    tag = f"[{agent_id}]"
    logger.info(f"{tag} Starting on display :{display_num}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Create display wrapper
    if display_num == 0:
        # Root agent on primary display - would use NativeDisplay
        # For now, just use XvfbDisplay pattern
        logger.warning(f"{tag} Display :0 should use NativeDisplay (not implemented yet)")

    display = XvfbDisplay(vm_ip, server_port, display_num)

    # Build system prompt
    chrome_port = 1337 + display_num
    system_prompt = (
        "You are a computer-use agent on Ubuntu 22.04 with openbox window manager. "
        f"Password: '{password}'. Home directory: /home/user. "
        "Right-click the desktop to see the application menu. "
        "You can open a terminal (xterm) from the menu to run commands. "
        f"If you launch Chrome from terminal: google-chrome --remote-debugging-port={chrome_port} "
        f"--user-data-dir=/tmp/chrome_display_{display_num} --no-first-run "
        f"--no-default-browser-check --disable-default-apps URL "
        "\n\n"
    )

    system_prompt += (
        "Your goal is to complete tasks correctly and efficiently. "
        "\n\n"
        "You can fork subtasks to child agents using fork_subtask. Each child runs on a separate display in parallel. "
        "\n"
        "When to fork: Fork when you have independent work that can run in parallel, where each child task would take "
        "more than 15 seconds of work. Forking has overhead (~5 seconds per child), so only fork when the time savings "
        "justify it. Don't fork trivial tasks.\n"
        "\n"
        "How to write setup: The setup config prepares the child's environment before it starts. Use setup to open "
        "applications, navigate to URLs, or prepare files. Setup is NOT instructions to the child - it actually executes "
        "before the child begins. Common setup types: chrome_open_tabs (open URLs in Chrome), launch (start applications), "
        "command (run shell commands). The child's display starts with setup already completed.\n"
        "\n"
        "How to write subtasks: Write a clear goal for the child, not step-by-step instructions. The child is autonomous "
        "and cannot see your screen. If setup opens Chrome to a URL, the subtask should be 'Search for X and report results', "
        "not 'Open Chrome and search for X'. The child starts in the state created by setup.\n"
        "\n"
        "Monitoring children: Use peek_child to check a child's progress without interrupting them. You'll see their current "
        "screenshot and full conversation history. This helps detect if a child is stuck in a loop or going down the wrong path. "
        "When a child completes, their result will automatically appear in your next observation - you don't need to request it."
    )

    if parent_id:
        system_prompt += (
            "\n\nYou are a child agent working on a specific subtask. Your display has been prepared via setup config. "
            "When done, just output DONE with your result. Your parent will automatically receive it."
        )

    system_prompt += "\n\n"

    system_prompt += (
        "When you complete your task, output DONE followed by a summary. "
        "Output FAIL if the task is impossible."
    )

    # Tools available to this agent
    tools = [
        COMPUTER_USE_TOOL,
        FORK_TOOL,
        KILL_CHILD_TOOL,
        PEEK_CHILD_TOOL,
    ]

    # Build initial message
    if parent_context:
        initial_text = (
            f"Parent context (compressed):\n{parent_context}\n\n"
            f"Your subtask:\n{task}"
        )
    else:
        initial_text = f"Task:\n{task}"

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": initial_text}]}
    ]

    last_tool_use_id: Optional[str] = None
    last_screenshot: Optional[bytes] = None
    final_response_text = ""
    resize_factor = (1920.0 / 1280.0, 1080.0 / 720.0)

    start_time = time.time()

    for step in range(1, max_steps + 1):
        logger.info(f"{tag} Step {step}/{max_steps}")

        # Check for child results (auto-inject)
        child_results = runtime.get_pending_child_results(agent_id)

        # Take screenshot
        shot = display.screenshot()
        if shot:
            shot = _resize_screenshot(shot)
            last_screenshot = shot
            if output_dir:
                shot_path = os.path.join(output_dir, f"step_{step:03d}.png")
                with open(shot_path, "wb") as f:
                    f.write(shot)

            obs_content: List[Dict[str, Any]] = []

            # Inject child results first
            if child_results:
                for child_result in child_results:
                    child_id = child_result.get("child_id", "unknown")
                    if child_result.get("status") == "completed":
                        result_data = child_result.get("result", {})
                        summary = result_data.get("summary", str(result_data))
                        obs_content.append({
                            "type": "text",
                            "text": f"Child {child_id} completed:\n{summary}"
                        })
                        logger.info(f"{tag} ← Result from {child_id}")
                    elif child_result.get("status") == "failed":
                        error = child_result.get("error", "unknown error")
                        obs_content.append({
                            "type": "text",
                            "text": f"Child {child_id} failed: {error}"
                        })
                        logger.info(f"{tag} ← Failure from {child_id}")

            obs_content.append({"type": "text", "text": f"Step {step}: current desktop state."})
            obs_content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(shot).decode(),
                },
            })
        else:
            logger.warning(f"{tag} Screenshot failed")
            obs_content = []

            # Inject child results even if screenshot failed
            if child_results:
                for child_result in child_results:
                    child_id = child_result.get("child_id", "unknown")
                    if child_result.get("status") == "completed":
                        result_data = child_result.get("result", {})
                        summary = result_data.get("summary", str(result_data))
                        obs_content.append({
                            "type": "text",
                            "text": f"Child {child_id} completed:\n{summary}"
                        })
                        logger.info(f"{tag} ← Result from {child_id}")
                    elif child_result.get("status") == "failed":
                        error = child_result.get("error", "unknown error")
                        obs_content.append({
                            "type": "text",
                            "text": f"Child {child_id} failed: {error}"
                        })
                        logger.info(f"{tag} ← Failure from {child_id}")

            obs_content.append({"type": "text", "text": f"Step {step}: screenshot unavailable."})

        # Prepend tool_result if needed
        if last_tool_use_id:
            obs_content.insert(0, {
                "type": "tool_result",
                "tool_use_id": last_tool_use_id,
                "content": "Action executed.",
            })
            last_tool_use_id = None

        messages.append({"role": "user", "content": obs_content})

        # Filter old images to save context
        filter_to_n_most_recent_images(messages, images_to_keep=10)

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
        logger.info(f"{tag} Response: {response_text[:200]}")
        final_response_text = response_text

        # Store in agent's conversation history (for peek_child)
        runtime.update_conversation(agent_id, {"step": step, "response": response_text})

        if output_dir:
            with open(os.path.join(output_dir, f"step_{step:03d}_response.txt"), "w") as f:
                f.write(response_text)

        # Handle tool calls
        tool_results = []
        for block in content_blocks:
            if not isinstance(block, dict) or block.get("type") != "tool_use":
                continue

            tool_name = block.get("name")
            tool_input = block.get("input", {})
            tool_use_id = block.get("id")

            if tool_name == "fork_subtask":
                # Handle fork
                subtask = tool_input.get("subtask", "")
                setup = tool_input.get("setup", [])

                logger.info(f"{tag} Forking subtask: {subtask[:60]}...")

                # Compress current context for child
                child_context = compress_context(messages)

                # Fork via runtime
                child_id = runtime.fork_agent(
                    parent_id=agent_id,
                    subtask=subtask,
                    config=setup,
                    context_summary=child_context,
                )

                if child_id:
                    result_text = f"Forked child {child_id} to work on: {subtask}"
                    logger.info(f"{tag} {result_text}")
                else:
                    result_text = f"Failed to fork child (no displays available?)"
                    logger.error(f"{tag} {result_text}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                })

            elif tool_name == "kill_child":
                # Kill a child agent
                child_id = tool_input.get("child_id", "")

                if not child_id:
                    result_text = "Error: No child_id provided"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_text,
                    })
                    continue

                logger.info(f"{tag} Killing child {child_id}")

                # Runtime enforces parent-child relationship
                runtime.kill_agent(agent_id=child_id, killer_id=agent_id)

                result_text = f"Child {child_id} has been terminated"
                logger.info(f"{tag} {result_text}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                })

            elif tool_name == "peek_child":
                # Peek at a child agent
                child_id = tool_input.get("child_id", "")

                if not child_id:
                    result_text = "Error: No child_id provided"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_text,
                    })
                    continue

                logger.info(f"{tag} Peeking at child {child_id}")

                # Runtime enforces parent-child relationship and returns state
                peek_data = runtime.peek_child(parent_id=agent_id, child_id=child_id)

                if not peek_data:
                    result_text = f"Error: Cannot peek at {child_id} (not your child or doesn't exist)"
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use_id,
                        "content": result_text,
                    })
                    continue

                # Format peek result for agent
                import base64
                screenshot_b64 = None
                if peek_data.get("screenshot"):
                    screenshot_b64 = base64.b64encode(peek_data["screenshot"]).decode()

                # Build text summary + image
                conversation = peek_data.get("conversation", [])
                conv_text = "\n".join([
                    f"Step {i+1}: {entry.get('response', '')[:200]}"
                    for i, entry in enumerate(conversation)
                ])

                result_content = [
                    {
                        "type": "text",
                        "text": (
                            f"Child {child_id} status:\n"
                            f"Status: {peek_data['status']}\n"
                            f"Steps: {peek_data['steps']}\n"
                            f"Duration: {peek_data['duration']:.1f}s\n\n"
                            f"Conversation history:\n{conv_text}"
                        ),
                    }
                ]

                if screenshot_b64:
                    result_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": screenshot_b64,
                        },
                    })

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_content,
                })

            elif tool_name == "computer":
                # Computer use action - parse and execute
                last_tool_use_id = tool_use_id

                actions = parse_computer_use_actions([block], resize_factor)
                action_code = next(
                    (a for a in actions if a not in ("DONE", "FAIL", "WAIT")),
                    None,
                )

                if action_code:
                    logger.info(f"{tag} Action: {action_code[:120]}")
                    display.run_action(action_code)
                    time.sleep(1)

        # If we have tool results to report, add them to messages
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
            # Continue to next step to get agent's response to tool results
            continue

        # Check for DONE/FAIL
        if "DONE" in final_response_text.upper():
            logger.info(f"{tag} DONE at step {step}")
            duration = time.time() - start_time
            result = {
                "status": "DONE",
                "summary": final_response_text,
                "steps_used": step,
                "duration": duration,
            }
            runtime.complete_agent(agent_id, result=result)
            return result

        if "FAIL" in final_response_text.upper():
            logger.info(f"{tag} FAIL at step {step}")
            duration = time.time() - start_time
            result = {
                "status": "FAIL",
                "summary": final_response_text,
                "steps_used": step,
                "duration": duration,
            }
            runtime.fail_agent(agent_id, error=final_response_text)
            return result

    # Max steps reached
    logger.warning(f"{tag} Max steps ({max_steps}) reached")
    duration = time.time() - start_time
    result = {
        "status": "MAX_STEPS",
        "summary": f"Reached max steps. Last response: {final_response_text}",
        "steps_used": max_steps,
        "duration": duration,
    }
    runtime.complete_agent(agent_id, result=result)
    return result
