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


# Message tool definitions
READ_MESSAGES_TOOL: Dict[str, Any] = {
    "name": "read_messages",
    "description": (
        "Check for messages from your children (if you're a parent) or from your "
        "parent (if you're a child). Returns a list of messages with sender and content."
    ),
    "input_schema": {
        "type": "object",
        "properties": {},
    },
}

SEND_MESSAGE_TOOL: Dict[str, Any] = {
    "name": "send_message",
    "description": (
        "Send a message to your parent (if you're a child) or to a specific child "
        "(if you're a parent). Use this to report results or request information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "to": {
                "type": "string",
                "description": "Agent ID to send to (use 'parent' if you're a child, or child_id if you're a parent)",
            },
            "content": {
                "type": "object",
                "description": "Message content (any structured data)",
            },
        },
        "required": ["to", "content"],
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
        "Children send results via send_message. Use read_messages to check for results. "
    )

    if parent_id:
        system_prompt += (
            "You are a child agent working on a specific subtask. Your display has been prepared via setup config. "
            "When done, use send_message to report your result to your parent. "
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
        READ_MESSAGES_TOOL,
        SEND_MESSAGE_TOOL,
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

        # Take screenshot
        shot = display.screenshot()
        if shot:
            shot = _resize_screenshot(shot)
            last_screenshot = shot
            if output_dir:
                shot_path = os.path.join(output_dir, f"step_{step:03d}.png")
                with open(shot_path, "wb") as f:
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
            logger.warning(f"{tag} Screenshot failed")
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

            elif tool_name == "read_messages":
                # Check for messages
                msgs = []
                while True:
                    msg = runtime.receive_message(agent_id, timeout=0)
                    if not msg:
                        break
                    msgs.append({
                        "from": msg.from_agent,
                        "content": msg.content,
                    })

                result_text = f"Received {len(msgs)} message(s): {msgs}" if msgs else "No messages"
                logger.info(f"{tag} {result_text}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                })

            elif tool_name == "send_message":
                # Send message
                to = tool_input.get("to", "")
                content = tool_input.get("content", {})

                # Resolve "parent" to actual parent_id
                if to == "parent":
                    if parent_id:
                        to = parent_id
                    else:
                        result_text = "Error: You have no parent (you're the root agent)"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": result_text,
                        })
                        continue

                runtime.send_message(from_agent=agent_id, to_agent=to, content=content)
                result_text = f"Message sent to {to}"
                logger.info(f"{tag} {result_text}")

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
