"""Agent runtime for fork-based parallel execution.

Manages agent lifecycle, display allocation, fork operations, and message passing.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from display_pool import DisplayPool
from setup_executor import SetupExecutor

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Status of an agent."""
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    KILLED = "killed"


@dataclass
class Agent:
    """Represents a running agent."""
    agent_id: str
    parent_id: Optional[str]
    display_num: int
    subtask: str
    status: AgentStatus = AgentStatus.RUNNING
    context_summary: Optional[str] = None
    children: Set[str] = field(default_factory=set)
    pending_child_results: List[Dict[str, Any]] = field(default_factory=list)
    pending_messages: List[str] = field(default_factory=list)
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    result: Optional[Any] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None


class AgentRuntime:
    """Runtime for managing fork-based parallel agent execution.

    Manages:
    - Display pool allocation
    - Agent lifecycle (spawn, track, cleanup)
    - Fork operations (setup + child spawn)
    - Message passing between parent and children

    Usage:
        runtime = AgentRuntime(vm_exec, num_displays=8)
        runtime.initialize()

        # Spawn root agent
        root_id = runtime.spawn_root_agent(
            task="Main task",
            display_num=0  # Primary display
        )

        # Agent requests fork
        child_id = runtime.fork_agent(
            parent_id=root_id,
            subtask="Subtask for child",
            config=[{"type": "chrome_open_tabs", ...}]
        )

        # Message passing
        runtime.send_message(from_agent=child_id, to_agent=root_id, content="Done!")
        msg = runtime.receive_message(agent_id=root_id)

        # Cleanup
        runtime.complete_agent(child_id, result={"status": "success"})
        runtime.shutdown()
    """

    def __init__(
        self,
        vm_exec: Callable[[str], Optional[dict]],
        num_displays: int = 8,
        password: str = "password",
    ):
        """Initialize agent runtime.

        Args:
            vm_exec: Function to execute commands on VM
            num_displays: Number of virtual displays
            password: VM sudo password
        """
        self.vm_exec = vm_exec
        self.display_pool = DisplayPool(
            vm_exec=vm_exec,
            num_displays=num_displays,
            password=password,
        )

        # Agent tracking
        self.agents: Dict[str, Agent] = {}
        self._agent_lock = threading.RLock()  # Reentrant lock to avoid deadlock
        self._next_child_num = 0

        # Global message bus (for cross-agent communication)
        self._message_lock = threading.RLock()

    def initialize(self) -> bool:
        """Initialize the runtime (start display pool).

        Returns:
            bool: True if successful
        """
        logger.info("Initializing agent runtime...")
        success = self.display_pool.initialize()
        if success:
            logger.info(f"✓ Runtime ready with {self.display_pool.get_idle_count()} displays")
        return success

    def spawn_root_agent(self, task: str, display_num: int = 0) -> str:
        """Spawn the root agent (no parent).

        Args:
            task: Main task for root agent
            display_num: Display to use (typically 0 for primary)

        Returns:
            str: Root agent ID
        """
        agent_id = "root"

        with self._agent_lock:
            agent = Agent(
                agent_id=agent_id,
                parent_id=None,
                display_num=display_num,
                subtask=task,
            )
            self.agents[agent_id] = agent

        logger.info(f"Spawned root agent on display :{display_num}")
        return agent_id

    def fork_agent(
        self,
        parent_id: str,
        subtask: str,
        config: List[Dict[str, Any]],
        context_summary: Optional[str] = None,
    ) -> Optional[str]:
        """Fork a new child agent from a parent.

        Args:
            parent_id: ID of parent agent requesting fork
            subtask: Task for child to complete
            config: Setup configuration steps (OSWorld format)
            context_summary: Compressed text summary of parent's context

        Returns:
            str: Child agent ID if successful, None if failed
        """
        with self._agent_lock:
            parent = self.agents.get(parent_id)
            if not parent:
                logger.error(f"Cannot fork: parent {parent_id} not found")
                return None

            if parent.status != AgentStatus.RUNNING:
                logger.error(f"Cannot fork: parent {parent_id} not running ({parent.status})")
                return None

        # Allocate display for child
        display_num = self.display_pool.allocate(agent_id=f"{parent_id}_child")
        if display_num is None:
            logger.error(f"Cannot fork: no displays available")
            return None

        logger.info(f"Forking child from {parent_id} on display :{display_num}")

        # Run setup on child's display
        executor = SetupExecutor(display_num=display_num, vm_exec=self.vm_exec)
        setup_success = executor.execute_config(config)

        if not setup_success:
            logger.error(f"Fork setup failed for {parent_id}")
            self.display_pool.release(display_num)
            return None

        # Take screenshot of child's initial state
        screenshot = executor.take_screenshot()
        if not screenshot:
            logger.warning(f"Could not capture initial screenshot for child")

        # Generate child ID
        with self._agent_lock:
            child_id = f"{parent_id}_child_{self._next_child_num}"
            self._next_child_num += 1

            # Create child agent
            child = Agent(
                agent_id=child_id,
                parent_id=parent_id,
                display_num=display_num,
                subtask=subtask,
                context_summary=context_summary,
            )
            self.agents[child_id] = child

            # Track child in parent
            parent.children.add(child_id)

        logger.info(
            f"✓ Forked {child_id} on display :{display_num} "
            f"({len(config)} setup steps, subtask: {subtask[:60]}...)"
        )

        return child_id

    def complete_agent(self, agent_id: str, result: Any):
        """Mark an agent as completed with a result.

        Args:
            agent_id: Agent ID
            result: Agent's result (any data)
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.error(f"Cannot complete: agent {agent_id} not found")
                return

            agent.status = AgentStatus.COMPLETED
            agent.result = result
            agent.end_time = time.time()

            # Inject result into parent's pending results
            if agent.parent_id:
                parent = self.agents.get(agent.parent_id)
                if parent:
                    # Capture final screenshot as visual evidence
                    screenshot = None
                    if agent.display_num > 0:
                        executor = SetupExecutor(display_num=agent.display_num, vm_exec=self.vm_exec)
                        screenshot = executor.take_screenshot()

                    parent.pending_child_results.append({
                        "child_id": agent_id,
                        "status": "completed",
                        "result": result,
                        "screenshot": screenshot,  # Visual evidence of completion
                    })
                    logger.info(f"📬 Result from {agent_id} queued for {agent.parent_id} (with screenshot)")

            # Release display back to pool
            if agent.display_num > 0:  # Don't release display :0 (primary)
                self.display_pool.release(agent.display_num)

            duration = agent.end_time - agent.start_time
            logger.info(f"✓ Agent {agent_id} completed ({duration:.1f}s)")

    def fail_agent(self, agent_id: str, error: str):
        """Mark an agent as failed.

        Args:
            agent_id: Agent ID
            error: Error message
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.error(f"Cannot fail: agent {agent_id} not found")
                return

            agent.status = AgentStatus.FAILED
            agent.result = {"error": error}
            agent.end_time = time.time()

            # Inject failure into parent's pending results
            if agent.parent_id:
                parent = self.agents.get(agent.parent_id)
                if parent:
                    parent.pending_child_results.append({
                        "child_id": agent_id,
                        "status": "failed",
                        "error": error,
                    })
                    logger.info(f"📬 Failure from {agent_id} queued for {agent.parent_id}")

            # Release display
            if agent.display_num > 0:
                self.display_pool.release(agent.display_num)

            logger.error(f"✗ Agent {agent_id} failed: {error}")

    def kill_agent(self, agent_id: str, killer_id: str):
        """Kill an agent (only parent can kill child).

        Args:
            agent_id: Agent to kill
            killer_id: Agent requesting the kill
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            killer = self.agents.get(killer_id)

            if not agent or not killer:
                logger.error(f"Cannot kill: invalid agent IDs")
                return

            # Only parent can kill child
            if agent.parent_id != killer_id:
                logger.error(f"Cannot kill: {killer_id} is not parent of {agent_id}")
                return

            # Don't overwrite terminal states
            if agent.status in [AgentStatus.COMPLETED, AgentStatus.FAILED]:
                logger.info(f"Agent {agent_id} already {agent.status.value}, ignoring kill request")
                return

            agent.status = AgentStatus.KILLED
            agent.end_time = time.time()

            # Release display
            if agent.display_num > 0:
                self.display_pool.release(agent.display_num)

            logger.warning(f"⚠ Agent {agent_id} killed by {killer_id}")

    def peek_child(self, parent_id: str, child_id: str) -> Optional[Dict[str, Any]]:
        """Peek at a child agent's progress without interrupting them.

        Args:
            parent_id: ID of parent agent requesting peek
            child_id: ID of child agent to peek at

        Returns:
            Dict with child status, screenshot, conversation history, or None if invalid
        """
        with self._agent_lock:
            parent = self.agents.get(parent_id)
            child = self.agents.get(child_id)

            if not parent or not child:
                logger.error(f"Cannot peek: invalid agent IDs")
                return None

            # Only parent can peek at child
            if child.parent_id != parent_id:
                logger.error(f"Cannot peek: {parent_id} is not parent of {child_id}")
                return None

            # Get child info
            duration = (
                (child.end_time - child.start_time)
                if child.end_time
                else (time.time() - child.start_time)
            )

            # Take screenshot of child's display
            from setup_executor import SetupExecutor
            executor = SetupExecutor(display_num=child.display_num, vm_exec=self.vm_exec)
            screenshot = executor.take_screenshot()

            # Return child state
            result = {
                "child_id": child_id,
                "status": child.status.value,
                "steps": len(child.conversation_history),
                "duration": duration,
                "screenshot": screenshot,  # bytes or None
                "conversation": child.conversation_history,
            }

            logger.info(f"👀 {parent_id} peeking at {child_id} (step {len(child.conversation_history)})")
            return result

    def message_child(self, parent_id: str, child_id: str, message: str) -> bool:
        """Send a message to child agent(s) from parent.

        The message will be injected into the child's next observation as guidance
        from the coordinator. Use this to provide hints, clarifications, or share
        discoveries between workers.

        Args:
            parent_id: ID of parent agent sending the message
            child_id: ID of child agent to message, or 'all' to broadcast
            message: The message content to send

        Returns:
            True if message sent successfully, False otherwise
        """
        with self._agent_lock:
            parent = self.agents.get(parent_id)
            if not parent:
                logger.error(f"Cannot message: parent {parent_id} not found")
                return False

            if child_id == "all":
                # Broadcast to all active children
                success_count = 0
                for c_id in parent.children:
                    child = self.agents.get(c_id)
                    if child and child.status == AgentStatus.RUNNING:
                        child.pending_messages.append(message)
                        success_count += 1

                if success_count > 0:
                    logger.info(f"📢 {parent_id} broadcast to {success_count} children: {message[:80]}...")
                    return True
                else:
                    logger.warning(f"📢 {parent_id} has no active children to broadcast to")
                    return False
            else:
                # Message specific child
                child = self.agents.get(child_id)
                if not child:
                    logger.error(f"Cannot message: child {child_id} not found")
                    return False

                # Only parent can message child
                if child.parent_id != parent_id:
                    logger.error(f"Cannot message: {parent_id} is not parent of {child_id}")
                    return False

                child.pending_messages.append(message)
                logger.info(f"📬 {parent_id} → {child_id}: {message[:80]}...")
                return True

    def update_conversation(self, agent_id: str, entry: Dict[str, Any]):
        """Add an entry to agent's conversation history.

        Args:
            agent_id: Agent ID
            entry: Conversation entry (e.g., {"step": 1, "response": "..."})
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            if agent:
                agent.conversation_history.append(entry)

    def get_pending_child_results(self, agent_id: str) -> List[Dict[str, Any]]:
        """Get and clear pending child results for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of pending child results (and clears the list)
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return []

            results = agent.pending_child_results.copy()
            agent.pending_child_results.clear()
            return results

    def get_pending_messages(self, agent_id: str) -> List[str]:
        """Get and clear pending messages for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            List of pending messages (and clears the list)
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return []

            messages = agent.pending_messages.copy()
            agent.pending_messages.clear()
            return messages

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Dict with agent status info, or None if not found
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                return None

            return {
                "agent_id": agent.agent_id,
                "parent_id": agent.parent_id,
                "display_num": agent.display_num,
                "subtask": agent.subtask,
                "status": agent.status.value,
                "context_summary": agent.context_summary,
                "children": list(agent.children),
                "result": agent.result,
                "duration": (
                    (agent.end_time - agent.start_time)
                    if agent.end_time
                    else (time.time() - agent.start_time)
                ),
            }

    def get_all_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all agents.

        Returns:
            Dict mapping agent_id -> status info
        """
        with self._agent_lock:
            return {
                agent_id: self.get_agent_status(agent_id)
                for agent_id in self.agents.keys()
            }

    def shutdown(self):
        """Shutdown the runtime and cleanup resources."""
        logger.info("Shutting down agent runtime...")

        # Release all displays
        with self._agent_lock:
            for agent in self.agents.values():
                if agent.display_num > 0 and agent.status == AgentStatus.RUNNING:
                    self.display_pool.release(agent.display_num)

        # Cleanup display pool
        self.display_pool.cleanup()

        logger.info("Runtime shutdown complete")
