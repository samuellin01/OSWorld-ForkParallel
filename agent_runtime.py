"""Agent runtime for fork-based parallel execution.

Manages agent lifecycle, display allocation, fork operations, and message passing.
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from queue import Queue

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
class Message:
    """Message between agents."""
    from_agent: str
    to_agent: str
    content: Any
    timestamp: float = field(default_factory=time.time)


@dataclass
class Agent:
    """Represents a running agent."""
    agent_id: str
    parent_id: Optional[str]
    display_num: int
    subtask: str
    status: AgentStatus = AgentStatus.RUNNING
    children: Set[str] = field(default_factory=set)
    message_queue: Queue = field(default_factory=Queue)
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
        self._agent_lock = threading.Lock()
        self._next_child_num = 0

        # Global message bus (for cross-agent communication)
        self._message_lock = threading.Lock()

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
            )
            self.agents[child_id] = child

            # Track child in parent
            parent.children.add(child_id)

        logger.info(
            f"✓ Forked {child_id} on display :{display_num} "
            f"({len(config)} setup steps, subtask: {subtask[:60]}...)"
        )

        return child_id

    def send_message(self, from_agent: str, to_agent: str, content: Any):
        """Send a message from one agent to another.

        Only parent-child communication is allowed.

        Args:
            from_agent: Sender agent ID
            to_agent: Recipient agent ID
            content: Message content (any JSON-serializable data)
        """
        with self._agent_lock:
            sender = self.agents.get(from_agent)
            recipient = self.agents.get(to_agent)

            if not sender or not recipient:
                logger.error(f"Cannot send message: invalid agent IDs")
                return

            # Verify parent-child relationship
            is_parent_to_child = recipient.parent_id == from_agent
            is_child_to_parent = sender.parent_id == to_agent

            if not (is_parent_to_child or is_child_to_parent):
                logger.error(
                    f"Cannot send message: {from_agent} and {to_agent} "
                    f"are not parent-child"
                )
                return

            # Queue message
            msg = Message(from_agent=from_agent, to_agent=to_agent, content=content)
            recipient.message_queue.put(msg)

            logger.info(f"Message: {from_agent} → {to_agent}")

    def receive_message(self, agent_id: str, timeout: float = 0) -> Optional[Message]:
        """Receive a message for an agent (non-blocking by default).

        Args:
            agent_id: Agent ID to receive message for
            timeout: Seconds to wait for message (0 = non-blocking)

        Returns:
            Message if available, None otherwise
        """
        with self._agent_lock:
            agent = self.agents.get(agent_id)
            if not agent:
                logger.error(f"Cannot receive message: agent {agent_id} not found")
                return None

        try:
            # Get message from queue (non-blocking by default)
            msg = agent.message_queue.get(timeout=timeout) if timeout > 0 else agent.message_queue.get_nowait()
            logger.info(f"Received: {msg.from_agent} → {agent_id}")
            return msg
        except:
            return None

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

            agent.status = AgentStatus.KILLED
            agent.end_time = time.time()

            # Release display
            if agent.display_num > 0:
                self.display_pool.release(agent.display_num)

            logger.warning(f"⚠ Agent {agent_id} killed by {killer_id}")

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
