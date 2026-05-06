"""Core data structures and orchestrator for signal/await parallel execution.

Architecture:
  - Orchestrator decomposes a task into agents, each on its own display
  - Each agent runs phases sequentially on the same display
  - Cross-agent data flows through named signals (signal/await)
  - All agents start immediately; they only block at await points
  - Setup (no cross-agent deps) runs in parallel across all agents

Example:
  Agent A: [open chrome] → [search] → [extract] → signal("results")
  Agent B: [open doc] → [navigate] → await("results") → [paste]

  Agent B's first two actions run in parallel with Agent A.
  B only blocks at the await point. Both agents start immediately.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from display_pool import DisplayPool
from setup_executor import SetupExecutor

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """A named data channel between agents."""
    name: str
    producer: str
    data: Optional[Dict[str, Any]] = None
    is_set: bool = False
    failed: bool = False
    error: Optional[str] = None


@dataclass
class Phase:
    """A phase of work within an agent. Phases run sequentially on the same display."""
    id: str
    task: str
    awaits: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    max_steps: int = 20
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class AgentPlan:
    """An agent's complete work plan: display setup + ordered phases."""
    id: str
    task: str
    phases: List[Phase] = field(default_factory=list)
    setup: List[Dict[str, Any]] = field(default_factory=list)
    display_num: Optional[int] = None
    status: str = "pending"
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class DAGPlan:
    """The full execution plan: agents + signals."""
    agents: Dict[str, AgentPlan] = field(default_factory=dict)
    signals: Dict[str, Signal] = field(default_factory=dict)
    root_task: str = ""
    created_at: float = field(default_factory=time.time)


class Orchestrator:
    """Signal/await orchestrator for parallel agent execution.

    All agents start immediately on separate displays. Each agent runs
    its phases sequentially, blocking only at await points where it
    needs another agent's signal data.
    """

    def __init__(
        self,
        plan: DAGPlan,
        display_pool: DisplayPool,
        vm_exec: Callable[[str], Optional[dict]],
        bedrock_factory: Callable[[str, str], Any],
        model: str,
        vm_ip: str,
        server_port: int,
        output_dir: str,
        task_timeout: float = 1200.0,
        password: str = "osworld-public-evaluation",
    ):
        self.plan = plan
        self.display_pool = display_pool
        self.vm_exec = vm_exec
        self.bedrock_factory = bedrock_factory
        self.model = model
        self.vm_ip = vm_ip
        self.server_port = server_port
        self.output_dir = output_dir
        self.task_timeout = task_timeout
        self.password = password

        self._lock = threading.RLock()
        self._signal_events: Dict[str, threading.Event] = {
            name: threading.Event() for name in plan.signals
        }
        self._agent_threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._bedrock_clients)

    # ------------------------------------------------------------------
    # Signal operations
    # ------------------------------------------------------------------

    def set_signal(self, name: str, data: Dict[str, Any]):
        with self._lock:
            signal = self.plan.signals.get(name)
            if not signal:
                logger.error("Unknown signal: %s", name)
                return
            signal.data = data
            signal.is_set = True
            logger.info("Signal '%s' set by %s", name, signal.producer)
        self._signal_events[name].set()

    def fail_signal(self, name: str, error: str):
        with self._lock:
            signal = self.plan.signals.get(name)
            if not signal:
                return
            signal.failed = True
            signal.error = error
            logger.warning("Signal '%s' failed: %s", name, error)
        self._signal_events[name].set()

    def wait_signal(self, name: str, timeout: Optional[float] = None) -> Optional[Dict[str, Any]]:
        event = self._signal_events.get(name)
        if not event:
            logger.error("Unknown signal: %s", name)
            return None

        event.wait(timeout=timeout)

        with self._lock:
            signal = self.plan.signals.get(name)
            if not signal or signal.failed:
                return None
            return signal.data

    def is_signal_failed(self, name: str) -> bool:
        with self._lock:
            signal = self.plan.signals.get(name)
            return signal.failed if signal else True

    # ------------------------------------------------------------------
    # Agent execution
    # ------------------------------------------------------------------

    def _run_agent_thread(self, agent: AgentPlan):
        tag = f"[{agent.id}]"
        try:
            agent.status = "running"
            agent.start_time = time.time()

            if agent.setup and agent.display_num is not None:
                executor = SetupExecutor(display_num=agent.display_num, vm_exec=self.vm_exec)
                setup_ok = executor.execute_config(agent.setup)
                if not setup_ok:
                    logger.warning("%s Setup failed, proceeding anyway", tag)

            agent_output = os.path.join(self.output_dir, agent.id)
            os.makedirs(agent_output, exist_ok=True)
            bedrock = self.bedrock_factory(agent_output, agent.id)
            with self._lock:
                self._bedrock_clients[agent.id] = bedrock

            for i, phase in enumerate(agent.phases):
                if self._start_time and (time.time() - self._start_time) > self.task_timeout:
                    phase.status = "failed"
                    phase.result = {"error": "task_timeout"}
                    raise TimeoutError("Task timeout")

                logger.info("%s Phase %d/%d: %s", tag, i + 1, len(agent.phases), phase.id)

                # Await signals
                signal_data: Dict[str, Any] = {}
                if phase.awaits:
                    phase.status = "blocked"
                    logger.info("%s Awaiting signals: %s", tag, phase.awaits)

                    for signal_name in phase.awaits:
                        remaining = None
                        if self._start_time:
                            remaining = max(1.0, self.task_timeout - (time.time() - self._start_time))

                        data = self.wait_signal(signal_name, timeout=remaining)

                        if self.is_signal_failed(signal_name):
                            phase.status = "failed"
                            phase.result = {"error": f"Awaited signal '{signal_name}' failed"}
                            raise RuntimeError(f"Signal '{signal_name}' failed")
                        if data is None:
                            phase.status = "failed"
                            phase.result = {"error": f"Timeout waiting for signal '{signal_name}'"}
                            raise TimeoutError(f"Signal '{signal_name}' timed out")

                        signal_data[signal_name] = data

                # Run CUA loop
                phase.status = "running"
                phase.start_time = time.time()

                from dag_worker import run_phase
                result = run_phase(
                    agent=agent,
                    phase=phase,
                    phase_index=i,
                    vm_ip=self.vm_ip,
                    server_port=self.server_port,
                    bedrock=bedrock,
                    model=self.model,
                    output_dir=agent_output,
                    password=self.password,
                    signal_data=signal_data if signal_data else None,
                )

                phase.end_time = time.time()
                phase.result = result

                if result.get("status") in ("DONE", "MAX_STEPS"):
                    phase.status = "done"
                    for signal_name in phase.signals:
                        self.set_signal(signal_name, result)
                else:
                    phase.status = "failed"
                    for signal_name in phase.signals:
                        self.fail_signal(signal_name, f"Phase {phase.id} failed")
                    raise RuntimeError(f"Phase {phase.id} failed: {result.get('summary', 'unknown')[:200]}")

            agent.status = "done"
            agent.end_time = time.time()
            duration = agent.end_time - (agent.start_time or agent.end_time)
            logger.info("%s Completed all %d phases (%.1fs)", tag, len(agent.phases), duration)

        except Exception as e:
            logger.error("%s Failed: %s", tag, e, exc_info=True)
            agent.status = "failed"
            agent.end_time = time.time()
            for phase in agent.phases:
                for signal_name in phase.signals:
                    with self._lock:
                        signal = self.plan.signals.get(signal_name)
                        if signal and not signal.is_set:
                            self.fail_signal(signal_name, str(e))

        finally:
            if agent.display_num is not None:
                self.display_pool.release(agent.display_num)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        self._start_time = time.time()
        num_agents = len(self.plan.agents)
        logger.info("Orchestrator starting: %d agents, %d signals",
                     num_agents, len(self.plan.signals))

        for agent in self.plan.agents.values():
            display_num = self.display_pool.allocate(agent_id=agent.id)
            if display_num is None:
                logger.error("No display available for agent %s", agent.id)
                agent.status = "failed"
                for phase in agent.phases:
                    for sig in phase.signals:
                        self.fail_signal(sig, "no display available")
                continue

            agent.display_num = display_num
            thread = threading.Thread(
                target=self._run_agent_thread,
                args=(agent,),
                daemon=True,
                name=f"agent-{agent.id}",
            )
            self._agent_threads[agent.id] = thread
            thread.start()
            logger.info("Started agent %s on display :%d", agent.id, display_num)

        for agent_id, thread in self._agent_threads.items():
            remaining = max(1.0, self.task_timeout - (time.time() - self._start_time))
            thread.join(timeout=remaining)
            if thread.is_alive():
                logger.warning("Agent %s still running after timeout", agent_id)
                self.plan.agents[agent_id].status = "failed"

        duration = time.time() - self._start_time
        all_done = all(a.status == "done" for a in self.plan.agents.values())
        overall_status = "DONE" if all_done else "FAIL"

        agent_summaries = {}
        for agent in self.plan.agents.values():
            phase_summaries = []
            for phase in agent.phases:
                summary = ""
                if phase.result:
                    summary = phase.result.get("summary", "")[:300]
                phase_summaries.append({
                    "id": phase.id,
                    "status": phase.status,
                    "summary": summary,
                })
            agent_summaries[agent.id] = {
                "status": agent.status,
                "phases": phase_summaries,
                "display_num": agent.display_num,
            }

        logger.info("Orchestrator finished: %s (%.1fs, %d agents)",
                     overall_status, duration, num_agents)

        return {
            "status": overall_status,
            "duration": duration,
            "agents": agent_summaries,
            "summary": self._build_summary(agent_summaries),
        }

    def _build_summary(self, agent_summaries: Dict[str, Any]) -> str:
        parts = []
        for agent_id, info in sorted(agent_summaries.items()):
            parts.append(f"[{agent_id}] {info['status']}")
            for phase in info.get("phases", []):
                parts.append(f"  {phase['id']}: {phase['status']} - {phase.get('summary', '')[:150]}")
        return "\n".join(parts)
