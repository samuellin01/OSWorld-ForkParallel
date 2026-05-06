"""Core data structures and orchestrator for signal/await parallel execution.

Architecture:
  Orchestrator (plan, signals, displays, task timeout)
  ├── Manager A (LLM, watches every step, spawns helpers, messages worker)
  │   └── Worker A (CUA: computer + await_signal)
  ├── Manager B
  │   └── Worker B
  │
  │  (Manager A spawns helper at runtime)
  ├── Manager H1
  │   └── Helper Worker 1

  - Orchestrator owns the plan, signals, and display allocation
  - Each agent gets a Manager that watches its worker's every step
  - Manager maintains a running assessment: work done, work remaining, pace
  - Manager spawns helpers when it sees separable remaining work
  - Manager messages worker to narrow focus ("helper is handling X, skip it")
  - Worker is a pure CUA executor with computer + await_signal tools
  - No hard step limits — manager nudges, task timeout is the wall
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

from display_pool import DisplayPool
from setup_executor import SetupExecutor

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

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
class StepRecord:
    """One step of a worker's execution, visible to its manager."""
    step_num: int
    timestamp: float
    elapsed: float
    action_summary: str


@dataclass
class Phase:
    """A phase of work within an agent. Phases run sequentially on the same display."""
    id: str
    task: str
    awaits: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    status: str = "pending"
    result: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    current_step: int = 0
    latest_response: str = ""
    step_history: List[StepRecord] = field(default_factory=list)


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


# ------------------------------------------------------------------
# Worker tool schemas
# ------------------------------------------------------------------

AWAIT_SIGNAL_TOOL = {
    "name": "await_signal",
    "description": (
        "Block until data from another agent is ready, then return it. "
        "Use this when you need results from a parallel agent before continuing. "
        "You can do independent setup work first, then call this when you actually need the data. "
        "The call will block until the data is available."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "signal_name": {
                "type": "string",
                "description": "Name of the signal to wait for (from the task description).",
            },
        },
        "required": ["signal_name"],
    },
}


# ------------------------------------------------------------------
# Manager: per-agent supervisor
# ------------------------------------------------------------------

_MANAGER_PROMPT = """\
You are managing a computer-use worker agent. You watch every step it takes \
and decide when to intervene.

Overall goal: {root_task}

Other agents working on this task (handled by their own managers — NOT your scope):
{sibling_agents}

Your worker's assigned task: {agent_task}
Current phase: {phase_task}
Free displays available for helpers: {idle_displays}

Helpers you have already spawned (do NOT duplicate):
{helpers_already_spawned}

Your previous assessment:
{previous_assessment}

New steps since last check:
{new_steps}

Worker's latest output:
{latest_response}

Updated totals: {total_steps} steps, {total_elapsed:.0f}s elapsed, {avg_step_time:.1f}s/step

Update your assessment and decide on action:

1. Write your updated assessment:
ASSESSMENT
work_completed: <what the worker has done so far>
work_remaining: <what's left within THIS worker's task>
estimated_remaining_steps: <number>
pace_notes: <observations about speed or struggles>

2. Decide on action:

CONTINUE — worker is on track, no intervention needed

SPAWN_HELPER — there is separable work WITHIN THIS WORKER'S TASK that a \
helper could do in parallel. Do NOT spawn helpers for work assigned to the \
other agents listed above — their managers handle that. Do NOT duplicate \
helpers already spawned above.
helper_task: <specific, self-contained subtask>
helper_setup: <"none" or a JSON setup action>
message_to_worker: <tell worker what to skip since helper handles it>

NUDGE — worker is going off track, stuck, or doing work that belongs to \
another agent. If the worker is doing another agent's job, tell it to stop. \
Do NOT provide task-specific data or answers yourself — only provide \
coordination guidance (what to do, what to skip, how to proceed).
message_to_worker: <guidance or course correction>"""


MAX_HELPERS_PER_MANAGER = 3


class Manager:
    """Per-agent supervisor that watches every worker step.

    Runs in its own thread alongside the worker. Processes each new step,
    maintains a running assessment of progress, and intervenes when needed
    (spawn helpers, message worker, nudge).

    Scoped to its own worker's task only — does not try to help other agents.
    """

    def __init__(
        self,
        agent: AgentPlan,
        orchestrator: "Orchestrator",
        bedrock: Any,
        model: str,
        sibling_agents: Optional[List[AgentPlan]] = None,
    ):
        self.agent = agent
        self.orchestrator = orchestrator
        self.bedrock = bedrock
        self.model = model
        self._last_step_seen = 0
        self._assessment = "(no assessment yet — worker just started)"
        self._helpers_spawned: List[str] = []
        self._helper_ids: List[str] = []
        self._sibling_info = self._format_siblings(sibling_agents or [])

    def run(self):
        """Main loop: watch steps, evaluate, then merge helper results if any."""
        tag = f"[mgr:{self.agent.id}]"
        logger.info("%s Manager started", tag)

        while self.agent.status == "running":
            phase = self._current_running_phase()
            if not phase:
                time.sleep(1)
                continue

            current_len = len(phase.step_history)
            if current_len <= self._last_step_seen:
                time.sleep(1)
                continue

            new_steps = phase.step_history[self._last_step_seen:current_len]
            self._last_step_seen = current_len

            try:
                self._evaluate(phase, new_steps, tag)
            except Exception as e:
                logger.warning("%s Evaluation failed: %s", tag, e)

        # Worker finished. If we spawned helpers, wait for them and merge results into deferred signals.
        if self._helper_ids:
            self._wait_and_merge(tag)

        logger.info("%s Manager finished (agent status: %s)", tag, self.agent.status)

    def _current_running_phase(self) -> Optional[Phase]:
        for phase in self.agent.phases:
            if phase.status == "running":
                return phase
        return None

    @staticmethod
    def _format_siblings(siblings: List[AgentPlan]) -> str:
        if not siblings:
            return "(none — this is the only agent)"
        lines = []
        for s in siblings:
            lines.append(f"  - {s.id}: {s.task}")
        return "\n".join(lines)

    def _format_new_steps(self, new_steps: List[StepRecord]) -> str:
        lines = []
        for rec in new_steps:
            lines.append(f"  step {rec.step_num} (+{rec.elapsed:.0f}s, {rec.elapsed - (new_steps[0].elapsed if new_steps[0] != rec else 0):.0f}s since prev): {rec.action_summary}")
        return "\n".join(lines) if lines else "(none)"

    def _evaluate(self, phase: Phase, new_steps: List[StepRecord], tag: str):
        total_steps = phase.current_step
        total_elapsed = new_steps[-1].elapsed if new_steps else 0
        avg_step_time = total_elapsed / total_steps if total_steps > 0 else 0
        idle_displays = self.orchestrator.display_pool.get_idle_count()

        # Format new steps with inter-step latency
        step_lines = []
        for i, rec in enumerate(new_steps):
            if i > 0:
                delta = f", {rec.elapsed - new_steps[i-1].elapsed:.1f}s since prev"
            else:
                delta = ""
            step_lines.append(f"  step {rec.step_num} (+{rec.elapsed:.0f}s{delta}): {rec.action_summary}")
        new_steps_text = "\n".join(step_lines) if step_lines else "(none)"

        # Format already-spawned helpers so the LLM doesn't repeat
        if self._helpers_spawned:
            helpers_text = "\n".join(f"  - {h}" for h in self._helpers_spawned)
        else:
            helpers_text = "(none)"

        # If at max helpers, don't show idle displays to avoid tempting the LLM
        effective_idle = idle_displays if len(self._helpers_spawned) < MAX_HELPERS_PER_MANAGER else 0

        prompt = _MANAGER_PROMPT.format(
            root_task=self.orchestrator.plan.root_task[:500],
            sibling_agents=self._sibling_info,
            agent_task=self.agent.task[:300],
            phase_task=phase.task[:300],
            idle_displays=effective_idle,
            helpers_already_spawned=helpers_text,
            previous_assessment=self._assessment,
            new_steps=new_steps_text,
            latest_response=phase.latest_response[:2000],
            total_steps=total_steps,
            total_elapsed=total_elapsed,
            avg_step_time=avg_step_time,
        )

        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]

        logger.info("%s Evaluating (step %d, %.0fs elapsed, %.1fs/step, %d idle displays)",
                     tag, total_steps, total_elapsed, avg_step_time, idle_displays)

        try:
            content_blocks, _ = self.bedrock.chat(
                messages=messages,
                system="You are a task manager. Be concise and decisive.",
                model=self.model,
                temperature=0.3,
                max_tokens=600,
            )
        except Exception as e:
            logger.warning("%s LLM call failed: %s", tag, e)
            return

        response = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )

        self._parse_and_act(response, tag)

    def _parse_and_act(self, response: str, tag: str):
        # Extract assessment
        if "ASSESSMENT" in response:
            assessment_lines = []
            in_assessment = False
            for line in response.split("\n"):
                stripped = line.strip()
                if stripped == "ASSESSMENT":
                    in_assessment = True
                    continue
                if in_assessment:
                    if stripped in ("CONTINUE", "SPAWN_HELPER", "NUDGE"):
                        break
                    assessment_lines.append(stripped)
            if assessment_lines:
                self._assessment = "\n".join(assessment_lines)

        if "SPAWN_HELPER" in response:
            if len(self._helpers_spawned) >= MAX_HELPERS_PER_MANAGER:
                logger.info("%s SPAWN_HELPER requested but at cap (%d helpers)", tag, MAX_HELPERS_PER_MANAGER)
            else:
                helper_task = ""
                helper_setup: List[Dict[str, Any]] = []
                message = ""

                for line in response.split("\n"):
                    line = line.strip()
                    if line.startswith("helper_task:"):
                        helper_task = line[len("helper_task:"):].strip()
                    elif line.startswith("helper_setup:"):
                        setup_str = line[len("helper_setup:"):].strip()
                        if setup_str.lower() != "none":
                            try:
                                parsed = json.loads(setup_str)
                                helper_setup = [parsed] if isinstance(parsed, dict) else parsed
                            except json.JSONDecodeError:
                                pass
                    elif line.startswith("message_to_worker:"):
                        message = line[len("message_to_worker:"):].strip()

                if helper_task:
                    logger.info("%s SPAWN_HELPER: %s", tag, helper_task[:100])
                    # Defer parent signals — we'll merge helper results before setting
                    phase = self._current_running_phase()
                    if phase:
                        for sig in phase.signals:
                            self.orchestrator.defer_signal(sig)
                    helper_id = self.orchestrator.spawn_helper(helper_task, helper_setup)
                    if helper_id:
                        self._helper_ids.append(helper_id)
                        self._helpers_spawned.append(f"{helper_id}: {helper_task[:80]}")
                    if message:
                        self.orchestrator.send_message(self.agent.id, message)
                else:
                    logger.info("%s SPAWN_HELPER but no task parsed", tag)

        elif "NUDGE" in response:
            message = ""
            for line in response.split("\n"):
                line = line.strip()
                if line.startswith("message_to_worker:"):
                    message = line[len("message_to_worker:"):].strip()
            if message:
                logger.info("%s NUDGE: %s", tag, message[:100])
                self.orchestrator.send_message(self.agent.id, message)

        elif "CONTINUE" in response:
            logger.info("%s CONTINUE", tag)

        else:
            logger.info("%s No clear action in response", tag)

    def _wait_and_merge(self, tag: str):
        """Wait for all spawned helpers to finish, merge their results into deferred signals."""
        logger.info("%s Waiting for %d helpers to finish...", tag, len(self._helper_ids))

        deadline = time.time() + 600
        for helper_id in self._helper_ids:
            helper = self.orchestrator._helpers.get(helper_id)
            if not helper:
                continue
            while helper.status not in ("done", "failed") and time.time() < deadline:
                time.sleep(2)
            logger.info("%s Helper %s finished: %s", tag, helper_id, helper.status)

        # Merge: worker result + all helper results
        worker_summary = ""
        for phase in self.agent.phases:
            if phase.result and phase.result.get("summary"):
                worker_summary += phase.result["summary"]

        helper_summaries = []
        for helper_id in self._helper_ids:
            helper = self.orchestrator._helpers.get(helper_id)
            if helper and helper.status == "done":
                for phase in helper.phases:
                    if phase.result and phase.result.get("summary"):
                        helper_summaries.append(phase.result["summary"])

        merged_summary = worker_summary
        if helper_summaries:
            merged_summary += "\n\n--- Additional results from helper agents ---\n\n"
            merged_summary += "\n\n".join(helper_summaries)

        merged_result = {"status": "DONE", "summary": merged_summary}

        # Set all deferred signals with merged data
        for phase in self.agent.phases:
            for signal_name in phase.signals:
                if self.orchestrator.is_signal_deferred(signal_name):
                    logger.info("%s Setting deferred signal '%s' with merged data (%d chars)",
                                 tag, signal_name, len(merged_summary))
                    self.orchestrator.set_signal(signal_name, merged_result)


# ------------------------------------------------------------------
# Orchestrator
# ------------------------------------------------------------------

class Orchestrator:
    """Signal/await orchestrator with per-agent managers.

    Owns the plan, signals, and display allocation. Does NOT have a
    global monitor — each agent gets its own Manager that watches
    every step and handles interventions.
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
        self._manager_threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._start_time: Optional[float] = None

        self._agent_messages: Dict[str, List[str]] = {}

        self._helpers: Dict[str, AgentPlan] = {}
        self._helper_count = 0
        self._deferred_signals: Set[str] = set()

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

    def register_signal(self, name: str, producer: str):
        with self._lock:
            if name not in self.plan.signals:
                self.plan.signals[name] = Signal(name=name, producer=producer)
                self._signal_events[name] = threading.Event()
                logger.info("Registered dynamic signal '%s' (producer=%s)", name, producer)

    def defer_signal(self, name: str):
        """Mark a signal as deferred — the manager will set it after helpers finish."""
        with self._lock:
            self._deferred_signals.add(name)
        logger.info("Signal '%s' deferred (manager will set after helper merge)", name)

    def is_signal_deferred(self, name: str) -> bool:
        with self._lock:
            return name in self._deferred_signals

    # ------------------------------------------------------------------
    # Messaging
    # ------------------------------------------------------------------

    def send_message(self, agent_id: str, message: str):
        with self._lock:
            self._agent_messages.setdefault(agent_id, []).append(message)
        logger.info("Message -> %s: %s", agent_id, message[:120])

    def get_pending_messages(self, agent_id: str) -> List[str]:
        with self._lock:
            return self._agent_messages.pop(agent_id, [])

    # ------------------------------------------------------------------
    # Helper spawning
    # ------------------------------------------------------------------

    def spawn_helper(self, task: str, setup: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        self._helper_count += 1
        helper_id = f"helper_{self._helper_count}"

        helper = AgentPlan(
            id=helper_id,
            task=task,
            phases=[Phase(id="execute", task=task)],
            setup=setup or [],
        )

        display_num = self.display_pool.allocate(agent_id=helper_id)
        if display_num is None:
            logger.warning("No display available for helper %s", helper_id)
            return None

        helper.display_num = display_num

        with self._lock:
            self._helpers[helper_id] = helper

        thread = threading.Thread(
            target=self._run_agent_thread,
            args=(helper,),
            daemon=True,
            name=f"agent-{helper_id}",
        )
        with self._lock:
            self._agent_threads[helper_id] = thread
        thread.start()

        logger.info("Spawned helper %s on display :%d — %s", helper_id, display_num, task[:80])
        return helper_id

    # ------------------------------------------------------------------
    # Agent execution (worker + manager)
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

            worker_bedrock = self.bedrock_factory(agent_output, agent.id)
            with self._lock:
                self._bedrock_clients[agent.id] = worker_bedrock

            # Start manager in its own thread
            manager_output = os.path.join(agent_output, "_manager")
            os.makedirs(manager_output, exist_ok=True)
            manager_bedrock = self.bedrock_factory(manager_output, f"mgr_{agent.id}")
            with self._lock:
                self._bedrock_clients[f"mgr_{agent.id}"] = manager_bedrock

            siblings = [a for a in self.plan.agents.values() if a.id != agent.id]
            manager = Manager(agent, self, manager_bedrock, self.model, sibling_agents=siblings)
            manager_thread = threading.Thread(
                target=manager.run, daemon=True, name=f"mgr-{agent.id}",
            )
            manager_thread.start()
            with self._lock:
                self._manager_threads[agent.id] = manager_thread

            # Run phases
            for i, phase in enumerate(agent.phases):
                if self._start_time and (time.time() - self._start_time) > self.task_timeout:
                    phase.status = "failed"
                    phase.result = {"error": "task_timeout"}
                    raise TimeoutError("Task timeout")

                logger.info("%s Phase %d/%d: %s", tag, i + 1, len(agent.phases), phase.id)

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

                phase.status = "running"
                phase.start_time = time.time()

                from dag_worker import run_phase
                result = run_phase(
                    agent=agent,
                    phase=phase,
                    phase_index=i,
                    vm_ip=self.vm_ip,
                    server_port=self.server_port,
                    bedrock=worker_bedrock,
                    model=self.model,
                    output_dir=agent_output,
                    password=self.password,
                    signal_data=signal_data if signal_data else None,
                    orchestrator=self,
                )

                phase.end_time = time.time()
                phase.result = result

                if result.get("status") in ("DONE",):
                    phase.status = "done"
                    for signal_name in phase.signals:
                        if not self.is_signal_deferred(signal_name):
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

        # Wait for all threads (agents + dynamically spawned helpers)
        deadline = self._start_time + self.task_timeout
        while time.time() < deadline:
            with self._lock:
                threads = dict(self._agent_threads)
            all_done = True
            for thread in threads.values():
                if thread.is_alive():
                    remaining = max(0.1, deadline - time.time())
                    thread.join(timeout=min(2.0, remaining))
                    if thread.is_alive():
                        all_done = False
            if all_done:
                break

        duration = time.time() - self._start_time
        all_agents = list(self.plan.agents.values()) + list(self._helpers.values())
        for agent in all_agents:
            t = self._agent_threads.get(agent.id)
            if t and t.is_alive():
                logger.warning("Agent %s still running after timeout", agent.id)
                agent.status = "failed"

        all_done = all(a.status == "done" for a in all_agents)
        overall_status = "DONE" if all_done else "FAIL"

        agent_summaries = {}
        for agent in all_agents:
            phase_summaries = []
            for phase in agent.phases:
                summary = ""
                if phase.result:
                    summary = phase.result.get("summary", "")[:300]
                phase_summaries.append({
                    "id": phase.id,
                    "status": phase.status,
                    "summary": summary,
                    "steps_used": phase.current_step,
                })
            agent_summaries[agent.id] = {
                "status": agent.status,
                "phases": phase_summaries,
                "display_num": agent.display_num,
            }

        total_agents = len(all_agents)
        helpers_spawned = len(self._helpers)
        logger.info("Orchestrator finished: %s (%.1fs, %d agents, %d helpers spawned)",
                     overall_status, duration, total_agents, helpers_spawned)

        return {
            "status": overall_status,
            "duration": duration,
            "agents": agent_summaries,
            "helpers_spawned": helpers_spawned,
            "summary": self._build_summary(agent_summaries),
        }

    def _build_summary(self, agent_summaries: Dict[str, Any]) -> str:
        parts = []
        for agent_id, info in sorted(agent_summaries.items()):
            parts.append(f"[{agent_id}] {info['status']}")
            for phase in info.get("phases", []):
                parts.append(f"  {phase['id']}: {phase['status']} ({phase.get('steps_used', 0)} steps) - {phase.get('summary', '')[:150]}")
        return "\n".join(parts)
