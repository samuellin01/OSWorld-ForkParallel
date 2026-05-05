"""Core data structures and scheduler for DAG-based parallel agent execution.

Replaces the fork-based agent runtime with a structured DAG scheduler that
extracts parallelism from task decomposition and assigns ready nodes to
available execution slots (displays).

Architecture:
- Coarse nodes (depth 0) each get their own display and run in parallel
- A worker on a display decomposes its node into a sub-DAG of finer steps
- The worker then executes the sub-DAG sequentially on its own display,
  recursing until it reaches atomic computer-use actions (click, type, key)
- Parallelism comes from multiple coarse nodes running on different displays
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from display_pool import DisplayPool
from setup_executor import SetupExecutor

logger = logging.getLogger(__name__)


@dataclass
class DAGNode:
    id: str
    task_description: str
    depends_on: List[str] = field(default_factory=list)
    status: str = "pending"  # pending | running | done | failed
    result: Optional[Dict[str, Any]] = None
    agent_id: Optional[str] = None
    display_num: Optional[int] = None
    parent_node_id: Optional[str] = None
    depth: int = 0
    setup_config: List[Dict[str, Any]] = field(default_factory=list)
    max_steps: int = 30
    start_time: Optional[float] = None
    timeout_seconds: float = 600.0
    retry_count: int = 0
    max_retries: int = 1


@dataclass
class DAGState:
    nodes: Dict[str, DAGNode] = field(default_factory=dict)
    root_task: str = ""
    max_depth: int = 3
    created_at: float = field(default_factory=time.time)


class DAGScheduler:
    """Scheduler that dispatches coarse DAG nodes to display slots.

    Each coarse node gets its own display. The worker on that display
    handles further decomposition and sequential execution internally.
    Parallelism comes from multiple coarse nodes running simultaneously.
    """

    def __init__(
        self,
        dag_state: DAGState,
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
        self.dag = dag_state
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
        self._worker_threads: Dict[str, threading.Thread] = {}
        self._bedrock_clients: Dict[str, Any] = {}
        self._start_time: Optional[float] = None
        self._completion_event = threading.Event()

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._bedrock_clients)

    def _find_ready_nodes(self) -> List[DAGNode]:
        ready = []
        with self._lock:
            for node in self.dag.nodes.values():
                if node.status != "pending":
                    continue
                deps_met = all(
                    self.dag.nodes[dep].status == "done"
                    for dep in node.depends_on
                    if dep in self.dag.nodes
                )
                if deps_met:
                    ready.append(node)
        return ready

    def _is_complete(self) -> bool:
        with self._lock:
            return all(
                n.status in ("done", "failed")
                for n in self.dag.nodes.values()
            )

    def _check_timeouts(self):
        now = time.time()
        with self._lock:
            for node in self.dag.nodes.values():
                if node.status != "running":
                    continue
                if node.start_time and (now - node.start_time) > node.timeout_seconds:
                    logger.warning(
                        "Node %s timed out (%.0fs > %.0fs)",
                        node.id, now - node.start_time, node.timeout_seconds,
                    )
                    self._handle_failure_locked(node, "timeout")

    def _gather_dependency_results(self, node: DAGNode) -> Dict[str, Any]:
        results = {}
        with self._lock:
            for dep_id in node.depends_on:
                dep = self.dag.nodes.get(dep_id)
                if dep and dep.result:
                    results[dep_id] = dep.result
        return results

    def _assign_node(self, node: DAGNode) -> bool:
        """Assign a coarse node to a free display and start a worker thread."""
        display_num = self.display_pool.allocate(agent_id=f"worker_{node.id}")
        if display_num is None:
            return False

        if node.setup_config:
            executor = SetupExecutor(display_num=display_num, vm_exec=self.vm_exec)
            setup_ok = executor.execute_config(node.setup_config)
            if not setup_ok:
                logger.warning(
                    "Setup failed for node %s on display :%d, proceeding anyway",
                    node.id, display_num,
                )

        with self._lock:
            node.status = "running"
            node.display_num = display_num
            node.agent_id = f"worker_{node.id}"
            node.start_time = time.time()

        import os
        worker_output = os.path.join(self.output_dir, node.agent_id)
        os.makedirs(worker_output, exist_ok=True)

        bedrock = self.bedrock_factory(worker_output, node.agent_id)
        with self._lock:
            self._bedrock_clients[node.agent_id] = bedrock

        dep_results = self._gather_dependency_results(node)

        thread = threading.Thread(
            target=self._run_worker_thread,
            args=(node, display_num, bedrock, worker_output, dep_results),
            daemon=True,
            name=f"worker-{node.id}",
        )
        with self._lock:
            self._worker_threads[node.id] = thread
        thread.start()

        logger.info(
            "Assigned node %s to display :%d (depth=%d, deps=%s)",
            node.id, display_num, node.depth, node.depends_on,
        )
        return True

    def _run_worker_thread(
        self,
        node: DAGNode,
        display_num: int,
        bedrock: Any,
        worker_output: str,
        dep_results: Dict[str, Any],
    ):
        try:
            from dag_worker import run_dag_worker
            result = run_dag_worker(
                node=node,
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                bedrock=bedrock,
                model=self.model,
                output_dir=worker_output,
                password=self.password,
                dependency_results=dep_results,
                max_depth=self.dag.max_depth,
            )
            self._report_completion(node.id, result)
        except Exception as e:
            logger.error("Worker for node %s crashed: %s", node.id, e, exc_info=True)
            self._report_failure(node.id, str(e))

    def _report_completion(self, node_id: str, result: Dict[str, Any]):
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node or node.status in ("done", "failed"):
                return

            if result.get("status") == "DONE":
                node.status = "done"
                node.result = result
                duration = time.time() - (node.start_time or time.time())
                logger.info("Node %s completed (%.1fs)", node_id, duration)
            else:
                self._handle_failure_locked(node, result.get("summary", "unknown"))

            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)

        self._completion_event.set()

    def _report_failure(self, node_id: str, error: str):
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node or node.status in ("done", "failed"):
                return
            self._handle_failure_locked(node, error)
            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)
        self._completion_event.set()

    def _handle_failure_locked(self, node: DAGNode, error: str):
        """Must be called with self._lock held."""
        if node.retry_count < node.max_retries and error != "cascade":
            node.retry_count += 1
            node.status = "pending"
            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)
            node.display_num = None
            node.agent_id = None
            node.start_time = None
            logger.info(
                "Node %s failed (%s), retrying (attempt %d/%d)",
                node.id, error[:200], node.retry_count, node.max_retries,
            )
            return

        node.status = "failed"
        node.result = {"error": error}
        logger.error("Node %s failed permanently: %s", node.id, error[:200])

        for other in self.dag.nodes.values():
            if node.id in other.depends_on and other.status == "pending":
                logger.warning("Cascading failure to downstream node %s", other.id)
                self._handle_failure_locked(other, "cascade")

    def run(self) -> Dict[str, Any]:
        """Main scheduler loop. Blocks until DAG is complete or timeout."""
        self._start_time = time.time()
        logger.info(
            "DAG Scheduler starting: %d nodes, max_depth=%d",
            len(self.dag.nodes), self.dag.max_depth,
        )

        while not self._is_complete():
            if time.time() - self._start_time > self.task_timeout:
                logger.warning("Task timeout (%.0fs)", self.task_timeout)
                with self._lock:
                    for n in self.dag.nodes.values():
                        if n.status in ("pending", "running"):
                            n.status = "failed"
                            n.result = {"error": "task_timeout"}
                break

            self._check_timeouts()

            ready = self._find_ready_nodes()
            assigned_any = False
            for node in ready:
                if self._assign_node(node):
                    assigned_any = True

            if not assigned_any:
                self._completion_event.wait(timeout=0.5)
                self._completion_event.clear()

        duration = time.time() - self._start_time

        with self._lock:
            all_done = all(n.status == "done" for n in self.dag.nodes.values())
            node_summaries = {}
            for n in self.dag.nodes.values():
                summary = n.result.get("summary", "") if n.result else ""
                node_summaries[n.id] = {
                    "status": n.status,
                    "depth": n.depth,
                    "summary": summary[:500] if summary else "",
                }

        overall_status = "DONE" if all_done else "FAIL"
        logger.info(
            "DAG Scheduler finished: %s (%.1fs, %d nodes)",
            overall_status, duration, len(self.dag.nodes),
        )

        return {
            "status": overall_status,
            "duration": duration,
            "nodes": node_summaries,
            "summary": self._build_final_summary(node_summaries),
        }

    def _build_final_summary(self, node_summaries: Dict[str, Any]) -> str:
        parts = []
        for nid, info in sorted(node_summaries.items()):
            status = info["status"]
            summary = info.get("summary", "")
            parts.append(f"[{nid}] {status}: {summary[:200]}")
        return "\n".join(parts)
