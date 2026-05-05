"""Core data structures and scheduler for hierarchical DAG-based parallel execution.

Architecture (from spec):
  - Scheduler maintains a global merged DAG
  - Assigns ready nodes to free slots (displays)
  - Workers either decompose (expand node → sub-DAG merged back) or execute
  - Recursive decomposition until nodes are atomic CUA agent tasks
  - Parallelism extracted at every level of the hierarchy
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
    """Scheduler that maintains a global DAG, assigns ready nodes to slots,
    and handles node expansion (sub-DAG merging) from workers.

    Core loop (spec §2):
      1. Find ready nodes (all deps done, status pending)
      2. Assign to free slots
      3. Wait for completions or expansions
      4. On completion: mark done, free slot
      5. On expansion: merge sub-DAG into global DAG, free slot,
         sub-nodes become schedulable
      6. Repeat until all nodes done/failed
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
        self._event = threading.Event()

    def get_all_bedrock_clients(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._bedrock_clients)

    # ------------------------------------------------------------------
    # DAG queries (lock must be held by caller or acquired internally)
    # ------------------------------------------------------------------

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

    def _gather_dependency_results(self, node: DAGNode) -> Dict[str, Any]:
        results = {}
        with self._lock:
            for dep_id in node.depends_on:
                dep = self.dag.nodes.get(dep_id)
                if dep and dep.result:
                    results[dep_id] = dep.result
        return results

    # ------------------------------------------------------------------
    # Node assignment
    # ------------------------------------------------------------------

    def _assign_node(self, node: DAGNode) -> bool:
        """Assign a node to a free display slot and start a worker."""
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

        worker_output = os.path.join(self.output_dir, node.agent_id)
        os.makedirs(worker_output, exist_ok=True)

        bedrock = self.bedrock_factory(worker_output, node.agent_id)
        with self._lock:
            self._bedrock_clients[node.agent_id] = bedrock

        dep_results = self._gather_dependency_results(node)

        thread = threading.Thread(
            target=self._run_worker_thread,
            args=(node, bedrock, worker_output, dep_results),
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
        bedrock: Any,
        worker_output: str,
        dep_results: Dict[str, Any],
    ):
        try:
            from dag_worker import run_dag_worker
            result = run_dag_worker(
                node=node,
                scheduler=self,
                vm_ip=self.vm_ip,
                server_port=self.server_port,
                bedrock=bedrock,
                model=self.model,
                output_dir=worker_output,
                password=self.password,
                dependency_results=dep_results,
            )
            # Worker returns None if it expanded (already handled via report_expansion)
            if result is not None:
                if result.get("status") == "DONE":
                    self._report_completion(node.id, result)
                else:
                    self._report_failure(node.id, result.get("summary", "unknown"))
        except Exception as e:
            logger.error("Worker for node %s crashed: %s", node.id, e, exc_info=True)
            self._report_failure(node.id, str(e))

    # ------------------------------------------------------------------
    # Worker callbacks
    # ------------------------------------------------------------------

    def _report_completion(self, node_id: str, result: Dict[str, Any]):
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node or node.status in ("done", "failed"):
                return
            node.status = "done"
            node.result = result
            duration = time.time() - (node.start_time or time.time())
            logger.info("Node %s completed (%.1fs, depth=%d)", node_id, duration, node.depth)
            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)
        self._event.set()

    def _report_failure(self, node_id: str, error: str):
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node or node.status in ("done", "failed"):
                return
            self._handle_failure_locked(node, error)
            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)
        self._event.set()

    def report_expansion(self, node_id: str, sub_dag_plan: List[Dict[str, Any]]):
        """Called by a worker to expand a node into a sub-DAG.

        The node is removed from the global DAG and replaced by sub-nodes.
        The worker's slot is freed. Sub-nodes become pending and will be
        scheduled to (potentially different) slots by the main loop.

        This is spec §3 (worker decompose path) + §4 (DAG expansion).
        """
        with self._lock:
            node = self.dag.nodes.get(node_id)
            if not node:
                return
            logger.info(
                "Node %s expanding into %d sub-nodes (depth %d→%d)",
                node_id, len(sub_dag_plan), node.depth, node.depth + 1,
            )
            if node.display_num is not None and node.display_num > 0:
                self.display_pool.release(node.display_num)
            self._expand_node(node, sub_dag_plan)
        self._event.set()

    # ------------------------------------------------------------------
    # DAG expansion (spec §4)
    # ------------------------------------------------------------------

    def _expand_node(self, node: DAGNode, sub_dag_plan: List[Dict[str, Any]]):
        """Replace a node with its sub-DAG in the global DAG.

        1. Create sub-nodes with namespaced IDs
        2. Entry sub-nodes (no internal deps) inherit original's external deps
        3. Find terminal sub-nodes (nothing in sub-DAG depends on them)
        4. Rewire: anything that depended on original now depends on terminals
        5. Remove original node

        Must be called with self._lock held.
        """
        sub_nodes = []
        for step in sub_dag_plan:
            sub_id = f"{node.id}__{step['id']}"
            internal_deps = [f"{node.id}__{d}" for d in step.get("depends_on", [])]
            sub_node = DAGNode(
                id=sub_id,
                task_description=step["task"],
                depends_on=internal_deps,
                status="pending",
                parent_node_id=node.id,
                depth=node.depth + 1,
                setup_config=step.get("setup", []),
                max_steps=step.get("max_steps", node.max_steps),
                timeout_seconds=node.timeout_seconds,
                max_retries=node.max_retries,
            )
            sub_nodes.append(sub_node)

        all_sub_ids = {sn.id for sn in sub_nodes}

        # Entry nodes: sub-nodes with no internal dependencies
        # They inherit the original node's external dependencies
        for sn in sub_nodes:
            has_internal_deps = any(d in all_sub_ids for d in sn.depends_on)
            if not has_internal_deps:
                sn.depends_on = list(node.depends_on)

        # Terminal nodes: sub-nodes that no other sub-node depends on
        depended_on_internally = set()
        for sn in sub_nodes:
            depended_on_internally.update(d for d in sn.depends_on if d in all_sub_ids)
        terminal_ids = list(all_sub_ids - depended_on_internally)

        # Rewire downstream: anything depending on original now depends on terminals
        for other in self.dag.nodes.values():
            if node.id in other.depends_on:
                other.depends_on.remove(node.id)
                other.depends_on.extend(terminal_ids)

        # Add sub-nodes, remove original
        for sn in sub_nodes:
            self.dag.nodes[sn.id] = sn
        del self.dag.nodes[node.id]

        logger.info(
            "Expanded %s → %d sub-nodes (entries: %s, terminals: %s)",
            node.id,
            len(sub_nodes),
            [sn.id for sn in sub_nodes if not any(d in all_sub_ids for d in sn.depends_on)],
            terminal_ids,
        )

    # ------------------------------------------------------------------
    # Failure handling
    # ------------------------------------------------------------------

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
                "Node %s failed (%s), retrying (%d/%d)",
                node.id, error[:200], node.retry_count, node.max_retries,
            )
            return

        node.status = "failed"
        node.result = {"error": error}
        logger.error("Node %s failed permanently: %s", node.id, error[:200])

        for other in self.dag.nodes.values():
            if node.id in other.depends_on and other.status == "pending":
                logger.warning("Cascading failure to node %s", other.id)
                self._handle_failure_locked(other, "cascade")

    def _check_timeouts(self):
        now = time.time()
        with self._lock:
            for node in self.dag.nodes.values():
                if node.status != "running":
                    continue
                if node.start_time and (now - node.start_time) > node.timeout_seconds:
                    logger.warning("Node %s timed out", node.id)
                    self._handle_failure_locked(node, "timeout")

    # ------------------------------------------------------------------
    # Main scheduler loop (spec §2)
    # ------------------------------------------------------------------

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
                self._event.wait(timeout=0.5)
                self._event.clear()

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
            parts.append(f"[{nid}] {info['status']}: {info.get('summary', '')[:200]}")
        return "\n".join(parts)
