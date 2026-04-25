"""Test script for AgentRuntime.

Tests fork-based parallel agent execution:
1. Spawn root agent
2. Root forks 2 children with different setups
3. Simulate children completing tasks
4. Children send results to parent
5. Parent receives and aggregates results

Usage:
    python test_agent_runtime.py --provider-name aws --region us-east-1
"""

import argparse
import logging
import os
import time
from typing import Optional

import requests

from agent_runtime import AgentRuntime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def simulate_child_work(runtime: AgentRuntime, child_id: str, work_duration: float, result: dict):
    """Simulate a child agent doing work and reporting back.

    Args:
        runtime: AgentRuntime instance
        child_id: Child agent ID
        work_duration: How long to "work" (seconds)
        result: Result to send back to parent
    """
    logger.info(f"[{child_id}] Starting work (will take {work_duration}s)...")
    time.sleep(work_duration)

    # Send result to parent
    agent_status = runtime.get_agent_status(child_id)
    parent_id = agent_status["parent_id"]

    logger.info(f"[{child_id}] Work complete, sending result to {parent_id}")
    runtime.send_message(
        from_agent=child_id,
        to_agent=parent_id,
        content=result,
    )

    # Mark self as completed
    runtime.complete_agent(child_id, result=result)


def main():
    parser = argparse.ArgumentParser(description="Test agent runtime")
    parser.add_argument("--provider-name", default="aws", help="Provider (aws/docker)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--output-dir", default="test_runtime_output", help="Output directory")
    args = parser.parse_args()

    password = "osworld-public-evaluation"
    os.makedirs(args.output_dir, exist_ok=True)

    # Boot VM
    logger.info("Booting VM...")
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.providers.aws.manager import IMAGE_ID_MAP

    screen_size = (1920, 1080)
    region_map = IMAGE_ID_MAP[args.region]
    ami_id = region_map.get(screen_size, region_map.get((1920, 1080)))

    env = DesktopEnv(
        provider_name=args.provider_name,
        action_space="pyautogui",
        screen_size=screen_size,
        headless=args.headless,
        os_type="Ubuntu",
        client_password=password,
        region=args.region,
        snapshot_name=ami_id,
    )
    env.reset()

    vm_ip = env.vm_ip
    port = env.server_port
    logger.info(f"VM ready at {vm_ip}:{port}")

    exec_url = f"http://{vm_ip}:{port}/setup/execute"

    def vm_exec(cmd: str, timeout: int = 120) -> Optional[dict]:
        try:
            r = requests.post(
                exec_url,
                json={"command": cmd, "shell": True},
                timeout=timeout,
            )
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            logger.error(f"vm_exec failed: {e}")
        return None

    # Wait for VM to be healthy
    logger.info("Waiting for VM server...")
    for attempt in range(30):
        try:
            r = requests.post(
                exec_url,
                json={"command": "echo ready", "shell": True},
                timeout=10,
            )
            if r.status_code == 200 and r.json().get("returncode") == 0:
                logger.info(f"VM server healthy (waited {attempt * 2}s)")
                break
        except Exception:
            pass
        time.sleep(2)

    # Initialize runtime
    logger.info("\n" + "=" * 60)
    logger.info("Initializing Agent Runtime")
    logger.info("=" * 60)

    runtime = AgentRuntime(vm_exec=vm_exec, num_displays=8, password=password)
    success = runtime.initialize()

    if not success:
        logger.error("Runtime initialization failed!")
        return

    logger.info("✓ Runtime initialized")

    # Test scenario: Parent forks 2 children to search for professor emails
    logger.info("\n" + "=" * 60)
    logger.info("Test Scenario: Parent Forks 2 Children")
    logger.info("=" * 60)

    # Spawn root agent
    root_task = "Find emails for Prof. Alice Smith and Prof. Bob Johnson"
    root_id = runtime.spawn_root_agent(task=root_task, display_num=0)

    logger.info(f"\n[{root_id}] Task: {root_task}")
    logger.info(f"[{root_id}] Planning: I'll fork 2 children to search in parallel")

    # Fork child 1: Find Alice's email
    logger.info(f"\n[{root_id}] Forking child 1 to find Alice's email...")

    child1_config = [
        {
            "type": "chrome_open_tabs",
            "parameters": {
                "urls_to_open": ["https://www.google.com/search?q=Alice+Smith+university+email"]
            }
        },
        {
            "type": "sleep",
            "parameters": {"seconds": 2}
        }
    ]

    child1_id = runtime.fork_agent(
        parent_id=root_id,
        subtask="Find Alice Smith's email address",
        config=child1_config,
        context_summary="Looking for professor emails. Alice Smith is first."
    )

    if not child1_id:
        logger.error("Failed to fork child 1!")
        return

    # Fork child 2: Find Bob's email
    logger.info(f"\n[{root_id}] Forking child 2 to find Bob's email...")

    child2_config = [
        {
            "type": "chrome_open_tabs",
            "parameters": {
                "urls_to_open": ["https://www.google.com/search?q=Bob+Johnson+university+email"]
            }
        },
        {
            "type": "sleep",
            "parameters": {"seconds": 2}
        }
    ]

    child2_id = runtime.fork_agent(
        parent_id=root_id,
        subtask="Find Bob Johnson's email address",
        config=child2_config,
        context_summary="Looking for professor emails. Bob Johnson is second."
    )

    if not child2_id:
        logger.error("Failed to fork child 2!")
        return

    # Show runtime state
    logger.info("\n" + "-" * 60)
    logger.info("Runtime State After Forking")
    logger.info("-" * 60)
    all_agents = runtime.get_all_agents()
    for agent_id, status in all_agents.items():
        logger.info(
            f"  {agent_id}: {status['status']} on display :{status['display_num']} "
            f"({len(status['children'])} children)"
        )

    # Simulate children working in parallel
    logger.info("\n" + "-" * 60)
    logger.info("Children Working in Parallel")
    logger.info("-" * 60)

    import threading

    # Child 1 takes 3 seconds
    thread1 = threading.Thread(
        target=simulate_child_work,
        args=(
            runtime,
            child1_id,
            3.0,
            {
                "status": "success",
                "email": "alice.smith@university.edu",
                "found_on": "faculty directory page"
            }
        )
    )

    # Child 2 takes 4 seconds
    thread2 = threading.Thread(
        target=simulate_child_work,
        args=(
            runtime,
            child2_id,
            4.0,
            {
                "status": "success",
                "email": "bob.johnson@university.edu",
                "found_on": "personal website"
            }
        )
    )

    start_time = time.time()
    thread1.start()
    thread2.start()

    # Parent waits for results
    logger.info(f"\n[{root_id}] Waiting for children to report back...")

    results = []
    for i in range(2):
        msg = runtime.receive_message(root_id, timeout=10)
        if msg:
            logger.info(f"[{root_id}] Received result from {msg.from_agent}: {msg.content}")
            results.append(msg.content)

    thread1.join()
    thread2.join()

    wall_clock_time = time.time() - start_time

    # Parent aggregates results
    logger.info("\n" + "-" * 60)
    logger.info("Parent Aggregating Results")
    logger.info("-" * 60)

    logger.info(f"[{root_id}] Both children completed!")
    logger.info(f"[{root_id}] Wall-clock time: {wall_clock_time:.1f}s (parallel execution)")
    logger.info(f"[{root_id}] Sequential would have taken: 7s (3s + 4s)")
    logger.info(f"[{root_id}] Speedup: {7.0 / wall_clock_time:.2f}x")

    logger.info(f"\n[{root_id}] Final results:")
    for result in results:
        logger.info(f"  - {result['email']} (found on {result['found_on']})")

    # Mark root as completed
    runtime.complete_agent(root_id, result={"emails": [r["email"] for r in results]})

    # Final runtime state
    logger.info("\n" + "-" * 60)
    logger.info("Final Runtime State")
    logger.info("-" * 60)
    all_agents = runtime.get_all_agents()
    for agent_id, status in all_agents.items():
        duration = status["duration"]
        logger.info(
            f"  {agent_id}: {status['status']} ({duration:.1f}s) "
            f"- result: {status['result']}"
        )

    # Shutdown
    logger.info("\n" + "-" * 60)
    logger.info("Shutting Down")
    logger.info("-" * 60)
    runtime.shutdown()

    logger.info("\n" + "=" * 60)
    logger.info("✓ Test Complete!")
    logger.info("=" * 60)
    logger.info(f"Fork-based parallel execution achieved {7.0 / wall_clock_time:.2f}x speedup")
    logger.info(f"Displays used: 3 (1 parent + 2 children)")
    logger.info(f"Total agents: 3 (1 root + 2 children)")

    # Keep VM alive for inspection
    logger.info(f"\nVM still running. VNC: http://{vm_ip}:5910/vnc.html")
    logger.info("Press Ctrl+C to terminate VM and exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down VM...")


if __name__ == "__main__":
    main()
