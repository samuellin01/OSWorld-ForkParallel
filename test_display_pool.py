"""Test script for DisplayPool.

Tests:
1. Initialize 8 displays
2. Allocate 5 displays
3. Release 3 displays
4. Verify pool state
5. Cleanup

Usage:
    python test_display_pool.py --provider-name aws --region us-east-1
"""

import argparse
import logging
import time
from typing import Optional

import requests

from display_pool import DisplayPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test display pool")
    parser.add_argument("--provider-name", default="aws", help="Provider (aws/docker)")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    args = parser.parse_args()

    password = "osworld-public-evaluation"

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

    # Test display pool
    logger.info("\n" + "=" * 60)
    logger.info("Testing Display Pool")
    logger.info("=" * 60)

    # Initialize pool
    pool = DisplayPool(vm_exec=vm_exec, num_displays=8, password=password)
    success = pool.initialize()

    if not success:
        logger.error("Display pool initialization failed!")
        return

    logger.info(f"\n✓ Pool initialized with {pool.get_idle_count()} idle displays")

    # Show initial status
    logger.info("\nInitial display status:")
    for display_num, status in pool.get_status().items():
        logger.info(f"  Display :{display_num} - {status}")

    # Test allocation
    logger.info("\n" + "-" * 60)
    logger.info("Test 1: Allocate 5 displays")
    logger.info("-" * 60)

    allocated = []
    for i in range(5):
        display_num = pool.allocate(agent_id=f"test_agent_{i}")
        if display_num:
            allocated.append(display_num)
            logger.info(f"  Allocated display :{display_num} to test_agent_{i}")
        else:
            logger.error(f"  Failed to allocate display for test_agent_{i}")

    logger.info(f"\nAllocated {len(allocated)} displays")
    logger.info(f"Idle displays remaining: {pool.get_idle_count()}")

    # Test release
    logger.info("\n" + "-" * 60)
    logger.info("Test 2: Release 3 displays")
    logger.info("-" * 60)

    for display_num in allocated[:3]:
        pool.release(display_num)
        logger.info(f"  Released display :{display_num}")

    logger.info(f"\nIdle displays after release: {pool.get_idle_count()}")

    # Show final status
    logger.info("\n" + "-" * 60)
    logger.info("Final display status:")
    logger.info("-" * 60)
    for display_num, status in pool.get_status().items():
        logger.info(f"  Display :{display_num} - {status}")

    # Test exhaustion
    logger.info("\n" + "-" * 60)
    logger.info("Test 3: Exhaust pool (allocate remaining displays)")
    logger.info("-" * 60)

    count = 0
    while True:
        display_num = pool.allocate(agent_id=f"exhaust_agent_{count}")
        if display_num:
            count += 1
            logger.info(f"  Allocated display :{display_num}")
        else:
            logger.info(f"  Pool exhausted after {count} allocations")
            break

    logger.info(f"\nIdle displays: {pool.get_idle_count()} (should be 0)")

    # Cleanup
    logger.info("\n" + "-" * 60)
    logger.info("Cleaning up...")
    logger.info("-" * 60)
    pool.cleanup()

    logger.info("\n" + "=" * 60)
    logger.info("✓ Display pool test complete!")
    logger.info("=" * 60)

    # Keep VM alive for manual inspection
    logger.info(f"\nVM still running. VNC: http://{vm_ip}:5910/vnc.html")
    logger.info("Press Ctrl+C to terminate VM and exit")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\nShutting down VM...")


if __name__ == "__main__":
    main()
