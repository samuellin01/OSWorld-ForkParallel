"""Parallel CUA prototype — two agents, two displays, one EC2 instance.

Boots a single DesktopEnv VM, creates a second Xvfb virtual display,
then runs two independent CUA agent loops concurrently via threads.

Agent A uses the native GNOME display (via the server's /screenshot and
/execute endpoints). Agent B uses an Xvfb display with openbox (via scrot
and DISPLAY=:2 python3).

Usage::

    python run_parallel_demo.py \
        --task-a "Open the terminal and run 'echo hello from agent A'" \
        --task-b "Run the command 'echo hello from agent B' in the terminal" \
        --provider-name aws --region us-east-1 --headless
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests

from bedrock_client import BedrockClient
from agent_utils import (
    COMPUTER_USE_TOOL,
    _resize_screenshot,
    filter_to_n_most_recent_images,
    parse_computer_use_actions,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_NATIVE = (
    "You are a computer-use agent on Ubuntu 22.04 with GNOME. "
    "Complete the task by interacting with the desktop. "
    "Password for sudo: '{password}'. Home directory: /home/user. "
    "If you launch Google Chrome, add --remote-debugging-port=1337. "
    "Output DONE when finished or FAIL if impossible."
)

_SYSTEM_PROMPT_XVFB = (
    "You are a computer-use agent on Ubuntu 22.04 with a minimal openbox "
    "desktop. An xterm terminal window is already open and ready for use. "
    "Right-click the desktop background for the openbox application menu. "
    "Password for sudo: '{password}'. Home directory: /home/user. "
    "Output DONE when finished or FAIL if impossible."
)


# ---------------------------------------------------------------------------
# Display wrappers
# ---------------------------------------------------------------------------

class NativeDisplay:
    """Uses the VM server's built-in /screenshot and /execute endpoints."""

    def __init__(self, vm_ip: str, server_port: int):
        self.vm_ip = vm_ip
        self.port = server_port
        self.display = ":native"
        self.base_url = f"http://{vm_ip}:{server_port}"

    def screenshot(self) -> Optional[bytes]:
        try:
            r = requests.get(f"{self.base_url}/screenshot", timeout=15)
            if r.status_code == 200 and len(r.content) > 100:
                return r.content
        except Exception as e:
            logger.warning("[native] screenshot failed: %s", e)
        return None

    def run_action(self, pyautogui_code: str) -> Optional[dict]:
        prefix = "import pyautogui; import time; pyautogui.FAILSAFE = False\n"
        full_code = prefix + pyautogui_code
        try:
            r = requests.post(
                f"{self.base_url}/execute",
                json={"command": ["python3", "-c", full_code], "shell": False},
                timeout=30,
            )
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.warning("[native] action failed: %s", e)
            return None


class XvfbDisplay:
    """Targets an Xvfb display via scrot + DISPLAY=:N python3."""

    def __init__(self, vm_ip: str, server_port: int, display_num: int):
        self.vm_ip = vm_ip
        self.port = server_port
        self.display = f":{display_num}"
        self.exec_url = f"http://{vm_ip}:{server_port}/setup/execute"

    def _shell(self, cmd: str, timeout: int = 60) -> Optional[dict]:
        try:
            r = requests.post(
                self.exec_url,
                json={"command": cmd, "shell": True},
                timeout=timeout,
            )
            return r.json() if r.status_code == 200 else None
        except Exception as e:
            logger.warning("[%s] shell failed: %s", self.display, e)
            return None

    def screenshot(self) -> Optional[bytes]:
        tmp = f"/tmp/par_shot_{self.display.replace(':', '')}.png"
        self._shell(f"DISPLAY={self.display} scrot -o {tmp}")
        result = self._shell(f"base64 -w0 {tmp}")
        if result and result.get("output"):
            try:
                return base64.b64decode(result["output"].strip())
            except Exception:
                pass
        return None

    def run_action(self, pyautogui_code: str) -> Optional[dict]:
        tmp = f"/tmp/par_action_{self.display.replace(':', '')}.py"
        full_code = (
            "import pyautogui; import time; pyautogui.FAILSAFE = False\n"
            + pyautogui_code
        )
        write_cmd = f"cat > {tmp} << 'ACTIONEOF'\n{full_code}\nACTIONEOF"
        self._shell(write_cmd)
        return self._shell(f"DISPLAY={self.display} python3 {tmp}", timeout=30)


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def agent_loop(
    display: NativeDisplay | XvfbDisplay,
    task: str,
    system_prompt: str,
    bedrock: BedrockClient,
    model: str,
    max_steps: int,
    output_dir: str,
) -> bool:
    """Run a CUA agent loop on one display. Returns True on DONE."""
    os.makedirs(output_dir, exist_ok=True)
    tag = display.display

    with open(os.path.join(output_dir, "task.txt"), "w") as f:
        f.write(task)

    tools = [COMPUTER_USE_TOOL]
    resize_factor = (1920.0 / 1280.0, 1080.0 / 720.0)

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": f"Task: {task}"}]},
    ]
    last_tool_use_id: Optional[str] = None

    for step in range(1, max_steps + 1):
        logger.info("[%s] Step %d/%d", tag, step, max_steps)

        shot = display.screenshot()
        if shot:
            shot = _resize_screenshot(shot)
            with open(os.path.join(output_dir, f"step_{step:03d}.png"), "wb") as f:
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
            logger.warning("[%s] Screenshot returned None", tag)
            obs_content = [
                {"type": "text", "text": f"Step {step}: screenshot unavailable."},
            ]

        if last_tool_use_id:
            obs_content.insert(0, {
                "type": "tool_result",
                "tool_use_id": last_tool_use_id,
                "content": "Action executed.",
            })
            last_tool_use_id = None

        messages.append({"role": "user", "content": obs_content})
        filter_to_n_most_recent_images(messages, images_to_keep=5, min_removal_threshold=5)

        content_blocks, _ = bedrock.chat(
            messages=messages, system=system_prompt, model=model, tools=tools,
        )
        messages.append({"role": "assistant", "content": content_blocks})

        response_text = "".join(
            b.get("text", "") for b in content_blocks
            if isinstance(b, dict) and b.get("type") == "text"
        )
        logger.info("[%s] Response: %s", tag, response_text[:200])

        with open(os.path.join(output_dir, f"step_{step:03d}_response.txt"), "w") as f:
            f.write(response_text)

        actions = parse_computer_use_actions(content_blocks, resize_factor)

        for block in content_blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                last_tool_use_id = block.get("id")
                break

        if "DONE" in actions:
            logger.info("[%s] DONE at step %d", tag, step)
            return True
        if "FAIL" in actions:
            logger.info("[%s] FAIL at step %d", tag, step)
            return False

        action_code = next(
            (a for a in actions if a not in ("DONE", "FAIL", "WAIT", "CALL_USER")),
            None,
        )
        if action_code:
            logger.info("[%s] Action: %s", tag, action_code[:120])
            display.run_action(action_code)
            time.sleep(1)
        else:
            time.sleep(2)

    logger.warning("[%s] Max steps reached", tag)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stdout,
    )

    parser = argparse.ArgumentParser(
        description="Parallel CUA demo: two agents, two displays, one VM.",
    )
    parser.add_argument("--task-a", required=True, help="Task for agent A (native GNOME)")
    parser.add_argument("--task-b", required=True, help="Task for agent B (Xvfb + openbox)")
    parser.add_argument("--model", default="claude-opus-4-6")
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--provider-name", default="aws")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", default="parallel_demo_results")
    parser.add_argument(
        "--keep-alive", action="store_true",
        help="Keep the VM running after agents finish so you can VNC in.",
    )
    args = parser.parse_args(argv)

    password = "osworld-public-evaluation"

    # --- 1. Boot one EC2 instance via DesktopEnv ---
    from desktop_env.desktop_env import DesktopEnv
    from desktop_env.providers.aws.manager import IMAGE_ID_MAP

    screen_size = (1920, 1080)
    region_map = IMAGE_ID_MAP[args.region]
    ami_id = region_map.get(screen_size, region_map.get((1920, 1080)))

    logger.info("Booting DesktopEnv (provider=%s) …", args.provider_name)
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
    logger.info("VM ready at %s:%s", vm_ip, port)

    exec_url = f"http://{vm_ip}:{port}/setup/execute"

    def vm_exec(cmd: str, timeout: int = 120) -> Optional[dict]:
        try:
            r = requests.post(exec_url, json={"command": cmd, "shell": True}, timeout=timeout)
            if r.status_code == 200:
                d = r.json()
                logger.info("  [setup] rc=%s out=%s", d.get("returncode"), (d.get("output") or "")[:200])
                return d
        except Exception as e:
            logger.warning("  [setup] %s: %s", cmd[:60], e)
        return None

    # --- 2. Wait for VM server to be healthy ---
    for attempt in range(30):
        try:
            r = requests.post(
                exec_url, json={"command": "echo ready", "shell": True}, timeout=10,
            )
            if r.status_code == 200 and r.json().get("returncode") == 0:
                logger.info("VM server healthy (waited %ds)", attempt * 2)
                break
        except Exception:
            pass
        time.sleep(2)

    # --- 3. Set up display :2 with openbox + xterm ---
    logger.info("Setting up second display …")
    # Only install if missing (saves ~2 min on repeated runs)
    vm_exec(
        "which Xvfb scrot openbox xterm xdotool tint2 x11vnc > /dev/null 2>&1 || "
        f"(echo '{password}' | sudo -S apt-get update -qq && "
        f"echo '{password}' | sudo -S apt-get install -y xvfb scrot openbox xterm xdotool tint2 x11vnc)"
    )

    # Start Xvfb, openbox, xterm in one shot
    vm_exec(
        f"export DISPLAY=:2; "
        f"nohup Xvfb :2 -screen 0 1920x1080x24 -ac >/dev/null 2>&1 & sleep 2; "
        f"nohup openbox >/dev/null 2>&1 & sleep 1; "
        f"xsetroot -solid '#2C3E50'; "
        f"nohup tint2 >/dev/null 2>&1 & sleep 1; "
        f"nohup xterm -geometry 120x40+50+50 -fa 'Monospace' -fs 12 >/dev/null 2>&1 &",
        timeout=30,
    )
    time.sleep(3)

    # Verify
    vm_exec("DISPLAY=:2 xdpyinfo | head -3")

    # --- 4. Create display wrappers and Bedrock clients ---
    display_a = NativeDisplay(vm_ip, port)
    display_b = XvfbDisplay(vm_ip, port, 2)

    out_a = os.path.join(args.output_dir, "agent_a")
    out_b = os.path.join(args.output_dir, "agent_b")
    bedrock_a = BedrockClient(region=args.region, log_dir=out_a)
    bedrock_b = BedrockClient(region=args.region, log_dir=out_b)

    prompt_a = _SYSTEM_PROMPT_NATIVE.format(password=password)
    prompt_b = _SYSTEM_PROMPT_XVFB.format(password=password)

    # --- 5. Wait for presenter to start ---
    print("\n" + "=" * 60)
    print(f"VM ready. Open VNC to watch: http://{vm_ip}:5910/vnc.html")
    print(f"Agent A [native]:    {args.task_a}")
    print(f"Agent B [display :2]: {args.task_b}")
    print("=" * 60)
    input("\nPress Enter to start agents …\n")

    # --- 6. Start recording both displays ---
    logger.info("Starting screen recordings …")
    try:
        requests.post(f"http://{vm_ip}:{port}/start_recording", timeout=10)
        logger.info("Recording display :0")
    except Exception as e:
        logger.warning("Could not start recording for display :0: %s", e)

    # Record display :2 via ffmpeg directly
    vm_exec(
        "nohup ffmpeg -y -f x11grab -draw_mouse 1 -s 1920x1080 "
        "-i :2.0 -c:v libx264 -r 30 /tmp/recording_display2.mp4 "
        ">/dev/null 2>&1 &",
        timeout=10,
    )

    # --- 6. Run both agents in parallel ---
    logger.info("Launching agents in parallel …")
    logger.info("  Agent A [native]: %s", args.task_a)
    logger.info("  Agent B [display :2]: %s", args.task_b)

    t_start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(
            agent_loop, display_a, args.task_a, prompt_a, bedrock_a,
            args.model, args.max_steps, out_a,
        )
        fut_b = pool.submit(
            agent_loop, display_b, args.task_b, prompt_b, bedrock_b,
            args.model, args.max_steps, out_b,
        )
        result_a = fut_a.result()
        result_b = fut_b.result()
    elapsed = time.monotonic() - t_start

    # --- 7. Stop recordings and download ---
    logger.info("Stopping screen recordings …")
    try:
        requests.post(f"http://{vm_ip}:{port}/end_recording", timeout=20)
    except Exception as e:
        logger.warning("Could not stop recording for display :0: %s", e)

    # Stop display :2 recording
    vm_exec("pkill -SIGINT -f 'ffmpeg.*recording_display2' || true", timeout=10)
    time.sleep(3)  # let ffmpeg finalize

    # Download recordings
    os.makedirs(args.output_dir, exist_ok=True)
    for remote_path, local_name in [
        ("/tmp/recording.mp4", "recording_display0.mp4"),
        ("/tmp/recording_display2.mp4", "recording_display2.mp4"),
    ]:
        result = vm_exec(f"base64 -w0 {remote_path}", timeout=30)
        if result and result.get("output"):
            local_path = os.path.join(args.output_dir, local_name)
            try:
                import base64 as b64mod
                with open(local_path, "wb") as f:
                    f.write(b64mod.b64decode(result["output"].strip()))
                logger.info("Saved %s", local_path)
            except Exception as e:
                logger.warning("Failed to save %s: %s", local_name, e)

    # --- 8. Report ---
    print("\n" + "=" * 60)
    print(f"Agent A (native):     {'DONE' if result_a else 'FAIL'}")
    print(f"Agent B (display :2): {'DONE' if result_b else 'FAIL'}")
    print(f"Wall-clock time: {elapsed:.1f}s")
    for name, bc in [("A", bedrock_a), ("B", bedrock_b)]:
        tu = bc.get_token_usage()
        print(
            f"Agent {name}: steps={tu['step_count']} "
            f"cost=${tu['total_cost_usd']:.4f} "
            f"latency={tu['total_latency_seconds']:.1f}s"
        )
    print(f"Screenshots & recordings saved to: {args.output_dir}/")
    print("=" * 60)

    if args.keep_alive:
        print(f"\n--keep-alive: VM is still running.")
        print(f"VNC: http://{vm_ip}:5910/vnc.html")
        input("Press Enter to terminate the instance …")

    env.close()
    sys.exit(0 if (result_a and result_b) else 1)


if __name__ == "__main__":
    main()
