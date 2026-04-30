"""Setup executor for running OSWorld config steps on virtual displays.

Executes setup configuration steps (chrome_open_tabs, launch, open, etc.)
on a specific display to prepare the environment for agent execution.
"""

import logging
import time
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class SetupExecutor:
    """Executes OSWorld setup config steps on a virtual display.

    Supports the standard OSWorld config types:
    - chrome_open_tabs: Open URLs in Chrome
    - launch: Start an application
    - open: Open a file with default app
    - command: Run shell command
    - execute: Run shell command (alias for command)
    - download: Download files from URLs
    - sleep: Wait for N seconds
    - activate_window: Focus a window by name

    Usage:
        executor = SetupExecutor(display_num=2, vm_exec=vm_exec_func)
        success = executor.execute_config([
            {"type": "chrome_open_tabs", "parameters": {"urls_to_open": ["https://google.com"]}},
            {"type": "sleep", "parameters": {"seconds": 2}}
        ])
    """

    def __init__(self, display_num: int, vm_exec: Callable[[str], Optional[dict]]):
        """Initialize setup executor.

        Args:
            display_num: Display number (e.g., 2 for :2)
            vm_exec: Function to execute commands on VM (cmd: str) -> dict
        """
        self.display_num = display_num
        self.vm_exec = vm_exec
        self.display_env = f"DISPLAY=:{display_num}"

    def execute_config(self, config_steps: List[Dict[str, Any]]) -> bool:
        """Execute a list of config steps sequentially.

        Args:
            config_steps: List of config step dicts with "type" and "parameters"

        Returns:
            bool: True if all steps succeeded, False otherwise
        """
        logger.info(f"Executing {len(config_steps)} setup steps on display :{self.display_num}")

        for i, step in enumerate(config_steps):
            step_type = step.get("type")
            parameters = step.get("parameters", {})

            logger.info(f"  Step {i+1}/{len(config_steps)}: {step_type}")

            success = self._execute_step(step_type, parameters)
            if not success:
                logger.error(f"Setup step {i+1} ({step_type}) failed")
                return False

        logger.info(f"✓ All {len(config_steps)} setup steps completed")
        return True

    def _execute_step(self, step_type: str, parameters: Dict[str, Any]) -> bool:
        """Execute a single setup step.

        Args:
            step_type: Type of step (chrome_open_tabs, launch, etc.)
            parameters: Parameters for the step

        Returns:
            bool: True if step succeeded
        """
        handlers = {
            "chrome_open_tabs": self._chrome_open_tabs,
            "launch": self._launch,
            "open": self._open,
            "command": self._command,
            "execute": self._command,  # execute is alias for command
            "download": self._download,
            "sleep": self._sleep,
            "activate_window": self._activate_window,
        }

        handler = handlers.get(step_type)
        if not handler:
            logger.warning(f"Unknown setup step type: {step_type}, skipping")
            return True  # Don't fail on unknown types

        try:
            return handler(parameters)
        except Exception as e:
            logger.error(f"Exception in {step_type}: {e}")
            return False

    def _chrome_open_tabs(self, params: Dict[str, Any]) -> bool:
        """Open URLs in Chrome on this display.

        Args:
            params: {"urls_to_open": ["url1", "url2", ...]}
        """
        urls = params.get("urls_to_open", [])
        if not urls:
            logger.warning("chrome_open_tabs: no URLs provided")
            return True

        # Check if Chrome is already running on this display
        check_cmd = f"{self.display_env} wmctrl -l | grep -i chrome"
        check_result = self.vm_exec(check_cmd)

        chrome_running = (
            check_result
            and check_result.get("returncode") == 0
            and check_result.get("output", "").strip()
        )

        if chrome_running:
            # Chrome already running, open URLs in new tabs
            logger.info(f"Chrome already running, opening {len(urls)} URL(s) in new tabs")
            for url in urls:
                cmd = f"{self.display_env} google-chrome --new-tab '{url}' >/dev/null 2>&1 &"
                self.vm_exec(cmd)
                time.sleep(0.5)
        else:
            # Launch Chrome with URLs
            # Use display-specific debugging port and user data dir to avoid conflicts
            debug_port = 1337 + self.display_num
            user_data_dir = f"/tmp/chrome_display_{self.display_num}"
            logger.info(f"Launching Chrome with {len(urls)} URL(s) on debug port {debug_port}")
            url_args = " ".join([f"'{url}'" for url in urls])
            cmd = (
                f"{self.display_env} nohup google-chrome "
                f"--remote-debugging-port={debug_port} "
                f"--user-data-dir={user_data_dir} "
                f"--no-first-run "
                f"--no-default-browser-check "
                f"--disable-default-apps "
                f"--start-maximized "
                f"{url_args} "
                f">/dev/null 2>&1 &"
            )
            result = self.vm_exec(cmd)
            if not result:
                return False
            time.sleep(2)  # Give Chrome time to start

        return True

    def _launch(self, params: Dict[str, Any]) -> bool:
        """Launch an application on this display.

        Args:
            params: {"command": ["app", "arg1", "arg2", ...]}
        """
        command = params.get("command", [])
        if not command:
            logger.warning("launch: no command provided")
            return True

        if isinstance(command, list):
            command = " ".join(command)

        cmd = f"{self.display_env} nohup {command} >/dev/null 2>&1 &"
        result = self.vm_exec(cmd)
        time.sleep(1.5)  # Give app time to start

        return result is not None

    def _open(self, params: Dict[str, Any]) -> bool:
        """Open a file with default application.

        Args:
            params: {"path": "/path/to/file"}
        """
        path = params.get("path")
        if not path:
            logger.warning("open: no path provided")
            return True

        cmd = f"{self.display_env} xdg-open '{path}' >/dev/null 2>&1 &"
        result = self.vm_exec(cmd)
        time.sleep(1.5)  # Give app time to open

        return result is not None

    def _command(self, params: Dict[str, Any]) -> bool:
        """Run a shell command.

        Args:
            params: {"command": ["cmd", "arg1", ...]} or {"command": "cmd arg1 ..."}
        """
        command = params.get("command", [])
        if not command:
            logger.warning("command: no command provided")
            return True

        if isinstance(command, list):
            command = " ".join(command)

        result = self.vm_exec(command)
        return result is not None and result.get("returncode") == 0

    def _download(self, params: Dict[str, Any]) -> bool:
        """Download files from URLs.

        Args:
            params: {"files": [{"path": "/dest/path", "url": "https://..."}, ...]}
        """
        files = params.get("files", [])
        if not files:
            logger.warning("download: no files provided")
            return True

        for file_info in files:
            path = file_info.get("path")
            url = file_info.get("url")

            if not path or not url:
                logger.warning(f"download: missing path or url in {file_info}")
                continue

            # Create parent directory if needed
            parent_dir = "/".join(path.rsplit("/", 1)[:-1])
            if parent_dir:
                self.vm_exec(f"mkdir -p '{parent_dir}'")

            # Download file
            cmd = f"wget -q -O '{path}' '{url}'"
            result = self.vm_exec(cmd)

            if not result or result.get("returncode") != 0:
                logger.error(f"Failed to download {url} to {path}")
                return False

            logger.info(f"Downloaded {url} to {path}")

        return True

    def _sleep(self, params: Dict[str, Any]) -> bool:
        """Sleep for N seconds.

        Args:
            params: {"seconds": N}
        """
        seconds = params.get("seconds", 0)
        if seconds > 0:
            logger.info(f"Sleeping {seconds}s")
            time.sleep(seconds)
        return True

    def _activate_window(self, params: Dict[str, Any]) -> bool:
        """Activate (focus) a window by name.

        Args:
            params: {"window_name": "Window Title", "strict": bool, "required": bool}
        """
        window_name = params.get("window_name")
        if not window_name:
            logger.warning("activate_window: no window_name provided")
            return True

        strict = params.get("strict", False)
        required = params.get("required", False)  # If False, don't fail config on error
        match_flag = "-e" if strict else "-i"  # exact match or case-insensitive

        cmd = f"{self.display_env} wmctrl {match_flag} -a '{window_name}'"
        result = self.vm_exec(cmd)

        # wmctrl returns 0 if window found and activated
        success = result is not None and result.get("returncode") == 0
        if not success:
            logger.warning(f"Could not activate window '{window_name}'")
            # If not required, treat as success (best effort)
            return success if required else True

        return success

    def take_screenshot(self) -> Optional[bytes]:
        """Take a screenshot of this display.

        Returns:
            bytes: PNG screenshot data, or None on failure
        """
        import base64

        tmp_path = f"/tmp/setup_screenshot_{self.display_num}.png"

        # Take screenshot with scrot
        cmd = f"{self.display_env} scrot -o '{tmp_path}'"
        result = self.vm_exec(cmd)

        if not result or result.get("returncode") != 0:
            logger.error(f"Failed to take screenshot on display :{self.display_num}")
            return None

        # Read screenshot as base64
        read_cmd = f"base64 -w0 '{tmp_path}'"
        result = self.vm_exec(read_cmd)

        if result and result.get("output"):
            try:
                screenshot_bytes = base64.b64decode(result["output"].strip())
                logger.info(f"Screenshot captured ({len(screenshot_bytes)} bytes)")
                return screenshot_bytes
            except Exception as e:
                logger.error(f"Failed to decode screenshot: {e}")

        return None
