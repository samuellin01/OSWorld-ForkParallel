"""Display pool for managing virtual displays in fork-based parallel execution.

Provides 8 virtual displays (2-9) for agent forking. Display :0 is reserved
for the primary GNOME desktop. Display :1 is skipped by Xvfb convention.
"""

import logging
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class DisplayStatus(Enum):
    """Status of a virtual display."""
    IDLE = "idle"
    ALLOCATED = "allocated"
    INITIALIZING = "initializing"
    ERROR = "error"


@dataclass
class Display:
    """Represents a single virtual display."""
    display_num: int
    status: DisplayStatus = DisplayStatus.INITIALIZING
    last_used_by: Optional[str] = None  # agent_id that last used this display
    error_msg: Optional[str] = None


class DisplayPool:
    """Thread-safe pool of virtual displays for parallel agent execution.

    Manages 8 Xvfb displays (numbers 2-9). Display :0 is the primary desktop.
    Display :1 is skipped by Xvfb convention.

    Usage:
        pool = DisplayPool(vm_exec_func, num_displays=8)
        pool.initialize()

        display_num = pool.allocate(agent_id="child_1")
        # ... use display ...
        pool.release(display_num)

        pool.cleanup()
    """

    def __init__(self, vm_exec, num_displays: int = 8, password: str = "password"):
        """Initialize display pool.

        Args:
            vm_exec: Function to execute commands on VM (cmd: str) -> dict
            num_displays: Number of displays to create (default 8)
            password: Sudo password for VM
        """
        self.vm_exec = vm_exec
        self.password = password
        self.num_displays = num_displays

        # Display numbers: 2, 3, 4, ..., num_displays+1
        # (Skip 0=primary, 1=convention)
        self.display_nums = list(range(2, num_displays + 2))

        # Track display state
        self.displays: Dict[int, Display] = {
            n: Display(display_num=n, status=DisplayStatus.INITIALIZING)
            for n in self.display_nums
        }

        # Thread safety
        self._lock = threading.Lock()

        # Track which displays are idle (available)
        self.idle_displays: Set[int] = set()

    def initialize(self):
        """Start all Xvfb displays and window managers.

        Returns:
            bool: True if all displays initialized successfully
        """
        logger.info(f"Initializing {self.num_displays} virtual displays...")

        # Install prerequisites (if not already present)
        logger.info("Ensuring display prerequisites installed...")
        result = self.vm_exec(
            "which Xvfb scrot openbox xterm xdotool > /dev/null 2>&1 || "
            f"(echo '{self.password}' | sudo -S apt-get update -qq && "
            f"echo '{self.password}' | sudo -S apt-get install -y xvfb scrot openbox xterm xdotool)"
        )
        if not result or result.get("returncode") != 0:
            logger.error("Failed to install display prerequisites")
            return False

        # Configure openbox keyboard shortcut for terminal recovery
        logger.info("Configuring openbox keyboard shortcuts...")
        openbox_config = """<?xml version="1.0" encoding="UTF-8"?>
<openbox_config xmlns="http://openbox.org/3.4/rc">
  <keyboard>
    <!-- Launch terminal with Ctrl+Alt+T -->
    <keybind key="C-A-t">
      <action name="Execute">
        <command>xterm -fa 'Monospace' -fs 14 -geometry 200x50 -xrm 'XTerm*selectToClipboard: true'</command>
      </action>
    </keybind>
  </keyboard>
</openbox_config>
"""
        config_cmd = (
            "mkdir -p ~/.config/openbox && "
            f"cat > ~/.config/openbox/rc.xml << 'OBEOF'\n{openbox_config}\nOBEOF"
        )
        result = self.vm_exec(config_cmd)
        if not result or result.get("returncode") != 0:
            logger.warning("Failed to configure openbox shortcuts (non-critical)")
        else:
            logger.info("✓ Openbox shortcuts configured (Ctrl+Alt+T=xterm)")

        # Start each display
        success_count = 0
        for display_num in self.display_nums:
            if self._start_display(display_num):
                success_count += 1
            else:
                logger.warning(f"Failed to start display :{display_num}")

        logger.info(f"Successfully initialized {success_count}/{self.num_displays} displays")
        return success_count == self.num_displays

    def _start_display(self, display_num: int) -> bool:
        """Start a single Xvfb display with openbox.

        Args:
            display_num: Display number (e.g., 2 for :2)

        Returns:
            bool: True if started successfully
        """
        logger.info(f"Starting display :{display_num}...")

        # Start Xvfb + openbox + set background (no taskbar for cleaner display)
        cmd = (
            f"export DISPLAY=:{display_num}; "
            f"nohup Xvfb :{display_num} -screen 0 1920x1080x24 -ac >/dev/null 2>&1 & sleep 2; "
            f"nohup openbox >/dev/null 2>&1 & sleep 1; "
            f"xsetroot -solid '#2C3E50' || true; "
        )

        result = self.vm_exec(cmd)
        if not result:
            with self._lock:
                self.displays[display_num].status = DisplayStatus.ERROR
                self.displays[display_num].error_msg = "Failed to execute start command"
            return False

        # Verify display is running
        verify_result = self.vm_exec(f"DISPLAY=:{display_num} xdpyinfo | head -3")
        if verify_result and verify_result.get("returncode") == 0:
            with self._lock:
                self.displays[display_num].status = DisplayStatus.IDLE
                self.idle_displays.add(display_num)
            logger.info(f"✓ Display :{display_num} ready")
            return True
        else:
            with self._lock:
                self.displays[display_num].status = DisplayStatus.ERROR
                self.displays[display_num].error_msg = "Display verification failed"
            logger.error(f"✗ Display :{display_num} failed verification")
            return False

    def allocate(self, agent_id: str) -> Optional[int]:
        """Allocate an idle display for an agent.

        Args:
            agent_id: Identifier for the agent using this display

        Returns:
            int: Display number if available, None if pool exhausted
        """
        with self._lock:
            if not self.idle_displays:
                logger.warning(f"No idle displays available for agent {agent_id}")
                return None

            # Pop first idle display
            display_num = self.idle_displays.pop()
            self.displays[display_num].status = DisplayStatus.ALLOCATED
            self.displays[display_num].last_used_by = agent_id

            logger.info(f"Allocated display :{display_num} to agent {agent_id}")
            return display_num

    def release(self, display_num: int):
        """Return a display to the idle pool.

        Args:
            display_num: Display number to release
        """
        with self._lock:
            if display_num not in self.displays:
                logger.warning(f"Attempted to release unknown display :{display_num}")
                return

            display = self.displays[display_num]
            if display.status != DisplayStatus.ALLOCATED:
                logger.warning(
                    f"Display :{display_num} was not allocated "
                    f"(status={display.status.value})"
                )

            # Reset display to clean slate
            self._reset_display(display_num)

            display.status = DisplayStatus.IDLE
            self.idle_displays.add(display_num)

            logger.info(
                f"Released display :{display_num} "
                f"(previously used by {display.last_used_by})"
            )

    def _reset_display(self, display_num: int):
        """Reset a display to clean state (close all windows).

        Args:
            display_num: Display number to reset
        """
        # Close all windows on this display
        cmd = (
            f"export DISPLAY=:{display_num}; "
            f"wmctrl -l | awk '{{print $1}}' | xargs -I{{}} wmctrl -ic {{}} 2>/dev/null || true"
        )
        self.vm_exec(cmd)
        logger.debug(f"Reset display :{display_num}")

    def get_status(self) -> Dict[int, str]:
        """Get status of all displays.

        Returns:
            Dict mapping display number -> status string
        """
        with self._lock:
            return {
                num: (
                    f"{display.status.value}"
                    f"{' - ' + display.last_used_by if display.last_used_by else ''}"
                    f"{' - ERROR: ' + display.error_msg if display.error_msg else ''}"
                )
                for num, display in self.displays.items()
            }

    def get_idle_count(self) -> int:
        """Get number of idle displays.

        Returns:
            int: Number of displays currently idle
        """
        with self._lock:
            return len(self.idle_displays)

    def cleanup(self):
        """Clean up all displays (kill Xvfb processes)."""
        logger.info("Cleaning up display pool...")

        for display_num in self.display_nums:
            cmd = f"pkill -f 'Xvfb :{display_num}' || true"
            self.vm_exec(cmd)

        logger.info("Display pool cleaned up")
