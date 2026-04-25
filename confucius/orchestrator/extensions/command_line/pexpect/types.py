# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import time
from typing import Optional

import pexpect
import pexpect.pty_spawn
from pydantic import BaseModel, Field

from .....common.code import CodeBlock


class SpawnTerminalInput(BaseModel):
    name: str = Field(..., description="Name of the terminal session for reference")
    command: str = Field(..., description="Command to run in the terminal")
    cwd: Optional[str] = Field(None, description="Working directory for the terminal")
    env: Optional[dict[str, str]] = Field(None, description="Environment variables")
    timeout: int = Field(
        default=30, description="Timeout in seconds for initial output"
    )
    expect_patterns: Optional[list[str]] = Field(
        None, description="Literal string patterns to wait for in the output (optional)"
    )
    expect_regex_patterns: Optional[list[str]] = Field(
        None, description="Regex patterns to wait for in the output (optional)"
    )


class InteractTerminalInput(BaseModel):
    session_id: int = Field(
        ..., description="Process ID of the terminal session to interact with"
    )
    send: Optional[str] = Field(
        None, description="Text to send to the terminal without a newline"
    )
    send_line: Optional[str] = Field(
        None, description="Text to send to the terminal followed by a newline"
    )
    expect_patterns: Optional[list[str]] = Field(
        None, description="Literal string patterns to wait for in the output (optional)"
    )
    expect_regex_patterns: Optional[list[str]] = Field(
        None, description="Regex patterns to wait for in the output (optional)"
    )
    timeout: int = Field(
        default=30, description="Timeout in seconds for waiting for patterns or output"
    )
    send_control: Optional[str] = Field(
        None, description="Control character to send (e.g., 'c' for Ctrl+C)"
    )
    send_eof: bool = Field(
        default=False, description="Send EOF (Ctrl+D) to the terminal"
    )


class TerminateTerminalInput(BaseModel):
    session_id: int = Field(
        ..., description="Process ID of the terminal session to terminate"
    )
    force: bool = Field(
        default=False, description="Force kill if terminal doesn't exit gracefully"
    )


class ListTerminalsInput(BaseModel):
    # No required parameters
    pass


class TerminalOutput(BaseModel):
    session_id: int = Field(..., description="Terminal session ID (process ID)")
    output: str = Field("", description="Terminal output text")
    exit_status: Optional[int] = Field(
        None, description="Exit status if process completed"
    )
    is_alive: bool = Field(True, description="Whether the terminal is still active")
    command: str = Field(..., description="Command that was executed")
    matched_pattern: Optional[str] = Field(
        None, description="Pattern that was matched (if using expect patterns)"
    )
    execution_time: float = Field(..., description="Execution time in seconds")

    def to_markdown(self) -> str:
        """Convert the terminal output to markdown format."""
        parts = []

        # Add session ID
        parts.append(f"**Session ID:** {self.session_id}")

        # Add command
        parts.append(f"**Command:** `{self.command}`")

        # Add status
        status = "Active" if self.is_alive else "Terminated"
        if not self.is_alive and self.exit_status is not None:
            status = f"Terminated (exit code: {self.exit_status})"
        parts.append(f"**Status:** {status}")

        # Add matched pattern if available
        if self.matched_pattern is not None:
            parts.append(f"**Matched Pattern:** `{self.matched_pattern}`")

        # Add execution time
        parts.append(f"**Execution time:** {self.execution_time:.2f}s")

        # Add output in a code block
        if self.output:
            parts.append(CodeBlock(content=self.output).to_markdown(language="console"))

        return "\n\n".join(parts)


class TerminalSessionInfo(BaseModel):
    """Model for terminal session information."""

    session_id: int = Field(
        ..., description="Process ID used as unique ID for this terminal session"
    )
    name: str = Field(..., description="Name of the terminal session for reference")
    command: str = Field(..., description="Command that was executed in the terminal")
    is_alive: bool = Field(..., description="Whether the terminal is still active")
    exit_status: Optional[int] = Field(
        None, description="Exit status if process completed"
    )
    spawn_time_seconds_ago: int = Field(
        ..., description="Seconds elapsed since terminal was spawned"
    )
    last_activity_seconds_ago: int = Field(
        ..., description="Seconds elapsed since last activity"
    )
    cwd: Optional[str] = Field(None, description="Working directory for the terminal")


class TerminalSession(BaseModel):
    """Wrapper for a pexpect terminal session with metadata."""

    session_id: int = Field(
        ..., description="Process ID used as unique ID for this terminal session"
    )
    input: SpawnTerminalInput = Field(
        ..., description="Input parameters used to create this session"
    )
    spawn_time: float = Field(..., description="Time when the terminal was spawned")
    last_interaction: float = Field(..., description="Time of last interaction")

    # The actual pexpect spawn object
    terminal: pexpect.pty_spawn.spawn = Field(exclude=True)

    exit_status: Optional[int] = Field(None, description="Exit status if terminated")

    class Config:
        arbitrary_types_allowed = True

    def is_alive(self) -> bool:
        """Check if the terminal process is still running."""
        return self.terminal.isalive()

    def update_last_interaction(self) -> None:
        """Update the timestamp of last interaction."""
        self.last_interaction = time.time()

    def terminate(self, force: bool = False) -> None:
        """Gracefully terminate the terminal session.

        Args:
            force: If True, forcefully kill the process instead of sending SIGTERM first
        """
        if self.is_alive():
            self.terminal.terminate(force=force)

            # Update status
            self.exit_status = self.terminal.exitstatus

    def __del__(self) -> None:
        """Kill the terminal process if it's still running when this object is garbage collected."""
        try:
            if hasattr(self, "terminal") and self.terminal and self.is_alive():
                self.terminal.terminate(
                    force=True
                )  # Force kill to ensure it terminates
        except Exception:
            # During interpreter shutdown, some modules might already be unloaded
            # so we need to be tolerant of exceptions here
            pass

    def get_info(self) -> TerminalSessionInfo:
        """Get information about the terminal session."""
        current_time = time.time()
        spawn_time_diff = current_time - self.spawn_time
        last_activity_diff = current_time - self.last_interaction
        return TerminalSessionInfo(
            session_id=self.session_id,
            name=self.input.name,
            command=self.input.command,
            is_alive=self.is_alive(),
            exit_status=self.exit_status,
            spawn_time_seconds_ago=int(spawn_time_diff),
            last_activity_seconds_ago=int(last_activity_diff),
            cwd=self.input.cwd,
        )


class TerminalSessionsInfo(BaseModel):
    """Model for terminal sessions information."""

    sessions: list[TerminalSessionInfo] = Field(
        ..., description="List of terminal sessions"
    )
