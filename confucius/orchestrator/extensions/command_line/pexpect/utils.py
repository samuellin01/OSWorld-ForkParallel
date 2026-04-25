# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import asyncio
import re
import time
from pathlib import Path
from typing import Tuple

import pexpect

from .....utils.asyncio import convert_to_async
from .....utils.timeutil import get_human_delta

from ..exceptions import InvalidCommandLineInput, SessionTerminated

from .types import (
    InteractTerminalInput,
    SpawnTerminalInput,
    TerminalOutput,
    TerminalSession,
    TerminalSessionsInfo,
)


async def _wait_for_patterns(
    terminal: pexpect.pty_spawn.spawn,
    expect_patterns: list[str] | None = None,
    expect_regex_patterns: list[str] | None = None,
    timeout: int = 30,
) -> Tuple[str, str | None, int]:
    """Wait for patterns in terminal output.

    Args:
        terminal: The pexpect terminal instance
        expect_patterns: List of literal string patterns to match
        expect_regex_patterns: List of regex patterns to match
        timeout: Timeout in seconds

    Returns:
        Tuple of (output, matched_pattern, index)
    """
    # Start with default patterns for timeout and EOF
    all_patterns = [pexpect.TIMEOUT, pexpect.EOF]

    # Add literal patterns (escaped for regex)
    literal_patterns = []
    for pattern in expect_patterns or []:
        escaped_pattern = re.escape(pattern)
        literal_patterns.append(re.compile(escaped_pattern))
    all_patterns.extend(literal_patterns)

    # Add regex patterns
    regex_patterns = []
    for pattern in expect_regex_patterns or []:
        regex_patterns.append(re.compile(pattern))
    all_patterns.extend(regex_patterns)

    index = await terminal.expect(all_patterns, timeout=timeout, async_=True)

    if index <= 1:  # TIMEOUT / EOF occurred
        output = terminal.before if terminal.before else ""
        # Clear the buffer on EOF as well for consistency
        if hasattr(terminal, "buffer"):
            terminal.buffer = ""
        return output, None, index
    else:  # One of the patterns matched
        output = terminal.before if hasattr(terminal, "before") else ""

        # If a pattern matched, include the matched pattern in the output
        # This is important for interactive shells where the prompt is part of the output
        if hasattr(terminal, "after"):
            after = terminal.after if terminal.after else ""
            output += after

        pattern_index = index - 2  # Adjust for TIMEOUT and EOF
        matched_pattern = None

        # Check if it's a literal pattern
        if expect_patterns and pattern_index < len(literal_patterns):
            matched_pattern = expect_patterns[pattern_index]
        # Otherwise it's a regex pattern
        elif expect_regex_patterns and pattern_index >= len(literal_patterns):
            regex_index = pattern_index - len(literal_patterns)
            if regex_index < len(expect_regex_patterns):
                matched_pattern = expect_regex_patterns[regex_index]

        return output, matched_pattern, index


async def create_terminal(
    inp: SpawnTerminalInput,
    sessions: dict[int, TerminalSession],
) -> TerminalOutput:
    """Create a new terminal session using pexpect."""
    start_time = time.time()

    # Validate that the working directory exists before spawning the terminal
    if inp.cwd is not None and not Path(inp.cwd).exists():
        raise InvalidCommandLineInput(f"Working directory does not exist: {inp.cwd}")

    # Use universal_newlines to handle line endings properly
    # Run command through bash to properly handle shell features like pipes,
    # redirections, command chaining, etc.
    terminal: pexpect.pty_spawn.spawn = await convert_to_async(
        pexpect.pty_spawn.spawn,
        "/bin/bash",
        ["-c", inp.command],
        cwd=inp.cwd,
        env=inp.env,
        encoding="utf-8",  # Force UTF-8 encoding
        codec_errors="replace",  # Replace invalid characters
        timeout=inp.timeout,
        echo=False,  # Disable echo
        use_poll=True,  # Use poll() for non-blocking I/O
    )

    output, matched_pattern, index = await _wait_for_patterns(
        terminal,
        expect_patterns=inp.expect_patterns,
        expect_regex_patterns=inp.expect_regex_patterns,
        timeout=inp.timeout,
    )

    is_alive = not index == 1  # index == 1 means EOF, so not is_alive

    # Get the process ID to use as session ID
    process_id = terminal.pid

    # Create session object
    session = TerminalSession(
        session_id=process_id,
        input=inp,
        spawn_time=start_time,
        last_interaction=time.time(),
        terminal=terminal,
    )

    # Store session in dictionary
    if is_alive:
        sessions[process_id] = session

    # Calculate execution time
    execution_time = time.time() - start_time

    return TerminalOutput(
        session_id=process_id,
        output=output,  # Already a string due to encoding='utf-8' in pexpect.spawn
        exit_status=terminal.exitstatus if not is_alive else None,
        is_alive=is_alive,
        command=inp.command,
        matched_pattern=matched_pattern,
        execution_time=execution_time,
    )


async def interact_terminal(
    inp: InteractTerminalInput,
    session: TerminalSession,
) -> TerminalOutput:
    """Interact with an existing terminal session."""
    if not session.is_alive():
        raise SessionTerminated("Terminal session has already terminated.")

    start_time = time.time()
    session.update_last_interaction()
    terminal = session.terminal

    if inp.send_control:
        await convert_to_async(terminal.sendcontrol, inp.send_control)
    elif inp.send is not None:
        await convert_to_async(terminal.send, inp.send)
    elif inp.send_line is not None:
        await convert_to_async(terminal.sendline, inp.send_line)

    if inp.send_eof:
        await convert_to_async(terminal.sendeof)

    output, matched_pattern, _ = await _wait_for_patterns(
        terminal,
        expect_patterns=inp.expect_patterns,
        expect_regex_patterns=inp.expect_regex_patterns,
        timeout=inp.timeout,
    )

    execution_time = time.time() - start_time

    return TerminalOutput(
        session_id=inp.session_id,
        output=output,
        exit_status=terminal.exitstatus if not terminal.isalive() else None,
        is_alive=terminal.isalive(),
        command=session.input.command,
        matched_pattern=matched_pattern,
        execution_time=execution_time,
    )


async def terminate_terminal(
    session: TerminalSession,
    force: bool = False,
) -> TerminalOutput:
    """Terminate a terminal session."""
    start_time = time.time()

    if not session.is_alive():
        return TerminalOutput(
            session_id=session.session_id,
            output="Terminal session was already terminated.",
            exit_status=session.exit_status,
            is_alive=False,
            command=session.input.command,
            matched_pattern=None,
            execution_time=0.0,
        )

    terminal = session.terminal

    session.terminate(force=force)

    if not force:
        await asyncio.sleep(0.5)
        if session.is_alive():  # If still alive after waiting
            session.terminate(force=True)  # Force kill

    return TerminalOutput(
        session_id=session.session_id,
        output="Terminal session terminated" + (" (forced)" if force else ""),
        exit_status=terminal.exitstatus,
        is_alive=False,  # Should be terminated now
        command=session.input.command,
        matched_pattern=None,
        execution_time=time.time() - start_time,
    )


def format_terminal_list(sessions: dict[int, TerminalSession]) -> str:
    """Format a list of terminal sessions as a markdown table."""
    if not sessions:
        return "No active terminal sessions."

    lines = [
        "| Session ID | Name | Command | Spawned | Last Activity | Status |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    current_time = time.time()
    for session_id, session in sessions.items():
        spawn_ago = get_human_delta(current_time - session.spawn_time) + " ago"
        last_activity_ago = (
            get_human_delta(current_time - session.last_interaction) + " ago"
        )

        status = (
            "Active"
            if session.is_alive()
            else f"Terminated (code: {session.exit_status})"
        )

        # Format command (truncate if too long)
        command = session.input.command
        if len(command) > 30:
            command = command[:27] + "..."

        # Add row
        lines.append(
            f"| PID {session_id} | {session.input.name} | `{command}` | {spawn_ago} | {last_activity_ago} | {status} |"
        )

    return "\n".join(lines)


def get_terminal_sessions_info(
    sessions: dict[int, TerminalSession],
) -> TerminalSessionsInfo:
    """Get detailed information about terminal sessions in a structured format."""
    return TerminalSessionsInfo(
        sessions=[session.get_info() for session in sessions.values()],
    )
