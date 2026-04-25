# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

SPAWN_TERMINAL_DESCRIPTION: str = """\
This tool starts a new interactive terminal session using pexpect.

Use this tool when you need to interact with commands that require multiple inputs over time, such as:
- Interactive shells
- Command-line tools with prompts and menus
- Programs with multi-step workflows

To start a new terminal session:
- Provide a descriptive name (for reference)
- Specify the command to execute
- Optionally set the working directory and other parameters
- Optionally specify patterns to wait for in the initial output

After starting a session, you'll receive a session_id that you can use for follow-up interactions.

Best Practices:
- Always specify `cwd` for better reliability
- Set a reasonable timeout based on the expected duration of the command:
  - Short timeouts (5-10s) for quick interactive commands
  - Longer timeouts (30-120s) for commands that involve compiling, linting, or testing
  - The timeout doesn't kill the process, it just determines when to return the current output
- Use `expect_patterns` for interactive shells to wait for the prompt

Example:
```json
{
  "name": "bash_session",
  "command": "bash",
  "cwd": "/tmp",
  "timeout": 5,
  "expect_patterns": ["$", "#"]  // Wait for bash prompt
}
```

Output Format:
```json
{
  "session_id": 12345,        # Process ID to use in future interactions
  "output": "output text",     # Terminal output so far
  "exit_status": null,        # Exit status (null if still running)
  "is_alive": true,          # Whether the terminal is still active
  "command": "bash",         # The command that was executed
  "matched_pattern": "$",     # Pattern matched (when using expect patterns)
  "execution_time": 0.123     # Time spent executing so far
}
```
"""

INTERACT_TERMINAL_DESCRIPTION: str = """\
This tool sends input or commands to an existing terminal session.

After creating a terminal session with `spawn_terminal`, use this tool to:
- Send text input to the terminal
- Send control characters (like Ctrl+C)
- Send EOF (Ctrl+D)
- Wait for specific patterns in the output
- Check if the session is still active

Parameters:
- `session_id`: ID from the spawn_terminal response (required)
- `send`: Text to send to the terminal WITHOUT appending a newline (for interactive applications like vi)
- `send_line`: Text to send to the terminal WITH a newline (for most terminal commands)
- `expect_patterns`: List of literal string patterns to wait for in the output
- `expect_regex_patterns`: List of regex patterns to wait for in the output
- `timeout`: How long to wait before returning output
- `send_control`: Send a control character (e.g., 'c' for Ctrl+C)
- `send_eof`: Whether to send EOF signal (Ctrl+D)

Note: You should use either `send`, `send_line`, `send_control`, or `send_eof` in a single request.

Best Practices:
- Set a reasonable timeout based on the expected duration of the command:
  - Short timeouts (5-10s) for quick interactive commands
  - Longer timeouts (30-120s) for commands that involve compiling, linting, or testing
  - The timeout doesn't kill the process, it just determines when to return the current output
- Use `expect_patterns` for literal string matching (safer and more reliable)
- Use `expect_regex_patterns` only when you need regex features
- For interactive shells, always wait for the prompt after sending a command

Examples:

1. Send a command to a bash session and wait for the prompt:
```json
{
  "session_id": 12345,
  "send_line": "ls -la",
  "expect_patterns": ["$", "#"],  // Wait for bash prompt
  "timeout": 5
}
```

2. Send a keypress to vi without a newline:
```json
{
  "session_id": 12345,
  "send": "i",  // Enter insert mode in vi
  "timeout": 2
}
```

3. Send a response to a prompt:
```json
{
  "session_id": 12345,
  "send_line": "y",
  "timeout": 5
}
```

4. Wait for specific patterns using both literal and regex matching:
```json
{
  "session_id": 12345,
  "send_line": "find / -name *.py",
  "expect_patterns": ["$", "#"],  // Literal prompts
  "expect_regex_patterns": ["Permission denied", "No such file"],  // Regex patterns
  "timeout": 30
}
```

4. Send Ctrl+C to interrupt a command:
```json
{
  "session_id": 12345,
  "send_control": "c",
  "timeout": 5
}
```

Output Format: Same as spawn_terminal.
"""

TERMINATE_TERMINAL_DESCRIPTION: str = """\
This tool terminates an existing terminal session.

Use this tool to gracefully or forcefully terminate a terminal session that was started with `spawn_terminal`.

Parameters:
- `session_id`: ID of the terminal session to terminate (required)
- `force`: Whether to force kill the terminal (default: false)

Example:
```json
{
  "session_id": 12345,
  "force": false
}
```

Output Format: Same as spawn_terminal, with status information.
"""

LIST_TERMINALS_DESCRIPTION: str = """\
This tool lists all active terminal sessions.

Use this tool to see what terminal sessions are currently active and available for interaction.

No parameters are required.

Example:
```json
{}
```

Output Format:
```json
{
  "sessions": [
    {
      "session_id": 12345,
      "name": "bash_session",
      "command": "bash",
      "is_alive": true,
      "exit_status": null,
      "spawn_time_seconds_ago": 60,
      "last_activity_seconds_ago": 30,
      "spawn_time_human": "1m ago",
      "last_activity_human": "30s ago",
      "cwd": "/tmp"
    },
    {
      "session_id": 12346,
      "name": "python_session",
      "command": "python -i",
      "is_alive": true,
      "exit_status": null,
      "spawn_time_seconds_ago": 120,
      "last_activity_seconds_ago": 45,
      "spawn_time_human": "2m ago",
      "last_activity_human": "45s ago",
      "cwd": "/home/user"
    }
  ]
}
```

This returns detailed information about all active terminal sessions.
"""
