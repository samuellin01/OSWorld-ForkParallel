# pyre-strict

from textwrap import dedent

COMMAND_LINE_BASIC_DESCRIPTION: str = dedent(
    """\
    The assistant is capable of making structured command-line calls during conversations.

    Some general guidelines for command-line calls:
    - Immediately before writing the command, think for one sentence in <thinking> tags about the thought process about whether to create or update the command.
    - Wrap the FULL command in opening and closing <{cli_tag_name}> tags.
    - Place the full command with its arguments within the tags.
    - Each <{cli_tag_name}> block must have an `identifier` attribute that serves as a unique ID for its purpose
    - The identifier should:
        * Use kebab-case (e.g., "example-command")
        * Be descriptive of the command's purpose
    - Optionally, you can set the `cwd` attribute to specify the current working directory for the command.
    - If the command output is too large, it will be truncated and the full result will be redirected to a tmp file. You can use <file_edit> or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output  (if `grep` is in the allow list).
    
    Example: 
    <{cli_tag_name} identifier="example-command" cwd="foo/bar">
    # FULL Command that starts with "command_name" with its arguments all together
    </{cli_tag_name}>
    """
)

COMMAND_LINE_BASH_SCRIPT_BASIC_DESCRIPTION: str = dedent(
    """\
    The assistant is capable of making structured command-line calls and writing bash scripts during conversations.

    Some general guidelines for command-line operations:
    - Immediately before writing the command or script, think for one sentence in <thinking> tags about the thought process about whether to create or update the operation.
    - Wrap the command or script in opening and closing <{cli_tag_name}> tags.
    - Each <{cli_tag_name}> block must have an `identifier` attribute that serves as a unique ID for its purpose
    - The identifier should:
        * Use kebab-case (e.g., "backup-database", "deploy-service")
        * Be descriptive of the operation's purpose
    - Optionally, you can set the `cwd` attribute to specify the current working directory for the operation.
    - If the command output is too large, it will be truncated and the full result will be redirected to a tmp file. You can use <file_edit> or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output (if `grep` is in the allow list).

    You can use the tags in two ways:

    1. For single command execution:
    <{cli_tag_name} identifier="example-command" cwd="foo/bar">
    # FULL Command that starts with "command_name" with its arguments all together
    </{cli_tag_name}>

    2. For bash scripts with multiple commands:
    <{cli_tag_name} identifier="backup-database" cwd="/var/backups">
    #!/bin/bash
    # Your bash script here
    command1 arg1 arg2
    command2 arg1
    if [ condition ]; then
        command3
    fi
    </{cli_tag_name}>

    When writing bash scripts:
    - Always include the shebang line (#!/bin/bash) at the start
    - Use proper bash syntax and indentation
    - Add comments when necessary to explain complex operations
    - Follow bash best practices for error handling and variable usage
    """
)

COMMAND_LINE_BASH_SCRIPT_TOOL_USE_DESCRIPTION: str = dedent(
    """\
    The assistant is capable of making structured command-line calls and writing bash scripts during conversations, using the `bash` tool.

    If the command output is too large, it will be truncated and the full result will be redirected to a tmp file. You can use str_replace_editor or `grep -n -B <lines before> -A <lines after> <query> <filename>` to confirm output (if `grep` is in the allow list).

    You can use the tool in two ways:

    1. For single command execution:
    ```bash
    # FULL Command that starts with "command_name" with its arguments all together
    ```

    2. For bash scripts with multiple commands:
    ```bash
    # Your bash script here
    command1 arg1 arg2
    command2 arg1
    if [ condition ]; then
        command3
    fi
    ```

    When writing bash scripts:
    - Use proper bash syntax and indentation
    - Add comments when necessary to explain complex operations
    - Follow bash best practices for error handling and variable usage
    """
)
