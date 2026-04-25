# pyre-strict


from bashlex import ast, parser
from pydantic import BaseModel, Field


def is_subcommand(potential_subcommand: str, command_line: str) -> bool:
    """
    Check if `potential_subcommand` is a subcommand of `command_line`, allowing for options in the subcommand.

    Args:
        potential_subcommand: The potential subcommand to check.
        command_line: The command line to check.

    Returns:
        Whether `potential_subcommand` is a subcommand of `command_line`.
    """
    # Split the command lines into components
    command_parts = command_line.lower().strip().split()
    subcommand_parts = potential_subcommand.lower().strip().split()

    # Check if the command parts are a prefix of the subcommand parts
    if len(command_parts) > len(subcommand_parts):
        return False

    # Check if the command is a prefix of the subcommand
    return subcommand_parts[: len(command_parts)] == command_parts


class CommandValidationResult(BaseModel):
    """Result of validating commands against allowed and disallowed lists."""

    allowed: set[str] = Field(
        default_factory=set,
        description="Commands that are explicitly allowed",
    )
    disallowed: set[str] = Field(
        default_factory=set,
        description="Commands that aren't in the allowed list",
    )
    explicitly_disallowed: set[str] = Field(
        default_factory=set,
        description="Commands that are explicitly disallowed",
    )
    disallowed_to_original: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of disallowed commands to original command strings",
    )


def get_allowed_and_disallowed_commands(
    bash_command: str,
    tokenized_allowed_commands: list[list[str]],
    tokenized_disallowed_commands: list[list[str]],
) -> CommandValidationResult:
    """
    Check all commands in bash_command to ensure they are all allowed and not explicitly disallowed.

    This solution is O(n * (k + d))
        where n is the # of commands in the input string
        where k is the # allowed commands
        where d is the # disallowed commands
        with the assumption that the number of tokens (split by " ") of each command is reasonable

    Args:
        bash_command: bash which may have 1 or more commands to check
        tokenized_allowed_commands: a list of allowed commands which are pre-tokenized (split by " ")
        tokenized_disallowed_commands: a list of disallowed commands which are pre-tokenized (split by " ")

    Returns:
        A CommandValidationResult object containing:
        - allowed: Commands that are explicitly allowed
        - disallowed: Commands that aren't in the allowed list
        - explicitly_disallowed: Commands that are explicitly disallowed
        - disallowed_to_original: Mapping of disallowed commands to original command strings
    """
    result = CommandValidationResult()
    tokens_for_each_command = get_command_tokens_from_bash(bash_command)

    # First check for explicitly disallowed commands
    for command_tokens in tokens_for_each_command:
        command_str = " ".join(command_tokens)
        for disallowed_command in tokenized_disallowed_commands:
            if _command_matches_allowed_command(command_tokens, disallowed_command):
                disallowed_str = " ".join(disallowed_command)
                result.explicitly_disallowed.add(disallowed_str)
                result.disallowed_to_original[disallowed_str] = command_str

    # Then check for allowed commands
    for command_tokens in tokens_for_each_command:
        # Skip if already marked as explicitly disallowed
        command_str = " ".join(command_tokens)
        if any(
            _command_matches_allowed_command(command_tokens, disallowed_command)
            for disallowed_command in tokenized_disallowed_commands
        ):
            continue

        matched = False
        for allowed_command in tokenized_allowed_commands:
            if _command_matches_allowed_command(command_tokens, allowed_command):
                # Use command from tokenized_allowed_commands for allowed list as it will contain a shorter name for display
                result.allowed.add(" ".join(allowed_command))
                matched = True
                break

        # None matched
        if not matched:
            result.disallowed.add(command_str)

    return result


def _command_matches_allowed_command(
    command: list[str], allowed_command: list[str]
) -> bool:
    """
    Checks if tokenized command matches a tokenized allowed command.
    We define matching as when the command starts with or matches the allowed_command
    We do this with tokens to ensure something like cli1 --foo doesn't match cli

    Returns true if the command matches the allowed command. False otherwise.
    """
    if len(allowed_command) > len(command):
        return False

    for i, command_token in enumerate(command):
        if i >= len(allowed_command):
            # Command matches all parts of allowed_command
            return True
        if command_token != allowed_command[i]:
            return False

    return True


class _NodeVisitor(ast.nodevisitor):
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def visitcommand(self, n: ast.node, parts: list[ast.node]) -> None:
        command_tokens = []

        for part in parts:
            # pyre-ignore[16]: `ast.node` has no attribute `kind`
            if part.kind == "word":
                # pyre-ignore[16]: `ast.node` has no attribute `word`
                command_tokens.append(part.word)
            elif command_tokens:
                # We've hit a non-word after collecting words, so we're done with this command
                break

        if command_tokens:
            self.commands.append(command_tokens)

    def visitredirect(
        self, n: ast.node, input: None, type: None, output: None, heredoc: None
    ) -> None:
        # Redirecting output can be abused by the agent to overwrite sensitive
        # files. However some agents depend on output redirection for their
        # standard workflows. So this feature needs to be gated. For now, just
        # commenting it out to unblock.
        pass
        # raise InvalidCommandLineInput(
        #     dedent("""\
        #     <warning title="Bash shell output redirect not allowed">
        #         Redirecting the output of a command is not allowed, as this can be
        #         used to bypass safeguards for preventing overwriting sensitive
        #         files. Please use a different approach to capture the output of the
        #         command.\
        #     </warning>
        # """)
        # )


def get_command_tokens_from_bash(command: str) -> list[list[str]]:
    """
    From a bash script which might contain multiple command calls,
    get all of the commands called (including subcommands, flags, etc.)

    Example:
    command: arc build --ignore-deps && arc canary
    returns: [["arc", "build", "--ignore-deps"], ["arc", "canary"]]
    """

    trees = parser.parse(command)
    visitor = _NodeVisitor()

    for tree in trees:
        visitor.visit(tree)

    return visitor.commands
