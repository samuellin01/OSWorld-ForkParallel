# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from textwrap import dedent

from .....core.analect.analect import AnalectRunContext
from ..exceptions import InvalidCommandLineInput
from ..runner import CommandLineInput
from .cli_command_validator import CliCommandValidator


OPTIONS_WITH_ARGS = [
    "-f",
    "--skip-fields",
    "-s",
    "--skip-chars",
    "-w",
    "--check-chars",
]


class UniqValidator(CliCommandValidator):
    async def run_validator(
        self,
        command_tokens: list[str],
        inp: CommandLineInput,
        context: AnalectRunContext,
    ) -> CommandLineInput:
        args = self.parse_posix_args(command_tokens, OPTIONS_WITH_ARGS)
        if len(args.positional) > 1:
            raise InvalidCommandLineInput(
                dedent(
                    """\
                    <warning>
                    You are not allowed to pass the output argument to `uniq`.
                    This is considered a security risk as it can bypass
                    safeguards in place to protect sensitive data.
                    </warning>\
                    """
                )
            )
        return inp
