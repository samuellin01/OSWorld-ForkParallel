# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from textwrap import dedent

from .....core.analect.analect import AnalectRunContext
from ..exceptions import InvalidCommandLineInput
from ..runner import CommandLineInput
from .cli_command_validator import CliCommandValidator


DISALLOWED_OPTIONS = [
    "-E",
    "--exec",
]

OPTIONS_WITH_ARGS = [
    "-f",
    "--file",
    "-F",
    "--field-separator",
    "-v",
    "--assign",
    "--dump-variables",
    "--debug",
    "-e",
    "--source",
    "-E",
    "--exec",
    "-i",
    "--include",
    "-l",
    "--load",
    "-L",
    "--lint",
    "--pretty-print",
    "--profile",
]


class AwkValidator(CliCommandValidator):
    async def run_validator(
        self,
        command_tokens: list[str],
        inp: CommandLineInput,
        context: AnalectRunContext,
    ) -> CommandLineInput:
        args = self.parse_posix_args(command_tokens, OPTIONS_WITH_ARGS)
        for arg in args.options:
            if arg[0] in DISALLOWED_OPTIONS:
                raise InvalidCommandLineInput(
                    dedent(
                        f"""\
                        <warning>
                        You are not allowed to use the `{arg[0]}` option with `awk`.
                        This is considered a security risk as it can bypass
                        safeguards in place to protect sensitive data.
                        </warning>\
                        """
                    )
                )
        return inp
