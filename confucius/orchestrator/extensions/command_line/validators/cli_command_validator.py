# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from pydantic import BaseModel

from .....core.analect.analect import AnalectRunContext
from ..runner import CommandLineInput


class ParsedArgs(BaseModel):
    command: str = ""
    options: list[list[str | bool]] = []
    positional: list[str] = []


class CliCommandValidator(BaseModel):
    def _handle_positional_arg(self, out: ParsedArgs, arg: str) -> None:
        out.positional.append(arg)

    def _handle_long_option(
        self, out: ParsedArgs, arg: str, options_with_args: list[str]
    ) -> str | None:
        parts = arg.split("=")
        option_name = parts[0]

        if len(parts) == 1 and option_name in options_with_args:
            # Option requires an argument but none provided in this token
            return option_name
        else:
            # Either option doesn't need an argument or it was provided with =
            value = True if len(parts) == 1 else parts[1]
            out.options.append([option_name, value])
            return None

    def _handle_short_options(
        self,
        out: ParsedArgs,
        arg: str,
        current_option: str | None,
        options_with_args: list[str],
    ) -> str | None:
        # If we have a pending option from before, add it with default value
        if current_option is not None:
            out.options.append([current_option, True])
            current_option = None

        # Process each character in the short option group
        for i, char in enumerate(arg[1:]):
            option_name = "-" + char
            is_last_char = i == len(arg[1:]) - 1

            if (
                is_last_char
                and option_name in options_with_args
                and current_option is None
            ):
                # Last option in group needs an argument
                current_option = option_name
            else:
                # Option doesn't need an argument or isn't last
                out.options.append([option_name, True])

        return current_option

    def _handle_option_value(
        self, out: ParsedArgs, arg: str, current_option: str | None
    ) -> str | None:
        if current_option is not None:
            # This arg is a value for the previous option
            out.options.append([current_option, arg])
            return None
        else:
            # This is a positional argument
            self._handle_positional_arg(out, arg)
            return None

    def parse_posix_args(
        self, args: list[str], options_with_args: list[str]
    ) -> ParsedArgs:
        out = ParsedArgs()
        current_option = None
        hit_end_of_options = False

        out.command = args[0]
        for arg in args[1:]:
            if hit_end_of_options:
                self._handle_positional_arg(out, arg)
                continue
            elif arg == "--":
                # Handle end of options
                hit_end_of_options = True
                continue

            # If we have a pending option that expected a value, but the next
            # token begins a new option (long or short), then the pending
            # option is a boolean flag. Finalize it with True before handling
            # the new token. This mirrors the behavior implemented for short
            # options and fixes cases like `--draft --stack` where `--draft`
            # was previously dropped.
            if current_option is not None and (arg.startswith("-") and arg != "-"):
                out.options.append([current_option, True])
                current_option = None

            if arg.startswith("--"):
                # Handle long options
                current_option = self._handle_long_option(out, arg, options_with_args)
            elif arg == "-":
                # Treat single dash as a value (e.g., stdin) rather than a new option
                current_option = self._handle_option_value(out, arg, current_option)
            elif arg.startswith("-"):
                # Handle short options
                current_option = self._handle_short_options(
                    out, arg, current_option, options_with_args
                )
            else:
                # Handle plain values
                current_option = self._handle_option_value(out, arg, current_option)

        # Handle any remaining option without a value
        if current_option is not None:
            out.options.append([current_option, True])

        return out

    async def run_validator(
        self,
        command_tokens: list[str],
        inp: CommandLineInput,
        context: AnalectRunContext,
    ) -> CommandLineInput:
        raise NotImplementedError("Subclasses must implement this method")
