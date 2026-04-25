# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

import plotly.graph_objs as go

from ...utils.validator import run_validator

from .. import types as cf


@dataclass
class Choice:
    name: str
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


EditOutputT = TypeVar("EditOutputT")


class IOInterface(ABC):
    @abstractmethod
    async def print(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """
        API to print text to IO interface.
        """
        ...

    async def log(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """
        API to print log to IO interface.
        """
        await self.print(text, **kwargs)

    async def error(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """
        API to print error to IO interface.
        """
        await self.print(text, **kwargs)

    async def warning(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """
        API to print error to IO interface.
        """
        await self.print(text, **kwargs)

    async def human(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """
        API to print human messages to IO interface.
        """
        await self.print(text, **kwargs)

    async def ai(
        self,
        text: str,
        **kwargs: Any,
    ) -> None:
        """
        API to print AI messages to IO interface.
        """
        await self.print(text, **kwargs)

    async def system(
        self,
        text: str,
        *,
        progress: int | None = None,
        run_status: cf.RunStatus | None = None,
        run_label: str | None = None,
        run_description: str | None = None,
        **kwargs: Any,
    ) -> None:
        """
        API to print system messages to IO interface.

        Args:
            text: The message to print as system role.
            kwargs: Additional arguments for different IO implementatins.
            For example, for UI specific IO: https://fburl.com/code/8269g6q7,
            we have the following extra options:
            - progress: A progress indicator to show the current progress.
        """
        await self.print(text, **kwargs)

    async def divider(self) -> None:
        """
        Print a visual divider between sections. Implementations may override.
        """
        await self.system("---")

    async def plotly(
        self,
        fig: go.Figure,
        **kwargs: Any,
    ) -> None:
        """
        API to visualize data using Plotly.
        """
        await self.print(fig.to_html(full_html=False, include_plotlyjs="cdn"), **kwargs)

    async def html(
        self,
        content: str,
        **kwargs: Any,
    ) -> None:
        """
        API to visualize data using HTML.
        """
        await self.print(content, **kwargs)

    async def display(
        self,
        obj: object,
        **kwargs: Any,
    ) -> None:
        """
        API to display data.

        This API is similar to IPython.display, it will display a Python object according to their type.

        Supported types:
        - Plotly
        - HTML
        - SVG
        - Markdown
        """
        # Plotly figures
        if isinstance(obj, go.Figure):
            return await self.plotly(fig=obj, **kwargs)

        # Rich representations
        method = getattr(obj, "_repr_svg_", None)
        if method is not None:
            content = method()
            if content is not None:
                return await self.html(content=content, **kwargs)

        method = getattr(obj, "_repr_markdown_", None)
        if method is not None:
            content = method()
            if content is not None:
                return await self.ai(text=content, **kwargs)

        method = getattr(obj, "_repr_html_", None)
        if method is not None:
            content = method()
            if content is not None:
                return await self.html(content=content, **kwargs)

        # Fallback
        await self.ai(repr(obj), **kwargs)

    async def _echo(self, text: str) -> None:
        await self.human(text)

    async def _echo_input(self, input: str) -> None:
        await self._echo(input)

    async def get_input(self, prompt: str = "", placeholder: str | None = None) -> str:
        """
        API to get user input.

        Args:
            prompt: The message to show before getting user input (deprecated, it is recommended to use self.ai(prompt) before calling this, which will have better UI display).
            placeholder: The message to show as a hint for user input.
        """
        inp = await self._get_input(prompt, placeholder)
        await self._echo_input(inp)
        return inp

    @abstractmethod
    async def _get_input(self, prompt: str, placeholder: str | None = None) -> str: ...

    async def _confirm(self, prompt: str, default: bool = False) -> bool:
        """
        Asks user for yes/no answer.
        """
        if default:
            prompt += " [Y/n]"
        else:
            prompt += " [y/N]"
        while True:
            await self.print(prompt)
            res = (await self._get_input("y/yes, or n/no")).lower().strip()
            if not res:
                return default
            elif res in ["y", "yes"]:
                return True
            elif res in ["n", "no"]:
                return False
            else:
                await self.error("Invalid input. Please type y/yes, or n/no.")

    async def _echo_confirm(self, result: bool) -> None:
        await self._echo("yes" if result else "no")

    async def confirm(self, prompt: str, default: bool = False) -> bool:
        """
        Asks user for yes/no answer.
        """
        res = await self._confirm(prompt=prompt, default=default)
        await self._echo_confirm(res)
        return res

    async def _echo_selection(
        self, choices: List[Choice], chosen_indices: List[int]
    ) -> None:
        await self._echo(f"Selected: {[choices[idx].name for idx in chosen_indices]}")

    async def choose_input(
        self,
        prompt: str,
        choices: List[Choice],
        min_count: int = 1,
        max_count: Optional[int] = None,
        default_choices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> List[int]:
        """
        Choose one or more items from the given choices.

        This method is expected to repeatedly ask user for input, until
        user provides a valid response.

        The default implementation uses 'print' and 'get_input', but different
        interfaces should override the implementation to provide more native
        user experience if available.

        Args:
        prompt:                     A prompt to show to user
        choices:                    A list of choices.
        min_count:                  Minimum number of choices required. By default
                                    we require at least one item to be selected.
        max_count:                  Max number of choices required. Use None to
                                    indicate no upper bound.
        default_choices:            An optional list of 0-based indices to use
                                    as default choices.
        kwargs:                     Additional arguments for different IO
                                    implementatins. For example, for UI
                                    specific IO:
                                    https://fburl.com/code/8269g6q7,
                                    we have the following extra options:
                                    - items_per_page_options: Use this when
                                    there are too many options and pagination
                                    is needed

        Returns:
            A list of 0-based indices of selected items.
        """
        chosen_indices = await self._choose_input(
            prompt,
            choices,
            min_count,
            max_count,
            default_choices,
            **kwargs,
        )
        await self._echo_selection(choices, chosen_indices)
        return chosen_indices

    async def _choose_input(
        self,
        prompt: str,
        choices: List[Choice],
        min_count: int = 1,
        max_count: Optional[int] = None,
        default_choices: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> List[int]:
        if not choices:
            raise ValueError("choices cannot be empty")
        if min_count < 0:
            raise ValueError(f"min_count must be >= 0, got: {min_count}")
        if max_count is not None and max_count <= 0:
            raise ValueError(f"max_count must be > 0 if provided, got: {max_count}")
        max_count = max_count or len(choices)

        prompt_with_choices_parts = [prompt]
        for i, choice in enumerate(choices, start=1):
            prompt_with_choices_parts.append(f"{i}: {choice.name}")
            if choice.description:
                prompt_with_choices_parts.append(f"\t{choice.description}")
        await self.print("\n".join(prompt_with_choices_parts))

        default_choices_one_based = (
            [c + 1 for c in default_choices] if default_choices else None
        )
        selection_prompt = self._get_selection_prompt(
            choices, min_count, max_count, default_choices_one_based
        )
        while True:
            user_input = await self._get_input(selection_prompt)
            try_res = self._try_parse_choices(
                user_input,
                choices,
                min_count,
                max_count,
                default_choices_one_based,
            )
            if isinstance(try_res, str):
                await self.error(f'Invalid input: "{user_input}". {try_res}')
            else:
                return try_res

    async def reset(self) -> None:  # noqa
        """
        Invoked when the current session is interrupted.
        Caller should reset the state to prepare a new session.
        """
        pass

    def _get_selection_prompt(
        self,
        choices: List[Choice],
        min_count: int,
        max_count: int,
        default_choices: Optional[List[int]],
    ) -> str:
        if not default_choices:
            if min_count == max_count:
                if min_count == 1:
                    selection_prompt = "Please choose 1 option (e.g. 1): "
                else:
                    selection_prompt = (
                        f"Please choose {min_count} options (e.g. 2,3,5): "
                    )
            else:
                selection_prompt = f"Please choose between {min_count} to {max_count} options (e.g. 2,3,5): "
            return selection_prompt
        else:
            if min_count == max_count:
                if min_count == 1:
                    selection_prompt = (
                        "Please choose 1 option (e.g. 1), "
                        f"or simply press enter to use defaults ({default_choices}): "
                    )
                else:
                    selection_prompt = (
                        f"Please choose {min_count} options (e.g. 2,3,5), "
                        f"or simply press enter to use defaults ({default_choices}): "
                    )
            else:
                selection_prompt = (
                    f"Please choose between {min_count} to {max_count} options (e.g. 2,3,5), "
                    f"or simply press enter to use defaults ({default_choices}): "
                )
            return selection_prompt

    def _try_parse_choices(
        self,
        user_input: str,
        choices: List[Choice],
        min_count: int,
        max_count: int,
        default_choices_one_based: Optional[List[int]],
    ) -> List[int] | str:
        """
        Either returns a list of 0-based indices or an error message.
        """
        if len(user_input.strip()) == 0:
            if default_choices_one_based is not None:
                user_choices = default_choices_one_based
            else:
                user_choices = []
        else:
            try:
                # by default, int() appears to already strip white spaces
                # e.g. int("  3\n") -> 3
                user_choices = {int(c) for c in user_input.split(",")}
            except ValueError as e:
                return f"Expecting a csv of indices, such as 1,3,5. Error: {e}."

        for c in user_choices:
            if c < 1 or c > len(choices):
                return f"Please choose from [1, {len(choices)}]."

        if len(user_choices) < min_count or len(user_choices) > max_count:
            return f"Expecting between {min_count} to {max_count} options, but got {len(user_choices)}."

        return sorted([c - 1 for c in user_choices])

    async def _edit(
        self,
        prompt: str,
        content: str,
        error_msg: str | None = None,
        enforce_syntax: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        API for prompting a candidate content to the user for editing.
        """
        if error_msg is not None:
            prompt = f"{error_msg} {prompt}"
        return await self.get_input(prompt=prompt, placeholder=content)

    async def _echo_edit(self, input: str) -> None:
        await self._echo(input)

    async def edit(
        self,
        prompt: str,
        content: str,
        validator: Callable[
            [str], Awaitable[EditOutputT] | EditOutputT
        ] = lambda val: val,
        enforce_syntax: bool = False,
        **kwargs: Any,
    ) -> EditOutputT:
        """
        API for prompting a candidate content to the user for editing.

        Args:
            prompt: A prompt to show to user.
            content: The content to show to user for editing.
            validator: A callable to validate the user input.
            enforce_syntax: Whether to enforce syntax validation. If True, the user won't be able to proceed if the syntax is invalid.
        """
        error_msg = None
        while True:
            out_str = await self._edit(
                prompt,
                content,
                error_msg=error_msg,
                enforce_syntax=enforce_syntax,
                **kwargs,
            )
            await self._echo_edit(out_str)
            try:
                return await run_validator(validator, out_str)
            except ValueError as exc:
                error_msg = f"Got error: {str(exc)}, please try again."
                content = out_str
                continue

    async def on_cancel(self) -> None:  # noqa
        pass
