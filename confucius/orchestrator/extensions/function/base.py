# pyre-strict

import json
from abc import ABC, abstractmethod
from typing import Any

import bs4
from langchain_core.runnables import Runnable
from langchain_core.runnables.utils import Input as LCInput, Output as LCOutput
from pydantic import Field, model_validator, PrivateAttr

from ....core.analect.analect import AnalectRunContext
from ....core.analect.base import Input, Output

from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ....core.memory import CfMessage
from ....utils.json import fix_invalid_escapes, remove_trailing_commas
from ....utils.string import truncate
from ...exceptions import OrchestratorInterruption
from ...tags import Tag, TagLike, unescaped_tag_content
from ..tag_with_id import TagWithIDExtension
from ..tool_use import ToolUseExtension

from .prompts import FUNCTION_CALL_BASIC_PROMPT

from .utils import (
    Function,
    generate_function_json_schema,
    generate_function_json_schema_dict,
    get_runnable,
)

RunnableInput = Input | LCInput
RunnableOutput = Output | LCOutput


class BaseRunnableExtension(TagWithIDExtension, ToolUseExtension, ABC):
    """
    Base class for runnable extensions.
    """

    max_output_lines: int = Field(
        default=100,
        description="Maximum number of lines of the output to include in the response string",
    )
    enable_tool_use: bool = True
    trace_tool_execution: bool = False  # Custom trace node will be generated
    _runnable_dict: dict[str, Runnable] = PrivateAttr(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True

    @model_validator(mode="after")
    def _validate_tool_use(self) -> "BaseRunnableExtension":  # noqa: B902
        if self.enable_tool_use:
            self.included_in_system_prompt = False
        return self

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._runnable_dict: dict[str, Runnable] = self._build_runnable_dict()

    @abstractmethod
    def _build_runnable_dict(self) -> dict[str, Runnable]:
        """
        Build a dictionary of runnables.
        """
        ...

    def _runnable_output_to_str(self, result: RunnableOutput) -> str:
        """
        Convert the result to a string.
        """
        return str(result)

    def _get_runnable_input(
        self, runnable: Runnable, content: str | dict[str, Any]
    ) -> RunnableInput:
        """
        Get the input for the runnable.
        """
        return json.loads(content) if isinstance(content, str) else content

    async def on_runnable_output(
        self,
        out: RunnableOutput,
        identifier: str,
        name: str,
        context: AnalectRunContext,
    ) -> RunnableOutput:
        """
        Callback to process the result of the runnable.
        """
        return out

    async def on_before_invoke_runnable(
        self,
        identifier: str,
        name: str,
        runnable: Runnable,
        inp: RunnableInput,
        context: AnalectRunContext,
    ) -> RunnableInput:
        """
        Callback before invoking the runnable.
        """
        return inp

    async def _invoke_runnable(
        self,
        identifier: str,
        name: str,
        runnable: Runnable,
        inp: RunnableInput,
        context: AnalectRunContext,
    ) -> RunnableOutput:
        inp = await self.on_before_invoke_runnable(
            identifier=identifier,
            name=name,
            runnable=runnable,
            inp=inp,
            context=context,
        )

        out = await context.invoke(runnable, inp, run_type=self.run_type)
        return await self.on_runnable_output(out, identifier, name, context)

    async def on_invoke_runnable_tag(
        self,
        identifier: str,
        content: str,
        context: AnalectRunContext,
        attrs: dict[str, str] | None = None,
    ) -> None:
        attrs = attrs or {}
        name: str = attrs["name"]

        if name not in self._runnable_dict:
            raise OrchestratorInterruption(f"Unknown {self.tag_name} name: '{name}'")

        runnable = self._runnable_dict[name]

        try:
            content = remove_trailing_commas(content.strip())
            content = fix_invalid_escapes(content)
            inp = self._get_runnable_input(runnable, content)
        except Exception as exc:
            raise OrchestratorInterruption(
                f"Failed to parse {self.tag_name} arguments: {exc}"
            )

        try:
            out = await self._invoke_runnable(
                identifier=identifier,
                name=name,
                runnable=runnable,
                inp=inp,
                context=context,
            )

        except Exception as exc:
            is_retryable = isinstance(exc, tuple(self.exceptions))
            if self.is_retryable_ex:
                is_retryable = self.is_retryable_ex(exc)

            if is_retryable:
                raise OrchestratorInterruption(
                    f"Failed to invoke {self.tag_name} '{name}': {exc}"
                )
            else:
                raise

        if out is not None:
            raise OrchestratorInterruption(
                messages=[
                    CfMessage(
                        content=Tag(
                            name=f"{self.tag_name}_response",
                            attributes={"identifier": identifier, "name": name},
                            contents=[
                                truncate(
                                    self._runnable_output_to_str(out),
                                    self.max_output_lines,
                                )
                            ],
                        ).prettify()
                    )
                ]
            )

    async def on_tag(self, tag: bs4.Tag, context: AnalectRunContext) -> None:
        if tag.name != self.tag_name:
            return

        identifier: str = ""
        try:
            identifier: str = (
                tag.get("identifier", self.default_identifier)
                if self.default_identifier is not None
                else tag["identifier"]
            )
            content: str = unescaped_tag_content(tag)
            await self.on_invoke_runnable_tag(
                identifier=identifier,
                content=content,
                context=context,
                attrs=tag.attrs,
            )

        except KeyError as exc:
            raise OrchestratorInterruption(
                f"Invalid tag, missing required attribute: {exc}"
            )

        except BaseException as exc:
            if not isinstance(exc, OrchestratorInterruption):
                context.memory_manager.add_messages(
                    [self.get_non_retryable_exceptions_message(identifier, exc)]
                )
            raise

    async def on_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        runnable = self._runnable_dict[tool_use.name]
        inp = self._get_runnable_input(runnable, tool_use.input)
        out = await self._invoke_runnable(
            identifier=tool_use.id,
            name=tool_use.name,
            runnable=runnable,
            inp=inp,
            context=context,
        )
        if isinstance(out, ant.MessageContentToolResult):
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=out.content,
            )

        if isinstance(out, (ant.MessageContentText, ant.MessageContentImage)):
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=[out],
            )

        if isinstance(out, list) and all(
            isinstance(o, (ant.MessageContentText, ant.MessageContentImage))
            for o in out
        ):
            return ant.MessageContentToolResult(
                tool_use_id=tool_use.id,
                content=out,
            )

        return ant.MessageContentToolResult(
            tool_use_id=tool_use.id,
            content=truncate(
                self._runnable_output_to_str(out),
                self.max_output_lines,
            ),
        )


class FunctionExtension(BaseRunnableExtension):
    """
    Extension for calling functions.

    Example usage:
    ```
    def test_func_1(a: int, b: int) -> str:
            return f"{a} + {b} = {a + b}"

    async def test_func_2(c: int, d: str) -> None:
        if d == "error":
            raise ValueError("test error")
        await asyncio.sleep(0.1)

    orchestrator = LLMOrchestrator(
        extensions=[
            FunctionExtension(functions=[test_func_1, test_func_2]),
            ...
        ]
    )

    task = "some_task"
    await context.invoke_analect(
        orchestrator,
        orchestrator.input(
            messages=[CfMessage(content="some input")],
            task=task,
        ),
    )
    """

    name: str = "function_call"
    included_in_system_prompt: bool = True
    tag_name: str = "function_call"
    run_type: str = "tool"
    functions: list[Function] = Field(
        default_factory=list, description="Functions to be called"
    )

    class Config:
        arbitrary_types_allowed = True

    def _build_runnable_dict(self) -> dict[str, Runnable]:
        return {func.__name__: get_runnable(func) for func in self.functions}

    def _get_function_desc(self, func: Function) -> Tag:
        contents = []
        if func.__doc__:
            contents.append(Tag(name="docstring", contents=[func.__doc__.strip()]))
        contents.append(
            Tag(
                name="input_schema",
                contents=[generate_function_json_schema(func)],
            )
        )
        return Tag(
            name="function",
            attributes={"name": func.__name__},
            contents=contents,
        )

    def _get_functions_desc(self) -> Tag:
        return Tag(
            name="functions",
            contents=[self._get_function_desc(func) for func in self.functions],
        )

    async def description(self) -> TagLike:
        return FUNCTION_CALL_BASIC_PROMPT.format(
            function_call_tag_name=self.tag_name,
            functions=self._get_functions_desc().prettify(),
        )

    @property
    async def tools(self) -> list[ant.ToolLike]:
        if self.enable_tool_use:
            return [
                ant.Tool(
                    name=func.__name__,
                    description=func.__doc__.strip() if func.__doc__ else None,
                    input_schema=generate_function_json_schema_dict(func),
                )
                for func in self.functions
            ]

        return []
