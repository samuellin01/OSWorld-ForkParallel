# pyre-strict


import bs4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages.base import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate
from pydantic import Field

from ..core.analect import AnalectRunContext
from ..core.llm_manager import LLMParams
from ..core.memory import CfMemory, HistoryVisibility
from ..output_parsers import XMLOutputParser
from .base import BaseOrchestrator
from .prompts import BASIC_ORCHESTRATOR_PROMPT
from .tags import Tag


class LLMOrchestrator(BaseOrchestrator):
    llm_params: list[LLMParams] = Field(
        default_factory=lambda: [LLMParams()],
        description="Parameters for the LLM.",
    )
    history_visibility: HistoryVisibility = Field(
        default=HistoryVisibility.SESSION,
        description="the history visibility level of the current analect",
    )
    include_child_history: bool = Field(
        default=True,
        description="whether or not to include history (messages, etc.) from children in LLM conversation",
    )
    raw_output_parser: XMLOutputParser | None = Field(
        default_factory=lambda: XMLOutputParser(parser="html.parser"),
        description="The parser to use for converting the LLM output to XML tags; if None, treat the entire output as plain text under a <root> tag",
    )
    prompt: ChatPromptTemplate = Field(
        BASIC_ORCHESTRATOR_PROMPT, description="System Prompt template"
    )
    use_stop_sequences: bool = Field(
        default=True,
        description="Whether to use stop sequences to terminate the LLM invocation early",
    )

    class Config:
        arbitrary_types_allowed = True

    async def on_invoke_llm(
        self,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> list[BaseMessage]:
        """
        Callback to process the LLM invocation. Default behavior is sequential processing of the LLM invocation based on the order of the extensions.

        Args:
            messages (list[BaseMessage]): The input messages for the LLM.
            context (AnalectRunContext): The context of the collector.

        Returns:
            list[BaseMessage]: The processed prompt and inputs of the LLM invocation.
        """
        for ext in self.extensions:
            messages = await ext.on_invoke_llm(messages, context)
        return messages

    async def on_invoke_llm_with_params(
        self,
        messages: list[BaseMessage],
        llm_params: LLMParams,
        context: AnalectRunContext,
    ) -> tuple[list[BaseMessage], LLMParams]:
        """
        Callback to process the LLM invocation and LLM params. Default behavior is sequential processing based on the order of the extensions.

        Args:
            messages (list[BaseMessage]): The input messages for the LLM.
            llm_params (LLMParams): The parameters for LLM invocation.
            context (AnalectRunContext): The context of the collector.

        Returns:
            tuple[list[BaseMessage], LLMParams]: The processed messages and llm_params.
        """
        for ext in self.extensions:
            messages, llm_params = await ext.on_invoke_llm_with_params(
                messages, llm_params, context
            )
        return messages, llm_params

    async def on_llm_response(
        self,
        message: BaseMessage,
        context: AnalectRunContext,
    ) -> BaseMessage:
        """
        Callback to process the LLM response. Default behavior is sequential processing of the LLM response based on the order of the extensions.

        Different from on_llm_output, this is called before the response is converted to plain text.

        Args:
            message (BaseMessage): The response of the LLM.
            context (AnalectRunContext): The context of the collector.

        Returns:
            BaseMessage: The processed response of the LLM.
        """
        for ext in self.extensions:
            message = await ext.on_llm_response(message, context)
        return message

    async def on_llm_output(
        self,
        text: str,
        context: AnalectRunContext,
    ) -> str:
        """
        Callback to process the LLM output. Default behavior is sequential processing of the LLM output based on the order of the extensions.

        Args:
            text (str): The text of the LLM output.
            context (AnalectRunContext): The context of the collector.

        Returns:
            The processed text of the LLM output.
        """
        for ext in self.extensions:
            text = await ext.on_llm_output(text, context)
        return text

    async def on_memory(self, memory: CfMemory, context: AnalectRunContext) -> CfMemory:
        """
        Callback to pre-process the memory before converting to LLM prompt.
        Default behavior is sequential processing of the memory based on the order of the extensions.

        Args:
            memory (CfMemory): The memory.
            context (AnalectRunContext): The context of the collector.

        Returns:
            The processed memory.
        """
        for ext in self.extensions:
            memory = await ext.on_memory(memory, context)
        return memory

    async def get_task(self, input_task: str) -> str:
        """
        Generate the task description based on the input task and the extensions.
        By default, extensions with the same name will be reduced to a single tag with the name and description of the last extension with included_in_system_prompt set to True.

        Args:
            input_task (str): The input task.

        Returns:
            str: The generated task description in the system prompt.
        """
        ext_descriptions: dict[str, Tag] = {
            ext.name: Tag(
                name="extension",
                attributes={"name": ext.name},
                contents=await ext.description(),
            )
            for ext in self.extensions
            if ext.included_in_system_prompt
        }
        if ext_descriptions:
            input_task += (
                "\n\n"
                + Tag(
                    name="extensions", contents=list(ext_descriptions.values())
                ).prettify()
            )
        return input_task

    async def get_llm_params(self) -> LLMParams:
        if len(self.llm_params) > 1:
            raise NotImplementedError(
                "Ensemble LLMs are not supported for LLMOrchestrator"
            )
        elif len(self.llm_params) == 0:
            raise ValueError("No LLM parameters provided")

        llm_params = self.llm_params[0]
        if self.use_stop_sequences and self.stop_sequences:
            llm_params.stop = self.stop_sequences
        return llm_params

    async def get_root_tag(self, task: str, context: AnalectRunContext) -> bs4.Tag:
        memory = context.memory_manager.get_memory_by_visibility(
            self.history_visibility, include_children=self.include_child_history
        )
        memory = await self.on_memory(memory, context)
        mem_messages = await memory.to_lc_messages()
        messages = (self.prompt + mem_messages).format_messages(
            task=await self.get_task(task)
        )
        messages = await self.on_invoke_llm(
            messages=messages,
            context=context,
        )
        llm_params = await self.get_llm_params()
        messages, llm_params = await self.on_invoke_llm_with_params(
            messages=messages,
            llm_params=llm_params,
            context=context,
        )
        chat = context.llm_manager._get_chat(params=llm_params)


        res: str = await self._invoke_llm_impl(
            chat=chat,
            messages=messages,
            context=context,
        )
        res = await self.on_llm_output(res, context)
        if self.raw_output_parser is None:
            soup = bs4.BeautifulSoup("", "html.parser")
            root = soup.new_tag("root")
            root.append(bs4.NavigableString(res))
            soup.append(root)
            return root
        parser = self.raw_output_parser
        parsed = await context.invoke(parser, res)
        root_tag = parsed.soup.find(parser.root_tag)
        return root_tag

    async def _invoke_llm_impl(
        self,
        chat: BaseChatModel,
        messages: list[BaseMessage],
        context: AnalectRunContext,
    ) -> str:
        response = await context.invoke(chat, messages)
        response = await self.on_llm_response(response, context)
        message_content = ""
        orig_content = response.content
        if isinstance(orig_content, str):
            message_content = orig_content
        else:
            assert isinstance(orig_content, list)
            for ct in orig_content:
                if isinstance(ct, str):
                    message_content += ct
                else:
                    message_content += ct.get("text")
        return message_content
