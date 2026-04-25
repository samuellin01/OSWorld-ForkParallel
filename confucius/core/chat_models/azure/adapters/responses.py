# pyre-strict

"""Responses API Adapter

This module provides the ResponsesAPIAdapter for OpenAI's responses API.
It handles advanced content types including:
- Text content with citations and annotations
- Image input/output
- PDF and document content
- Advanced tool calls (12+ types)
- Reasoning content preservation
- File input/output handling
"""

import json
import uuid

from typing import Any, cast, Literal, Union

import openai.types.responses.response_input_item as rii
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai._types import NOT_GIVEN
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseInputFile,
    ResponseOutputItem,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.function_tool_param import FunctionToolParam
from openai.types.responses.response_create_params import ToolChoice
from openai.types.responses.response_input_content import ResponseInputContent
from openai.types.responses.response_input_image import ResponseInputImage
from openai.types.responses.response_input_text import ResponseInputText
from openai.types.responses.response_output_item import (
    ImageGenerationCall,
    ResponseCodeInterpreterToolCall,
    ResponseComputerToolCall,
    ResponseFileSearchToolCall,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_text import (
    AnnotationContainerFileCitation,
    AnnotationFileCitation,
    AnnotationFilePath,
    AnnotationURLCitation,
)
from openai.types.responses.response_usage import ResponseUsage
from openai.types.responses.tool_choice_function_param import ToolChoiceFunctionParam
from openai.types.responses.tool_param import ToolParam

from ...bedrock.api.invoke_model import anthropic as ant
from ...bedrock.utils import lc_message_to_ant_message, lc_message_to_ant_system

from ..base import OpenAIBase
from ..model import get_model
from .chat_completions import ant_thinking_to_reasoning_effort, is_thinking_model

MessageRole = Literal["user", "system", "developer"]


class ResponsesAPIAdapter(OpenAIBase):
    """Specialized adapter for OpenAI responses API with comprehensive content type support.

    This adapter handles advanced features including:
    - Text content with citations and annotations
    - Multiple input/output content types
    - Advanced tool calls with detailed metadata
    - Reasoning content preservation
    - File/document input and output
    - Complex output item processing (12+ types)
    """

    async def _invoke_api(self, messages: list[BaseMessage], **kwargs: Any) -> Response:
        """Invoke the responses API with input + instructions format.

        The Responses API uses a different structure:
        - input: Array of input messages/content
        - instructions: System/instruction messages
        - model: Model identifier
        - max_output_tokens: Token limit (different from max_tokens)
        """
        # Convert messages to responses API format
        input_messages, instructions = await self._convert_messages_to_responses_format(
            messages
        )

        # Prepare tools for Responses API (different format from Chat Completions API)
        model = get_model(self.model)
        is_thinking = is_thinking_model(model)

        # Prepare reasoning parameter (must be an object with 'effort' field)
        reasoning_effort = ant_thinking_to_reasoning_effort(self.thinking)
        reasoning_param = (
            {"effort": reasoning_effort, "summary": "detailed"}
            if reasoning_effort
            else NOT_GIVEN
        )
        responses_tools = self._get_response_tools()
        responses_tool_choice = self._get_response_tool_choice()

        # Create responses API call with all parameters
        response = await self.client.responses.create(
            input=[msg.to_dict() for msg in input_messages],
            instructions=instructions,
            model=model,
            max_output_tokens=self.max_tokens,
            temperature=self.temperature if not is_thinking else NOT_GIVEN,
            top_p=self.top_p if not is_thinking else NOT_GIVEN,
            parallel_tool_calls=True,  # Enable parallel tool calls
            tools=responses_tools or NOT_GIVEN,
            tool_choice=responses_tool_choice or NOT_GIVEN,
            reasoning=reasoning_param,
            **kwargs,
        )
        return response

    def _get_response_tools(self) -> list[ToolParam]:
        """Get the list of tools for the response.

        Returns:
            list of tools for the response, or None if no tools are specified.
        """
        # Convert tools to Responses API format
        responses_tools: list[ToolParam] = []
        for tool in self.tools or []:
            if isinstance(tool, ant.Tool):
                # Standard Anthropic Tool with all fields
                responses_tool = FunctionToolParam(
                    type="function",
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema,
                    strict=None,
                )
            elif isinstance(tool, ant.TextEditor):
                # TextEditor tool
                responses_tool = FunctionToolParam(
                    type="function",
                    name=tool.name,
                    description=ant.TEXT_EDITOR_DESCRIPTION,
                    parameters=ant.TEXT_EDITOR_SCHEMA,
                    strict=None,
                )
            elif isinstance(tool, ant.BashTool):
                # BashTool
                responses_tool = FunctionToolParam(
                    type="function",
                    name=tool.name,
                    description=ant.BASH_TOOL_DESCRIPTION,
                    parameters=ant.BASH_TOOL_SCHEMA,
                    strict=None,
                )
            else:
                raise ValueError(f"Unsupported tool type: {type(tool)}")

            responses_tools.append(responses_tool)

        return responses_tools

    def _get_response_tool_choice(self) -> ToolChoice | None:
        """Get the tool choice for the response.

        Returns:
            Tool choice for the response, or None if no tool choice is specified.
        """
        # Convert tool_choice to appropriate format for Responses API
        tool_choice: ToolChoice | None = None
        ant_tool_choice = self.tool_choice
        if ant_tool_choice:
            # Anthropic ToolChoice objects
            if ant_tool_choice.type == ant.ToolChoiceType.AUTO:
                tool_choice = cast(ToolChoice, "auto")
            elif ant_tool_choice.type == ant.ToolChoiceType.ANY:
                tool_choice = cast(ToolChoice, "required")
            elif (
                ant_tool_choice.type == ant.ToolChoiceType.TOOL and ant_tool_choice.name
            ):
                tool_choice = ToolChoiceFunctionParam(
                    type="function", name=ant_tool_choice.name
                )
            elif ant_tool_choice.type == ant.ToolChoiceType.NONE:
                tool_choice = cast(ToolChoice, "none")
            else:
                raise ValueError(
                    f"Unsupported tool choice type: {type(ant_tool_choice)}"
                )

        return tool_choice

    def _convert_response(self, raw_response: Response) -> ant.Response:
        """Convert responses API response to Anthropic format.

        The Responses API returns an output array with multiple item types:
        - message items (text with potential citations)
        - function_call items (tool calls)
        - reasoning items (thinking content)
        - file_search_call, computer_call, code_interpreter_call items
        - Various other specialized item types
        """
        return self._responses_api_to_ant_response(raw_response)

    async def _convert_messages_to_responses_format(
        self, messages: list[BaseMessage]
    ) -> tuple[list[rii.ResponseInputItem], str]:
        """Convert Langchain messages to responses API input + instructions format.

        The responses API separates:
        - input: Main conversation content, files, images
        - instructions: System messages, behavioral instructions

        Returns:
            Tuple of (input_messages, instructions_string)
        """
        input_messages: list[rii.ResponseInputItem] = []
        instructions_parts: list[str] = []

        for msg in messages:
            if isinstance(msg, SystemMessage):
                # System messages become instructions
                ant_system = lc_message_to_ant_system(msg)
                if isinstance(ant_system, str):
                    instructions_parts.append(ant_system)
                elif isinstance(ant_system, list):
                    instructions_parts.extend(item.text for item in ant_system)

            elif isinstance(msg, (HumanMessage, AIMessage)):
                # Regular messages become input
                ant_msg = await lc_message_to_ant_message(msg)
                input_messages.extend(await self._convert_ant_message_to_input(ant_msg))

        instructions = " ".join(instructions_parts) if instructions_parts else ""
        return input_messages, instructions

    # ---- Input conversion helpers (reduce complexity of _convert_ant_message_to_input) ----
    def _input_from_text(
        self, role: ant.MessageRole, part: ant.MessageContentText
    ) -> rii.Message | ResponseOutputMessage:
        if role == ant.MessageRole.USER:
            return rii.Message(
                role="user",
                type="message",
                content=[ResponseInputText(text=part.text, type="input_text")],
            )
        elif role == ant.MessageRole.DEVELOPER:
            return rii.Message(
                role="developer",
                type="message",
                content=[ResponseInputText(text=part.text, type="input_text")],
            )
        elif role == ant.MessageRole.ASSISTANT:
            # Convert Anthropic citations to Responses API annotations if present
            annotations = self._ant_citations_to_annotations(part.citations)
            return ResponseOutputMessage(
                id="msg_" + str(uuid.uuid4()),
                role="assistant",
                type="message",
                status="completed",
                content=[
                    ResponseOutputText(
                        annotations=annotations,
                        text=part.text,
                        type="output_text",
                    )
                ],
            )
        else:
            raise ValueError(f"Unsupported role: {role} for text input")

    def _input_from_image(
        self, role: ant.MessageRole, part: ant.MessageContentImage
    ) -> rii.Message:
        if role in [ant.MessageRole.USER, ant.MessageRole.DEVELOPER]:
            return rii.Message(
                role="user" if role == ant.MessageRole.USER else "developer",
                type="message",
                content=[self._convert_image_to_file_input(part)],
            )

        raise ValueError(f"Unsupported role: {role} for image input")

    def _input_from_document(
        self, role: ant.MessageRole, part: ant.MessageContentDocument
    ) -> rii.Message:
        if role in [ant.MessageRole.USER, ant.MessageRole.DEVELOPER]:
            file_input = self._convert_document_to_file_input(part)
            return rii.Message(
                role="user" if role == ant.MessageRole.USER else "developer",
                type="message",
                content=[file_input],
            )

        raise ValueError(f"Unsupported role: {role} for document input")

    def _input_from_tool_use(
        self, part: ant.MessageContentToolUse
    ) -> rii.ResponseFunctionToolCall:
        return rii.ResponseFunctionToolCall(
            arguments=json.dumps(part.input),
            call_id=part.id,
            name=part.name,
            type="function_call",
        )

    def _inputs_from_tool_result(
        self, part: ant.MessageContentToolResult
    ) -> list[rii.ResponseInputItem]:
        items: list[rii.ResponseInputItem] = []
        content = part.content
        images: list[ResponseInputContent] = []
        texts: list[str] = []
        if isinstance(content, list):
            for ct in content:
                if isinstance(ct, ant.MessageContentImage):
                    images.append(self._convert_image_to_file_input(ct))
                else:
                    texts.append(ct.text)

        items.append(
            rii.FunctionCallOutput(
                call_id=part.tool_use_id,
                output=content if isinstance(content, str) else "\n".join(texts),
                type="function_call_output",
            )
        )
        if images:
            prefix: list[ResponseInputContent] = [
                ResponseInputText(text="[Images]", type="input_text")
            ]
            items.append(
                rii.Message(role="user", type="message", content=prefix + images)
            )
        return items

    async def _convert_ant_message_to_input(
        self, message: ant.Message
    ) -> list[rii.ResponseInputItem]:
        """Convert Anthropic message to responses API input message format.

        Handles various content types:
        - Text content
        - Image content (as input files)
        - Document content (PDFs, etc.)
        - Tool use and tool results
        """
        input_items: list[rii.ResponseInputItem] = []

        content_parts = (
            message.content if isinstance(message.content, list) else [message.content]
        )

        # Dispatch handlers by type to lower cyclomatic complexity
        for part in content_parts:
            if isinstance(part, ant.MessageContentText):
                input_items.append(self._input_from_text(message.role, part))
            elif isinstance(part, ant.MessageContentImage):
                input_items.append(self._input_from_image(message.role, part))
            elif isinstance(part, ant.MessageContentDocument):
                input_items.append(self._input_from_document(message.role, part))
            elif isinstance(part, ant.MessageContentToolUse):
                input_items.append(self._input_from_tool_use(part))
            elif isinstance(part, ant.MessageContentToolResult):
                input_items.extend(self._inputs_from_tool_result(part))

        return input_items

    def _convert_image_to_file_input(
        self, image_content: ant.MessageContentImage
    ) -> ResponseInputImage:
        """Convert image content to responses API file input format."""
        source = image_content.source

        return ResponseInputImage(
            detail="auto",
            type="input_image",
            image_url=f"data:{source.media_type};base64,{source.data}",
        )

    def _convert_document_to_file_input(
        self, doc_content: ant.MessageContentDocument
    ) -> ResponseInputFile:
        """Convert document content to responses API file input format."""
        source = doc_content.source

        # Handle different document source types
        if isinstance(source, ant.MessageContentPDFSource):
            # PDF source has media_type "application/pdf" and data
            return ResponseInputFile(
                type="input_file",
                file_data=source.data,
                file_url=None,
                filename=None,
            )
        elif isinstance(source, ant.MessageContentPlainTextSource):
            # Plain text source has media_type "text/plain" and data
            return ResponseInputFile(
                type="input_file",
                file_data=source.data,
                file_url=None,
                filename=None,
            )
        elif isinstance(source, ant.MessageContentContentBlockSource):
            # Content block source has content field
            return ResponseInputFile(
                type="input_file",
                file_data=source.content,
                file_url=None,
                filename=None,
            )
        else:
            # Fallback for any unknown source type
            return ResponseInputFile(
                type="input_file",
                file_data=None,
                file_url=None,
                filename=None,
            )

    def _responses_api_to_ant_response(self, response: Response) -> ant.Response:
        """Convert responses API response to Anthropic response format.

        Processes all output item types:
        - message: Text content with potential citations
        - function_call: Tool calls
        - reasoning: Thinking/reasoning content
        - file_search_call, computer_call, code_interpreter_call: Advanced tool calls
        - And 8+ other specialized item types
        """
        content: list[ant.ResponseContent] = []

        # Process output items from the response
        if response.output:
            content = self._process_output_items(response.output)

        # Extract metadata
        response_id = response.id
        model = response.model

        # Calculate usage
        usage = (
            ant.Usage(input_tokens=0, output_tokens=0)
            if response.usage is None
            else self._convert_usage(response.usage)
        )

        # Determine stop reason
        stop_reason = self._determine_stop_reason(response)

        return ant.Response(
            content=content,
            id=response_id,
            model=model,
            stop_reason=stop_reason,
            usage=usage,
        )

    def _process_output_items(
        self, output_items: list[ResponseOutputItem]
    ) -> list[ant.ResponseContent]:
        """Process the output items array from responses API.

        Uses dispatch pattern to handle 12+ output item types efficiently.
        """
        content: list[ant.ResponseContent] = []

        for item in output_items:
            converted_item = self._convert_output_item(item)
            if converted_item:
                content.append(converted_item)

        return content

    def _convert_output_item(
        self, item: ResponseOutputItem
    ) -> ant.ResponseContent | None:
        """Convert a single output item using dispatch pattern."""
        item_type = item.type

        # Dispatch table for output item type handlers
        type_handlers = {
            "message": self._convert_message_output,
            "function_call": self._convert_tool_calls,
            "reasoning": self._convert_reasoning_content,
            "image_generation": self._convert_image_generation,
            "error": self._convert_error_output,
        }

        # Handler for advanced tool calls (multiple types)
        advanced_tool_types = [
            "file_search_call",
            "computer_call",
            "code_interpreter_call",
        ]
        if item_type in advanced_tool_types:
            return self._convert_advanced_tool_call(item)

        # Handler for document/file outputs (multiple types)
        document_types = ["document_analysis", "file_output"]
        if item_type in document_types:
            return self._convert_document_output(item)

        # Handler for search results (multiple types)
        search_types = ["web_search", "search_result"]
        if item_type in search_types:
            return self._convert_search_result(item)

        # Use dispatch table or fallback to unknown handler
        handler = type_handlers.get(item_type, self._convert_unknown_output)

        # Add type narrowing for specific handlers that need it
        if item_type == "reasoning" and handler == self._convert_reasoning_content:
            # Type narrow to ResponseReasoningItem
            return handler(item)  # type: ignore[arg-type]

        return handler(item)  # type: ignore[arg-type]

    def _convert_text_content(
        self, output_text: ResponseOutputText
    ) -> ant.MessageContentText | None:
        """Convert text output with citations to Anthropic format."""
        if not output_text.text:
            return None

        # Extract citations if available in the response
        citations = []
        if output_text.annotations:
            for annotation in output_text.annotations:
                # Convert OpenAI annotation format to Anthropic citation format
                # Handle different annotation types with proper type checking
                if isinstance(annotation, AnnotationURLCitation):
                    # URL citations have start_index, end_index, title
                    citation = ant.ResponseCharLocationCitation(
                        cited_text=annotation.title,
                        document_title=annotation.title,
                        document_index=1,  # URL citations don't have document index
                        start_char_index=annotation.start_index,
                        end_char_index=annotation.end_index,
                    )
                    citations.append(citation)
                elif isinstance(annotation, AnnotationFileCitation):
                    # File citations have filename and index
                    citation = ant.ResponseCharLocationCitation(
                        cited_text=annotation.filename,
                        document_title=annotation.filename,
                        document_index=annotation.index,
                        start_char_index=0,  # File citations don't have char indices
                        end_char_index=0,
                    )
                    citations.append(citation)
                elif isinstance(annotation, AnnotationContainerFileCitation):
                    # Container file citations have filename, start_index, end_index
                    citation = ant.ResponseCharLocationCitation(
                        cited_text=annotation.filename,
                        document_title=annotation.filename,
                        document_index=1,  # Container citations don't have document index
                        start_char_index=annotation.start_index,
                        end_char_index=annotation.end_index,
                    )
                    citations.append(citation)
                elif isinstance(annotation, AnnotationFilePath):
                    # File path annotations have file_id and index
                    citation = ant.ResponseCharLocationCitation(
                        cited_text=f"File: {annotation.file_id}",
                        document_title=f"File: {annotation.file_id}",
                        document_index=annotation.index,
                        start_char_index=0,
                        end_char_index=0,
                    )
                    citations.append(citation)

        # Return text content with citations if available
        if citations:
            return ant.MessageContentText(text=output_text.text, citations=citations)
        else:
            return ant.MessageContentText(text=output_text.text)

    def _convert_message_output(
        self, message_item: ResponseOutputMessage
    ) -> ant.ResponseContent | None:
        """Convert ResponseOutputMessage to Anthropic format by processing its content array."""
        if not message_item.content:
            return None

        # ResponseOutputMessage contains a content array with ResponseOutputText items
        # Process each content item - typically there's only one ResponseOutputText
        for content_item in message_item.content:
            # Handle ResponseOutputText
            if isinstance(content_item, ResponseOutputText):
                return self._convert_text_content(content_item)

            # Handle other potential content types in the future
            elif isinstance(content_item, ResponseOutputRefusal):
                return ant.MessageContentText(text=f"[Refusal]: {content_item.refusal}")

        # If no matching content items found, return None
        return None

    def _convert_tool_calls(
        self, tool_call: ResponseFunctionToolCall
    ) -> ant.MessageContentToolUse | None:
        """Convert function tool calls to Anthropic format."""
        if not tool_call.name:
            return None

        call_id = tool_call.call_id or tool_call.id
        if not call_id:
            call_id = f"call_{tool_call.name}_{uuid.uuid4().hex[:8]}"

        arguments = tool_call.arguments
        try:
            input_args = (
                json.loads(arguments) if isinstance(arguments, str) else arguments
            )
        except json.JSONDecodeError:
            input_args = {}

        return ant.MessageContentToolUse(
            id=call_id, name=tool_call.name, input=input_args
        )

    def _convert_reasoning_content(
        self, reasoning_item: ResponseReasoningItem
    ) -> ant.MessageContentThinking | ant.MessageContentRedactedThinking | None:
        """Convert reasoning content to Anthropic thinking format.

        Handles both encrypted reasoning content (for privacy) and plain reasoning content.
        """
        # Handle encrypted/redacted reasoning content first
        if self._is_encrypted_reasoning(reasoning_item):
            encrypted_data = reasoning_item.encrypted_content
            if encrypted_data:  # Null safety check
                return ant.MessageContentRedactedThinking(data=encrypted_data)
            # If encrypted_content is None, fall through to plain text extraction

        # Extract plain reasoning content
        thinking_text = self._extract_reasoning_text(reasoning_item)
        if not thinking_text:
            return None

        # Get signature/ID for the thinking content
        signature = self._extract_reasoning_signature(reasoning_item)

        return ant.MessageContentThinking(signature=signature, thinking=thinking_text)

    def _is_encrypted_reasoning(self, reasoning_item: ResponseReasoningItem) -> bool:
        """Check if reasoning content is encrypted/redacted."""
        return bool(
            hasattr(reasoning_item, "encrypted_content")
            and reasoning_item.encrypted_content
        )

    def _extract_reasoning_text(self, reasoning_item: ResponseReasoningItem) -> str:
        """Extract reasoning text from various possible formats."""
        # Try content field first
        text = self._extract_from_content_field(reasoning_item)
        if text:
            return text

        # Try summary field
        text = self._extract_from_summary_field(reasoning_item)
        if text:
            return text

        # Try text field as fallback
        return self._extract_from_text_field(reasoning_item)

    def _extract_from_content_field(self, reasoning_item: ResponseReasoningItem) -> str:
        """Extract text from content field (list format)."""
        # content is list[Content] | None according to the type definition
        if not reasoning_item.content:
            return ""

        # content is a list of Content objects
        return self._extract_from_content_list(reasoning_item.content)

    def _extract_from_content_list(self, content_list: list[Any]) -> str:
        """Extract text from list of content blocks."""
        text_parts = []
        for content_block in content_list:
            # Content objects have 'text' attribute
            if hasattr(content_block, "text"):
                text_parts.append(content_block.text)
            # Handle string fallback (though this shouldn't happen with proper typing)
            elif isinstance(content_block, str):
                text_parts.append(content_block)
        return " ".join(text_parts)

    def _extract_from_summary_field(self, reasoning_item: ResponseReasoningItem) -> str:
        """Extract text from summary field."""
        # summary is list[Summary] according to the type definition
        if not reasoning_item.summary:
            return ""

        # Extract text from Summary objects
        text_parts = []
        for summary_item in reasoning_item.summary:
            # Summary objects have 'text' attribute
            text_parts.append(summary_item.text)

        return " ".join(text_parts)

    def _extract_from_text_field(self, reasoning_item: ResponseReasoningItem) -> str:
        """Extract text from text field as final fallback."""
        # ResponseReasoningItem doesn't have a 'text' field directly
        # It has 'summary' and optional 'content'
        # This method should likely not be called, but keeping for safety
        return ""

    def _extract_reasoning_signature(
        self, reasoning_item: ResponseReasoningItem
    ) -> str:
        """Extract signature/ID for the reasoning content."""
        return reasoning_item.id

    def _ant_citations_to_annotations(
        self,
        citations: list[
            Union[ant.ResponseCharLocationCitation, ant.ResponsePageLocationCitation]
        ]
        | None,
    ) -> list[
        Union[
            AnnotationContainerFileCitation,
            AnnotationFileCitation,
            AnnotationFilePath,
            AnnotationURLCitation,
        ]
    ]:
        """Convert Anthropic citations to OpenAI Responses API annotations.

        Currently supports mapping ResponseCharLocationCitation -> AnnotationContainerFileCitation.
        Unknown citation types are ignored.
        """
        annotations: list[
            Union[
                AnnotationContainerFileCitation,
                AnnotationFileCitation,
                AnnotationFilePath,
                AnnotationURLCitation,
            ]
        ] = []
        if not citations:
            return annotations

        for cit in citations:
            if isinstance(cit, ant.ResponseCharLocationCitation):
                filename = cit.document_title or cit.cited_text
                start_idx = cit.start_char_index or 0
                end_idx = cit.end_char_index or 0
                annotations.append(
                    AnnotationContainerFileCitation(
                        type="container_file_citation",
                        filename=filename,
                        file_id="file",
                        start_index=start_idx,
                        end_index=end_idx,
                        container_id="container",
                    )
                )
            # TODO: Support other Anthropic citation variants if introduced

        return annotations

    def _convert_advanced_tool_call(
        self, tool_call_item: Any
    ) -> ant.MessageContentToolUse | None:
        """Convert advanced tool calls (file_search, computer, code_interpreter) to Anthropic format."""
        # Handle different tool call types with proper type checking
        if isinstance(tool_call_item, ResponseFileSearchToolCall):
            call_id = tool_call_item.id
            tool_name = "file_search"
        elif isinstance(tool_call_item, ResponseComputerToolCall):
            call_id = tool_call_item.id
            tool_name = "computer_tool"
        elif isinstance(tool_call_item, ResponseCodeInterpreterToolCall):
            call_id = tool_call_item.id
            tool_name = "code_interpreter"
        else:
            # Fallback for unexpected types
            call_id = f"call_unknown_{uuid.uuid4().hex[:8]}"
            tool_name = "unknown"

        # Extract tool arguments/parameters
        tool_args = {}
        if hasattr(tool_call_item, "parameters"):
            tool_args = tool_call_item.parameters
        elif hasattr(tool_call_item, "arguments"):
            try:
                tool_args = (
                    json.loads(tool_call_item.arguments)
                    if isinstance(tool_call_item.arguments, str)
                    else tool_call_item.arguments
                )
            except json.JSONDecodeError:
                tool_args = {}

        return ant.MessageContentToolUse(id=call_id, name=tool_name, input=tool_args)

    def _determine_stop_reason(self, response: Response) -> ant.StopReason:
        """Determine stop reason from responses API response.

        Maps OpenAI responses API finish reasons to Anthropic stop reasons.
        Uses dispatch pattern to reduce complexity.
        """
        finish_reason = self._extract_finish_reason(response)
        if finish_reason:
            return self._map_finish_reason_to_stop_reason(finish_reason)
        return ant.StopReason.END_TURN

    def _convert_usage(self, usage: ResponseUsage) -> ant.Usage:
        """Convert OpenAI ResponseUsage into ant.Usage.

        - Maps input_tokens directly
        - Maps output_tokens plus reasoning_tokens into ant.output_tokens
        - Maps input_tokens_details.cached_tokens to ant.cache_read_input_tokens
        """
        # Base counts
        total_input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens

        # Cached tokens (prompt caching reads)
        cached_tokens = usage.input_tokens_details.cached_tokens

        # Anthropic mapping rules:
        # 1) ant.input_tokens excludes cached read tokens
        # 2) ant.output_tokens equals output_tokens (reasoning tokens are already included)
        input_tokens = max(total_input_tokens - cached_tokens, 0)

        return ant.Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_input_tokens=cached_tokens,
            cache_creation_input_tokens=None,
        )

    def _extract_finish_reason(self, response: Response) -> str | None:
        """Extract finish reason from response status or incomplete_details."""
        # Primary: Check response status
        if response.status:
            return response.status

        # Secondary: Check incomplete_details for specific reasons
        if response.incomplete_details and response.incomplete_details.reason:
            return response.incomplete_details.reason

        return None

    def _map_finish_reason_to_stop_reason(self, finish_reason: str) -> ant.StopReason:
        """Map finish reason string to Anthropic StopReason using lookup table."""
        reason_lower = finish_reason.lower()

        # Lookup table for reason mappings (reduces cyclomatic complexity)
        reason_mappings = {
            # OpenAI Response API status values
            **dict.fromkeys(["completed"], ant.StopReason.END_TURN),
            **dict.fromkeys(["max_output_tokens"], ant.StopReason.MAX_TOKENS),
            **dict.fromkeys(
                [
                    "content_filter",
                    "failed",
                    "cancelled",
                    "error",
                    "incomplete",
                    "in_progress",
                    "queued",
                ],
                ant.StopReason.END_TURN,
            ),
            # Legacy mappings for backward compatibility
            **dict.fromkeys(["stop", "end_turn", "finished"], ant.StopReason.END_TURN),
            **dict.fromkeys(
                ["length", "max_tokens", "token_limit"], ant.StopReason.MAX_TOKENS
            ),
            **dict.fromkeys(
                ["tool_calls", "function_call", "tool_use", "function_calls"],
                ant.StopReason.TOOL_USE,
            ),
        }

        return reason_mappings.get(reason_lower, ant.StopReason.END_TURN)

    def _convert_image_generation(
        self, image_item: Any
    ) -> ant.MessageContentImage | ant.MessageContentText | None:
        """Convert image generation output to Anthropic image content format."""
        # Handle ImageGenerationCall type
        if isinstance(image_item, ImageGenerationCall):
            # ImageGenerationCall has 'result' field with base64 data
            if image_item.result:
                # Default to PNG format since OpenAI doesn't specify format
                return ant.MessageContentImage(
                    type=ant.MessageContentType.IMAGE,
                    source=ant.MessageContentImageSource(
                        type="base64",
                        media_type=ant.MessageContentSourceMediaType.IMAGE_PNG,
                        data=image_item.result,
                    ),
                )
            else:
                # Image generation in progress or failed
                status_text = f"[Image generation {image_item.status}]"
                return ant.MessageContentText(
                    type=ant.MessageContentType.TEXT,
                    text=status_text,
                )

        # Fallback for any other image-related types
        # Check if it has base64 data in a 'data' field
        elif hasattr(image_item, "data") and image_item.data:
            # Default to PNG for unknown image types
            return ant.MessageContentImage(
                type=ant.MessageContentType.IMAGE,
                source=ant.MessageContentImageSource(
                    type="base64",
                    media_type=ant.MessageContentSourceMediaType.IMAGE_PNG,
                    data=image_item.data,
                ),
            )

        # Handle URL-based images (fallback to text description)
        elif hasattr(image_item, "url") and image_item.url:
            return ant.MessageContentText(
                type=ant.MessageContentType.TEXT,
                text=f"[Image generated - available at: {image_item.url}]",
            )

        # If no data available, return None
        return None

    def _convert_document_output(self, doc_item: Any) -> ant.MessageContentText | None:
        """Convert document analysis output to Anthropic text format."""
        if not hasattr(doc_item, "content") and not hasattr(doc_item, "data"):
            return None

        # Extract document analysis results as text
        content_text = ""

        if hasattr(doc_item, "analysis") and doc_item.analysis:
            # If there's analysis results, use that
            content_text = str(doc_item.analysis)
        elif hasattr(doc_item, "summary") and doc_item.summary:
            # Use summary if available
            content_text = str(doc_item.summary)
        elif hasattr(doc_item, "content") and doc_item.content:
            # Use raw content, truncated for readability
            raw_content = str(doc_item.content)
            content_text = (
                raw_content[:500] + "..." if len(raw_content) > 500 else raw_content
            )
        elif hasattr(doc_item, "data") and doc_item.data:
            # Use data, truncated for readability
            raw_data = str(doc_item.data)
            content_text = raw_data[:500] + "..." if len(raw_data) > 500 else raw_data

        if not content_text:
            return None

        # Add document type information
        # Since we don't have specific document output types from OpenAI,
        # we handle this generically with hasattr checks
        media_type = (
            doc_item.media_type if hasattr(doc_item, "media_type") else "document"
        )
        doc_name = ""
        if hasattr(doc_item, "name"):
            doc_name = doc_item.name
        elif hasattr(doc_item, "filename"):
            doc_name = doc_item.filename

        result_text = "[Document Analysis"
        if doc_name:
            result_text += f" - {doc_name}"
        if "pdf" in media_type.lower():
            result_text += " (PDF)"
        result_text += f"]\n{content_text}"

        return ant.MessageContentText(text=result_text)

    def _convert_search_result(self, search_item: Any) -> ant.MessageContentText | None:
        """Convert search result to Anthropic text format with metadata."""
        result_text = ""

        # Extract search result content
        if hasattr(search_item, "results") and search_item.results:
            results = (
                search_item.results
                if isinstance(search_item.results, list)
                else [search_item.results]
            )
            result_parts = []

            for result in results:
                if hasattr(result, "title") and hasattr(result, "content"):
                    result_parts.append(f"**{result.title}**: {result.content}")
                elif hasattr(result, "text"):
                    result_parts.append(result.text)
                elif isinstance(result, str):
                    result_parts.append(result)

            result_text = "\n".join(result_parts)

        elif hasattr(search_item, "content") and search_item.content:
            result_text = search_item.content
        elif hasattr(search_item, "text") and search_item.text:
            result_text = search_item.text

        if not result_text:
            return None

        # Add search metadata prefix
        # Handle ResponseFunctionWebSearch type
        query = ""
        if isinstance(search_item, ResponseFunctionWebSearch):
            # ResponseFunctionWebSearch has 'action' which might contain query info
            # For now, we just use status or ID as metadata
            query = f"Web Search {search_item.id}"
        else:
            # Generic fallback for unknown search types
            if hasattr(search_item, "query"):
                query = search_item.query

        if query:
            result_text = f"[Search Results for: {query}]\n{result_text}"
        else:
            result_text = f"[Search Results]\n{result_text}"

        return ant.MessageContentText(text=result_text)

    def _convert_error_output(self, error_item: Any) -> ant.MessageContentText | None:
        """Convert error output to Anthropic text format."""
        error_text = ""

        # Extract error information
        if hasattr(error_item, "message") and error_item.message:
            error_text = error_item.message
        elif hasattr(error_item, "error") and error_item.error:
            error_text = error_item.error
        elif hasattr(error_item, "text") and error_item.text:
            error_text = error_item.text

        if not error_text:
            return None

        # Add error prefix and context
        error_code = error_item.code if hasattr(error_item, "code") else None

        prefix = "[Error"
        if error_code:
            prefix += f" {error_code}"
        prefix += "]: "

        return ant.MessageContentText(text=f"{prefix}{error_text}")

    def _convert_unknown_output(
        self, unknown_item: Any
    ) -> ant.MessageContentText | None:
        """Convert unknown output type to Anthropic text format with type information."""
        item_type = unknown_item.type if hasattr(unknown_item, "type") else "unknown"

        # Try to extract any text content
        text_content = ""
        if hasattr(unknown_item, "content") and unknown_item.content:
            text_content = str(unknown_item.content)
        elif hasattr(unknown_item, "text") and unknown_item.text:
            text_content = str(unknown_item.text)
        elif hasattr(unknown_item, "data") and unknown_item.data:
            text_content = str(unknown_item.data)
        else:
            # If no obvious content, just indicate the type was encountered
            text_content = f"[Unsupported output item type: {item_type}]"

        return ant.MessageContentText(text=f"[{item_type}]: {text_content}")
