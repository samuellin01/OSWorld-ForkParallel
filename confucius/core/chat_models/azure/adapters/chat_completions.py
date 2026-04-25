# pyre-strict

"""Chat Completions API Adapter

This module provides the ChatCompletionsAdapter for OpenAI's chat.completions API.
It handles basic text, image, and tool call functionality.
"""

import json
from typing import Any, List, Optional, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from openai import NOT_GIVEN
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionContentPartParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
)
from openai.types.chat.chat_completion_assistant_message_param import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageToolCallUnionParam,
    ContentArrayOfContentPart,
)
from openai.types.chat.chat_completion_content_part_image_param import (
    ChatCompletionContentPartImageParam,
    ImageURL,
)

from openai.types.chat.chat_completion_content_part_text_param import (
    ChatCompletionContentPartTextParam,
)
from openai.types.chat.chat_completion_message_custom_tool_call import (
    ChatCompletionMessageCustomToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_named_tool_choice_param import (
    ChatCompletionNamedToolChoiceParam,
)
from openai.types.chat.chat_completion_system_message_param import (
    ChatCompletionSystemMessageParam,
)
from openai.types.chat.chat_completion_tool_choice_option_param import (
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.chat_completion_tool_message_param import (
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.chat.chat_completion_user_message_param import (
    ChatCompletionUserMessageParam,
)
from openai.types.shared.reasoning_effort import ReasoningEffort
from openai.types.shared_params.function_definition import FunctionDefinition

from ...bedrock.api.invoke_model import anthropic as ant
from ...bedrock.utils import lc_message_to_ant_message, lc_message_to_ant_system

from ..base import OpenAIBase
from ..model import get_model


def ant_content_to_openai_content(
    content: ant.MessageContent,
) -> ChatCompletionContentPartParam:
    """Convert Anthropic message content to OpenAI content format."""
    if isinstance(content, ant.MessageContentText):
        return ChatCompletionContentPartTextParam(type="text", text=content.text)

    if isinstance(content, ant.MessageContentToolUse):
        # Note: Tool use is handled separately in ant_tools_to_openai_tools
        # This is just a fallback to represent it as text
        return ChatCompletionContentPartTextParam(
            type="text", text=f"[Tool Use: {content.name}({content.input})]"
        )

    if isinstance(content, ant.MessageContentImage):
        source = content.source
        if source.type == "base64":
            return ChatCompletionContentPartImageParam(
                type="image_url",
                image_url=ImageURL(
                    url=f"data:{source.media_type};base64,{source.data}"
                ),
            )

    if isinstance(content, ant.MessageContentThinking):
        # OpenAI doesn't have a direct equivalent for thinking steps
        # We'll convert it to text with a prefix
        return ChatCompletionContentPartTextParam(
            type="text", text=f"[Thinking] {content.thinking}"
        )

    # For unsupported content types, return a placeholder
    return ChatCompletionContentPartTextParam(
        type="text", text=f"[Unsupported content type: {str(content.type)}]"
    )


def ant_tool_to_openai_tool(tool: ant.ToolLike) -> ChatCompletionToolParam:
    """Convert an Anthropic tool to OpenAI tool format."""
    if isinstance(tool, ant.TextEditor):
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=ant.TEXT_EDITOR_DESCRIPTION,
                parameters=ant.TEXT_EDITOR_SCHEMA,
            ),
        )
    elif isinstance(tool, ant.BashTool):
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=ant.BASH_TOOL_DESCRIPTION,
                parameters=ant.BASH_TOOL_SCHEMA,
            ),
        )
    elif isinstance(tool, ant.Tool):
        return ChatCompletionToolParam(
            type="function",
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.input_schema,
            ),
        )
    else:
        raise ValueError(f"Unsupported tool type: {type(tool)}")


def ant_tools_to_openai_tools(
    tools: List[ant.ToolLike],
) -> List[ChatCompletionToolParam]:
    """Convert a list of Anthropic tools to OpenAI tools format."""
    return [ant_tool_to_openai_tool(tool) for tool in tools]


def _normalize_content_parts(
    content: Union[str, List[ant.MessageContent]],
) -> List[ant.MessageContent]:
    """Normalize any content format to a list of content parts."""
    if isinstance(content, str):
        return [ant.MessageContentText(text=content)]
    elif isinstance(content, list):
        return content
    else:
        return [ant.MessageContentText(text=str(content))]


def _create_tool_result_message(
    part: ant.MessageContentToolResult,
) -> ChatCompletionToolMessageParam:
    """Create a tool result message from an Anthropic tool result."""
    tool_content = part.content if isinstance(part.content, str) else str(part.content)
    tool_content_parts = [
        ChatCompletionContentPartTextParam(type="text", text=tool_content)
    ]
    return ChatCompletionToolMessageParam(
        role="tool",
        content=tool_content_parts,
        tool_call_id=part.tool_use_id,
    )


def _create_tool_call(
    part: ant.MessageContentToolUse,
) -> ChatCompletionMessageToolCallUnionParam:
    """Create a tool call from an Anthropic tool use."""
    return {
        "id": part.id,
        "type": "function",
        "function": {
            "name": part.name,
            "arguments": json.dumps(part.input),
        },
    }


def _process_user_message(
    content_parts: List[ant.MessageContent],
) -> List[ChatCompletionMessageParam]:
    """Process content parts for a user message."""
    messages: List[ChatCompletionMessageParam] = []
    current_user_content: List[ChatCompletionContentPartParam] = []

    def _flush_user_content() -> None:
        """Flush accumulated user content as a user message."""
        if current_user_content:
            messages.append(
                ChatCompletionUserMessageParam(
                    role="user", content=current_user_content.copy()
                )
            )
            current_user_content.clear()

    for part in content_parts:
        _flush_user_content()
        if isinstance(part, ant.MessageContentToolResult):
            messages.append(_create_tool_result_message(part))
        elif isinstance(part, ant.MessageContentToolUse):
            raise ValueError(
                "Tool use found in user message - tool use can only come from assistant messages"
            )
        else:
            converted_part = ant_content_to_openai_content(part)
            current_user_content.append(converted_part)

    # Add user message if we have regular content
    _flush_user_content()

    return messages


def _process_assistant_message(
    content_parts: List[ant.MessageContent],
) -> List[ChatCompletionAssistantMessageParam]:
    """Process content parts for an assistant message.

    Tool use becomes tool_calls within the main assistant message.
    Returns a list with typically one assistant message.
    """
    assistant_content: List[ContentArrayOfContentPart] = []
    tool_calls: List[ChatCompletionMessageToolCallUnionParam] = []

    for part in content_parts:
        if isinstance(part, ant.MessageContentToolResult):
            # Tool results should NOT appear in assistant messages
            raise ValueError(
                "Tool result found in assistant message - tool results can only come from user messages"
            )
        elif isinstance(part, ant.MessageContentToolUse):
            # Tool use can only come from assistant messages - becomes tool_calls
            tool_calls.append(_create_tool_call(part))
        else:
            # Regular content (text, thinking) - images not allowed for assistants
            converted_part = ant_content_to_openai_content(part)
            if converted_part.get("type") == "text":
                assistant_content.append(converted_part)  # type: ignore [arg-type]
            # Skip non-text content for assistants (like images)

    main_message = ChatCompletionAssistantMessageParam(
        role="assistant", content=assistant_content
    )
    if tool_calls:
        main_message["tool_calls"] = tool_calls

    return [main_message]


def ant_message_to_openai(message: ant.Message) -> List[ChatCompletionMessageParam]:
    """Convert an Anthropic message to OpenAI message format.

    Always returns a list of messages to maintain consistent API.
    Maintains 1:1 mapping between Anthropic content parts and OpenAI content parts.
    """
    content_parts = _normalize_content_parts(message.content)

    if message.role == ant.MessageRole.USER:
        result_messages = _process_user_message(content_parts)
    elif message.role == ant.MessageRole.ASSISTANT:
        result_messages = _process_assistant_message(content_parts)
    else:
        raise ValueError(f"Unsupported Anthropic message role: {message.role}")

    # Type cast to satisfy the broader ChatCompletionMessageParam union
    result: List[ChatCompletionMessageParam] = result_messages  # type: ignore [misc]
    return result


def ant_system_to_openai(
    system: str | list[ant.MessageContentText] | None,
) -> ChatCompletionMessageParam:
    """Convert Anthropic system message to OpenAI system message format."""
    if system is None:
        return ChatCompletionSystemMessageParam(role="system", content="")

    if isinstance(system, str):
        return ChatCompletionSystemMessageParam(role="system", content=system)

    # For list of MessageContentText, concatenate all texts
    if isinstance(system, list):
        system_text = " ".join(item.text for item in system)
        return ChatCompletionSystemMessageParam(role="system", content=system_text)

    raise ValueError(f"Invalid system type: {type(system)}")


def _extract_tool_calls(
    message: ChatCompletionMessage,
) -> List[ant.MessageContentToolUse]:
    """Extract tool calls from an OpenAI message."""
    tool_uses: List[ant.MessageContentToolUse] = []

    # Handle tool_calls if present
    tool_calls = message.tool_calls
    if tool_calls is not None and len(tool_calls) > 0:
        for tool_call in tool_calls:
            # Parse the arguments string to a dict
            if isinstance(tool_call, ChatCompletionMessageCustomToolCall):
                tool_uses.append(
                    ant.MessageContentToolUse(
                        id=tool_call.id,
                        name=tool_call.custom.name,
                        input=json.loads(tool_call.custom.input),
                    )
                )
            elif isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                tool_uses.append(
                    ant.MessageContentToolUse(
                        id=tool_call.id,
                        name=tool_call.function.name,
                        input=json.loads(tool_call.function.arguments),
                    )
                )
            else:
                raise ValueError(f"Unsupported type of ToolCall: {type(tool_call)}")

    # Handle function_call if present (older OpenAI API format)
    function_call = message.function_call
    if function_call is not None:
        tool_uses.append(
            ant.MessageContentToolUse(
                id=function_call.name,  # Use name as ID since function_call doesn't have an ID
                name=function_call.name,
                input=json.loads(function_call.arguments),
            )
        )

    return tool_uses


def _extract_text_content(
    message: ChatCompletionMessage,
) -> List[ant.MessageContentText]:
    """Extract text content from an OpenAI message."""
    text_contents = []

    # Handle string content
    if isinstance(message.content, str) and message.content:
        text_contents.append(ant.MessageContentText(text=message.content))

    # Handle list content
    elif isinstance(message.content, list):
        for item in message.content:
            if item.get("type") == "text" and item.get("text"):
                text_contents.append(ant.MessageContentText(text=item["text"]))

    return text_contents


def _map_finish_reason_to_stop_reason(
    finish_reason: str,
) -> ant.StopReason:
    """Map OpenAI finish reason to Anthropic stop reason."""

    if finish_reason == "stop":
        return ant.StopReason.END_TURN
    elif finish_reason == "length":
        return ant.StopReason.MAX_TOKENS
    elif finish_reason == "tool_calls" or finish_reason == "function_call":
        return ant.StopReason.TOOL_USE
    else:
        # No direct mapping, return END_TURN
        return ant.StopReason.END_TURN


def openai_response_to_ant_response(
    response: ChatCompletion,
) -> ant.Response:
    """Convert an OpenAI chat completion response to Anthropic response format."""
    # Ensure there's at least one choice
    if not response.choices or len(response.choices) == 0:
        raise ValueError("No choices in OpenAI response")

    choice = response.choices[0]
    message = choice.message
    content: List[ant.ResponseContent] = []

    # Extract tool calls if present
    tool_uses = _extract_tool_calls(message)
    if tool_uses:
        content.extend(tool_uses)

    # Extract text content
    text_contents = _extract_text_content(message)
    if text_contents:
        content.extend(text_contents)

    # If no content was extracted, add an empty text content
    if not content:
        content.append(ant.MessageContentText(text=""))

    # Calculate token usage
    usage = response.usage
    input_tokens = usage.prompt_tokens if usage is not None else 0
    output_tokens = usage.completion_tokens if usage is not None else 0

    # Map finish reason to stop reason
    stop_reason = _map_finish_reason_to_stop_reason(choice.finish_reason)

    return ant.Response(
        content=content,
        id=response.id,
        model=response.model,
        stop_reason=stop_reason,
        usage=ant.Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


async def convert_lc_messages_to_openai_messages(
    messages: list[BaseMessage],
) -> list[ChatCompletionMessageParam]:
    """Convert Langchain messages to OpenAI messages format."""
    openai_messages: list[ChatCompletionMessageParam] = []
    for msg in messages:
        if isinstance(msg, (HumanMessage, AIMessage)):
            ant_msg = await lc_message_to_ant_message(msg)
            converted = ant_message_to_openai(ant_msg)
            openai_messages.extend(converted)
        elif isinstance(msg, SystemMessage):
            ant_system = lc_message_to_ant_system(msg)
            openai_messages.append(ant_system_to_openai(ant_system))
        else:
            raise ValueError(f"Invalid message type: {type(msg)} for OpenAI API")

    return openai_messages


def prepare_openai_tool_choice(
    tools: list[ant.ToolLike] | None, tool_choice: ant.ToolChoice | None
) -> tuple[
    Optional[List[ChatCompletionToolParam]],
    Optional[ChatCompletionToolChoiceOptionParam],
]:
    """Prepare OpenAI tools and tool_choice from Anthropic format."""
    openai_tools: Optional[List[ChatCompletionToolParam]] = None
    openai_tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None

    if tools is not None and tools:
        openai_tools = ant_tools_to_openai_tools(tools)

        if tool_choice is not None:
            tool_choice_type = tool_choice.type
            if tool_choice_type == ant.ToolChoiceType.AUTO:
                openai_tool_choice = "auto"  # type: ignore [assignment]
            elif tool_choice_type == ant.ToolChoiceType.NONE:
                openai_tool_choice = "none"  # type: ignore [assignment]
            elif tool_choice_type == ant.ToolChoiceType.ANY:
                openai_tool_choice = "required"  # type: ignore [assignment]
            elif tool_choice_type == ant.ToolChoiceType.TOOL:
                tool_name = getattr(tool_choice, "name", None)
                if tool_name is not None:
                    openai_tool_choice = ChatCompletionNamedToolChoiceParam(
                        type="function",
                        function={"name": tool_name},
                    )
            else:
                # Add catch-all else case as requested in comment
                raise ValueError(f"Unknown tool choice type: {tool_choice_type}")

    return openai_tools, openai_tool_choice  # type: ignore [return-value]


def is_thinking_model(model: str) -> bool:
    return any(keyword in model for keyword in ["o1", "o3", "o4", "gpt-5"])


def ant_thinking_to_reasoning_effort(
    thinking: ant.Thinking | None,
) -> ReasoningEffort | None:
    """Convert Anthropic thinking to OpenAI reasoning effort."""
    if thinking is None or thinking.type == ant.ThinkingType.DISABLED:
        return None

    budget_tokens = thinking.budget_tokens
    if budget_tokens is None:
        return None
    if budget_tokens <= 2048:
        return "minimal"
    if budget_tokens <= 8192:
        return "low"
    if budget_tokens <= 16384:
        return "medium"
    return "high"


class ChatCompletionsAdapter(OpenAIBase):
    """Specialized adapter for OpenAI chat.completions API.

    This adapter handles:
    - Basic text content
    - Image content (input only)
    - Tool calls and tool results
    - Thinking content (converted to text)
    """

    async def _invoke_api(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> ChatCompletion:
        """Invoke the chat.completions API."""
        # Convert messages to OpenAI format
        openai_messages = await convert_lc_messages_to_openai_messages(messages)

        # Prepare tools and tool_choice
        openai_tools, openai_tool_choice = prepare_openai_tool_choice(
            self.tools, self.tool_choice
        )
        model = get_model(self.model)
        is_thinking = is_thinking_model(model)

        # List of unsupported params for thinking model: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/reasoning?tabs=python-secure%2Cpy#not-supported
        response = await self.client.chat.completions.create(
            messages=openai_messages,
            model=model,
            top_p=self.top_p if not is_thinking else NOT_GIVEN,
            frequency_penalty=self.frequency_penalty if not is_thinking else NOT_GIVEN,
            temperature=self.temperature if not is_thinking else NOT_GIVEN,
            max_tokens=self.max_tokens if not is_thinking else NOT_GIVEN,
            max_completion_tokens=self.max_tokens if is_thinking else NOT_GIVEN,
            tools=openai_tools or NOT_GIVEN,
            tool_choice=openai_tool_choice or NOT_GIVEN,
            reasoning_effort=ant_thinking_to_reasoning_effort(self.thinking),
        )
        return response

    def _convert_response(self, raw_response: ChatCompletion) -> ant.Response:
        """Convert chat completion response to Anthropic format."""
        return openai_response_to_ant_response(raw_response)
