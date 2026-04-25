# pyre-strict

import base64
from typing import Any

from google.genai import types

from ..bedrock.api.invoke_model import anthropic as ant

from .exceptions import UnexpectedFinishReason


def ant_content_to_part(content: ant.MessageContent) -> types.Part:
    if isinstance(content, ant.MessageContentText):
        return types.Part(text=content.text)

    if isinstance(content, (ant.MessageContentImage, ant.MessageContentDocument)):
        source = content.source
        if isinstance(source, ant.MessageContentContentBlockSource):
            return types.Part.from_text(text=source.content)

        if source.type == "text":
            return types.Part.from_text(text=source.data)

        if source.type == "base64":
            return types.Part.from_bytes(
                data=base64.b64decode(source.data),
                mime_type=source.media_type,
            )

    if isinstance(content, ant.MessageContentToolUse):
        return types.Part(
            function_call=types.FunctionCall(
                id=content.id,
                name=content.name,
                args=content.input,
            )
        )

    if isinstance(content, ant.MessageContentToolResult):
        response = {}
        if content.is_error:
            response["error"] = str(content.content)
        else:
            response["output"] = str(content.content)

        return types.Part(
            function_response=types.FunctionResponse(
                id=content.tool_use_id,
                response=response,
                name=content.tool_use_id,  # We use the tool id as the name for now
            )
        )

    if isinstance(content, ant.MessageContentThinking):
        return types.Part(thought=True, text=content.thinking)

    raise ValueError(f"Invalid content type: {type(content)}")


def part_to_ant_content(part: types.Part) -> ant.ResponseContent | None:
    if part.text is not None:
        if part.thought:
            return ant.MessageContentThinking(
                thinking=part.text, signature=hex(hash(part.text))
            )
        else:
            return ant.MessageContentText(text=part.text)

    if part.function_call is not None:
        name = part.function_call.name or "__tool_name__"
        return ant.MessageContentToolUse(
            id=part.function_call.id or name,
            name=name,
            input=part.function_call.args or {},
        )

    return None


def ant_system_to_google(
    system: str | list[ant.MessageContentText] | None,
    # pyre-fixme[11]: Annotation `types.ContentUnion` is not defined as a type in the stub
) -> types.ContentUnion | None:
    if system is None:
        return None
    if isinstance(system, str):
        return system

    if isinstance(system, list):
        return [ant_content_to_part(content) for content in system]

    raise ValueError(f"Invalid system type: {type(system)}")


def ant_message_role_content_role(role: ant.MessageRole) -> str:
    if role == ant.MessageRole.USER:
        return "user"

    if role == ant.MessageRole.ASSISTANT:
        return "model"

    raise ValueError(f"Invalid role type: {type(role)}")


def ant_message_to_google(message: ant.Message) -> types.ContentUnion:
    return types.Content(
        role=ant_message_role_content_role(message.role),
        parts=[ant_content_to_part(content) for content in message.content],
    )


def json_schema_to_google(schema: dict[str, Any]) -> types.Schema:
    """
    Convert a JSON schema dictionary to Google GenAI Schema format.
    This function handles schemas that may contain $ref references to definitions.

    Args:
        schema: A JSON schema dictionary, typically from Pydantic's model_json_schema()

    Returns:
        A Google GenAI Schema object
    """
    # Early return for None or empty schema
    if not schema:
        return types.Schema.model_validate({})

    _remove_additional_properties(schema)

    # Extract definitions if they exist (from Pydantic model_json_schema output)
    definitions = schema.get("$defs", {})

    # Quick path: if no definitions or refs, use simple conversion
    if not definitions and "$ref" not in str(
        schema
    ):  # Fast check for $ref anywhere in schema
        return types.Schema.model_validate(schema)

    # Enhanced conversion for schemas with references
    expanded_schema = _expand_refs(schema, definitions, {})

    # Remove $defs as it's not needed anymore and not part of Google Schema
    if "$defs" in expanded_schema:
        del expanded_schema["$defs"]

    return types.Schema.model_validate(expanded_schema)


def _expand_refs(
    schema_dict: dict[str, Any],
    definitions: dict[str, Any],
    memo: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Recursively expand all $ref references in a schema dictionary.
    Uses memoization to avoid redundant processing of the same references.

    Args:
        schema_dict: A schema dictionary that may contain $ref references
        definitions: A dictionary of schema definitions to resolve references
        memo: Optional memoization cache to avoid redundant processing

    Returns:
        A new schema dictionary with all references expanded
    """
    # Initialize memoization dictionary if not provided
    if memo is None:
        memo = {}

    # Early return for non-dict values
    if not isinstance(schema_dict, dict):
        return schema_dict

    # Handle $ref by resolving the reference
    if "$ref" in schema_dict:
        ref = schema_dict["$ref"]
        # Extract the definition name from the reference
        if ref.startswith("#/$defs/"):
            def_name = ref[len("#/$defs/") :]

            # Use memoized result if available
            if def_name in memo:
                # Merge any other properties from the original schema_dict
                result = {k: v for k, v in schema_dict.items() if k != "$ref"}
                # The definition takes precedence over the original properties
                result.update(memo[def_name])
                return result

            if def_name in definitions:
                # Get the definition and expand any nested references
                definition = _expand_refs(definitions[def_name], definitions, memo)
                # Store in memo for future reuse
                memo[def_name] = definition
                # Merge any other properties from the original schema_dict
                result = {k: v for k, v in schema_dict.items() if k != "$ref"}
                # The definition takes precedence over the original properties
                result.update(definition)
                return result

    # Process each property recursively
    result = {}
    for key, value in schema_dict.items():
        if isinstance(value, dict):
            result[key] = _expand_refs(value, definitions, memo)
        elif isinstance(value, list):
            result[key] = [
                (
                    _expand_refs(item, definitions, memo)
                    if isinstance(item, dict)
                    else item
                )
                for item in value
            ]
        else:
            result[key] = value

    return result


def _remove_additional_properties(schema: dict[str, Any]) -> None:
    if isinstance(schema, dict):
        if "additionalProperties" in schema:
            del schema["additionalProperties"]
        if "additional_properties" in schema:
            del schema["additional_properties"]

        for _, value in schema.items():
            _remove_additional_properties(value)
    elif isinstance(schema, list):
        for item in schema:
            _remove_additional_properties(item)


def ant_tool_to_function_declaration(tool: ant.ToolLike) -> types.FunctionDeclaration:
    if isinstance(tool, ant.Tool):
        return types.FunctionDeclaration(
            description=tool.description,
            name=tool.name,
            parameters=json_schema_to_google(tool.input_schema),
        )

    if isinstance(tool, ant.TextEditor):
        return types.FunctionDeclaration(
            description=ant.TEXT_EDITOR_DESCRIPTION,
            name=tool.name,
            parameters=json_schema_to_google(ant.TEXT_EDITOR_SCHEMA),
        )

    if isinstance(tool, ant.BashTool):
        return types.FunctionDeclaration(
            description=ant.BASH_TOOL_DESCRIPTION,
            name=tool.name,
            parameters=json_schema_to_google(ant.BASH_TOOL_SCHEMA),
        )

    raise ValueError(f"Invalid tool type: {type(tool)}")


def ant_tools_to_google(tools: list[ant.ToolLike]) -> types.ToolListUnion:
    return [
        types.Tool(
            function_declarations=[
                ant_tool_to_function_declaration(tool) for tool in tools
            ]
        )
    ]


def ant_tool_choice_type_to_function_calling_config_mode(
    tc_type: ant.ToolChoiceType,
) -> types.FunctionCallingConfigMode:
    if tc_type == ant.ToolChoiceType.NONE:
        return types.FunctionCallingConfigMode.NONE

    if tc_type == ant.ToolChoiceType.AUTO:
        return types.FunctionCallingConfigMode.AUTO

    if tc_type == ant.ToolChoiceType.ANY:
        return types.FunctionCallingConfigMode.ANY

    return types.FunctionCallingConfigMode.MODE_UNSPECIFIED


def ant_tool_choice_to_tool_config(tool_choice: ant.ToolChoice) -> types.ToolConfig:
    return types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(
            mode=ant_tool_choice_type_to_function_calling_config_mode(tool_choice.type),
            allowed_function_names=(
                [tool_choice.name] if tool_choice.name is not None else None
            ),
        )
    )


def ant_thinking_to_thinking_config(thinking: ant.Thinking) -> types.ThinkingConfig:
    return types.ThinkingConfig(
        include_thoughts=True, thinking_budget=thinking.budget_tokens
    )


def finish_reason_to_stop_reason(
    finish_reason: types.FinishReason,
) -> ant.StopReason:
    if finish_reason == types.FinishReason.MAX_TOKENS:
        return ant.StopReason.MAX_TOKENS

    elif finish_reason == types.FinishReason.STOP:
        return ant.StopReason.END_TURN

    else:
        raise UnexpectedFinishReason(f"Unexpected finish reason: {finish_reason}")


def usage_metadata_to_usage(
    usage: types.GenerateContentResponseUsageMetadata | None,
) -> ant.Usage:
    if usage is None:
        return ant.Usage(input_tokens=0, output_tokens=0)

    return ant.Usage(
        input_tokens=usage.prompt_token_count or 0,
        output_tokens=(usage.total_token_count or 0) - (usage.prompt_token_count or 0),
        cache_read_input_tokens=usage.cached_content_token_count,
    )


def generate_content_response_to_ant_response(
    response: types.GenerateContentResponse,
) -> ant.Response:
    assert (
        response.candidates is not None and len(response.candidates) > 0
    ), "No candidates in response"
    content: list[ant.ResponseContent] = []
    candidate = response.candidates[0]
    c_cont = candidate.content
    if c_cont is not None:
        for part in c_cont.parts or []:
            ct = part_to_ant_content(part)
            if ct is not None:
                content.append(ct)

    return ant.Response(
        content=content,
        id=response.response_id or "",
        model=response.model_version or "",
        stop_reason=(
            finish_reason_to_stop_reason(candidate.finish_reason)
            if candidate.finish_reason is not None
            else None
        ),
        usage=usage_metadata_to_usage(response.usage_metadata),
    )
