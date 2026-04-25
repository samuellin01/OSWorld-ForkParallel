# pyre-strict


import enum
from typing import Any, Literal

from langchain_core.load.serializable import Serializable

from pydantic import BaseModel, Field, validator

# API doc: https://docs.anthropic.com/en/api/messages


class CacheControl(BaseModel):
    type: Literal["ephemeral"] = Field("ephemeral")
    ttl: Literal["5m", "1h"] | None = Field(
        default="5m",
        description="Cache TTL duration. '5m' uses the default 5-minute TTL. '1h' uses 1-hour TTL at 2x base input token price.",
    )


class ResponseCharLocationCitation(BaseModel):
    cited_text: str
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    document_index: int = Field(..., gt=0)
    document_title: str | None
    end_char_index: int
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    start_char_index: int = Field(..., gt=0)
    type: Literal["char_location"] = Field("char_location")


class ResponsePageLocationCitation(BaseModel):
    cited_text: str
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    document_index: int = Field(..., gt=0)
    document_title: str | None
    end_page_number: int
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    start_page_number: int = Field(..., gt=1)
    type: Literal["page_location"] = Field("page_location")


class MessageContentType(str, enum.Enum):
    TEXT = "text"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    IMAGE = "image"
    THINKING = "thinking"
    REDACTED_THINKING = "redacted_thinking"
    DOCUMENT = "document"


class MessageContentText(BaseModel):
    type: Literal[MessageContentType.TEXT] = Field(MessageContentType.TEXT)
    text: str
    citations: (
        list[ResponseCharLocationCitation | ResponsePageLocationCitation] | None
    ) = Field(default=None)
    cache_control: CacheControl | None = Field(default=None)


class MessageContentSourceMediaType(str, enum.Enum):
    IMAGE_JPEG = "image/jpeg"
    IMAGE_PNG = "image/png"
    IMAGE_GIF = "image/gif"
    IMAGE_WEBP = "image/webp"
    APPLICATION_PDF = "application/pdf"
    TEXT_PLAIN = "text/plain"


class MessageContentImageSource(BaseModel):
    type: Literal["base64"] = Field("base64")
    media_type: MessageContentSourceMediaType
    data: str

    @validator("media_type")
    def validate_media_type(
        cls,  # noqa
        value: MessageContentSourceMediaType,
    ) -> MessageContentSourceMediaType:
        if value not in [
            MessageContentSourceMediaType.IMAGE_JPEG,
            MessageContentSourceMediaType.IMAGE_PNG,
            MessageContentSourceMediaType.IMAGE_GIF,
            MessageContentSourceMediaType.IMAGE_WEBP,
        ]:
            raise ValueError(f"Invalid media type {value}. Must be image type.")
        return value


class MessageContentPDFSource(BaseModel):
    type: Literal["base64"] = Field("base64")
    media_type: Literal[MessageContentSourceMediaType.APPLICATION_PDF] = Field(
        MessageContentSourceMediaType.APPLICATION_PDF
    )
    data: str


class MessageContentPlainTextSource(BaseModel):
    type: Literal["text"] = Field("text")
    media_type: Literal[MessageContentSourceMediaType.TEXT_PLAIN] = Field(
        MessageContentSourceMediaType.TEXT_PLAIN
    )
    data: str


class MessageContentContentBlockSource(BaseModel):
    type: Literal["content"] = Field("content")
    content: str


class MessageContentImage(BaseModel):
    type: Literal[MessageContentType.IMAGE] = Field(MessageContentType.IMAGE)
    source: MessageContentImageSource
    cache_control: CacheControl | None = Field(default=None)


class MessageContentDocument(BaseModel):
    type: Literal[MessageContentType.DOCUMENT] = Field(MessageContentType.DOCUMENT)
    source: (
        MessageContentPDFSource
        | MessageContentPlainTextSource
        | MessageContentContentBlockSource
    )
    cache_control: CacheControl | None = Field(default=None)


class MessageContentToolUse(BaseModel):
    id: str
    name: str = Field(..., min_length=1, max_length=64)
    input: dict[str, Any]
    type: Literal[MessageContentType.TOOL_USE] = Field(MessageContentType.TOOL_USE)
    cache_control: CacheControl | None = Field(default=None)


class MessageContentToolResult(BaseModel):
    tool_use_id: str
    content: str | list[MessageContentText | MessageContentImage]
    type: Literal[MessageContentType.TOOL_RESULT] = Field(
        MessageContentType.TOOL_RESULT
    )
    is_error: bool | None = Field(default=None)
    cache_control: CacheControl | None = Field(default=None)


class MessageContentThinking(BaseModel):
    signature: str
    thinking: str
    type: Literal[MessageContentType.THINKING] = Field(MessageContentType.THINKING)


class MessageContentRedactedThinking(BaseModel):
    data: str
    type: Literal[MessageContentType.REDACTED_THINKING] = Field(
        MessageContentType.REDACTED_THINKING
    )


class MessageRole(str, enum.Enum):
    USER = "user"
    ASSISTANT = "assistant"
    DEVELOPER = "developer"  # for OpenAI compatibility


MessageContent = (
    MessageContentText
    | MessageContentToolUse
    | MessageContentToolResult
    | MessageContentImage
    | MessageContentDocument
    | MessageContentThinking
    | MessageContentRedactedThinking
)


class Message(BaseModel):
    role: MessageRole
    content: list[MessageContent]


class ThinkingType(str, enum.Enum):
    ENABLED = "enabled"
    DISABLED = "disabled"


class Thinking(Serializable):
    type: ThinkingType
    budget_tokens: int | None = Field(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class ToolChoiceType(str, enum.Enum):
    AUTO = "auto"
    ANY = "any"
    TOOL = "tool"
    NONE = "none"


class ToolChoice(Serializable):
    type: ToolChoiceType = Field(ToolChoiceType.AUTO)
    disable_parallel_tool_use: bool = Field(default=False)
    name: str | None = Field(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class ToolType(str, enum.Enum):
    TEXT_EDITOR_20250429 = "text_editor_20250429"
    TEXT_EDITOR_20250124 = "text_editor_20250124"
    TEXT_EDITOR_20250728 = "text_editor_20250728"
    TEXT_EDITOR_20241022 = "text_editor_20241022"
    BASH_20250124 = "bash_20250124"
    BASH_20241022 = "bash_20241022"


class ComputerToolType(str, enum.Enum):
    COMPUTER_20251124 = "computer_20251124"


class Tool(Serializable):
    name: str
    description: str | None = Field(default=None)
    input_schema: dict[str, Any]
    cache_control: CacheControl | None = Field(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class TextEditor(Serializable):
    name: Literal["str_replace_editor", "str_replace_based_edit_tool"] = Field(
        "str_replace_based_edit_tool"
    )
    type: Literal[
        ToolType.TEXT_EDITOR_20250429,
        ToolType.TEXT_EDITOR_20250124,
        ToolType.TEXT_EDITOR_20250728,
        ToolType.TEXT_EDITOR_20241022,
    ] = Field(ToolType.TEXT_EDITOR_20250728)
    cache_control: CacheControl | None = Field(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class TextEditorCommand(str, enum.Enum):
    VIEW = "view"
    CREATE = "create"
    STR_REPLACE = "str_replace"
    INSERT = "insert"
    UNDO_EDIT = "undo_edit"


class TextEditorInput(BaseModel):
    command: TextEditorCommand = Field(
        ...,
        description="The commands to run",
    )
    path: str = Field(
        ...,
        description="Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
    )
    file_text: str | None = Field(
        default=None,
        description="Required parameter of `create` command, with the content of the file to be created.",
    )
    old_str: str | None = Field(
        default=None,
        description="Required parameter of `str_replace` command containing the string in `path` to replace.",
    )
    new_str: str | None = Field(
        default=None,
        description="Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
    )
    insert_line: int | None = Field(
        default=None,
        description="Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
    )
    view_range: list[int] | None = Field(
        default=None,
        description="Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
    )


class BashTool(Serializable):
    name: Literal["bash"] = Field("bash")
    type: Literal[ToolType.BASH_20250124, ToolType.BASH_20241022] = Field(
        ToolType.BASH_20250124
    )
    cache_control: CacheControl | None = Field(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


class BashInput(BaseModel):
    command: str | None = Field(
        default=None,
        description="The bash command to run. Required unless the tool is being restarted.",
    )
    restart: bool = Field(
        default=False,
        description="Specifying true will restart this tool. Otherwise, leave this unspecified.",
    )


class ComputerTool(Serializable):
    name: Literal["computer"] = Field("computer")
    type: ComputerToolType = Field(ComputerToolType.COMPUTER_20251124)
    display_width_px: int = Field(default=1280)
    display_height_px: int = Field(default=720)
    display_number: int = Field(default=1)
    cache_control: CacheControl | None = Field(default=None)

    @classmethod
    def is_lc_serializable(cls) -> bool:
        return True


ToolLike = Tool | TextEditor | BashTool | ComputerTool


class Body(BaseModel):
    max_tokens: int = Field(default=512)
    messages: list[Message] = Field(default=[])
    stop_sequences: list[str] | None = Field(default=None)
    system: str | list[MessageContentText] | None = Field(default=None)
    temperature: float | None = Field(default=None)
    thinking: Thinking | None = Field(default=None)
    tool_choice: ToolChoice | None = Field(default=None)
    tools: list[ToolLike] | None = Field(default=None)
    top_p: float | None = Field(default=None)
    anthropic_version: str = Field(...)
    anthropic_beta: list[str] | None = Field(default=None)


class Usage(BaseModel):
    cache_creation_input_tokens: int | None = Field(default=None)
    cache_read_input_tokens: int | None = Field(default=None)
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    input_tokens: int = Field(..., ge=0)
    # pyre-fixme[6]: Incompatible parameter type [6]: T220454231
    output_tokens: int = Field(..., ge=0)


ResponseContent = (
    MessageContentText
    | MessageContentToolUse
    | MessageContentThinking
    | MessageContentRedactedThinking
)


class StopReason(str, enum.Enum):
    END_TURN = "end_turn"
    MAX_TOKENS = "max_tokens"
    STOP_SEQUENCE = "stop_sequence"
    TOOL_USE = "tool_use"
    PAUSE_TURN = "pause_turn"
    REFUSAL = "refusal"


class Response(BaseModel):
    content: list[ResponseContent]
    id: str
    model: str
    role: Literal[MessageRole.ASSISTANT] = Field(MessageRole.ASSISTANT)
    stop_reason: StopReason | None
    stop_sequence: str | None = Field(default=None)
    type: Literal["message"] = Field("message")
    usage: Usage
    metadata: dict[str, Any] | None = Field(default=None)


# https://docs.anthropic.com/en/docs/agents-and-tools/computer-use#text-editor-tool

TEXT_EDITOR_DESCRIPTION: str = """\
Custom editing tool for viewing, creating and editing files
* State is persistent across command calls and discussions with the user
* If `path` is a file, `view` displays the result of applying `cat -n`. If `path` is a directory, `view` lists non-hidden files and directories up to 2 levels deep
* The `create` command cannot be used if the specified `path` already exists as a file
* If a `command` generates a long output, it will be truncated and marked with `<response clipped>`
* The `undo_edit` command will revert the last edit made to the file at `path`

Notes for using the `str_replace` command:
* The `old_str` parameter should match EXACTLY one or more consecutive lines from the original file. Be mindful of whitespaces!
* If the `old_str` parameter is not unique in the file, the replacement will not be performed. Make sure to include enough context in `old_str` to make it unique
* The `new_str` parameter should contain the edited lines that should replace the `old_str`
"""

TEXT_EDITOR_SCHEMA: dict[str, Any] = {
    "properties": {
        "command": {
            "description": "The commands to run. Allowed options are: `view`, `create`, `str_replace`, `insert`, `undo_edit`.",
            "enum": ["view", "create", "str_replace", "insert", "undo_edit"],
            "type": "string",
        },
        "file_text": {
            "description": "Required parameter of `create` command, with the content of the file to be created.",
            "type": "string",
        },
        "insert_line": {
            "description": "Required parameter of `insert` command. The `new_str` will be inserted AFTER the line `insert_line` of `path`.",
            "type": "integer",
        },
        "new_str": {
            "description": "Optional parameter of `str_replace` command containing the new string (if not given, no string will be added). Required parameter of `insert` command containing the string to insert.",
            "type": "string",
        },
        "old_str": {
            "description": "Required parameter of `str_replace` command containing the string in `path` to replace.",
            "type": "string",
        },
        "path": {
            "description": "Absolute path to file or directory, e.g. `/repo/file.py` or `/repo`.",
            "type": "string",
        },
        "view_range": {
            "description": "Optional parameter of `view` command when `path` points to a file. If none is given, the full file is shown. If provided, the file will be shown in the indicated line number range, e.g. [11, 12] will show lines 11 and 12. Indexing at 1 to start. Setting `[start_line, -1]` shows all lines from `start_line` to the end of the file.",
            "items": {"type": "integer"},
            "type": "array",
        },
    },
    "required": ["command", "path"],
    "type": "object",
}

BASH_TOOL_DESCRIPTION: str = """\
Run commands in a bash shell
* When invoking this tool, the contents of the "command" parameter does NOT need to be XML-escaped.
* You have access to a mirror of common linux and python packages via apt and pip.
* State is persistent across command calls and discussions with the user.
* To inspect a particular line range of a file, e.g. lines 10-25, try 'sed -n 10,25p /path/to/the/file'.
* Please avoid commands that may produce a very large amount of output.
* Please run long lived commands in the background, e.g. 'sleep 10 &' or start a server in the background.
"""

BASH_TOOL_SCHEMA: dict[str, Any] = {
    "properties": {
        "command": {
            "description": "The bash command to run. Required unless the tool is being restarted.",
            "type": "string",
        },
        "restart": {
            "description": "Specifying true will restart this tool. Otherwise, leave this unspecified.",
            "type": "boolean",
        },
    },
    "type": "object",
}
