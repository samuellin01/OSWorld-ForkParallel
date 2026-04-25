# pyre-strict
import logging
import json

import math
from typing import Any

from langchain_core.messages import BaseMessage

from ....core.memory import CfMessage
from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant

logger: logging.Logger = logging.getLogger(__name__)
EXCLUDE_KEYS: list[str] = ["signature", "id", "tool_use_id", "cache_control"]
# Image, document, and redacted_thinking blocks contain binary/base64 data that
# inflates character counts far beyond what their token cost reflects.  Excluding
# them keeps the chars-per-token estimate accurate and prevents the first-turn
# token estimate (used to seed last_checkpoint in BasePromptCaching.on_memory)
# from being set astronomically high when a screenshot is present.
EXCLUDE_TYPES: list[str] = ["thinking", "image", "document", "redacted_thinking"]
# Anthropic charges approximately (width * height) / 750 tokens for an image.
# For the 1280×720 screenshots used in OSWorld computer-use tasks:
#   1280 * 720 / 750 ≈ 1229 tokens per image.
# This constant is now only used as a fallback when image dimensions cannot be
# extracted from the image block's base64 data.
ESTIMATED_TOKENS_PER_IMAGE = 1334


def _serialize_data(data: Any) -> str:
    """
    Safely serialize a data to JSON string.

    Handles multiple types:
    - Pydantic models: Use model_dump_json()
    - Dicts: Use json.dumps()
    - Other types: Fallback to str()

    Args:
        data: The data to serialize

    Returns:
        JSON string representation of the data
    """
    try:
        # Try Pydantic model serialization first
        if hasattr(data, "model_dump_json"):
            return data.model_dump_json()  # pyre-ignore[16]
        # Try dict serialization
        elif isinstance(data, dict):
            return json.dumps(data)
        # Fallback to string representation
        else:
            return str(data)
    except Exception as e:
        logger.warning(f"Failed to serialize data, using string fallback: {e}")
        return str(data)

def calculate_image_tokens_from_dimensions(width: int, height: int) -> int:
    """
    Calculate image token cost using Anthropic's deterministic pixel-based formula.

    Anthropic resizes images before billing: the longest side is scaled to at most
    1568 px and the shortest side is scaled to at most 768 px (proportionally).
    Tokens are then computed as ``ceil(width * height / 750)``.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        int: Token count for the image
    """
    long_side = max(width, height)
    short_side = min(width, height)

    scale = 1.0
    if long_side > 1568:
        scale = min(scale, 1568.0 / long_side)
    if short_side > 768:
        scale = min(scale, 768.0 / short_side)

    if scale < 1.0:
        width = int(width * scale)
        height = int(height * scale)

    tokens = math.ceil(width * height / 750)

    return tokens


def get_image_dimensions_from_block(block: dict[str, Any]) -> tuple[int, int] | None:
    """
    Extract image width and height from a base64-encoded image content block.

    The block is expected to have the structure::

        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "<base64-encoded bytes>",
            },
        }

    PIL/Pillow is imported inside the function to avoid a heavy module-level
    import in contexts where image processing is not needed.

    Args:
        block: An image content block dict.

    Returns:
        ``(width, height)`` tuple, or ``None`` if dimensions cannot be extracted.
    """
    try:
        source = block.get("source", {})
        if not isinstance(source, dict):
            return None
        data_str = source.get("data", "")
        if not isinstance(data_str, str) or not data_str:
            return None

        import base64
        from io import BytesIO
        from PIL import Image  # pyre-ignore[21]

        image_bytes = base64.b64decode(data_str)
        with Image.open(BytesIO(image_bytes)) as img:
            size: tuple[int, int] = img.size
        return size
    except Exception as exc:
        logger.debug("Failed to extract image dimensions from block: %s", exc)
        return None


def calculate_image_tokens(
    messages: list[BaseMessage] | list[CfMessage],
) -> int:
    """
    Calculate the total token cost for all image blocks in a list of messages.

    Iterates over every image block — including those nested inside
    ``tool_result`` content blocks — extracts their pixel dimensions using
    :func:`get_image_dimensions_from_block`, and applies
    :func:`calculate_image_tokens_from_dimensions`.

    If dimensions cannot be extracted for an image (e.g. corrupt data),
    ``ESTIMATED_TOKENS_PER_IMAGE`` is used as a fallback for that image, and a
    warning is logged.

    Args:
        messages: The list of messages to inspect.

    Returns:
        int: Total image token cost across all messages.
    """
    total = 0
    for msg in messages:
        content = msg.content
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "image":
                dims = get_image_dimensions_from_block(item)
                if dims is not None:
                    total += calculate_image_tokens_from_dimensions(dims[0], dims[1])
                else:
                    logger.warning(
                        "Could not extract dimensions from image block; "
                        "falling back to ESTIMATED_TOKENS_PER_IMAGE=%d",
                        ESTIMATED_TOKENS_PER_IMAGE,
                    )
                    total += ESTIMATED_TOKENS_PER_IMAGE
            elif item.get("type") == "tool_result" and isinstance(
                item.get("content"), list
            ):
                for nested in item["content"]:
                    if isinstance(nested, dict) and nested.get("type") == "image":
                        dims = get_image_dimensions_from_block(nested)
                        if dims is not None:
                            total += calculate_image_tokens_from_dimensions(
                                dims[0], dims[1]
                            )
                        else:
                            logger.warning(
                                "Could not extract dimensions from nested image block; "
                                "falling back to ESTIMATED_TOKENS_PER_IMAGE=%d",
                                ESTIMATED_TOKENS_PER_IMAGE,
                            )
                            total += ESTIMATED_TOKENS_PER_IMAGE
    return total


def get_content_str(
    content: str | list[str | dict[str, Any]],
    *,
    exclude_keys: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> str:
    exclude_keys = exclude_keys or EXCLUDE_KEYS
    exclude_types = exclude_types or EXCLUDE_TYPES

    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        res = []
        for item in content:
            if isinstance(item, str):
                res.append(item)
            elif isinstance(item, dict):
                # Filter out excluded keys and types
                type_ = item.get("type")
                if type_ in exclude_types:
                    continue

                # For tool_result blocks, recursively process nested content
                # so that embedded image/document blocks are excluded from
                # character counting (same as top-level exclusions).
                if type_ == "tool_result" and isinstance(item.get("content"), list):
                    nested_str = get_content_str(
                        item["content"],
                        exclude_keys=exclude_keys,
                        exclude_types=exclude_types,
                    )
                    filtered = {k: v for k, v in item.items() if k not in exclude_keys and k != "content"}
                    if nested_str:
                        filtered["content"] = nested_str
                    res.append(_serialize_data(filtered))
                    continue

                res.append(
                    _serialize_data(
                        {k: v for k, v in item.items() if k not in exclude_keys}
                    )
                )
            else:
                raise ValueError(f"Unexpected content type: {type(item)}")
        return "\n".join(res)


async def _get_text_attachment_length(msg: BaseMessage | CfMessage) -> int:
    """
    Get the total length of text attachments in a message.
    """
    total_length = 0

    if isinstance(msg, CfMessage):
        # Approximate by summing lengths of file attachments' data/urls if present.
        for att in msg.attachments:
            try:
                content = att.content
                # Cf types union: FileAttachment | LinkAttachment | ArtifactInfoAttachment
                data = getattr(content, "data", None) or getattr(content, "url", None)
                if isinstance(data, str):
                    total_length += len(data)
            except Exception:
                # Be conservative on failures
                continue
        return total_length

    return total_length


async def get_prompt_char_lengths(
    messages: list[BaseMessage] | list[CfMessage],
    tools: list[ant.ToolLike] | None = None,
    exclude_keys: list[str] | None = None,
    exclude_types: list[str] | None = None
) -> list[int]:
    """
    Get the lengths of a prompt in characters per message. Text attachments are counted,
    but binary/base64 content blocks (image, document, redacted_thinking) are not counted
    because their character length vastly overstates their true token cost.

    Args:
        messages (list[BaseMessage]): The list of messages to get the lengths of.
        tools (list[ant.ToolLike] | None): Optional list of tools to include in the length calculation.

    Returns:
        list[int]: The lengths of the prompt in characters per message. If tools are provided,
                   the total length of all tool definitions (as JSON) is appended as the last element.
    """
    lengths = []
    for msg in messages:
        attachment_length = await _get_text_attachment_length(msg)
        lengths.append(
            len(
                get_content_str(
                    msg.content, exclude_keys=exclude_keys, exclude_types=exclude_types
                )
            )
            + attachment_length
        )

    # Validate and serialize tools if provided
    if tools is not None:
        # Type safety: ensure tools is actually a list
        if not isinstance(tools, list):
            logger.warning(
                f"tools parameter should be a list, got {type(tools).__name__}. Ignoring tools."
            )
        else:
            # Use helper to safely serialize each tool
            total_tool_length = sum(len(_serialize_data(tool)) for tool in tools)
            lengths.append(total_tool_length)

    return lengths


def count_image_blocks(
    messages: list[BaseMessage] | list[CfMessage],
) -> int:
    """
    Count the number of image content blocks across all messages.

    Mirrors the content inspection logic used by ``get_content_str`` but
    specifically counts blocks whose ``type`` is ``"image"``.

    Args:
        messages: The list of messages to inspect.

    Returns:
        The total number of image blocks found across all messages.
    """
    count = 0
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image":
                    count += 1
                # Also count images nested inside tool_result blocks
                elif isinstance(item, dict) and item.get("type") == "tool_result" and isinstance(
                    item.get("content"), list
                ):
                    for nested in item["content"]:
                        if isinstance(nested, dict) and nested.get("type") == "image":
                            count += 1

    return count


async def get_prompt_token_lengths(
    messages: list[BaseMessage] | list[CfMessage],
    num_chars_per_token: float = 3.0,
    tokens_per_image: int = ESTIMATED_TOKENS_PER_IMAGE,
    tools: list[ant.ToolLike] | None = None,
    exclude_keys: list[str] | None = None,
    exclude_types: list[str] | None = None,
) -> list[int]:
    """
    Get the lengths of a prompt in tokens per message. Text attachments are counted,
    but binary/base64 content blocks (image, document, redacted_thinking) are not counted
    in the char-based estimate; instead image tokens are added back deterministically.

    Args:
        messages (list[BaseMessage]): The list of messages to get the lengths of.
        num_chars_per_token (float, optional): The number of characters per token. Defaults to 3.0. This is a rough estimate, based on https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them.
        tokens_per_image (int, optional): Token cost per image block. Defaults to ESTIMATED_TOKENS_PER_IMAGE.
        tools (list[ant.ToolLike] | None): Optional list of tools to include in the length calculation.

    Returns:
        list[int]: The lengths of the prompt in tokens per message. If tools are provided,
                   the total token length of all tool definitions is appended as the last element.
    """
    # Use rough estimate by converting character lengths to token lengths
    char_lengths = await get_prompt_char_lengths(
        messages, tools, exclude_keys, exclude_types
    )
    token_lengths = [math.ceil(char_len / num_chars_per_token) for char_len in char_lengths]

    # Add image tokens per message (deterministic: images aren't in char_lengths)
    for i, msg in enumerate(messages):
        if i < len(token_lengths):  # don't touch the tools entry if appended
            content = msg.content
            if isinstance(content, list):
                msg_image_tokens = 0
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "image":
                        dims = get_image_dimensions_from_block(item)
                        if dims is not None:
                            msg_image_tokens += calculate_image_tokens_from_dimensions(
                                dims[0], dims[1]
                            )
                        else:
                            msg_image_tokens += tokens_per_image
                    # Also count images nested inside tool_result blocks
                    elif item.get("type") == "tool_result" and isinstance(
                        item.get("content"), list
                    ):
                        for nested in item["content"]:
                            if isinstance(nested, dict) and nested.get("type") == "image":
                                dims = get_image_dimensions_from_block(nested)
                                if dims is not None:
                                    msg_image_tokens += calculate_image_tokens_from_dimensions(
                                        dims[0], dims[1]
                                    )
                                else:
                                    msg_image_tokens += tokens_per_image
                token_lengths[i] += msg_image_tokens

    return token_lengths
