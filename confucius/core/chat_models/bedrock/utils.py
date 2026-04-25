# pyre-strict


from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .api.invoke_model import anthropic as ant


async def lc_message_attachment_to_ant_content(
    attachment: dict[str, Any],
) -> ant.MessageContentImage | ant.MessageContentDocument:
    mime: str = attachment.get("mime", "image/jpeg")
    data = attachment.get("data", "")
    if mime.startswith("image/"):
        return ant.MessageContentImage(
            source=ant.MessageContentImageSource(
                media_type=ant.MessageContentSourceMediaType(mime), data=data
            )
        )
    elif mime == "application/pdf":
        return ant.MessageContentDocument(source=ant.MessageContentPDFSource(data=data))
    elif mime == "text/plain":
        return ant.MessageContentDocument(
            source=ant.MessageContentPlainTextSource(data=data)
        )
    else:
        raise ValueError(f"Unsupported mime type: {mime}")


def append_stop_sequence(response: ant.Response) -> ant.Response:
    """Append the stop sequence to the response."""
    if (
        response.stop_reason != ant.StopReason.STOP_SEQUENCE
        or not response.stop_sequence
    ):
        return response

    stop_sequence = response.stop_sequence
    for i in range(len(response.content) - 1, -1, -1):
        content = response.content[i]
        if isinstance(content, ant.MessageContentText):
            content.text += stop_sequence
            break
    return response


async def lc_message_to_ant_message(msg: AIMessage | HumanMessage | SystemMessage) -> ant.Message:
    """Process a langchain message into an anthropic message."""
    content = msg.content
    ant_content: list[dict[str, Any]] = []
    if isinstance(content, str) and content:
        ant_content.append(ant.MessageContentText(text=content).dict(exclude_none=True))
    elif isinstance(content, list):
        for ct in content:
            if isinstance(ct, str) and ct:
                ant_content.append(
                    ant.MessageContentText(text=ct).dict(exclude_none=True)
                )
            elif isinstance(ct, dict):
                ant_content.append(ct)
            else:
                raise ValueError(f"Unknown content type: {type(ct)}")

    for attachment in msg.additional_kwargs.get("attachments", []):
        ant_content.append(
            (await lc_message_attachment_to_ant_content(attachment)).dict(
                exclude_none=True
            )
        )

    if isinstance(msg, AIMessage):
        role = ant.MessageRole.ASSISTANT
    elif isinstance(msg, HumanMessage):
        role = ant.MessageRole.USER
    elif isinstance(msg, SystemMessage):
        role = ant.MessageRole.DEVELOPER
    else:
        raise ValueError(
            f"Invalid message type: {type(msg)} for anthropic messages API"
        )

    return ant.Message.parse_obj(
        {
            "role": role,
            "content": ant_content,
        }
    )


def lc_message_to_ant_system(
    msg: SystemMessage,
) -> str | list[ant.MessageContentText] | None:
    """Process a langchain message into an anthropic system prompt."""
    content = msg.content
    if isinstance(content, str) and content:
        return content
    elif isinstance(content, list):
        ant_content: list[ant.MessageContentText] = []
        for ct in content:
            if isinstance(ct, str) and ct:
                ant_content.append(ant.MessageContentText(text=ct))
            elif isinstance(ct, dict):
                ant_content.append(ant.MessageContentText.parse_obj(ct))
            else:
                raise ValueError(f"Unknown content type: {type(ct)}")
        if ant_content:
            return ant_content
