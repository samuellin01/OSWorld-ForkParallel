# pyre-strict
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from ..token.utils import get_content_str

logger: logging.Logger = logging.getLogger(__name__)


def has_attachment(msg: BaseMessage) -> bool:
    """
    Check if a message has attachments.
    """
    return msg.additional_kwargs.get("attachments") is not None


def prompt_to_convo_tag(messages: list[BaseMessage]) -> list[BaseMessage]:
    """
    Convert LC prompt value to a list of LC Messages that represents a conversation XML tag.
    """
    out_msgs: list[BaseMessage] = [HumanMessage(content="<conversation>")]
    for msg in messages:
        out_msgs[-1].content += f'\n<message role="{msg.type}">'
        if isinstance(msg, (AIMessage, SystemMessage)):
            out_msgs[-1].content += "\n" + get_content_str(msg.content)
        else:
            assert isinstance(msg, HumanMessage)
            if has_attachment(msg):
                out_msgs.append(msg)
                out_msgs.append(HumanMessage(content=""))
            else:
                out_msgs[-1].content += "\n" + get_content_str(msg.content)

        out_msgs[-1].content += "\n</message>"
    out_msgs[-1].content += "\n</conversation>"
    return out_msgs


EXCLUDE_KEYS = ["signature"]
