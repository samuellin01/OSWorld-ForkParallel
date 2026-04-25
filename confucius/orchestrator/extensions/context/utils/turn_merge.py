# pyre-strict

"""Turn merging utilities for smart context management.

This module provides helper functions for merging consecutive fully-ignored
turns into a single compact system message to save tokens.

A "turn" is defined as a sequence starting with an AI message (which could be
thinking/text) followed by more AI messages and HUMAN messages (tool results).
When all tool uses in consecutive turns have been ignored by context management,
these turns can be merged into a single system message summarizing what was cleared.
"""

import logging
from collections import Counter
from collections.abc import Callable

from .....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from .....core import types as cf
from .....core.memory import CfMemoryManager, CfMessage
from ....tags import Tag
from pydantic import BaseModel, Field

logger: logging.Logger = logging.getLogger(__name__)

# Flag to mark messages that have been merged
MERGED_TURN_KEY = "context_management_merged_turn"


def _has_tool_uses(msg: CfMessage) -> bool:
    """Check if message contains tool uses (AI) or tool results (HUMAN).

    Args:
        msg: The message to check

    Returns:
        True if message has tool uses (AI) or tool results (HUMAN)
    """
    if not isinstance(msg.content, list):
        return False

    for item in msg.content:
        if msg.type == cf.MessageType.AI:
            try:
                ant.MessageContentToolUse.model_validate(item)
                return True
            except Exception:
                continue
        elif msg.type == cf.MessageType.HUMAN:
            try:
                ant.MessageContentToolResult.model_validate(item)
                return True
            except Exception:
                continue
    return False


def _extract_tool_uses(msg: CfMessage) -> list[ant.MessageContentToolUse]:
    """Extract all tool_use objects from an AI message.

    Args:
        msg: The AI message to extract tool uses from

    Returns:
        List of ToolUse objects (can access .id and .name)
    """
    tool_uses: list[ant.MessageContentToolUse] = []

    if msg.type != cf.MessageType.AI or not isinstance(msg.content, list):
        return tool_uses

    for item in msg.content:
        try:
            tool_use = ant.MessageContentToolUse.model_validate(item)
            tool_uses.append(tool_use)
        except Exception:
            continue

    return tool_uses


class Turn(BaseModel):
    """Represents a complete tool-use turn in conversation.

    A turn:
    - STARTS when AI message follows HUMAN/SYS (or is first message)
    - CONTINUES when AI message follows AI (consecutive AI messages)
    - CONTAINS AI messages + HUMAN tool results + SYS messages
    - ENDS when next AI message follows a HUMAN/SYS message
    """

    messages: list[CfMessage] = Field(default_factory=list)
    start_idx: int = 0
    end_idx: int = 0  # exclusive

    @property
    def has_tool_uses(self) -> bool:
        """Check if turn contains any tool use OR tool result.

        Returns:
            True if this is a "tool turn" (has tool-related content).
            False if just regular conversation (thinking/text/questions).
        """
        return any(_has_tool_uses(msg) for msg in self.messages)

    def is_fully_ignored(
        self,
        is_ignored_fn: Callable[[CfMessage], bool],
    ) -> bool:
        """Check if all tool-related messages in this turn are ignored.

        Only checks messages with tool uses or tool results.
        Thinking/text messages are ignored for this check.

        Args:
            is_ignored_fn: Function to check if a message is ignored

        Returns:
            True if ALL tool_use/tool_result messages are ignored
        """
        tool_messages = [msg for msg in self.messages if _has_tool_uses(msg)]
        if not tool_messages:
            return False  # No tool messages = not a tool turn
        return all(is_ignored_fn(msg) for msg in tool_messages)


def group_messages_into_turns(
    messages: list[CfMessage],
) -> list[Turn]:
    """Group messages into turns.

    A turn:
    - STARTS when AI message follows HUMAN/SYS (or is first message)
    - CONTINUES when AI message follows AI (consecutive AI messages)
    - CONTAINS AI messages + HUMAN tool results + SYS messages
    - ENDS when next AI message follows a HUMAN/SYS message

    Example:
        HUMAN: text              <- skipped (before first AI)
        --- turn 1 ---
        AI: thinking             <- new turn (AI after HUMAN)
        AI: text                 <- same turn (AI after AI)
        AI: tool_use_1           <- same turn (AI after AI)
        AI: tool_use_2           <- same turn (AI after AI)
        HUMAN: tool_result_1     <- same turn
        HUMAN: tool_result_2     <- same turn
        --- turn 2 ---
        AI: thinking             <- new turn (AI after HUMAN)
        AI: text                 <- same turn (AI after AI)
        AI: tool_use_3           <- same turn (AI after AI)
        HUMAN: tool_result_3     <- same turn
        --- turn 3 ---
        AI: thinking             <- new turn (AI after HUMAN)
        AI: text                 <- same turn (AI after AI)
        [end]

    Args:
        messages: List of conversation messages

    Returns:
        List of Turn objects representing conversation turns
    """
    turns: list[Turn] = []
    current_turn: Turn | None = None
    prev_msg_type: cf.MessageType | None = None

    for idx, msg in enumerate(messages):
        # Skip already merged messages
        if msg.additional_kwargs.get(MERGED_TURN_KEY):
            continue

        if msg.type == cf.MessageType.AI:
            if current_turn is None or prev_msg_type != cf.MessageType.AI:
                # New turn: first AI, or AI after HUMAN/SYS
                if current_turn is not None:
                    current_turn.end_idx = idx
                    turns.append(current_turn)
                current_turn = Turn(start_idx=idx, messages=[msg])
            else:
                # Same turn: consecutive AI messages
                current_turn.messages.append(msg)

        else:  # HUMAN or SYS
            if current_turn is not None:
                current_turn.messages.append(msg)
            # Skip if no current turn (messages before first AI)

        prev_msg_type = msg.type

    # Handle final turn
    if current_turn is not None:
        current_turn.end_idx = len(messages)
        turns.append(current_turn)

    return turns


def find_mergeable_turn_groups(
    turns: list[Turn],
    is_ignored_fn: Callable[[CfMessage], bool],
) -> list[list[Turn]]:
    """Find consecutive groups of fully-ignored tool turns.

    Args:
        turns: List of turns to analyze
        is_ignored_fn: Function to check if a message is ignored

    Returns:
        List of turn groups, where each group contains consecutive fully-ignored
        tool turns that can be merged together
    """
    groups: list[list[Turn]] = []
    current_group: list[Turn] = []

    for turn in turns:
        # Skip non-tool turns entirely
        if not turn.has_tool_uses:
            # Flush current group if any
            if current_group:
                groups.append(current_group)
                current_group = []
            continue

        # Check if this tool turn is fully ignored
        if turn.is_fully_ignored(is_ignored_fn):
            current_group.append(turn)
        else:
            # Not fully ignored - flush current group
            if current_group:
                groups.append(current_group)
                current_group = []

    if current_group:
        groups.append(current_group)

    return groups


def create_merge_summary_content(
    turns: list[Turn],
) -> str:
    """Create summary content string for merged turns.

    Args:
        turns: List of turns to summarize

    Returns:
        Formatted summary string wrapped in system tag
    """
    # Count tool calls by name
    tool_counts = Counter(
        tool_use.name
        for turn in turns
        for msg in turn.messages
        for tool_use in _extract_tool_uses(msg)
    )

    # Format as "name (×count)" for multiple calls
    parts = []
    for name, count in sorted(tool_counts.items()):
        if count > 1:
            parts.append(f"{name} (×{count})")
        else:
            parts.append(name)

    # Use Tag helper for consistency with existing patterns
    content = Tag(
        name="system",
        contents=f"Function calls cleared: {', '.join(parts)}",
    ).prettify()

    return content


def replace_turns_with_summary(
    memory_manager: CfMemoryManager,
    turns: list[Turn],
    summary_content: str,
) -> None:
    """Replace turn group with single summary message.

    Strategy:
    1. Take the FIRST message of the first turn
    2. Convert it to a SYS message with summary content
    3. Delete ALL other messages in the group

    This preserves:
    - Message ordering (via sequence_id of first message)
    - Memory structure integrity

    Args:
        memory_manager: The memory manager to modify
        turns: List of turns to replace
        summary_content: The summary text to use
    """
    if not turns:
        return

    # Get all messages to process
    all_messages_in_group: list[CfMessage] = []
    for turn in turns:
        all_messages_in_group.extend(turn.messages)

    if not all_messages_in_group:
        return

    # 1. Modify FIRST message to become summary
    first_msg = all_messages_in_group[0]
    first_msg.type = cf.MessageType.SYS
    first_msg.content = summary_content
    first_msg.additional_kwargs[MERGED_TURN_KEY] = True  # Mark as merged

    # 2. Delete ALL OTHER messages
    for msg in all_messages_in_group[1:]:
        memory_manager.delete_message(msg)


def merge_fully_ignored_turns(
    memory_manager: CfMemoryManager,
    is_ignored_fn: Callable[[CfMessage], bool],
) -> None:
    """Main entry point: Merge all fully-ignored turn groups in memory.

    This function is called from extension's on_memory callback.

    Args:
        memory_manager: The memory manager to process
        is_ignored_fn: Function to check if a message is ignored
    """
    # 1. Group messages into turns
    messages = memory_manager.get_session_memory().messages
    turns = group_messages_into_turns(messages)

    # 2. Find consecutive fully-ignored turn groups
    mergeable_groups = find_mergeable_turn_groups(turns, is_ignored_fn)

    # 3. For each group, create summary message and replace originals
    # Process in reverse order to preserve indices
    for group in reversed(mergeable_groups):
        summary_content = create_merge_summary_content(group)
        replace_turns_with_summary(memory_manager, group, summary_content)
