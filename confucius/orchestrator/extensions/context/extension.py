# pyre-strict

"""Smart LLM-driven context management extension."""

import asyncio
import datetime
import json
import logging
from enum import Enum
from typing import Any, Awaitable, Callable, cast, override
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, PrivateAttr
from ....core.analect import AnalectRunContext, get_current_context
from ....core.chat_models.bedrock.api.invoke_model import anthropic as ant
from ....core import types as cf
from ....core.memory import CfMessage
from ....core.llm_manager import LLMParams
from ...exceptions import OrchestratorInterruption
from ...base import BaseOrchestrator
from ..token.estimator import TokenEstimatorExtension
from ..tool_use import ToolUseExtension, ToolUseObserver
from ..caching.base import BasePromptCaching, CACHE_BREAKPOINT_KEY
from ..token.utils import calculate_image_tokens_from_dimensions, get_image_dimensions_from_block
from ...tags import Tag
from .prompts import (
    AUTO_FORCE_MESSAGE_TEMPLATE,
    CONTEXT_EDIT_TOOL_NAME,
    CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE,
    CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE_TEMPLATE,
    CONTEXT_MANAGEMENT_REMINDER_MESSAGE_TEMPLATE,
    get_context_edit_tool_description,
)
from .utils.turn_merge import merge_fully_ignored_turns

context_edit_logger = logging.getLogger("desktopenv.context_edit")

# Memory tool names used for context management exemption
MEMORY_TOOL_NAMES: list[str] = ["write_memory", "read_memory", "delete_memory", "edit_memory", "import_memory", "search_memory"]

SYS_INFO_TOOL_USE_ID_KEY = "system_info_tool_use_id"
PENDING_DATA_KEY = "pending_data"
# Marker placed in tool_use input when a tool use has been cleared by context management
CLEARED_TOOL_USE_KEY = "_context_management_cleared"


class ContextEditOperation(str, Enum):
    """Operations for context editing."""

    IGNORE = "IGNORE"
    COMPACT = "COMPACT"


class ContextEdit(BaseModel):
    """Single context edit specification."""

    tool_use_id: str = Field(
        description="ID of tool use / function call to edit (e.g., 'toolu_bdrk_abc123', 'call_xyz123'), as shown in the <system_info> tag AFTER each function call result"
    )

    operation: ContextEditOperation = Field(
        description=(
            "IGNORE: Remove tool use entirely from context. "
            "COMPACT: Replace with summary to save tokens."
        )
    )

    tool_use_input_replacement: dict[str, Any] | None = Field(
        default=None,
        description="For COMPACT: Summarized tool input to replace verbose original",
    )

    tool_result_content_replacement: str | None = Field(
        default=None,
        description="For COMPACT: Summary to replace verbose tool result",
    )

    reason: str | None = Field(
        default=None, description="Why this edit is needed (for debugging/logging)"
    )


class ContextEditInput(BaseModel):
    """Input schema for context_edit tool."""

    edits: list[ContextEdit] = Field(description="List of context edits to apply")


class RejectedEdit(BaseModel):
    """An individual edit that was rejected during validation."""

    edit: ContextEdit
    reason: str


class ReplacementInfo(BaseModel):
    """Information about an edit that replaced a pending edit."""

    tool_use_id: str
    old_operation: ContextEditOperation
    new_operation: ContextEditOperation
    old_savings: int
    new_savings: int
    delta: int


class ValidationResult(BaseModel):
    """Result of edit validation with clear success/failure semantics.

    Attributes:
        is_safe: Whether validation passed (True) or failed (False)
        is_pending: Whether edits are valid but need more for savings threshold (True) or not (False)
        valid_edits: Edits that passed validation and can be applied
        rejected_edits: Individual edits that were rejected (with reasons)
        failure_reason: Global validation failure reason (when is_safe=False, all edits rejected)
        pending_reason: Why edits are pending (when is_pending=True)
        estimated_savings: Estimated token savings from the edits
        edit_savings_map: Map of tool_use_id to individual savings
        replacement_info: Information about edits that replaced pending edits
    """

    is_safe: bool
    is_pending: bool = False
    valid_edits: list[ContextEdit] = Field(default_factory=list)
    rejected_edits: list[RejectedEdit] = Field(default_factory=list)
    failure_reason: str | None = None
    pending_reason: str | None = None
    estimated_savings: int = 0
    edit_savings_map: dict[str, int] = Field(default_factory=dict)
    replacement_info: list[ReplacementInfo] = Field(default_factory=list)


class EditStats(BaseModel):
    """Statistics about applied edits."""

    removed_count: int
    compacted_count: int
    kept_count: int
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    savings_percent: float


class _ToolUseMessages(BaseModel):
    """Message references for a single tool use (CfMessage references)."""

    # Store direct CfMessage references for in-place edits
    tool_use: CfMessage | None = None
    tool_result: CfMessage | None = None
    tool_use_info: CfMessage | None = None

    # Convenience methods to check existence without triggering lookup
    def has_tool_use(self) -> bool:
        """Check if tool_use is set without triggering message lookup."""
        return self.tool_use is not None

    def has_tool_result(self) -> bool:
        """Check if tool_result is set without triggering message lookup."""
        return self.tool_result is not None

    def has_tool_use_info(self) -> bool:
        """Check if tool_use_info is set without triggering message lookup."""
        return self.tool_use_info is not None


_ToolUseMessageMap = dict[str, _ToolUseMessages]


class _PendingData(BaseModel):
    edits: list[ContextEdit] = Field(default_factory=list)
    estimated_savings: int = 0
    edit_savings_map: dict[str, int] = Field(default_factory=dict)

    def clear(self) -> None:
        """Clear all pending data."""
        self.edits = []
        self.estimated_savings = 0
        self.edit_savings_map = {}


class _ToolUseResult(BaseModel):
    tool_use: ant.MessageContentToolUse
    tool_result: ant.MessageContentToolResult


class SmartContextManagementExtension(
    TokenEstimatorExtension, ToolUseExtension, ToolUseObserver
):
    """
    Smart LLM-driven context management via direct memory manipulation.

    This extension provides the context_edit tool that allows the LLM to
    intelligently manage conversation context by selectively removing or
    compacting tool uses based on their relevance to the current task.

    Example usage in an Entry:

        extensions=[
            SmartContextManagementExtension(
                keep=10,  # Keep at least 10 tool uses
                clear_at_least=75000,  # Clear 75K+ tokens per edit
                input_tokens_trigger=150000,  # Trigger enforcement at 150K tokens
            ),
        ]

    Cache Consideration:
    Prompt caching allows reusing prompt prefixes across requests.
    However, context editing invalidates the cache because message history changes.

    The clear_at_least threshold ensures edits save enough tokens to justify
    the cache invalidation cost. Set this based on your usage patterns:
    - High frequency, long sessions: Set higher (100K+)
    - Low frequency, short sessions: Set lower (25K-50K)
    """

    included_in_system_prompt: bool = False

    # Configuration fields
    keep: int = Field(
        default=0,
        description="Minimum number of tool uses to retain",
    )
    clear_at_least: int = Field(
        default=50000,
        description="Minimum tokens that must be cleared per edit to justify the operation",
    )
    enforce_clear_at_least: bool = Field(
        default=True,
        description="Whether to enforce the clear_at_least threshold",
    )
    enforce_clear_at_least_tolerance: float = Field(
        default=0.75,
        description="Tolerance for enforcing the clear_at_least threshold, to account for token estimation errors",
    )
    input_tokens_trigger: int = Field(
        default=100000,
        description="Input token count that triggers context optimization enforcement",
    )
    reminder_enabled: bool = Field(
        default=True,
        description="Whether to inject the soft reminder SystemMessage when approaching the trigger threshold",
    )
    reminder_ratio: float = Field(
        default=0.8,
        description="Ratio of input_tokens_trigger at which to start showing reminders (only used when reminder_enabled=True)",
    )
    reminder_message_template: str = Field(
        default=CONTEXT_MANAGEMENT_REMINDER_MESSAGE_TEMPLATE,
        description="The reminder message to inject when triggered",
    )
    enforcement_message_template: str = Field(
        default=CONTEXT_MANAGEMENT_ENFORCEMENT_MESSAGE_TEMPLATE,
        description="Enforcement message template for context management",
    )
    remove_context_edit_history: bool = Field(
        default=True,
        description="Remove context_edit tool messages from history after execution",
    )
    max_continuous_edits: int = Field(
        default=3,
        description="Maximum number of continuous context_edit attempts before auto-forcing additional IGNORE edits on oldest tool uses",
    )
    merge_fully_ignored_turns_enabled: bool = Field(
        default=True,
        description="Merge consecutive fully-ignored turns into single system message to save tokens",
    )
    log_dir: str | None = Field(
        default=None,
        description="Directory to write context_edits.jsonl log file. If None, file logging is disabled.",
    )
    _tool_use_message_map: _ToolUseMessageMap = PrivateAttr(default_factory=dict)
    _ignored_tool_use_ids: set[str] = PrivateAttr(default_factory=set)
    _last_added_tool_use_id: str | None = PrivateAttr(default=None)
    _context_edit_tool_uses: dict[str, _ToolUseResult] = PrivateAttr(
        default_factory=dict
    )
    _compressible_tokens: int | None = PrivateAttr(default=None)
    _continuous_edit_count: int = PrivateAttr(default=0)
    _enforce_mode: bool = PrivateAttr(default=False)

    def _get_tool_use_messages(self, tool_use_id: str) -> _ToolUseMessages:
        """Get the messages for a single tool use."""
        return self._tool_use_message_map.setdefault(tool_use_id, _ToolUseMessages())

    def _log_context_edit(self, data: dict) -> None:
        """Log a context edit event at INFO level and optionally to a JSONL file."""
        summary = data.get("event", "context_edit")
        details = {k: v for k, v in data.items() if k != "event"}
        context_edit_logger.info("%s: %s", summary, details)
        if self.log_dir is not None:
            import os
            log_path = os.path.join(self.log_dir, "context_edits.jsonl")
            record = {"timestamp": datetime.datetime.utcnow().isoformat(), **data}
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception as exc:
                context_edit_logger.warning("Failed to write context_edits.jsonl: %s", exc)

    @property
    def _pending_data(self) -> _PendingData:
        """Get the pending data for the current context edit session."""
        context = get_current_context()
        return cast(
            _PendingData,
            context.session_storage[self.__class__.__name__].setdefault(
                PENDING_DATA_KEY, _PendingData()
            ),
        )

    @property
    def practical_clear_at_least(self) -> int:
        """
        Get the practical minimum token threshold with tolerance applied.

        Takes the smaller of _compressible_tokens and clear_at_least, then applies
        tolerance to account for token estimation errors. This ensures we don't
        require clearing more tokens than are actually available while being
        forgiving about estimation errors.
        """
        # Start with clear_at_least as the base
        effective_threshold = self.clear_at_least

        # If compressible tokens is known and smaller, use that instead
        if (
            self._compressible_tokens is not None
            and self._compressible_tokens < self.clear_at_least
        ):
            effective_threshold = self._compressible_tokens

        # Apply tolerance to the effective threshold
        return int(effective_threshold * self.enforce_clear_at_least_tolerance)

    @property
    def enforced_input_tokens_trigger(self) -> int:
        """Get the enforced input tokens trigger (now just returns input_tokens_trigger directly)."""
        return self.input_tokens_trigger

    def _is_message_fully_ignored(self, msg: CfMessage) -> bool:
        """Check if ALL tool_use_ids in this message are ignored."""
        tool_use_ids = self._extract_tool_use_ids_from_message(msg)
        if not tool_use_ids:
            return False
        return all(tid in self._ignored_tool_use_ids for tid in tool_use_ids)

    def _create_tool_use_input_ignore_placeholder(self) -> dict[str, Any]:
        """Create placeholder input for ignored tool_use messages.

        Returns a dict with explicit marker that prevents LLM confusion
        by clearly indicating this is a cleared system placeholder.

        Returns:
            Dict with cleared marker
        """
        return {CLEARED_TOOL_USE_KEY: True}

    def _create_tool_result_ignore_placeholder(self) -> str:
        """Create placeholder text for ignored tool_result messages.

        Returns:
            Placeholder text string
        """
        return "The result was cleared"

    async def _calculate_compressible_tokens(self, context: AnalectRunContext) -> int:
        """
        Calculate total tokens from all tracked tool use messages (roofline).

        This represents the maximum possible token savings if all tool uses
        were removed or compacted to zero tokens.

        Args:
            context: The analect run context.

        Returns:
            Total tokens from all non-ignored tool use, tool result, and tool_use_info messages.
        """
        self._build_tool_use_message_map(context)
        total = 0
        for tool_use_id, tool_use_msgs in self._tool_use_message_map.items():
            # Only count non-ignored messages
            # Cache tool_use lookup to avoid double property access
            tool_use = tool_use_msgs.tool_use
            if tool_use is not None and tool_use_id not in self._ignored_tool_use_ids:
                # Collect all three messages for this tool use
                messages_to_count = [tool_use]  # Already verified non-None
                if tool_use_msgs.tool_result is not None:
                    messages_to_count.append(tool_use_msgs.tool_result)
                if tool_use_msgs.tool_use_info is not None:
                    messages_to_count.append(tool_use_msgs.tool_use_info)

                if messages_to_count:
                    token_lengths = await self.get_prompt_token_lengths(
                        messages_to_count
                    )
                    # Calculate placeholder sizes for both tool_use and tool_result
                    tool_use_placeholder = await self._get_tool_use_placeholder_size(
                        tool_use_id, tool_use
                    )
                    tool_result_placeholder = (
                        await self._get_tool_result_placeholder_size(
                            tool_use_id, tool_use
                        )
                        if tool_use_msgs.tool_result is not None
                        else 0
                    )
                    total_placeholder_size = (
                        tool_use_placeholder + tool_result_placeholder
                    )
                    total += max(0, sum(token_lengths) - total_placeholder_size)
        return total

    @override
    @property
    async def tools(self) -> list[ant.ToolLike]:
        """Return context_edit tool."""
        return [
            ant.Tool(
                name=CONTEXT_EDIT_TOOL_NAME,
                description=get_context_edit_tool_description(
                    keep=self.keep,
                    clear_at_least=self.clear_at_least,
                    enforce_clear_at_least=self.enforce_clear_at_least,
                ),
                input_schema=ContextEditInput.model_json_schema(),
            )
        ]

    @override
    async def on_invoke_llm_with_params(
        self,
        messages: list[BaseMessage],
        llm_params: LLMParams,
        context: AnalectRunContext,
    ) -> tuple[list[BaseMessage], LLMParams]:
        # Call super to allow other extensions to process
        (
            messages,
            llm_params,
        ) = await super().on_invoke_llm_with_params(messages, llm_params, context)

        prompt_lengths = await self.get_prompt_token_lengths(
            messages, tools=(llm_params.additional_kwargs or {}).get("tools")
        )
        total_length = sum(prompt_lengths)
        
        # Calculate reminder and enforcement triggers
        reminder_trigger = int(self.input_tokens_trigger * self.reminder_ratio) if self.reminder_enabled else self.input_tokens_trigger
        enforcement_trigger = self.input_tokens_trigger
        
        # Calculate _compressible_tokens when above the minimum trigger threshold
        if total_length >= min(reminder_trigger, enforcement_trigger):
            self._compressible_tokens = await self._calculate_compressible_tokens(
                context
            )
        
        # Inject reminder message if enabled and above reminder threshold
        if self.reminder_enabled and total_length >= reminder_trigger:
            messages.append(
                SystemMessage(
                    content=self.reminder_message_template.format(
                        token_count=total_length,
                        compressible_tokens=self._compressible_tokens,
                        tool_name=CONTEXT_EDIT_TOOL_NAME,
                        keep=self.keep,
                        clear_at_least=self.practical_clear_at_least,
                    )
                )
            )
        
        # Inject enforcement message if in enforce mode
        if self._enforce_mode:
            if self._compressible_tokens is None:
                self._compressible_tokens = await self._calculate_compressible_tokens(
                    context
                )
            messages.append(
                SystemMessage(
                    content=self.enforcement_message_template.format(
                        token_count=total_length,
                        compressible_tokens=self._compressible_tokens,
                        tool_name=CONTEXT_EDIT_TOOL_NAME,
                        keep=self.keep,
                        clear_at_least=self.practical_clear_at_least,
                    )
                )
            )

        return messages, llm_params

    def _extract_tool_use_ids_from_message(self, msg: CfMessage) -> set[str]:
        """
        Extract all tool_use_ids from a message.

        Handles three types of messages:
        - AI messages with tool uses (extracts tool_use.id)
        - HUMAN messages with tool results (extracts tool_result.tool_use_id)
        - SYS messages with tool_use_id in additional_kwargs

        Args:
            msg: The message to extract tool_use_ids from

        Returns:
            Set of tool_use_ids found in the message
        """
        tool_use_ids = set()

        # Check SYS message with tool_use_id in additional_kwargs
        if msg.type == cf.MessageType.SYS:
            tool_use_id = msg.additional_kwargs.get(SYS_INFO_TOOL_USE_ID_KEY)
            if tool_use_id:
                tool_use_ids.add(tool_use_id)
            return tool_use_ids

        # Check message content blocks
        if not isinstance(msg.content, list):
            return tool_use_ids

        for content_item in msg.content:
            # Extract from AI messages (tool uses)
            if msg.type == cf.MessageType.AI:
                try:
                    tool_use = ant.MessageContentToolUse.model_validate(content_item)
                    tool_use_ids.add(tool_use.id)
                except Exception:
                    # Not a tool use, skip to next item
                    continue

            # Extract from HUMAN messages (tool results)
            elif msg.type == cf.MessageType.HUMAN:
                try:
                    tool_result = ant.MessageContentToolResult.model_validate(
                        content_item
                    )
                    tool_use_ids.add(tool_result.tool_use_id)
                except Exception:
                    # Not a tool result, skip to next item
                    continue

        return tool_use_ids

    def _build_tool_use_message_map(self, context: AnalectRunContext) -> None:
        """
        Build map of tool_use_id to CfMessage references.

        This allows fast lookup when applying edits.
        """
        self._tool_use_message_map.clear()
        self._ignored_tool_use_ids.clear()  # Reset and rebuild from message content
        messages = context.memory_manager.get_session_memory().messages

        for msg in messages:
            tool_use_ids = self._extract_tool_use_ids_from_message(msg)

            for tool_use_id in tool_use_ids:
                # Store message reference based on message type
                if msg.type == cf.MessageType.AI:
                    self._get_tool_use_messages(tool_use_id).tool_use = msg
                    # Check if this specific tool_use_id has been cleared
                    if self._is_tool_use_content_cleared(msg, tool_use_id):
                        self._ignored_tool_use_ids.add(tool_use_id)
                elif msg.type == cf.MessageType.HUMAN:
                    self._get_tool_use_messages(tool_use_id).tool_result = msg
                elif msg.type == cf.MessageType.SYS:
                    self._get_tool_use_messages(tool_use_id).tool_use_info = msg

    def _is_tool_use_content_cleared(self, msg: CfMessage, tool_use_id: str) -> bool:
        """Check if a specific tool_use_id within a message has been cleared."""
        if not isinstance(msg.content, list):
            return False
        for item in msg.content:
            try:
                tool_use = ant.MessageContentToolUse.model_validate(item)
                if tool_use.id == tool_use_id:
                    return isinstance(tool_use.input, dict) and tool_use.input.get(CLEARED_TOOL_USE_KEY) is True
            except Exception:
                continue
        return False

    def _is_context_edit_message(self, msg: CfMessage) -> bool:
        """Check if message is related to context_edit tool use."""
        tool_use_ids = self._extract_tool_use_ids_from_message(msg)
        return bool(tool_use_ids & set(self._context_edit_tool_uses.keys()))

    async def _delete_context_edit_messages(self, context: AnalectRunContext) -> None:
        """
        Delete all context_edit tool messages from conversation history.

        For each tool_use_id in _context_edit_tool_uses, deletes:
        - Tool use message (AI message)
        - Tool result message (HUMAN message)
        - Tool use info message (SYS message with <system_info>)

        Uses a comprehensive predicate to identify context_edit messages without
        relying on _tool_use_message_map which might not be built yet.
        """
        if not self._context_edit_tool_uses:
            return

        self._build_tool_use_message_map(context)

        # 1. First: Merge fully-ignored turns from PREVIOUS context_edit operations
        if self.merge_fully_ignored_turns_enabled:
            merge_fully_ignored_turns(
                memory_manager=context.memory_manager,
                is_ignored_fn=self._is_message_fully_ignored,
            )

        # 2. Then: Ignore THIS context_edit's tool uses
        for tool_use_id, tool_use_and_result in self._context_edit_tool_uses.items():
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)
            await self._ignore_tool_use_messages(
                tool_use_msgs,
                tool_use_id,
                context,
                placeholder_text=Tag(
                    name="system",
                    contents=f"A function call to `{tool_use_and_result.tool_use.name}` was cleared. "
                    + (
                        "The context window failed to optimize."
                        if tool_use_and_result.tool_result.is_error is True
                        else "The context window has been successfully optimized. Please continue the conversation."
                    ),
                ).prettify(),
            )

        # 3. Clear the tracking list
        self._context_edit_tool_uses.clear()

    async def on_before_tool_use(
        self, tool_use: ant.MessageContentToolUse, context: AnalectRunContext
    ) -> None:
        # Reset continuous edit counter when using non-context_edit tool
        if tool_use.name != CONTEXT_EDIT_TOOL_NAME:
            self._continuous_edit_count = 0

        # Enforce context management if input tokens exceed threshold
        last_prompt_token_length = self.get_last_prompt_token_length() or 0
        if last_prompt_token_length >= self.input_tokens_trigger:
            if tool_use.name not in MEMORY_TOOL_NAMES + [CONTEXT_EDIT_TOOL_NAME]:
                # Set flag before raising interruption - it will be read in the next
                # on_invoke_llm_with_params call to inject the enforcement message
                self._enforce_mode = True
                raise OrchestratorInterruption()

    @override
    async def on_after_tool_use_result(
        self,
        tool_use: ant.MessageContentToolUse,
        tool_result: ant.MessageContentToolResult,
        context: AnalectRunContext,
    ) -> None:
        # Check if tool_use_info already added (deduplication via flag)
        # This extension is both ToolUseExtension and ToolUseObserver, so it will be called twice
        if self._last_added_tool_use_id == tool_result.tool_use_id:
            return  # Already added, skip

        # Get estimated token lengths for tool_use and tool_result
        tool_use_tokens, tool_result_tokens = await self.get_prompt_token_lengths(
            [
                AIMessage(content=[tool_use.dict()]),
                HumanMessage(content=[tool_result.dict()]),
            ]
        )

        # Count image blocks in the tool result and calculate their token cost
        image_count = 0
        image_tokens = 0
        if isinstance(tool_result.content, list):
            for item in tool_result.content:
                if isinstance(item, ant.MessageContentImage):
                    dims = get_image_dimensions_from_block(item.model_dump(mode="json"))
                    if dims is not None:
                        image_tokens += calculate_image_tokens_from_dimensions(dims[0], dims[1])
                    image_count += 1

        # Build system_info tag contents
        sys_info_contents: list[Tag | str] = [
            Tag(
                name="tool_use_id",
                contents=tool_result.tool_use_id,
            ),
            Tag(
                name="estimated_tokens",
                contents=f"tool_use: {tool_use_tokens}, tool_result: {tool_result_tokens}, total: {tool_use_tokens + tool_result_tokens}",
            ),
        ]
        if image_count > 0:
            sys_info_contents.append(
                Tag(
                    name="images",
                    contents=f"{image_count} image{'s' if image_count != 1 else ''} (~{image_tokens} token{'s' if image_tokens != 1 else ''} of {tool_result_tokens} tool_result tokens)",
                )
            )

        # Create and add sys info message
        tool_use_info = CfMessage(
            type=cf.MessageType.SYS,
            content=Tag(
                name="system_info",
                contents=sys_info_contents,
            ).prettify(),
            additional_kwargs={SYS_INFO_TOOL_USE_ID_KEY: tool_result.tool_use_id},
        )
        context.memory_manager.add_messages([tool_use_info])

        # Mark as added to prevent duplicate
        self._last_added_tool_use_id = tool_result.tool_use_id

        # Record context_edit tool use IDs for cleanup
        if tool_use.name == CONTEXT_EDIT_TOOL_NAME:
            self._context_edit_tool_uses[tool_use.id] = _ToolUseResult(
                tool_use=tool_use,
                tool_result=tool_result,
            )

        # Clean up context_edit tool messages if enabled and no pending edits
        if (
            tool_use.name == CONTEXT_EDIT_TOOL_NAME
            and self.remove_context_edit_history
            and not self._pending_data.edits
        ):
            await self._delete_context_edit_messages(context)

    def _merge_pending_with_new_edits(
        self, new_edits: list[ContextEdit]
    ) -> tuple[list[ContextEdit], int, list[tuple[str, ContextEditOperation]]]:
        """
        Merge pending edits with new edits, deduplicating by tool_use_id.

        New edits override old edits for the same tool_use_id.

        Args:
            new_edits: New edits from current context_edit call

        Returns:
            Tuple of (all_edits, pending_count, replaced_edits) where:
            - all_edits: Merged and deduplicated edits
            - pending_count: Number of edits that were pending before merge
            - replaced_edits: List of (tool_use_id, old_operation) for replaced edits
        """
        pending_count = len(self._pending_data.edits)
        if not self._pending_data.edits:
            return new_edits, 0, []

        # Track which edits are being replaced
        replaced_edits: list[tuple[str, ContextEditOperation]] = []

        # Create a dict to deduplicate by tool_use_id (new edits override old)
        edit_map = {edit.tool_use_id: edit for edit in self._pending_data.edits}
        for edit in new_edits:
            if edit.tool_use_id in edit_map:
                # This is a replacement
                old_edit = edit_map[edit.tool_use_id]
                replaced_edits.append((edit.tool_use_id, old_edit.operation))
            edit_map[edit.tool_use_id] = edit  # New overrides old
        all_edits = list(edit_map.values())
        return all_edits, pending_count, replaced_edits

    async def _calculate_replacement_info(
        self,
        replaced_edits: list[tuple[str, ContextEditOperation]],
        new_edits: list[ContextEdit],
        context: AnalectRunContext,
    ) -> list[ReplacementInfo]:
        """
        Calculate replacement info with token savings delta.

        Args:
            replaced_edits: List of (tool_use_id, old_operation) for replaced edits
            new_edits: New edits from current context_edit call
            context: Analect run context

        Returns:
            List of ReplacementInfo with token savings delta
        """
        replacement_info: list[ReplacementInfo] = []

        for tool_use_id, old_operation in replaced_edits:
            # Find the new edit
            new_edit = next(
                (e for e in new_edits if e.tool_use_id == tool_use_id), None
            )
            if not new_edit:
                continue

            # Get old savings from map
            old_savings = self._pending_data.edit_savings_map.get(tool_use_id, 0)

            # Calculate new savings
            new_savings = await self._estimate_edit_token_savings(new_edit, context)

            # Calculate delta
            delta = new_savings - old_savings

            replacement_info.append(
                ReplacementInfo(
                    tool_use_id=tool_use_id,
                    old_operation=old_operation,
                    new_operation=new_edit.operation,
                    old_savings=old_savings,
                    new_savings=new_savings,
                    delta=delta,
                )
            )

        return replacement_info

    async def _handle_empty_edits_with_pending(
        self, tool_use_id: str, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """
        Handle case where new edits are empty but pending edits exist.

        Keeps pending and returns pending message asking for more edits.

        Args:
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with pending message
        """
        msg = self._format_pending_message(
            pending_count=len(self._pending_data.edits),
            estimated_savings=self._pending_data.estimated_savings,
            rejected_edits=[],
        )
        await context.io.system(
            msg,
            run_label="Context Management",
            run_status=cf.RunStatus.IN_PROGRESS,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=msg,
            is_error=False,
        )

    async def _handle_no_edits(
        self, tool_use_id: str, context: AnalectRunContext
    ) -> ant.MessageContentToolResult:
        """
        Handle case where no edits were provided at all.

        Args:
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result indicating no changes
        """
        msg = "No edits provided; no changes applied."
        await context.io.system(
            msg,
            run_label="Context Management",
            run_status=cf.RunStatus.COMPLETED,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=msg,
            is_error=False,
        )

    async def _show_processing_message(
        self,
        all_edits: list[ContextEdit],
        pending_count: int,
        new_edits: list[ContextEdit],
        context: AnalectRunContext,
    ) -> None:
        """
        Show processing message indicating operation has started.

        Args:
            all_edits: All edits being processed (pending + new)
            pending_count: Number of edits that were pending
            new_edits: New edits from current call
            context: Analect run context
        """
        edit_msg = f"Processing {len(all_edits)} context edit(s)"
        if pending_count > 0:
            edit_msg += f" ({pending_count} pending + {len(new_edits)} new)"
        edit_msg += "..."
        await context.io.system(
            edit_msg,
            run_label="Context Management",
            run_status=cf.RunStatus.IN_PROGRESS,
        )

    async def _handle_pending_validation(
        self,
        validation_result: ValidationResult,
        edit_savings_map: dict[str, int],
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle pending validation result.

        Stores valid edits and returns pending message asking for more.
        If max continuous edits exceeded, triggers auto-force instead.

        Args:
            validation_result: Result with is_pending=True
            edit_savings_map: Map of tool_use_id to individual savings
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with pending message or auto-force result
        """
        # Check if max continuous edits exceeded - trigger auto-force
        if self._continuous_edit_count >= self.max_continuous_edits:
            return await self._handle_auto_force_edits(
                validation_result, edit_savings_map, tool_use_id, context
            )

        # Normal pending flow
        self._pending_data.edits = validation_result.valid_edits
        self._pending_data.estimated_savings = validation_result.estimated_savings
        self._pending_data.edit_savings_map = edit_savings_map
        pending_msg = self._format_pending_message(
            pending_count=len(validation_result.valid_edits),
            estimated_savings=validation_result.estimated_savings,
            rejected_edits=validation_result.rejected_edits,
            replacement_info=validation_result.replacement_info,
        )
        self._log_context_edit({
            "event": "context_edit_pending",
            "tool_use_id": tool_use_id,
            "pending_edits": len(validation_result.valid_edits),
            "estimated_savings": validation_result.estimated_savings,
            "rejected_count": len(validation_result.rejected_edits),
        })
        await context.io.system(
            pending_msg,
            run_label="Context Management",
            run_status=cf.RunStatus.WARNING,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=pending_msg,
            is_error=True,
        )

    async def _handle_validation_failure(
        self,
        validation_result: ValidationResult,
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle hard validation failure.

        Clears pending edits and returns error message.

        Args:
            validation_result: Result with is_safe=False
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with error message
        """
        self._pending_data.clear()  # Clear pending on hard failure
        error_msg = self._format_validation_failure(validation_result)
        self._log_context_edit({
            "event": "context_edit_validation_failure",
            "tool_use_id": tool_use_id,
            "failure_reason": validation_result.failure_reason,
            "rejected_count": len(validation_result.rejected_edits),
        })
        await context.io.system(
            error_msg,
            run_label="Context Management",
            run_status=cf.RunStatus.WARNING,
        )
        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=error_msg,
            is_error=True,
        )

    async def _handle_successful_edits(
        self,
        validation_result: ValidationResult,
        pending_count: int,
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle successful edits.

        Applies all edits, clears pending, and returns success message.

        Args:
            validation_result: Result with is_safe=True
            pending_count: Number of edits that were pending before success
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with success message
        """
        # Build per-edit image info for logging BEFORE applying edits (images may be removed)
        edit_log_items = []
        for edit in validation_result.valid_edits:
            item: dict[str, Any] = {
                "tool_use_id": edit.tool_use_id,
                "operation": edit.operation.value,
            }
            tool_use_msgs = self._tool_use_message_map.get(edit.tool_use_id)
            if tool_use_msgs is not None and tool_use_msgs.tool_result is not None:
                result_msg = tool_use_msgs.tool_result
                result_content = result_msg.content
                img_count = 0
                img_tokens = 0
                if isinstance(result_content, list):
                    for outer_block in result_content:
                        try:
                            tr = ant.MessageContentToolResult.model_validate(outer_block)
                            if tr.tool_use_id == edit.tool_use_id and isinstance(tr.content, list):
                                for inner_block in tr.content:
                                    if isinstance(inner_block, ant.MessageContentImage):
                                        dims = get_image_dimensions_from_block(inner_block.model_dump(mode="json"))
                                        if dims is not None:
                                            img_tokens += calculate_image_tokens_from_dimensions(dims[0], dims[1])
                                        img_count += 1
                        except Exception:
                            continue
                if img_count > 0:
                    item["image_count"] = img_count
                    item["image_tokens_saved"] = img_tokens
            edit_log_items.append(item)

        stats = await self._apply_edits(validation_result.valid_edits, context)
        self._pending_data.clear()  # Clear pending on success
        self._continuous_edit_count = 0  # Reset counter on success

        self._log_context_edit({
            "event": "context_edit_applied",
            "tool_use_id": tool_use_id,
            "tokens_before": stats.tokens_before,
            "tokens_after": stats.tokens_after,
            "tokens_saved": stats.tokens_saved,
            "savings_percent": round(stats.savings_percent, 1),
            "removed_count": stats.removed_count,
            "compacted_count": stats.compacted_count,
            "edits": edit_log_items,
        })

        # Show success with statistics
        result = self._format_success_message(stats, validation_result, pending_count)
        await context.io.system(
            result,
            run_label="Context Management",
            run_status=cf.RunStatus.COMPLETED,
        )

        # Reset cache checkpoint after context edits to enable proper cache hits
        # The post-edit state is the new stable prefix that should be cached
        if context.runnable is not None:
            # Get BasePromptCaching extension from the orchestrator
            if isinstance(context.runnable, BaseOrchestrator):
                # Note: Only one BasePromptCaching extension should be configured per orchestrator
                # We process the first one found and break to avoid duplicate resets
                for ext in context.runnable.extensions:
                    if isinstance(ext, BasePromptCaching):
                        # Reset last_checkpoint to the new token count
                        messages = context.memory_manager.get_session_memory().messages
                        total_length = stats.tokens_after
                        messages[-1].additional_kwargs[CACHE_BREAKPOINT_KEY] = True
                        ext.set_last_checkpoint(total_length)
                        # Invalidate cached token length so on_memory() uses fresh value
                        ext.set_last_prompt_token_length(total_length)
                        break

        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=result,
            is_error=False,
        )

    def _should_include_tool_use(
        self, tool_use_id: str, tool_use_msg: CfMessage | None
    ) -> bool:
        """
        Check if a tool use should be included in context management operations.

        Excludes:
        - Tool uses with no message (None)
        - Already-ignored tool uses
        - context_edit tool uses themselves

        Args:
            tool_use_id: The tool use ID
            tool_use_msg: The tool use message (can be None)

        Returns:
            True if tool use should be included, False otherwise
        """
        if tool_use_msg is None:
            return False

        # Skip if already ignored
        if tool_use_id in self._ignored_tool_use_ids:
            return False

        # Skip context_edit tool uses
        tool_name = self._extract_tool_name(tool_use_id, tool_use_msg)
        if tool_name == CONTEXT_EDIT_TOOL_NAME:
            return False

        return True

    def _get_oldest_non_ignored_tool_uses(
        self,
        context: AnalectRunContext,
        exclude_ids: set[str],
    ) -> list[str]:
        """
        Get tool_use_ids sorted by message order (oldest first).

        Excludes:
        - Already-ignored tool uses
        - Tool uses in exclude_ids (already pending)
        - context_edit tool uses themselves

        Args:
            context: The analect run context
            exclude_ids: Set of tool_use_ids to exclude (already in pending)

        Returns:
            List of tool_use_ids from oldest to newest
        """
        messages = context.memory_manager.get_session_memory().messages
        ordered_ids: list[str] = []

        for msg in messages:
            if msg.type != cf.MessageType.AI:
                continue

            tool_use_ids = self._extract_tool_use_ids_from_message(msg)
            for tool_use_id in tool_use_ids:
                # Skip if in exclude set
                if tool_use_id in exclude_ids:
                    continue

                # Get tool use messages
                tool_use_msgs = self._get_tool_use_messages(tool_use_id)
                tool_use_msg = tool_use_msgs.tool_use

                # Use shared filtering logic
                if not self._should_include_tool_use(tool_use_id, tool_use_msg):
                    continue

                ordered_ids.append(tool_use_id)

        return ordered_ids

    async def _auto_force_ignore_edits(
        self,
        context: AnalectRunContext,
        current_savings: int,
        valid_edits: list[ContextEdit],
    ) -> tuple[list[ContextEdit], int]:
        """
        Automatically generate IGNORE edits for oldest tool uses until threshold is met.

        Args:
            context: The analect run context
            current_savings: Current estimated token savings from pending edits
            valid_edits: Current cycle's validated edits to exclude from auto-force

        Returns:
            Tuple of (forced_edits, additional_savings)
        """
        remaining_needed = self.practical_clear_at_least - current_savings
        if remaining_needed <= 0:
            return [], 0

        # Get IDs from current cycle's validated edits (not stale _pending_data)
        pending_ids = {edit.tool_use_id for edit in valid_edits}

        # Get oldest tool uses to force-ignore
        oldest_ids = self._get_oldest_non_ignored_tool_uses(context, pending_ids)

        # Calculate how many tool uses we can safely ignore (respect keep constraint)
        # Exclude already-ignored tool uses from count (same pattern as _validate_minimum_retention)
        total_tool_uses = sum(
            1
            for tool_use_id, msgs in self._tool_use_message_map.items()
            if msgs.tool_use is not None and tool_use_id not in self._ignored_tool_use_ids
        ) - len(self._context_edit_tool_uses)
        pending_ignore_count = sum(
            1
            for e in self._pending_data.edits
            if e.operation == ContextEditOperation.IGNORE
        )
        can_ignore = max(0, total_tool_uses - self.keep - pending_ignore_count)

        forced_edits: list[ContextEdit] = []
        additional_savings = 0

        for tool_use_id in oldest_ids:
            if additional_savings >= remaining_needed:
                break

            # Respect keep constraint
            if len(forced_edits) >= can_ignore:
                break

            # Create IGNORE edit
            edit = ContextEdit(
                tool_use_id=tool_use_id,
                operation=ContextEditOperation.IGNORE,
                reason="Auto-forced due to max continuous edits exceeded",
            )

            # Estimate savings
            savings = await self._estimate_edit_token_savings(edit, context)
            forced_edits.append(edit)
            additional_savings += savings

        return forced_edits, additional_savings

    async def _show_auto_force_notification(
        self,
        forced_edits: list[ContextEdit],
        total_savings: int,
        pending_count: int,
        context: AnalectRunContext,
    ) -> None:
        """
        Show UI notification about auto-force triggering.

        Args:
            forced_edits: List of auto-forced edits
            total_savings: Total estimated token savings
            pending_count: Number of pending edits before auto-force
            context: The analect run context
        """
        forced_list = "\n".join(f"- {edit.tool_use_id}" for edit in forced_edits)
        message = AUTO_FORCE_MESSAGE_TEMPLATE.format(
            max_attempts=self.max_continuous_edits,
            forced_count=len(forced_edits),
            forced_list=forced_list,
            pending_count=pending_count,
            total_savings=total_savings,
        )
        await context.io.system(
            message,
            run_label="Context Management",
            run_status=cf.RunStatus.WARNING,
        )

    def _format_auto_force_success_message(
        self,
        stats: EditStats,
        forced_edits: list[ContextEdit],
    ) -> str:
        """
        Format success message noting which tool uses were auto-forced.

        Args:
            stats: Statistics about the applied edits
            forced_edits: List of auto-forced edits

        Returns:
            Formatted success message with auto-force details
        """
        auto_ignored_list = ", ".join(edit.tool_use_id for edit in forced_edits)

        return f"""✅ **Context Edit Applied** (with auto-forced edits)

**Results:**
- Removed: {stats.removed_count} tool use(s)
- Compacted: {stats.compacted_count} tool use(s)
- Kept unchanged: {stats.kept_count} tool use(s)

**Token savings:**
- Before: ~{stats.tokens_before or 0:,} tokens
- After: ~{stats.tokens_after or 0:,} tokens
- Saved: ~{stats.tokens_saved or 0:,} tokens ({stats.savings_percent or 0.0:.1f}%)

**Auto-ignored tool uses ({len(forced_edits)}):** {auto_ignored_list}
"""

    async def _handle_auto_force_edits(
        self,
        validation_result: ValidationResult,
        edit_savings_map: dict[str, int],
        tool_use_id: str,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Handle auto-forcing IGNORE edits when max attempts exceeded.

        Args:
            validation_result: Current validation result with pending edits
            edit_savings_map: Map of tool_use_id to individual savings
            tool_use_id: ID of the tool use
            context: Analect run context

        Returns:
            Tool result with success or partial success message
        """
        # Generate forced edits
        forced_edits, additional_savings = await self._auto_force_ignore_edits(
            context, validation_result.estimated_savings, validation_result.valid_edits
        )

        pending_count = len(validation_result.valid_edits)

        # If no more tool uses to force-ignore, apply what we have
        if not forced_edits:
            # Apply pending edits even if below threshold - exit gracefully
            stats = await self._apply_edits(validation_result.valid_edits, context)
            self._pending_data.clear()
            self._continuous_edit_count = 0

            result = f"""⚠️ **Context Edit Applied** (partial - threshold not met)

No more tool uses available to auto-ignore due to `keep` constraint.
Applied {len(validation_result.valid_edits)} pending edit(s).

**Results:**
- Removed: {stats.removed_count} tool use(s)
- Compacted: {stats.compacted_count} tool use(s)
- Kept unchanged: {stats.kept_count} tool use(s)

**Token savings:**
- Before: ~{stats.tokens_before or 0:,} tokens
- After: ~{stats.tokens_after or 0:,} tokens
- Saved: ~{stats.tokens_saved or 0:,} tokens ({stats.savings_percent or 0.0:.1f}%)
"""
            await context.io.system(
                result,
                run_label="Context Management",
                run_status=cf.RunStatus.COMPLETED,
            )
            return ant.MessageContentToolResult(
                tool_use_id=tool_use_id,
                content=result,
                is_error=False,
            )

        # Combine pending + forced edits
        all_edits = validation_result.valid_edits + forced_edits
        total_savings = validation_result.estimated_savings + additional_savings

        # Show auto-force notification
        await self._show_auto_force_notification(
            forced_edits, total_savings, pending_count, context
        )

        # Apply all edits
        stats = await self._apply_edits(all_edits, context)
        self._pending_data.clear()
        self._continuous_edit_count = 0

        # Format success message with auto-force note
        result = self._format_auto_force_success_message(stats, forced_edits)
        await context.io.system(
            result,
            run_label="Context Management",
            run_status=cf.RunStatus.COMPLETED,
        )

        return ant.MessageContentToolResult(
            tool_use_id=tool_use_id,
            content=result,
            is_error=False,
        )

    @override
    async def on_tool_use(
        self,
        tool_use: ant.MessageContentToolUse,
        context: AnalectRunContext,
    ) -> ant.MessageContentToolResult:
        """
        Execute context_edit tool.

        Directly manipulates context.memory_manager.memory.messages

        Flow:
        1. Parse input and merge with pending edits
        2. Handle special cases (empty edits, no edits)
        3. Validate all edits
        4. Handle validation outcome (pending, failure, or success)
        """
        # Reset enforce mode flag
        self._enforce_mode = False

        # Increment continuous edit counter
        self._continuous_edit_count += 1

        # Parse and merge edits
        input_data = ContextEditInput.model_validate(tool_use.input)
        new_edits = input_data.edits
        all_edits, pending_count, replaced_edits = self._merge_pending_with_new_edits(
            new_edits
        )

        self._log_context_edit({
            "event": "context_edit_requested",
            "tool_use_id": tool_use.id,
            "new_edits": [e.model_dump(mode="json", exclude_none=True) for e in new_edits],
            "total_edits": len(all_edits),
            "pending_count": pending_count,
        })

        # Handle empty edits with pending
        if not new_edits and self._pending_data.edits:
            return await self._handle_empty_edits_with_pending(tool_use.id, context)

        # Handle no edits at all
        if not all_edits:
            return await self._handle_no_edits(tool_use.id, context)

        # Show progress and validate
        await self._show_processing_message(
            all_edits, pending_count, new_edits, context
        )
        self._build_tool_use_message_map(context)

        # Calculate replacement info before validation
        replacement_info = await self._calculate_replacement_info(
            replaced_edits, new_edits, context
        )

        validation_result = await self._validate_edits(
            all_edits, context, replacement_info
        )

        # Handle different validation outcomes
        if validation_result.is_pending:
            return await self._handle_pending_validation(
                validation_result,
                validation_result.edit_savings_map,
                tool_use.id,
                context,
            )
        if not validation_result.is_safe:
            return await self._handle_validation_failure(
                validation_result, tool_use.id, context
            )

        return await self._handle_successful_edits(
            validation_result, pending_count, tool_use.id, context
        )

    @override
    async def on_process_messages_complete(self, context: AnalectRunContext) -> None:
        """Check if progress needs to be updated and raise interruption if not complete"""
        if not self._pending_data.edits or (
            (self.get_last_prompt_token_length() or 0) < self.input_tokens_trigger
        ):
            return

        reminder_message = self._format_pending_reminder_message(
            pending_count=len(self._pending_data.edits),
            estimated_savings=self._pending_data.estimated_savings,
        )
        raise OrchestratorInterruption(reminder_message)

    def _format_success_message(
        self,
        stats: EditStats,
        validation_result: ValidationResult,
        pending_count: int = 0,
    ) -> str:
        """
        Format success message with statistics and warnings.

        Override this method to customize the success message format.

        Args:
            stats: Statistics about the applied edits
            validation_result: Validation result containing rejected edits
            pending_count: Number of edits that were pending before this success

        Returns:
            Formatted success message string
        """
        result = "✅ Context optimized"
        if pending_count > 0:
            result += f" (accumulated {pending_count} pending edits)"
        result += "\n"

        result += f"""
Changes:
- {stats.removed_count} tool uses removed (IGNORE)
- {stats.compacted_count} tool uses compacted (COMPACT)
- {stats.kept_count} tool uses kept unchanged

Token Savings:
- Before: ~{stats.tokens_before or 0:,} tokens
- After: ~{stats.tokens_after or 0:,} tokens
- Saved: ~{stats.tokens_saved or 0:,} tokens ({stats.savings_percent or 0.0:.1f}% reduction)

NOTE:
* Token counts are rough estimates and exclude the tool definitions in the system prompt.
"""

        # Add warning if tokens saved is less than clear_at_least threshold
        if stats.tokens_saved < self.practical_clear_at_least:
            result += f"""

⚠️ **Warning: Insufficient token savings**

You cleared ~{stats.tokens_saved or 0:,} tokens, which is less than the recommended {self.practical_clear_at_least:,} tokens.

This may not justify the cost of invalidating the prompt cache. Next time, please try to:
- Remove more outdated tool uses
- Compact more verbose inputs and/or results
- Clear older, less relevant content

This helps make each context edit more worthwhile.
"""

        # Add rejected edits information if any
        result += self._format_rejected_edits_section(validation_result.rejected_edits)

        return result

    def _format_rejected_edits_section(self, rejected_edits: list[RejectedEdit]) -> str:
        """
        Format rejected edits section for display in messages.

        Args:
            rejected_edits: List of edits that were rejected with reasons

        Returns:
            Formatted string showing rejected edits with reasons, or empty string if none
        """
        if not rejected_edits:
            return ""

        result = f"\n\n⚠️ {len(rejected_edits)} edit(s) were rejected:\n"
        result += "\n".join(
            f"- {r.edit.tool_use_id} ({r.edit.operation.value}): {r.reason}"
            for r in rejected_edits
        )
        return result

    def _format_replacement_info_section(
        self, replacement_info: list[ReplacementInfo]
    ) -> str:
        """
        Format replacement info section showing edits that replaced pending edits.

        Args:
            replacement_info: List of replacement information

        Returns:
            Formatted string showing replacements with token delta, or empty string if none
        """
        if not replacement_info:
            return ""

        result = (
            f"\n\n⚠️ {len(replacement_info)} edit(s) replaced already-pending edits:\n"
        )
        for info in replacement_info:
            op_change = f"{info.old_operation.value} → {info.new_operation.value}"
            if info.delta > 0:
                delta_str = f"+{info.delta:,} tokens (savings increased)"
            elif info.delta < 0:
                delta_str = f"{info.delta:,} tokens (savings decreased)"
            else:
                delta_str = "±0 tokens (no change)"
            result += f"- {info.tool_use_id} ({op_change}): {delta_str}\n"

        result += "\n**Note:** Replacing edits recalculates token savings. Consider if these changes are beneficial."
        return result

    def _format_pending_message(
        self,
        pending_count: int,
        estimated_savings: int,
        rejected_edits: list[RejectedEdit],
        replacement_info: list[ReplacementInfo] | None = None,
    ) -> str:
        """
        Format pending message when edits are accumulated but savings insufficient.

        Args:
            pending_count: Number of edits currently pending
            estimated_savings: Estimated token savings from pending edits
            rejected_edits: List of edits that were rejected with reasons
            replacement_info: Optional list of replacement information

        Returns:
            Formatted pending message string
        """
        estimated_savings = estimated_savings or 0
        result = f"""⚠️ INSUFFICIENT EDITS - MORE AGGRESSIVE COMPRESSION REQUIRED

Current Status:
- {pending_count} edit(s) pending (NOT ENOUGH)
- ~{estimated_savings:,} tokens saved so far (INSUFFICIENT)
- Required minimum: {self.practical_clear_at_least:,} tokens
- Additional needed: ~{max(0, self.practical_clear_at_least - estimated_savings):,} tokens

**CRITICAL: You MUST be MORE AGGRESSIVE in context compression to avoid repeated editing loops.**

Your current edits are too conservative. Be significantly more aggressive:

1. **IGNORE more liberally** - Remove ALL outdated/exploratory/redundant tool uses:
   - Failed attempts that were superseded
   - Exploratory queries no longer relevant
   - Duplicate or similar operations
   - Old context that's been saved to memory

2. **COMPACT more extensively** - Heavily summarize verbose inputs and/or results:
   - Large file contents → brief summaries
   - Long query results → key insights only
   - Verbose inputs/outputs → essential information only

3. **Prioritize OLDER content** - Recent work is more likely relevant:
   - Start from the earliest tool uses
   - Work your way forward chronologically

4. **Think in BATCHES** - Don't edit one at a time:
   - Review 5-10 tool uses at once
   - Submit comprehensive edits together

Your {pending_count} pending edit(s) are saved. Call `context_edit` NOW with significantly more aggressive edits.
"""

        # Add replacement info if any
        if replacement_info:
            result += self._format_replacement_info_section(replacement_info)

        # Add rejected edits information if any
        result += self._format_rejected_edits_section(rejected_edits)

        return result

    def _format_pending_reminder_message(
        self,
        pending_count: int,
        estimated_savings: int,
    ) -> str:
        """
        Format reminder message for interruption when pending edits exist.

        This message is shown when the orchestrator is about to exit but there
        are still pending edits that haven't reached the minimum threshold.

        Args:
            pending_count: Number of edits currently pending
            estimated_savings: Estimated token savings from pending edits
            rejected_edits: List of edits that were rejected with reasons

        Returns:
            Formatted reminder message string for interruption context
        """
        estimated_savings = estimated_savings or 0
        result = f"""⚠️ **Pending context edits incomplete**

You have {pending_count} pending context edit(s) that need to be completed.

Status:
- {pending_count} edit(s) pending
- ~{estimated_savings:,} tokens saved so far
- Target: {self.practical_clear_at_least:,} tokens minimum
- Need: ~{max(0, self.practical_clear_at_least - estimated_savings):,} more tokens

**Required Action:**
You MUST call `context_edit` again with more edits to reach the minimum threshold before you can exit.

Tips to reach the threshold:
- Remove more outdated tool uses (IGNORE operation)
- Compact more verbose inputs and/or results (COMPACT operation)
- Focus on older, less relevant content
- Your previous {pending_count} edit(s) are already saved and will accumulate
"""

        return result

    def _format_validation_failure(self, validation_result: ValidationResult) -> str:
        """
        Format validation failure message.

        Override this method to customize the error message format.

        Args:
            validation_result: Validation result with failure_reason

        Returns:
            Formatted error message string
        """
        return f"Context edit rejected:\n{validation_result.failure_reason}"

    def _validate_tool_use_ids_exist(
        self, edits: list[ContextEdit], context: AnalectRunContext
    ) -> tuple[list[ContextEdit], list[RejectedEdit]]:
        """
        Check 1: Filter edits to only valid tool_use_ids.

        Args:
            edits: List of context edits to validate
            context: The analect run context

        Returns:
            (valid_edits, rejected_edits): Valid edits and rejected edits with reasons
        """
        valid_edits = []
        rejected_edits = []

        for edit in edits:
            tool_use_id = edit.tool_use_id
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)

            # Check if tool_use exists (use has_* for efficiency)
            if not tool_use_msgs.has_tool_use():
                rejected_edits.append(
                    RejectedEdit(edit=edit, reason="Unknown tool_use_id")
                )
            elif edit.operation == ContextEditOperation.IGNORE:
                # For IGNORE, also need tool_result
                if not tool_use_msgs.has_tool_result():
                    rejected_edits.append(
                        RejectedEdit(edit=edit, reason="Tool result not found")
                    )
                else:
                    # tool_use is guaranteed to exist since has_tool_use() passed
                    if edit.tool_use_id in self._ignored_tool_use_ids:
                        rejected_edits.append(
                            RejectedEdit(edit=edit, reason="Tool use already ignored")
                        )
                    else:
                        valid_edits.append(edit)
            else:
                # COMPACT operations
                valid_edits.append(edit)

        return valid_edits, rejected_edits

    def _validate_minimum_retention(
        self, edits: list[ContextEdit], context: AnalectRunContext
    ) -> list[str]:
        """Check 2: Validate minimum tool uses retention.

        Args:
            edits: List of context edits to validate
            context: The analect run context

        Returns:
            List of error messages if validation fails, empty list otherwise
        """
        errors = []
        ignore_count = len(
            [e for e in edits if e.operation == ContextEditOperation.IGNORE]
        )
        # Count actual tool uses (exclude already-ignored)
        total_tool_uses = sum(
            1
            for tool_use_id, msgs in self._tool_use_message_map.items()
            if msgs.tool_use is not None and tool_use_id not in self._ignored_tool_use_ids
        )
        remaining_tool_uses = total_tool_uses - ignore_count
        if remaining_tool_uses < self.keep:
            errors.append(
                f"Would keep only {remaining_tool_uses} tool uses, minimum is {self.keep}"
            )
        return errors

    def _extract_tool_name(self, tool_use_id: str, tool_use_msg: CfMessage) -> str:
        """Extract tool name from tool use message."""
        tool_name = "unknown"
        if isinstance(tool_use_msg.content, list):
            for item in tool_use_msg.content:
                try:
                    tool_use_obj = ant.MessageContentToolUse.model_validate(item)
                    if tool_use_obj.id == tool_use_id:
                        tool_name = tool_use_obj.name
                        break
                except Exception:
                    continue
        return tool_name

    async def _get_tool_use_placeholder_size(
        self, tool_use_id: str, tool_use_msg: CfMessage
    ) -> int:
        """Get placeholder size for tool_use with cleared marker input."""
        # Extract tool name to reconstruct the tool_use
        tool_name = self._extract_tool_name(tool_use_id, tool_use_msg)

        # Create temp message with cleared marker to measure token size
        temp_msg = CfMessage(
            type=tool_use_msg.type,  # Keep original type
            content=[
                ant.MessageContentToolUse(
                    id=tool_use_id,
                    name=tool_name,
                    input=self._create_tool_use_input_ignore_placeholder(),
                ).model_dump(mode="json", exclude_none=True)
            ],
        )
        placeholder_size = (await self.get_prompt_token_lengths([temp_msg]))[0]
        return placeholder_size

    async def _get_tool_result_placeholder_size(
        self, tool_use_id: str, tool_use_msg: CfMessage
    ) -> int:
        """Get placeholder size for tool_result with placeholder text."""
        # Use existing helper to create placeholder text
        placeholder_text = self._create_tool_result_ignore_placeholder()

        # Create temp message to measure token size
        temp_msg = CfMessage(
            type=cf.MessageType.HUMAN,  # tool_result is always HUMAN type
            content=[
                ant.MessageContentToolResult(
                    tool_use_id=tool_use_id,
                    content=placeholder_text,
                ).model_dump(mode="json", exclude_none=True)
            ],
        )
        placeholder_size = (await self.get_prompt_token_lengths([temp_msg]))[0]
        return placeholder_size

    async def _estimate_edit_token_savings(
        self, edit: ContextEdit, context: AnalectRunContext
    ) -> int:
        """Calculate token savings for a single edit."""
        tool_use_id = edit.tool_use_id
        tool_use_msgs = self._get_tool_use_messages(tool_use_id)
        savings = 0

        if edit.operation == ContextEditOperation.IGNORE:
            # Cache property accesses to avoid repeated lookups
            tool_use = tool_use_msgs.tool_use
            tool_result = tool_use_msgs.tool_result
            tool_use_info = tool_use_msgs.tool_use_info

            # For tool_use: Calculate difference (original - placeholder with cleared marker)
            if tool_use is not None:
                original_size = (await self.get_prompt_token_lengths([tool_use]))[0]
                placeholder_size = await self._get_tool_use_placeholder_size(
                    tool_use_id, tool_use
                )
                savings += max(0, original_size - placeholder_size)

            # For tool_result: Calculate difference (original - placeholder text)
            # Changed from full deletion to compaction
            if tool_result is not None and tool_use is not None:
                original_size = (await self.get_prompt_token_lengths([tool_result]))[0]
                placeholder_size = await self._get_tool_result_placeholder_size(
                    tool_use_id, tool_use
                )
                savings += max(0, original_size - placeholder_size)

            # For tool_use_info: Still fully deleted, count full size
            if tool_use_info is not None:
                savings += (await self.get_prompt_token_lengths([tool_use_info]))[0]

        elif edit.operation == ContextEditOperation.COMPACT:
            replacement = edit.tool_result_content_replacement
            # Cache property access
            tool_result = tool_use_msgs.tool_result
            if replacement is not None and tool_result is not None:
                original_size = (await self.get_prompt_token_lengths([tool_result]))[0]

                # Use consistent token counting method
                temp_msg = CfMessage(
                    type=cf.MessageType.HUMAN,
                    content=[
                        ant.MessageContentToolResult(
                            tool_use_id=tool_use_id,
                            content=replacement,
                        ).model_dump(mode="json", exclude_none=True)
                    ],
                )
                new_size = (await self.get_prompt_token_lengths([temp_msg]))[0]
                savings = max(0, original_size - new_size)

        return savings

    async def _estimate_context_edit_cleanup_savings(
        self, context: AnalectRunContext
    ) -> int:
        """
        Estimate token savings from cleaning up tracked context_edit tool calls.

        When remove_context_edit_history=True, successful context edits trigger
        cleanup of all tracked context_edit tool call messages. This method
        calculates the token savings from that cleanup.

        Note: Assumes _tool_use_message_map is already built (it is built in
        on_tool_use before validation, line 964).

        Args:
            context: The analect run context

        Returns:
            Estimated token savings from removing all tracked context_edit messages,
            or 0 if remove_context_edit_history is False or no tracked calls exist.
        """
        if not self.remove_context_edit_history or not self._context_edit_tool_uses:
            return 0

        total_savings = 0
        # Map is already built at line 964 before validation, no need to rebuild

        for tool_use_id in self._context_edit_tool_uses.keys():
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)

            # Collect messages that will be deleted
            messages_to_count = []
            if tool_use_msgs.tool_use is not None:
                messages_to_count.append(tool_use_msgs.tool_use)
            if tool_use_msgs.tool_result is not None:
                messages_to_count.append(tool_use_msgs.tool_result)
            if tool_use_msgs.tool_use_info is not None:
                messages_to_count.append(tool_use_msgs.tool_use_info)

            if messages_to_count:
                token_lengths = await self.get_prompt_token_lengths(messages_to_count)
                total_savings += sum(token_lengths)

        return total_savings

    async def _validate_minimum_savings(
        self, edits: list[ContextEdit], context: AnalectRunContext
    ) -> tuple[int, bool, dict[str, int]]:
        """Check 3: Validate estimated token savings meets threshold.

        Returns:
            Tuple of (estimated_savings, should_pend, edit_savings_map):
            - estimated_savings: Estimated token savings from the edits plus cleanup
            - should_pend: Whether to pend edits (True) or not (False)
            - edit_savings_map: Map of tool_use_id to individual savings
        """
        # Parallel token estimation for better performance
        savings_list = await asyncio.gather(
            *[self._estimate_edit_token_savings(edit, context) for edit in edits]
        )

        # Create map of individual savings
        edit_savings_map = {
            edit.tool_use_id: savings for edit, savings in zip(edits, savings_list)
        }
        estimated_savings = sum(savings_list)

        # Add savings from cleaning up context_edit tool calls
        # When edits are applied successfully, all tracked context_edit messages
        # will be removed, so we include those savings in the total
        cleanup_savings = await self._estimate_context_edit_cleanup_savings(context)
        estimated_savings += cleanup_savings

        # If not enforcing, pass validation
        if not self.enforce_clear_at_least and (
            (self.get_last_prompt_token_length() or 0) >= self.input_tokens_trigger
        ):
            return (estimated_savings, False, edit_savings_map)

        # If savings insufficient, return pending state
        if estimated_savings < self.practical_clear_at_least:
            return (estimated_savings, True, edit_savings_map)

        # Savings sufficient
        return (estimated_savings, False, edit_savings_map)

    def _filter_valid_replacement_info(
        self,
        replacement_info: list[ReplacementInfo] | None,
        valid_edits: list[ContextEdit],
    ) -> list[ReplacementInfo]:
        """
        Filter replacement info to only include valid edits.

        Filters out:
        1. Replacements where the new edit was rejected (tool_use_id not in valid_edits)
        2. IGNORE -> IGNORE replacements (noisy estimation errors with no operation change)

        Args:
            replacement_info: Raw replacement info (may include rejected edits)
            valid_edits: List of edits that passed validation

        Returns:
            Filtered list of ReplacementInfo
        """
        if not replacement_info:
            return []

        # Create set of valid tool_use_ids for fast lookup
        valid_tool_use_ids = {edit.tool_use_id for edit in valid_edits}

        # Filter to valid edits and exclude IGNORE -> IGNORE
        return [
            info
            for info in replacement_info
            if info.tool_use_id in valid_tool_use_ids
            and not (
                info.old_operation == ContextEditOperation.IGNORE
                and info.new_operation == ContextEditOperation.IGNORE
            )
        ]

    async def _validate_edits(
        self,
        edits: list[ContextEdit],
        context: AnalectRunContext,
        replacement_info: list[ReplacementInfo] | None = None,
    ) -> ValidationResult:
        """
        Validate context edits before applying.

        Safety checks:
        1. Filter to valid tool_use_ids (forgiving - rejects invalid, keeps valid)
        2. Minimum tool uses retention (strict - rejects all if violated)
        3. Estimated token savings meets clear_at_least threshold (may pend or fail)

        Returns:
            ValidationResult with valid_edits and rejected_reasons
        """
        # Check 1: Filter to valid tool_use_ids (forgiving)
        valid_edits, rejected_edits = self._validate_tool_use_ids_exist(edits, context)

        # If no valid edits remain after filtering, reject all
        if not valid_edits:
            rejected_summary = "\n".join(
                f"- {r.edit.tool_use_id} ({r.edit.operation.value}): {r.reason}"
                for r in rejected_edits
            )
            return ValidationResult(
                is_safe=False,
                failure_reason=f"All edits were rejected due to invalid tool_use_ids:\n{rejected_summary}",
                valid_edits=[],
                rejected_edits=rejected_edits,
            )

        # Check 2: Minimum retention (strict)
        retention_errors = self._validate_minimum_retention(valid_edits, context)

        # Check 3: Minimum savings (may pend)
        (
            estimated_savings,
            should_pend,
            edit_savings_map,
        ) = await self._validate_minimum_savings(valid_edits, context)

        # If retention fails (strict check), reject ALL edits
        if retention_errors:
            # Filter replacement info to only valid edits (empty in this case)
            filtered_replacement_info = self._filter_valid_replacement_info(
                replacement_info, []
            )
            return ValidationResult(
                is_safe=False,
                failure_reason="\n".join(retention_errors),
                valid_edits=[],
                rejected_edits=rejected_edits,
                estimated_savings=estimated_savings,
                edit_savings_map=edit_savings_map,
                replacement_info=filtered_replacement_info,
            )

        # If savings fails but should pend, return pending state
        if should_pend:
            # Filter replacement info to only valid edits and exclude IGNORE -> IGNORE
            filtered_replacement_info = self._filter_valid_replacement_info(
                replacement_info, valid_edits
            )
            return ValidationResult(
                is_safe=False,
                is_pending=True,
                valid_edits=valid_edits,
                rejected_edits=rejected_edits,
                edit_savings_map=edit_savings_map,
                replacement_info=filtered_replacement_info,
                pending_reason=(
                    f"Estimated token savings ({estimated_savings:,}) is less than required "
                    f"minimum ({self.practical_clear_at_least:,}). Please add more edits to reach the threshold."
                ),
                estimated_savings=estimated_savings,
            )

        # All validation passed
        # Filter replacement info to only valid edits and exclude IGNORE -> IGNORE
        filtered_replacement_info = self._filter_valid_replacement_info(
            replacement_info, valid_edits
        )
        return ValidationResult(
            is_safe=True,
            valid_edits=valid_edits,
            rejected_edits=rejected_edits,
            failure_reason=None,
            estimated_savings=estimated_savings,
            edit_savings_map=edit_savings_map,
            replacement_info=filtered_replacement_info,
        )

    async def _apply_edits(
        self,
        edits: list[ContextEdit],
        context: AnalectRunContext,
    ) -> EditStats:
        """
        Apply context edits to memory.

        Directly modifies context.memory_manager.memory.messages
        """
        messages = context.memory_manager.get_session_memory().messages

        # Calculate tokens before using inherited token estimation
        token_lengths = await self.get_prompt_token_lengths(messages)
        tokens_before = sum(token_lengths)

        # Calculate statistics BEFORE applying edits
        total_tool_uses = sum(
            1
            for tool_use_id, msgs in self._tool_use_message_map.items()
            if msgs.tool_use is not None and tool_use_id not in self._ignored_tool_use_ids
        )
        ignored_count = len(
            [e for e in edits if e.operation == ContextEditOperation.IGNORE]
        )
        kept_count = total_tool_uses - ignored_count

        # Apply edits
        for edit in edits:
            tool_use_id = edit.tool_use_id
            tool_use_msgs = self._get_tool_use_messages(tool_use_id)

            if edit.operation == ContextEditOperation.IGNORE:
                await self._ignore_tool_use_messages(
                    tool_use_msgs, tool_use_id, context
                )
            elif edit.operation == ContextEditOperation.COMPACT:
                await self._compact_tool_use_messages(tool_use_msgs, edit, context)

        # Calculate tokens after using inherited token estimation
        messages_after = context.memory_manager.get_session_memory().messages
        token_lengths_after = await self.get_prompt_token_lengths(messages_after)
        tokens_after = sum(token_lengths_after)

        return EditStats(
            removed_count=ignored_count,
            compacted_count=len(
                [e for e in edits if e.operation == ContextEditOperation.COMPACT]
            ),
            kept_count=kept_count,
            tokens_before=tokens_before,
            tokens_after=tokens_after,
            tokens_saved=tokens_before - tokens_after,
            savings_percent=(
                ((tokens_before - tokens_after) / tokens_before * 100)
                if tokens_before > 0
                else 0
            ),
        )

    def _compact_tool_use_input(
        self,
        message: CfMessage,
        tool_use_id: str,
        new_input: dict[str, Any],
    ) -> None:
        """Replace tool use input with compacted version (in-place)."""
        # Modify the original message's content directly (no copy)
        content = message.content
        if isinstance(content, list):
            for i, item in enumerate(content):
                try:
                    # Use model_validate for type-safe handling
                    tool_use = ant.MessageContentToolUse.model_validate(item)
                    if tool_use.id == tool_use_id:
                        tool_use.input = new_input
                        content[i] = tool_use.model_dump(mode="json", exclude_none=True)
                        break
                except Exception:
                    # Not a tool_use or validation failed, skip to next item
                    continue

    def _compact_tool_result(
        self,
        message: CfMessage,
        tool_use_id: str,
        new_content: str,
    ) -> None:
        """Replace tool result content with compacted version (in-place)."""
        # Modify the original message's content directly (no copy)
        content = message.content
        if isinstance(content, list):
            for i, item in enumerate(content):
                try:
                    # Use model_validate for type-safe handling
                    tool_result = ant.MessageContentToolResult.model_validate(item)
                    if tool_result.tool_use_id == tool_use_id:
                        tool_result.content = new_content
                        content[i] = tool_result.model_dump(
                            mode="json", exclude_none=True
                        )
                        break
                except Exception:
                    # Not a tool_result or validation failed, skip to next item
                    continue

    async def _ignore_tool_use_messages(
        self,
        tool_use_msgs: _ToolUseMessages,
        tool_use_id: str,
        context: AnalectRunContext,
        placeholder_text: str | None = None,
    ) -> None:
        """Replace tool_use with placeholder and delete tool_result and tool_use_info.

        This unified method:
        1. Keeps the tool_use message but replaces content with placeholder
        2. Deletes the tool_result message
        3. Deletes the tool_use_info message

        Args:
            tool_use_msgs: The message references from the map
            tool_use_id: The tool use ID
            context: The analect run context
            placeholder_text: Optional custom placeholder text
        """
        # Replace tool_use with placeholder (keep the message, modify content)
        tool_use = tool_use_msgs.tool_use
        if tool_use is not None:
            self._compact_tool_use_input(
                tool_use,
                tool_use_id,
                self._create_tool_use_input_ignore_placeholder(),
            )
            self._ignored_tool_use_ids.add(tool_use_id)

        tool_result = tool_use_msgs.tool_result
        if tool_result is not None:
            self._compact_tool_result(
                tool_result,
                tool_use_id,
                placeholder_text or self._create_tool_result_ignore_placeholder(),
            )
            # Mark by the same tool_use_id (both sides share the same ID)
            self._ignored_tool_use_ids.add(tool_use_id)

        # Delete tool_use_info message (SYS message with <system_info>)
        tool_use_info_msg = tool_use_msgs.tool_use_info
        if tool_use_info_msg is not None:
            context.memory_manager.delete_message(tool_use_info_msg)

        # Remove from map to keep it clean (tool_use placeholder doesn't need tracking)
        self._tool_use_message_map.pop(tool_use_id, None)

    async def _compact_tool_use_messages(
        self,
        tool_use_msgs: _ToolUseMessages,
        edit: ContextEdit,
        context: AnalectRunContext,
    ) -> None:
        """Compact tool use messages with replacements (in-place modification).

        This unified method coordinates compaction of both tool_use input and tool_result.
        It delegates to the existing _compact_tool_use_input and _compact_tool_result methods
        for the actual in-place modifications.

        Args:
            tool_use_msgs: The message references from the map
            edit: The context edit with replacement values
            context: The analect run context
        """
        # Compact tool_use input if replacement provided
        # Cache property access to avoid repeated lookups
        tool_use = tool_use_msgs.tool_use
        if edit.tool_use_input_replacement is not None and tool_use is not None:
            self._compact_tool_use_input(
                tool_use,
                edit.tool_use_id,
                edit.tool_use_input_replacement,
            )

        # Compact tool_result if replacement provided
        # Cache property access to avoid repeated lookups
        tool_result = tool_use_msgs.tool_result
        if edit.tool_result_content_replacement is not None and tool_result is not None:
            self._compact_tool_result(
                tool_result,
                edit.tool_use_id,
                edit.tool_result_content_replacement,
            )
