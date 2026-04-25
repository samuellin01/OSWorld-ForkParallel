# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from __future__ import annotations

import logging
import threading

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Set

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field, PrivateAttr, validator

from . import types as cf

logger: logging.Logger = logging.getLogger()


def _get_filtered_messages_by_entry_name(
    memory_manager: CfMemoryManager, entry_name: str
) -> List[CfMessage]:
    return [
        msg for msg in memory_manager.memory.messages if msg.entry_name == entry_name
    ]


def _collect_session_messages(memory_manager: CfMemoryManager) -> List[CfMessage]:
    if not memory_manager.parent_memory:
        return memory_manager.memory.messages

    return (
        _collect_session_messages(memory_manager.parent_memory)
        + memory_manager.memory.messages
    )


def _collect_entry_messages(
    memory_manager: CfMemoryManager,
    entry_name: str,
) -> List[CfMessage]:
    parent_memory = memory_manager.parent_memory
    if parent_memory is None:
        return _get_filtered_messages_by_entry_name(memory_manager, entry_name)

    return _collect_entry_messages(
        parent_memory, entry_name
    ) + _get_filtered_messages_by_entry_name(memory_manager, entry_name)


def _get_filtered_messages_by_runnable(
    memory_manager: CfMemoryManager, runnable: Runnable
) -> List[CfMessage]:
    """Filter messages in memory by the class name of the runnable."""
    runnable_name = runnable.get_name()
    return [
        msg
        for msg in memory_manager.memory.messages
        if msg.runnable_name is not None and msg.runnable_name == runnable_name
    ]


def _collect_runnable_messages(
    memory_manager: CfMemoryManager,
    runnable: Runnable,
) -> List[CfMessage]:
    """Collect all messages for a specific runnable class from current and parent memories."""
    parent_memory = memory_manager.parent_memory
    if parent_memory is None:
        return _get_filtered_messages_by_runnable(memory_manager, runnable)

    return _collect_runnable_messages(
        parent_memory, runnable
    ) + _get_filtered_messages_by_runnable(memory_manager, runnable)


def _filter_analect_messages(
    retained_message_types: Set[cf.MessageType], messages: List[CfMessage]
) -> CfMemory:
    filtered_memory: List[CfMessage] = [
        message for message in messages if message.type in retained_message_types
    ]
    return CfMemory(messages=filtered_memory)


class HistoryVisibility(Enum):
    SESSION = "session"
    ENTRY = "entry"
    ANALECT = "analect"
    RUNNABLE = "runnable"


class ChildContextMemoryOptions(BaseModel):
    included_message_types: Set[cf.MessageType] = Field(
        default_factory=lambda: {
            cf.MessageType.AI,
            cf.MessageType.HUMAN,
            cf.MessageType.SYS,
        },
        description="list of message types allowed by child",
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


class ChildContextOptions(BaseModel):
    memory_options: ChildContextMemoryOptions = Field(
        default_factory=ChildContextMemoryOptions, description="memory options"
    )


class CfMessageCounter:
    def __init__(self) -> None:
        self._counter: int = 0
        self._lock: threading.Lock = threading.Lock()

    def get_next(self) -> int:
        with self._lock:
            self._counter += 1
            return self._counter

    def reset(self) -> None:
        with self._lock:
            self._counter = 0


_global_messages_counter = CfMessageCounter()


class CfMessage(BaseModel):
    entry_name: Optional[str] = Field(None, description="entry name of this message")
    runnable_name: Optional[str] = Field(
        None, description="class name of the runnable that generated this message"
    )
    path: List[str] = Field(
        default_factory=list, description="hierarchical path of the message context"
    )
    type: cf.MessageType = Field(
        cf.MessageType.HUMAN, description="the type of this message"
    )
    content: str | list[str | dict[str, Any]] = Field(
        "", description="the content of this message"
    )
    attachments: List[cf.MessageAttachment] = Field(
        default_factory=list, description="list of attachments of this message"
    )
    additional_kwargs: dict[str, Any] = Field(default_factory=dict)
    _sequence_id: int = PrivateAttr(default_factory=_global_messages_counter.get_next)

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def __lt__(self, other: CfMessage) -> bool:
        return self._sequence_id < other._sequence_id

    async def to_lc_messages(self) -> List[BaseMessage]:
        if self.type == cf.MessageType.AI:
            return [
                AIMessage(
                    content=self.content, additional_kwargs=self.additional_kwargs
                )
            ]
        elif self.type == cf.MessageType.SYS:
            return [
                SystemMessage(
                    content=self.content, additional_kwargs=self.additional_kwargs
                )
            ]
        elif self.type == cf.MessageType.HUMAN:
            human_msgs: list[BaseMessage] = []
            if self.content:
                human_msgs.append(
                    HumanMessage(
                        content=self.content, additional_kwargs=self.additional_kwargs
                    )
                )

            for attachment in self.attachments:
                if isinstance(attachment.content, cf.FileAttachment):
                    human_msgs.append(
                        HumanMessage(
                            content="",
                            additional_kwargs={
                                **self.additional_kwargs,
                                # pyre-ignore: Undefined attribute [16]
                                "attachments": [attachment.content.to_dict()],
                            },
                        )
                    )

            return human_msgs

        raise ValueError(
            f"Unknown message type: {self.type} when converting to langchain message"
        )


class CfMemory(BaseModel):
    """Memory that stores a list of messages"""

    messages: List[CfMessage] = Field(
        default_factory=list, description="list of messages"
    )

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @validator("messages")
    def sort_messages(
        cls,  # noqa
        v: list[CfMessage],
    ) -> list[CfMessage]:
        return sorted(v)

    async def to_lc_messages(self) -> List[BaseMessage]:
        return [
            lc_msg for msg in self.messages for lc_msg in (await msg.to_lc_messages())
        ]

    def add_message(self, message: CfMessage) -> None:
        self.messages.append(message)
        self.messages.sort()

    def add_messages(self, messages: Iterable[CfMessage]) -> None:
        self.messages.extend(messages)
        self.messages.sort()

    def clear(self) -> None:
        self.messages.clear()

    def delete_message(self, target: int | CfMessage) -> Optional[CfMessage]:
        """
        Delete a single message by its sequence_id or by CfMessage object.

        Args:
            target: Either an integer sequence_id or a CfMessage object
                   (will use its sequence_id for matching)

        Returns:
            The deleted CfMessage if found, None otherwise
        """
        sequence_id = target if isinstance(target, int) else target._sequence_id

        for i, msg in enumerate(self.messages):
            if msg._sequence_id == sequence_id:
                return self.messages.pop(i)
        return None

    def delete_messages(
        self, predicate: Callable[[CfMessage], bool]
    ) -> List[CfMessage]:
        """
        Delete all messages matching the given predicate function.

        Args:
            predicate: A function that takes a CfMessage and returns True
                      if the message should be deleted

        Returns:
            List of all deleted CfMessages (in their original order)
        """
        deleted_messages: List[CfMessage] = []
        remaining_messages: List[CfMessage] = []

        for msg in self.messages:
            if predicate(msg):
                deleted_messages.append(msg)
            else:
                remaining_messages.append(msg)

        self.messages = remaining_messages
        return deleted_messages


class CfMemoryManager(BaseModel):
    memory: CfMemory = Field(
        default=CfMemory(), description="the memory of the current analect"
    )
    parent_memory: Optional[CfMemoryManager] = Field(
        default=None, description="pointer to parent memory"
    )
    entry_name: Optional[str] = Field(None, description="the name of the current entry")
    path: List[str] = Field(
        default_factory=list, description="hierarchical path of the memory manager"
    )
    runnable: Runnable | None = Field(
        None, description="the runnable that is currently being executed", exclude=True
    )
    _memory_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        if self.parent_memory and not self.path:
            self.path = self.parent_memory.path.copy()

        if self.runnable:
            self.path.append(self.runnable.get_name())

    class Config:
        arbitrary_types_allowed = True

    def _message_is_from_child(self, message: CfMessage) -> bool:
        other_path = message.path

        if len(other_path) <= len(self.path):
            # Path is shorter so can't be a child
            return False

        for i, path_part in enumerate(self.path):
            # We already know this can't happen due to the check above but defensive programming
            assert i < len(other_path)

            if path_part != other_path[i]:
                return False

        # self.path is a sub-list of other_path meaning that message is a child of this memory manager
        return True

    def _filter_out_child_messages(self, messages: list[CfMessage]) -> list[CfMessage]:
        return [
            message for message in messages if not self._message_is_from_child(message)
        ]

    def get_session_memory(self, include_children: bool = True) -> CfMemory:
        with self._memory_lock:
            messages = _collect_session_messages(self)

            if not include_children:
                messages = self._filter_out_child_messages(messages)

            return CfMemory(messages=messages)

    def get_entry_memory(self, include_children: bool = True) -> CfMemory:
        with self._memory_lock:
            if not self.entry_name:
                return CfMemory()
            messages = _collect_entry_messages(self, self.entry_name)

            if not include_children:
                messages = self._filter_out_child_messages(messages)

            return CfMemory(messages=messages)

    def get_analect_memory(self, include_children: bool = True) -> CfMemory:
        with self._memory_lock:
            messages = self.memory.messages

            if not include_children:
                messages = self._filter_out_child_messages(messages)

            return CfMemory(messages=messages)

    def get_runnable_memory(self) -> CfMemory:
        with self._memory_lock:
            if not self.runnable:
                return CfMemory()
            return CfMemory(messages=_collect_runnable_messages(self, self.runnable))

    def get_memory_by_visibility(
        self, visibility: HistoryVisibility, include_children: bool = True
    ) -> CfMemory:
        match visibility:
            case HistoryVisibility.SESSION:
                return self.get_session_memory(include_children)
            case HistoryVisibility.ENTRY:
                return self.get_entry_memory(include_children)
            case HistoryVisibility.ANALECT:
                return self.get_analect_memory(include_children)
            case HistoryVisibility.RUNNABLE:
                return self.get_runnable_memory()
            case _:
                raise ValueError(f"Unknown history visibility: {visibility}")

    def add_messages(self, messages: List[CfMessage]) -> None:
        with self._memory_lock:
            new_messages = []
            for message in messages:
                update_fields = {}
                if self.entry_name:
                    update_fields["entry_name"] = self.entry_name
                if not message.runnable_name and self.runnable:
                    update_fields["runnable_name"] = self.runnable.get_name()
                if not message.path and self.path:
                    update_fields["path"] = self.path.copy()
                new_messages.append(message.model_copy(update=update_fields))
            self.memory.add_messages(new_messages)

    def consolidate_messages(
        self,
        child_memory_manager: CfMemoryManager,
        child_context_options: Optional[ChildContextOptions] = None,
        child_retained_message_types: Optional[Set[cf.MessageType]] = None,
    ) -> None:
        with self._memory_lock:
            if child_context_options is None:
                child_context_options = ChildContextOptions()

            if child_retained_message_types is None:
                child_retained_message_types = {
                    cf.MessageType.HUMAN,
                    cf.MessageType.AI,
                    cf.MessageType.SYS,
                }

            retained_message_types: Set[cf.MessageType] = (
                child_context_options.memory_options.included_message_types
                & child_retained_message_types
            )
            child_memory: CfMemory = _filter_analect_messages(
                retained_message_types, child_memory_manager.memory.messages
            )
        self.add_messages(child_memory.messages)

    def clear_messages(self) -> None:
        with self._memory_lock:
            self.memory.clear()

    def delete_message(self, target: int | CfMessage) -> Optional[CfMessage]:
        """
        Delete a single message from the memory.

        Args:
            target: Either an integer sequence_id or a CfMessage object

        Returns:
            The deleted CfMessage if found, None otherwise
        """
        with self._memory_lock:
            return self.memory.delete_message(target)

    @property
    def is_empty(self) -> bool:
        with self._memory_lock:
            return len(self.memory.messages) == 0

    async def save(self, path: str, overwrite: bool = False) -> None:
        if self.is_empty:
            logger.info("No data to save")
            return

        logger.info(f"Saving memory to {path}")
        with self._memory_lock:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            if overwrite or not Path(path).exists():
                Path(path).write_text(self.model_dump_json(exclude_defaults=True))

    async def load(self, path: str) -> CfMemoryManager:
        logger.info(f"Loading memory from {path}")
        self = CfMemoryManager.model_validate_json(Path(path).read_text())
        return self
