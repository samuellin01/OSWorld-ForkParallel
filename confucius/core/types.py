# pyre-strict
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel


class MessageType(str, Enum):
    UNKNOWN = "UNKNOWN"
    HUMAN = "HUMAN"
    AI = "AI"
    SYS = "SYS"
    ERR = "ERR"
    LOG = "LOG"
    WARN = "WARN"
    CONTEXT = "CONTEXT"
    ARTIFACT = "ARTIFACT"


class RunStatus(str, Enum):
    UNKNOWN = "UNKNOWN"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    WARNING = "WARNING"


class Tag(BaseModel):
    label: str
    value: Optional[str] = None
    color: Optional[str] = None


class ArtifactInfoAttachment(BaseModel):
    name: str
    version: int
    display_name: Optional[str] = None


class FileAttachment(BaseModel):
    name: Optional[str] = None
    mime: Optional[str] = None
    data: Optional[str] = None
    everstore_handle: Optional[str] = None
    manifold_path: Optional[str] = None
    url: Optional[str] = None


class LinkAttachment(BaseModel):
    href: str
    target: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None


# Use a simple Union for content instead of a custom wrapper
MessageAttachmentContent = ArtifactInfoAttachment | FileAttachment | LinkAttachment


class MessageAttachment(BaseModel):
    uuid: str
    content: MessageAttachmentContent
