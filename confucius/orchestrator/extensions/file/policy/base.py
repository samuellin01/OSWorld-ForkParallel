# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

from .....core.analect.analect import AnalectRunContext


@dataclass
class FileAccessResult:
    """Result of a file access check with optional error message."""

    allowed: bool
    message: Optional[str] = None

    @classmethod
    def allow(cls) -> "FileAccessResult":
        """Allow the operation."""
        return cls(allowed=True)

    @classmethod
    def deny(cls, message: str) -> "FileAccessResult":
        """Deny the operation with an error message."""
        return cls(allowed=False, message=message)


class FileAccessPolicyBase(BaseModel):
    """Policy controlling file operations with detailed permissions and error messages."""

    async def check_read(
        self,
        path: Path,
        is_directory: bool = False,
        context: Optional[AnalectRunContext] = None,
    ) -> FileAccessResult:
        """Check if reading from a path is allowed.

        Args:
            path: The file or directory path to check
            is_directory: True if the path is a directory, False for a file
            context: The run context (optional)

        Returns:
            FileAccessResult: The result with allowed status and optional message
        """
        raise NotImplementedError()

    async def check_create(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if creating a file at path is allowed.

        Args:
            path: The file path to check
            context: The run context (optional)

        Returns:
            FileAccessResult: The result with allowed status and optional message
        """
        raise NotImplementedError()

    async def check_update(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if updating a file at path is allowed.

        Args:
            path: The file path to check
            context: The run context (optional)

        Returns:
            FileAccessResult: The result with allowed status and optional message
        """
        raise NotImplementedError()

    async def check_delete(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if deleting a file at path is allowed.

        Args:
            path: The file path to check
            context: The run context (optional)

        Returns:
            FileAccessResult: The result with allowed status and optional message
        """
        raise NotImplementedError()
