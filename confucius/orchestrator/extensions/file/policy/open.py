# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict
from pathlib import Path
from typing import Optional

from .....core.analect.analect import AnalectRunContext

from .base import FileAccessPolicyBase, FileAccessResult


class OpenFileAccessPolicy(FileAccessPolicyBase):
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
        return FileAccessResult.allow()

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
        return FileAccessResult.allow()

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
        return FileAccessResult.allow()

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
        return FileAccessResult.allow()
