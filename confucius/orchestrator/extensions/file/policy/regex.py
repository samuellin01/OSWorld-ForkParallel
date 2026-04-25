# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import re
from pathlib import Path
from typing import Any, List, Literal, Optional, Pattern, Union

from pydantic import Field

from .....core.analect.analect import AnalectRunContext

from .base import FileAccessPolicyBase, FileAccessResult

# Operation constants
OperationType = Literal["create", "read", "update", "delete"]
ALL_OPERATIONS: List[OperationType] = ["create", "read", "update", "delete"]


class RegularExpressionAccessPolicy(FileAccessPolicyBase):
    """Policy that controls file access based on regular expression patterns.

    This policy allows/blocks operations based on regex patterns applied to file paths.
    Operations not explicitly covered are always allowed (default permissive).

    You can customize error messages with format variables:
    - {path}: The full file path that was blocked
    - {file_name}: Just the file name portion of the path
    - {operation}: The operation that was attempted (create, read, update, delete)
    - {matched_pattern}: The specific pattern that caused the block
    """

    allowed_patterns: List[Union[str, Pattern[str]]] = Field(
        default=[],
        description="Patterns for paths that are explicitly allowed (empty list means allow all non-blocked)",
    )

    blocked_patterns: List[Union[str, Pattern[str]]] = Field(
        default=[],
        description="Patterns for paths that are explicitly blocked",
    )

    operations: List[OperationType] = Field(
        default=ALL_OPERATIONS,
        description="Operations covered by this policy (others will always be allowed)",
    )

    blocked_message: str = Field(
        default="The {operation} operation is not allowed for path {path} (matches blocked pattern '{matched_pattern}')",
        description="Message template for blocked files (supports format variables)",
    )

    not_allowed_message: str = Field(
        default="The {operation} operation is not allowed for path {path} (doesn't match any allowed patterns)",
        description="Message template when no allowed patterns match (supports format variables)",
    )

    allow_directory_read: bool = Field(
        default=True,
        description="Whether to bypass regex when listing directory contents",
    )

    @classmethod
    def _compile_list(
        cls, patterns: List[Union[str, Pattern[str]]]
    ) -> List[Pattern[str]]:
        """Compile a list of patterns to regex objects."""
        return [
            re.compile(p, re.IGNORECASE) if isinstance(p, str) else p for p in patterns
        ]

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self._allowed_patterns: List[Pattern[str]] = self._compile_list(
            self.allowed_patterns
        )
        self._blocked_patterns: List[Pattern[str]] = self._compile_list(
            self.blocked_patterns
        )

    def _is_operation_covered(self, operation: OperationType) -> bool:
        """Check if an operation is covered by this policy."""
        return operation in self.operations

    async def _on_blocked(
        self,
        path: Path,
        operation: OperationType,
        pattern: Pattern[str],
        context: Optional[AnalectRunContext] = None,
    ) -> Optional[FileAccessResult]:
        """Hook called when a path is blocked by a pattern.

        Subclasses can override this method to customize behavior when a path matches a blocked pattern.
        Return a FileAccessResult to override the default deny behavior, or None to use the default.

        Args:
            path: The file path being checked
            operation: The operation being performed (read, create, update, delete)
            pattern: The pattern that matched and caused the block
            context: The run context (optional)

        Returns:
            Optional[FileAccessResult]: Override result or None to use default deny
        """
        return None

    async def _on_not_allowed(
        self,
        path: Path,
        operation: OperationType,
        context: Optional[AnalectRunContext] = None,
    ) -> Optional[FileAccessResult]:
        """Hook called when a path doesn't match any allowed pattern.

        Subclasses can override this method to customize behavior when a path doesn't match any allowed pattern.
        Return a FileAccessResult to override the default deny behavior, or None to use the default.

        Args:
            path: The file path being checked
            operation: The operation being performed (read, create, update, delete)
            context: The run context (optional)

        Returns:
            Optional[FileAccessResult]: Override result or None to use default deny
        """
        return None

    async def _on_allowed(
        self,
        path: Path,
        operation: OperationType,
        pattern: Pattern[str],
        context: Optional[AnalectRunContext] = None,
    ) -> Optional[FileAccessResult]:
        """Hook called when a path is allowed by a pattern.

        Subclasses can override this method to customize behavior when a path matches an allowed pattern.
        Return a FileAccessResult to override the default allow behavior, or None to use the default.

        Args:
            path: The file path being checked
            operation: The operation being performed (read, create, update, delete)
            pattern: The pattern that matched and allowed access
            context: The run context (optional)

        Returns:
            Optional[FileAccessResult]: Override result or None to use default allow
        """
        return None

    async def _on_bypassed(
        self,
        path: Path,
        operation: OperationType,
        context: Optional[AnalectRunContext] = None,
    ) -> Optional[FileAccessResult]:
        """Hook called when a policy check is bypassed (not covered by this policy).

        Subclasses can override this method to customize behavior when an operation is not covered by this policy.
        Return a FileAccessResult to override the default allow behavior, or None to use the default.

        Args:
            path: The file path being checked
            operation: The operation being performed (read, create, update, delete)
            context: The run context (optional)

        Returns:
            Optional[FileAccessResult]: Override result or None to use default allow
        """
        return None

    async def _check_path(
        self,
        path: Path,
        operation: OperationType,
        context: Optional[AnalectRunContext] = None,
    ) -> FileAccessResult:
        """Common path checking logic used by all operation methods."""
        # Skip checks for operations not covered by this policy
        if not self._is_operation_covered(operation):
            return (
                await self._on_bypassed(path, operation, context)
                or FileAccessResult.allow()
            )

        path_str = str(path)

        # Check blocked patterns first
        for pattern in self._blocked_patterns:
            if pattern.search(path_str):
                # Format error message with variables
                message = self.blocked_message.format(
                    path=path_str,
                    file_name=path.name,
                    base_name=path.stem,
                    extension=path.suffix,
                    operation=operation,
                    matched_pattern=pattern.pattern,
                )
                # Allow hook method to override the block decision
                return await self._on_blocked(
                    path, operation, pattern, context
                ) or FileAccessResult.deny(message)

        # If we have allowed patterns, at least one must match
        if self._allowed_patterns:
            for pattern in self._allowed_patterns:
                if pattern.search(path_str):
                    return (
                        await self._on_allowed(path, operation, pattern, context)
                        or FileAccessResult.allow()
                    )
            # No allowed patterns matched
            message = self.not_allowed_message.format(
                path=path_str,
                file_name=path.name,
                operation=operation,
                matched_pattern="N/A",
            )
            # Allow hook method to override the not allowed decision
            return await self._on_not_allowed(
                path, operation, context
            ) or FileAccessResult.deny(message)

        # No allowed patterns specified and no blocked patterns matched
        return (
            await self._on_bypassed(path, operation, context)
            or FileAccessResult.allow()
        )

    async def check_read(
        self,
        path: Path,
        is_directory: bool = False,
        context: Optional[AnalectRunContext] = None,
    ) -> FileAccessResult:
        """Check if reading from the path is allowed based on regex patterns."""
        if self.allow_directory_read and is_directory:
            return FileAccessResult.allow()

        return await self._check_path(path, "read", context=context)

    async def check_create(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if creating a file at the path is allowed based on regex patterns."""
        return await self._check_path(path, "create", context=context)

    async def check_update(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if updating a file at the path is allowed based on regex patterns."""
        return await self._check_path(path, "update", context=context)

    async def check_delete(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if deleting a file at the path is allowed based on regex patterns."""
        return await self._check_path(path, "delete", context=context)
