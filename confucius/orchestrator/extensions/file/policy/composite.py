# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from pathlib import Path
from typing import List, Optional, TypeVar

from pydantic import Field

from .....core.analect.analect import AnalectRunContext

from .base import FileAccessPolicyBase, FileAccessResult

# Type variable for policy check methods
T = TypeVar("T")


class CompositeFileAccessPolicy(FileAccessPolicyBase):
    """Composite policy that combines multiple policies with an AND operation.

    This policy allows an operation only if ALL constituent policies allow it.
    The first denial result will be returned, with its error message.
    """

    policies: List[FileAccessPolicyBase] = Field(
        default_factory=list,
        description="List of policies to apply (all must allow for operation to proceed)",
    )

    async def _check_policy_method(
        self,
        path: Path,
        method_name: str,
        context: Optional[AnalectRunContext] = None,
        **kwargs: T,
    ) -> FileAccessResult:
        """Common implementation for all check methods.

        Args:
            path: The file or directory path to check
            method_name: Name of the policy method to call
            context: The run context (optional)
            **kwargs: Additional arguments to pass to the policy method

        Returns:
            FileAccessResult: Result from the first denying policy, or allow if all allow
        """
        for policy in self.policies:
            # Get the policy method by name and call it
            method = getattr(policy, method_name)
            result = await method(path, context=context, **kwargs)

            # Short-circuit on first denial
            if not result.allowed:
                return result

        return FileAccessResult.allow()

    async def check_read(
        self,
        path: Path,
        is_directory: bool = False,
        context: Optional[AnalectRunContext] = None,
    ) -> FileAccessResult:
        """Check if all policies allow reading from the path."""
        return await self._check_policy_method(
            path, "check_read", context=context, is_directory=is_directory
        )

    async def check_create(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if all policies allow creating a file at the path."""
        return await self._check_policy_method(path, "check_create", context=context)

    async def check_update(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if all policies allow updating a file at the path."""
        return await self._check_policy_method(path, "check_update", context=context)

    async def check_delete(
        self, path: Path, context: Optional[AnalectRunContext] = None
    ) -> FileAccessResult:
        """Check if all policies allow deleting a file at the path."""
        return await self._check_policy_method(path, "check_delete", context=context)
