# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from typing import cast, List, Pattern, Union

from pydantic import Field

from .regex import OperationType, RegularExpressionAccessPolicy


class BlockTemporaryFilesPolicy(RegularExpressionAccessPolicy):
    """Policy that blocks the creation of temporary or investigation files.

    This policy specifically targets files with extensions or naming patterns
    commonly used for temporary work, investigations, or experiments.
    It only blocks creation operations, allowing reading, updating, or deleting
    such files if they already exist.
    """

    blocked_patterns: List[Union[str, Pattern[str]]] = Field(
        default_factory=lambda: cast(
            List[Union[Pattern[str], str]],
            [
                # Common temporary file extensions
                r"\.(txt|tmp|temp|temporary|test|taxonomy|new|bak|swp|log|out)$",
                # Common temporary file prefixes
                r"/(tmp|temp|testing|investigation)(_|-).*",
                # Common temporary file suffixes
                r"(experiment|scratch|junk|delete|deleteme|remove|todo)\.",
                # Files that are likely just for testing or investigation
                r"test_?output\.\w+$",
                r"debug_?output\.\w+$",
                r"example_?output\.\w+$",
            ],
        ),
        description="Patterns for paths that are explicitly blocked",
    )

    operations: List[OperationType] = Field(
        default=cast(List[OperationType], ["create"]),
        description="Operations covered by this policy (others will always be allowed)",
    )
