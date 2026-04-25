# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from textwrap import dedent
from typing import cast, List, Pattern, Union

from pydantic import Field

from .regex import RegularExpressionAccessPolicy


# Define documentation file extensions
allowed_extensions = r"\.(md|txt|rst|adoc|rdoc|textile|org|wiki|confluence)$"

# Define source code file extensions to block
source_code_extensions = (
    r"\.(py|js|jsx|ts|tsx|c|cpp|cc|cxx|h|hpp|hxx|go|java|rb|php|scala|"
    r"swift|m|mm|kt|cs|fs|rs|sql|sh|bash|pl|elm|clj|hs|ex|lua)"
)

# Define patterns for allowed documentation files
allowed_regex_list: list[str | Pattern[str]] = [
    # Documentation directories
    r"/(docs|wiki|documentation|manual|guide)/.*",
    # README files (case insensitive)
    r"[Rr][Ee][Aa][Dd][Mm][Ee]" + allowed_extensions,
    r"[Rr][Ee][Aa][Dd][Mm][Ee](?:\.\w+)?" + allowed_extensions,
    # Documentation-specific files
    r"(CONTRIBUTING|LICENSE|CHANGELOG|AUTHORS|INSTALL|ARCHITECTURE|DESIGN)"
    + allowed_extensions,
    # Any markdown file
    r"\.md$",
    # Documentation directories with any extension
    r"/(docs|wiki|documentation)/.*" + allowed_extensions,
    # Specific documentation files that might not be in doc directories
    r"(tutorial|howto|guide|manual|reference|doc)" + allowed_extensions,
    # GitHub/GitLab specific documentation files
    r"\.github/.*" + allowed_extensions,
    r"CODEOWNERS",
    r"SECURITY\.md$",
    # OpenAPI/Swagger documentation
    r"swagger\.ya?ml$",
    r"openapi\.ya?ml$",
]


def get_blocked_message() -> str:
    valid_patterns = "\n".join(
        f"<pattern>{pattern}</pattern>" for pattern in allowed_regex_list
    )
    return dedent(f"""
        <error>
        You are not permitted to read the source code files themselves. Please only read the documentation files.
        </error>
        <next_steps>
        Valid documentation file patterns:
        {valid_patterns}
        </next_steps>
    """)


class AllowOnlyReadmesPolicy(RegularExpressionAccessPolicy):
    """Policy that allows access only to documentation files, not implementation files.

    This policy is designed for scenarios where agents should have access to
    documentation but not to source code or implementation details. It's useful
    for documentation-focused agents that need to understand system architecture
    without examining implementation details.
    """

    allowed_patterns: List[Union[str, Pattern[str]]] = Field(
        default_factory=lambda: allowed_regex_list,
        description="Patterns for paths that are explicitly allowed (empty list means allow all non-blocked)",
    )

    blocked_patterns: List[Union[str, Pattern[str]]] = Field(
        default_factory=lambda: cast(
            List[Union[str, Pattern[str]]],
            [
                # Source code files
                source_code_extensions + "$",
                # Build files
                r"(Makefile|CMakeLists\.txt|\.cmake$|BUCK|TARGETS|\.bzl$|package\.json|pom\.xml)$",
                # Configuration files
                r"\.(yaml|yml|json|xml|ini|cfg|conf|properties|toml|env)$",
                # Binary files
                r"\.(exe|bin|o|a|so|dll|lib|jar|war|class|pyc|pyd)$",
                # Data files
                r"\.(csv|tsv|xls|xlsx|db|sqlite|parquet|avro)$",
                # Image files
                r"\.(jpg|jpeg|png|gif|bmp|svg|ico|webp)$",
                # Miscellaneous files unlikely to be documentation
                r"\.(zip|tar|gz|tgz|rar|7z|log|cache)$",
            ],
        ),
        description="Patterns for paths that are explicitly blocked",
    )

    blocked_message: str = Field(
        default_factory=get_blocked_message,
        description="Message template for blocked files (supports format variables)",
    )
