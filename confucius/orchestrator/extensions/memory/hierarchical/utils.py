# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

import fnmatch
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .types import MemoryNode


def parse_frontmatter(content: str) -> tuple[List[str], str]:
    """Parse YAML frontmatter from markdown content."""
    if not content.startswith("---\n"):
        return [], content

    try:
        parts = content.split("---\n", 2)
        if len(parts) < 3:
            return [], content

        frontmatter_str = parts[1]
        actual_content = parts[2]

        frontmatter = yaml.safe_load(frontmatter_str)
        if isinstance(frontmatter, dict) and "tags" in frontmatter:
            tags = frontmatter["tags"]
            if isinstance(tags, list):
                return tags, actual_content
    except Exception:
        pass

    return [], content


def create_content_with_frontmatter(content: str, tags: List[str]) -> str:
    """Create markdown content with YAML frontmatter."""
    if not tags:
        return content

    frontmatter = {"tags": tags}
    yaml_str = yaml.dump(frontmatter, default_flow_style=False)
    return f"---\n{yaml_str}---\n{content}"


def should_merge_memory_dir(node: MemoryNode) -> bool:
    """Check if a directory node should be merged with its single child directory."""
    return (
        len(node.children) == 1
        and len(node.children[0].children) > 0  # Child is also a directory
        and not node.content  # Parent has no content
        and not node.tags  # Parent has no tags
    )


def merge_single_child_memory_dirs(node: MemoryNode) -> MemoryNode:
    """Merge directories that have only one child directory."""
    # First, merge children recursively
    merged_children = [merge_single_child_memory_dirs(child) for child in node.children]
    node.children = merged_children

    # Then check if this node should be merged with its single child
    if should_merge_memory_dir(node):
        child = node.children[0]
        # Combine the names with a separator for display
        merged_name = f"{node.name}/{child.name}"
        return MemoryNode(
            path=child.path,  # Use child's full path
            name=merged_name,  # Combined display name
            content=child.content,
            tags=child.tags,
            children=child.children,
        )

    return node


def matches_path_pattern(
    node: MemoryNode, pattern: Optional[str], base_dir: Path
) -> bool:
    """Check if node's relative path from memory base matches the given pattern."""
    if pattern is None:
        return True

    # Compute relative path from memory base directory
    try:
        relative_path = node.path.relative_to(base_dir)
        # For .md files, remove the extension for pattern matching
        relative_path_str = str(relative_path)
        if relative_path_str.endswith(".md"):
            relative_path_str = relative_path_str[:-3]
        return fnmatch.fnmatch(relative_path_str, pattern)
    except ValueError:
        # If node.path is not relative to base_dir, fall back to name matching
        return fnmatch.fnmatch(node.name, pattern)


def matches_content_pattern(node: MemoryNode, pattern: Optional[str]) -> bool:
    """Check if node content matches the given pattern."""
    if pattern is None:
        return True

    try:
        return bool(re.search(pattern, node.content, re.IGNORECASE))
    except re.error:
        # If regex is invalid, do simple substring search
        return pattern.lower() in node.content.lower()


def matches_tags(node: MemoryNode, required_tags: Optional[List[str]]) -> bool:
    """Check if node has all required tags."""
    if not required_tags:
        return True

    return all(tag in node.tags for tag in required_tags)


def node_matches_criteria(
    node: MemoryNode,
    path_pattern: Optional[str],
    content_pattern: Optional[str],
    tags: Optional[List[str]],
    base_dir: Path,
) -> bool:
    """Check if a node matches all search criteria."""
    return (
        matches_path_pattern(node, path_pattern, base_dir)
        and matches_content_pattern(node, content_pattern)
        and matches_tags(node, tags)
    )


def create_search_result(node: MemoryNode, base_dir: Path) -> Dict[str, Any]:
    """Create a search result dictionary from a node."""
    content_preview = (
        node.content[:100] + "..." if len(node.content) > 100 else node.content
    )

    # Use relative path from memory base directory for search results
    try:
        relative_path = node.path.relative_to(base_dir)
        file_path = str(relative_path)
        # Keep .md extension for display
    except ValueError:
        # Fallback to node name if relative path calculation fails
        file_path = node.name

    return {
        "path": file_path,
        "tags": node.tags,
        "content_preview": content_preview,
    }


def collect_matching_nodes(
    nodes: List[MemoryNode],
    path_pattern: Optional[str],
    content_pattern: Optional[str],
    tags: Optional[List[str]],
    max_results: int,
    results: List[Dict[str, Any]],
    base_dir: Path,
) -> None:
    """Recursively collect nodes that match search criteria."""
    for node in nodes:
        if len(results) >= max_results:
            return

        if node_matches_criteria(node, path_pattern, content_pattern, tags, base_dir):
            results.append(create_search_result(node, base_dir))

        if node.children:
            collect_matching_nodes(
                node.children,
                path_pattern,
                content_pattern,
                tags,
                max_results,
                results,
                base_dir,
            )


def cleanup_empty_parent_directories(file_path: Path, base_dir: Path) -> None:
    """Clean up empty parent directories after file deletion.

    Args:
        file_path: Path to the deleted file
        base_dir: Base directory to stop at (won't delete this directory)
    """
    parent_dir = file_path.parent

    # Remove empty parent directories up to base_dir
    while parent_dir != base_dir and parent_dir.exists():
        try:
            if not any(parent_dir.iterdir()):  # Directory is empty
                parent_dir.rmdir()
                parent_dir = parent_dir.parent
            else:
                break  # Directory not empty, stop here
        except OSError:
            break  # Error occurred, stop cleanup


# Cloud storage functions not implemented in this version
# Memory operations use local filesystem only
