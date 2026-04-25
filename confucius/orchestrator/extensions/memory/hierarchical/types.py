# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

# pyre-strict

from pathlib import Path

from pydantic import BaseModel, Field


class MemoryNode(BaseModel):
    """
    A node in the memory graph.
    """

    path: Path = Field(..., description="Full path of file or directory")
    name: str = Field(..., description="Display name of the node")
    content: str = Field("", description="the content of the node")
    tags: list[str] = Field(
        default_factory=list, description="tags associated with the node"
    )
    children: list["MemoryNode"] = Field(
        default_factory=list, description="children of the node"
    )


class Memory(BaseModel):
    """
    A memory graph.
    """

    nodes: list[MemoryNode] = Field(
        default_factory=list, description="nodes in the memory graph"
    )
