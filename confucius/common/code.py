# pyre-strict
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class CodeBlock(BaseModel):
    """A simple code block representation for OSS compatibility."""

    content: str = Field("", description="content of the code block")
    name: str = Field("", description="name of the code block")

    def __str__(self) -> str:
        """Return the string representation of the code block."""
        return self.to_markdown()

    def _repr_markdown_(self) -> str:
        """Return the string representation of the code block."""
        return self.to_markdown()

    def to_markdown(self, language: str | None = None) -> str:
        """Convert the code block to markdown."""
        return f"```{language or self.name}\n{self.content}\n```"

    def to_file(
        self, directory: str | Path | None = None, ext: str | None = None
    ) -> str:
        """Save the code block to a file.

        Args:
            directory: The directory where the file should be saved. If not specified,
                the current working directory will be used.
            ext: The extension of the file. If not specified, no extension will be added.

        Returns:
            The full path to the saved file.
        """
        if directory is None:
            directory = Path.cwd()
        if ext is None:
            ext = ""
        elif not ext.startswith("."):
            ext = f".{ext}"
        assert self.name != "", "code block must have a name for saving"
        file_name = f"{self.name}{ext}"
        file_path = Path(directory) / file_name
        file_path.parent.mkdir(
            parents=True, exist_ok=True
        )  # ensure the directory exists
        file_path.write_text(self.content)
        return str(file_path)

    @classmethod
    def from_file(cls, file_path: str | Path) -> CodeBlock:
        """Load the code block from a file.

        Args:
            file_path: The path to the file.

        Returns:
            The code block loaded from the file.
        """
        file_path = Path(file_path)
        return cls(name=file_path.name, content=file_path.read_text())


class CodeBlocks(BaseModel):
    blocks: list[CodeBlock] = Field(..., description="code blocks")

    def __str__(self) -> str:
        """Return the string representation of the code block."""
        return self.to_markdown()

    def to_markdown(self) -> str:
        """Convert the code gen output to markdown."""
        return "\n\n".join(block.to_markdown() for block in self.blocks)

    def to_files(
        self, directory: str | Path | None = None, ext: str | None = None
    ) -> list[str]:
        """Save the code blocks to files.

        Args:
            directory: The directory where the files should be saved. If not specified,
                the current working directory will be used.
            ext: The extension of the file. If not specified, no extension will be added.

        Returns:
            The full paths to the saved files.
        """
        return [block.to_file(directory, ext) for block in self.blocks]

    @classmethod
    def from_files(cls, file_paths: list[str | Path]) -> CodeBlocks:
        """Load the code blocks from files.

        Args:
            file_paths: The paths to the files.

        Returns:
            The code blocks loaded from the files.
        """
        return cls(blocks=[CodeBlock.from_file(file_path) for file_path in file_paths])
