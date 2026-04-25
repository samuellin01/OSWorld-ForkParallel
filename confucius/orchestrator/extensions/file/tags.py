# pyre-strict

from pathlib import Path

from pydantic import Field, model_validator

from ...tags import Tag


class File(Tag):
    name: str = "file"
    file_path: str | Path | None = Field(None, description="file path")

    @model_validator(mode="after")
    def append_attrs(self) -> "File":  # noqa: B902
        if self.file_path is not None:
            self.attributes["file_path"] = str(self.file_path)
        return self
