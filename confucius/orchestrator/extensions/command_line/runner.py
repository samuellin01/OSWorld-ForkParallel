# pyre-strict


import asyncio
from subprocess import PIPE

from pydantic import BaseModel, Field

from ....common.code import CodeBlock
from ....utils.string import truncate
from ...tags import Tag


class CommandLineInput(BaseModel):
    identifier: str = Field(
        "__cmd_run__", description="Identifier of the command line execution"
    )
    command: str = Field(..., description="Command to execute")
    cwd: str | None = Field(
        None, description="Current working directory of the command line"
    )
    attrs: dict[str, str] | None = Field(
        default=None, description="Attributes of the command line"
    )
    max_output_lines: int = Field(
        default=100,
        description="Maximum number of lines of the output to include in the markdown string",
    )
    max_output_length: int | None = Field(
        default=None,
        description="Maximum length of the output to include in the response string",
    )
    env: dict[str, str] | None = Field(
        default=None, description="Environment variables for the command line execution"
    )


class CommandLineOutput(BaseModel):
    identifier: str = Field(..., description="Identifier of the command line execution")
    cwd: str | None = Field(
        ..., description="Current working directory of the command line"
    )
    stdout: str = Field(
        ..., description="Standard output of the command line execution"
    )
    stderr: str = Field(..., description="Standard error of the command line execution")
    returncode: int | None = Field(
        ..., description="Return code of the command line execution"
    )

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def to_tag(self) -> Tag:
        """
        Convert the result to a message string in XML tag.

        Returns:
            Tag: The XML tag representation of the result.
        """
        output_contents: list[Tag | str] = []
        if self.stderr:
            output_contents.append(Tag(name="stderr", contents=self.stderr))
        if self.stdout:
            output_contents.append(Tag(name="stdout", contents=self.stdout))
        attributes = {
            "identifier": self.identifier,
        }
        if self.returncode is not None:
            attributes["returncode"] = str(self.returncode)
        if self.cwd is not None:
            attributes["cwd"] = self.cwd
        return Tag(
            name="command_line_output",
            attributes=attributes,
            contents=output_contents,
        )

    def to_markdown(self) -> str:
        """
        Convert the result to a markdown string.

        Returns:
            str: The markdown string representation of the result.
        """
        res = []
        if self.stderr:
            res.append(CodeBlock(content=self.stderr).to_markdown(language="console"))
        if self.stdout:
            res.append(CodeBlock(content=self.stdout).to_markdown(language="console"))
        return "\n\n".join(res) + "\n" if res else "No output"


async def run_command_line(inp: CommandLineInput) -> CommandLineOutput:
    """
    Execute a command line instruction asynchronously.

    Args:
        inp (CommandLineInput): The input parameters for the command line execution,
        including the command, current working directory, and other attributes.

    Returns:
        CommandLineOutput: The output of the command line execution, including
        standard output, standard error, return code, and the working directory.
    """

    process = await asyncio.create_subprocess_shell(
        inp.command, env=inp.env, stdout=PIPE, stderr=PIPE, cwd=inp.cwd
    )
    stdout, stderr = await process.communicate()
    return CommandLineOutput(
        identifier=inp.identifier,
        stdout=truncate(
            s=stdout.decode("utf-8") if stdout else "",
            max_lines=inp.max_output_lines,
            max_length=inp.max_output_length,
        ),
        stderr=truncate(
            s=stderr.decode("utf-8") if stderr else "",
            max_lines=inp.max_output_lines,
            max_length=inp.max_output_length,
        ),
        returncode=process.returncode,
        cwd=inp.cwd,
    )
