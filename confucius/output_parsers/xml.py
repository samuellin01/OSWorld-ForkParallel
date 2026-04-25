# pyre-strict


from typing import List

from bs4 import BeautifulSoup
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.outputs import Generation
from pydantic import BaseModel, Field

from ..core.analect import AnalectRunContext

DEFAULT_XML_FORMAT_INSTRUCTIONS = """\
The output should be formatted as a XML file. Remember to always open and close all the tags.

As an example, for the tags ["foo", "bar", "baz"]:
1. String "<foo>\n   <bar>\n      <baz></baz>\n   </bar>\n</foo>" is a well-formatted instance of the schema. 
2. String "<foo>\n   <bar>\n   </foo>" is a badly-formatted instance.
"""


class XMLOutput(BaseModel):
    soup: BeautifulSoup = Field(..., description="BeautifulSoup object")

    class Config:
        arbitrary_types_allowed = True


class XMLOutputParser(BaseOutputParser[XMLOutput]):
    """
    Parses a xml string into a xml.Element object.
    """

    format_instructions: str | None = Field(
        default=None,
        description="The user defined format instructions that will override the automatic generated one",
    )
    parser: str = Field("lxml-xml", description="The parser to use")
    root_tag: str = Field("root", description="The root tag of the xml")

    class Config:
        arbitrary_types_allowed = True

    def parse(self, text: str) -> XMLOutput:
        raise NotImplementedError("Only the aparse function is implemented")

    async def aparse(self, text: str) -> XMLOutput:
        try:
            soup = BeautifulSoup(text, self.parser)
            if soup.find(self.root_tag) is None:
                soup = BeautifulSoup(
                    f"<{self.root_tag}>{text}</{self.root_tag}>", self.parser
                )
            return XMLOutput(soup=soup)
        except Exception as exc:
            msg = f"Failed to parse the document using {self.parser} parser. Got: {exc}"
            raise OutputParserException(msg, llm_output=text)

    async def aparse_result(
        self, result: List[Generation], *, partial: bool = False
    ) -> XMLOutput:
        return await self.aparse(result[0].text)

    def get_format_instructions(self) -> str:
        if self.format_instructions is not None:
            return self.format_instructions

        return DEFAULT_XML_FORMAT_INSTRUCTIONS

    async def aget_format_instructions(self, context: AnalectRunContext) -> str:
        return self.get_format_instructions()

    @property
    def _type(self) -> str:
        return "xml"
