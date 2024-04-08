from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List


class ParseOutputFromModel(BaseModel):
    summary: str = Field(description="Summary of the information gathered on the topic")
    facts: List[str] = Field(
        description="Three facts about the keyword been researched up on"
    )

    def to_dict(self):
        return {"summary": self.summary, "facts": self.facts}


data_parser: PydanticOutputParser = PydanticOutputParser(
    pydantic_object=ParseOutputFromModel
)
