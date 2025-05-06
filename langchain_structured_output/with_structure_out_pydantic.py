from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal
from pydantic import BaseModel,Field

load_dotenv()

model = ChatOpenAI(model="gpt-4o-2024-08-06")

# Define a TypedDict for the structured output
# class Review(TypedDict):
#     summary: str
#     sentiment: str

## the above mention form may createsome ambiguity in near furure because the model may not be able to understand the content of the TypedDict
## so we can use annotated to make it clear

class Review(BaseModel):

    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    
    summary: str = Field(description="The summary of the review")
    sentiment: Literal['pos','neg','neu'] = Field(description="The sentiment of the review either positive negative or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros of the review inside a list")  ## optional feature there indicates the model does'nt need to cumplusorily need to add this feild if the prompt does'nt have any pros
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons of the review inside a list")

    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also the UI looks outdated compared to other brands. Hoping for a software update to fix this."""
)


print('\n')
print(result.summary)
print('\n')
print(result.sentiment)