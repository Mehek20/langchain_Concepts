from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()

model = ChatOpenAI(model="gpt-4o-2024-08-06")

# Define a TypedDict for the structured output
# class Review(TypedDict):
#     summary: str
#     sentiment: str

## the above mention form may createsome ambiguity in near furure because the model may not be able to understand the content of the TypedDict
## so we can use annotated to make it clear

class Review(TypedDict):
    key_themes: Annotated[str, "Write down all the key themes discussed in the review"]
    summary: Annotated[str, "The summary of the review"]
    # sentiment: Annotated[str, "The sentiment of the review either positive negative or neutral"]
    sentiment: Annotated[Literal['pos','neg','neu'], "The sentiment of the review either positive negative or neutral"] ## this will make sure that the db will only accept the values pos,neg,neu and model will provide it accodingly
    pros: Annotated[Optional[list[str]], "Write down all the pros of the review inside a list"]  ## optional feature there indicates the model does'nt need to cumplusorily need to add this feild if the prompt does'nt have any pros
    cons: Annotated[Optional[list[str]], "Write down all the cons of the review inside a list"]

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also the UI looks outdated compared to other brands. Hoping for a software update to fix this."""
)

print(result)
print('\n')
print(result['summary'])
print('\n')
print(result['sentiment'])