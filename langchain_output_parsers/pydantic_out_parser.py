from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

## pydantic object class
class Person(BaseModel):
    name: str = Field(description="The person's name")
    age: int = Field(gt=18,description="The person's age")
    city: str = Field(description="The city where the person lives")

parcer = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional person of place {place} \n {format_instruction}',
    input_variables=['place'],
    partial_variables={'format_instruction': parcer.get_format_instructions()}
)
# prompt = template.format(place="New York")
# result = model.invoke(prompt)
# final_result = parcer.parse(result.content)
# print(prompt)

chain = template | model | parcer
result = chain.invoke({'place':'New York'}) ## we should pass a dictionary in invoke method
print(result) # this will be a dictionary with the keys as the name of the fields in the pydantic object

