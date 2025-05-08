from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResposneSchema

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    repo_id="google/gemma-3-27b-it",
    task="text-generation"
)

model = ChatHuggingFace(llm=llm)

## Schema
schema = [
    ResposneSchema(name="fact1", description="First fact about the topic"),
    ResposneSchema(name="fact2", description="Second fact about the topic"),    
    ResposneSchema(name="fact3", description="Third fact about the topic"),
]
parcer = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parcer.get_format_instructions()}
)
chain = template | model | parcer

# prompt = template.format(topic="black hole")
# result = model.invoke(prompt)
# final_result=parcer.parse(result.content)

final_result = chain.invoke({'topic':'black hole'}) ## we should pass a dictionary in invoke method

print(final_result)