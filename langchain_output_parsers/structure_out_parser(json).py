from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

# Define the model
llm = HuggingFaceEndpoint(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    task="text-generation",
    # model="mistralai",

)




## Schema
schema = [
    ResponseSchema(name="fact1", description="First fact about the topic"),
    ResponseSchema(name="fact2", description="Second fact about the topic"),    
    ResponseSchema(name="fact3", description="Third fact about the topic"),
]
parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template='Give me 3 facts about {topic} \n {format_instruction}',
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)
chain = template | llm | parser

# prompt = template.format(topic="black hole")
# result = llm.invoke(prompt)
# final_result=parser.parse()

final_result = chain.invoke({'topic':'black hole'}) ## we should pass a dictionary in invoke method

print(final_result)