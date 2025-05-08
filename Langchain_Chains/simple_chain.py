from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = PromptTemplate(
    template = 'generate 5 intresting facts about {topic}',
    input_variables = ['topic']
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"topic": "cricket"})
print(result)
# The above code is a simple chain that generates 5 interesting facts about a given topic using Langchain.

chain.get_graph().print_ascii() ## This will print the graph of the chain in ASCII format.
# The graph will show the flow of data from the prompt to the model and then to the output parser.