from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel,RunnableBranch,RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model1 = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):

    sentiment: Literal['positive', 'negative'] = Field(description="The sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following text into positive or negetive : \n {feedback} \n {format_instructions}',
    input_variables=['feedback'],
    partial_variables={'format_instructions': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model1 | parser2

# result = classifier_chain.invoke({"feedback": "The product is great!"}).sentiment
# print(result)  # Output: positive

prompt2 = PromptTemplate(
    template = 'Write an appropriate response to this positive feedback: \n {feedback}',
    input_variables = ['feedback']
)

prompt3 = PromptTemplate(
    template = 'Write an appropriate response to this negetive feedback: \n {feedback}',
    input_variables = ['feedback']
)

# branch_chain = RunnableBranch(
#     (cond1,chain1),
#     (cond2,chain2),
#     default chain
# )

branch_chain = RunnableBranch(
    (lambda x:x['sentiment'] == 'positive',prompt2 | model1 | parser),
    (lambda x:x['sentiment'] == 'negetive',prompt3 | model1 | parser),
    RunnableLambda(lambda x :"could not find sentiment") # default chain this will not not run directly as it is not a part of chain so we have to convert it into a runnable
)

chain = classifier_chain | branch_chain
result = chain.invoke({"feedback": "The product is great!"}) # Output: "Thank you for your positive feedback! We're glad you liked the product."

chain.get_graph().print_ascii() ## This will print the graph of the chain in ASCII format.