## Runnable Passthrough 
## This module defines a simple passthrough runnable that returns the input as output.
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough

load_dotenv()

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template = "explain the following joke - {text}",
    input_variables = ["text"]
)

joke_genchain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explaination': RunnableSequence(prompt2, model, parser)
}
)
final_chain = RunnableSequence(joke_genchain, parallel_chain)
print(final_chain.invoke({'topic':'cats'}))