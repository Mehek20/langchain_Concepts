from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableSequence,RunnablePassthrough,RunnableLambda

load_dotenv()

prompt1 = PromptTemplate(
    template = "Write a joke about {topic}",
    input_variables = ["topic"]
)

model = ChatOpenAI()

parser = StrOutputParser()

## Defining a simple function that counts the number of words in a given text.
def word_count(text):
    return len(text.split())

joke_genchain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
}
)
final_chain = RunnableSequence(joke_genchain, parallel_chain)
result = final_chain.invoke({'topic':'marraige'})

final_result = """{} \n Word Count: {}""".format(result['joke'], result['word_count'])
print(final_result)