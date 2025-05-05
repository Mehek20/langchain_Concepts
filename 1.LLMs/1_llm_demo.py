from langchain_openai import OpenAI
from dotenv import load_dotenv   # used to load the .env file

load_dotenv()   # load the .env file

llm=OpenAI(model='gpt-3.5-turbo-instruct')    # create an instance of the OpenAI class

result = llm.invoke("What is the capital of India")

print(result)   # print the result

