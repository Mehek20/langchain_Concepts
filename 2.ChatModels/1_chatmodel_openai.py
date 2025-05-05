from langchain_openai import ChatOpenAI
from dotenv import load_dotenv   # used to load the .env file

load_dotenv()   # load the .env file

model = ChatOpenAI(model='gpt-4',temperature=1.5,max_completion_tokens=10)    # create an instance of the OpenAI class
result = model.invoke("Write a 5 line poem on cricket")

print(result.content)   # print the result