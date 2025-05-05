from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
# from langchain_openai import ChatOpenAI
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatOpenAI()

chat_template=ChatPromptTemplate([
    ('system', "You are a helpful {domain} expert"),
    ('human', "Explain the terms, what is {topic}")
    # SystemMessage(content="you are a helpful {domain} expert"),
    # HumanMessage(content="Explain the terms, what is {topic}")
])

prompt = chat_template.invoke({
    "domain": "Cricket",
    "topic": "Wicket"
})

print(prompt)
