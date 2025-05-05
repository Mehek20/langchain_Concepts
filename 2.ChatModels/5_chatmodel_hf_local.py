from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint ## to use api of huggingface
from dotenv import load_dotenv   # used to load the .env file

import os
os.environ["HF_HOME"] = "E:/huggingface_cache"   # set the huggingface directory


load_dotenv()   # load the .env file

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
    
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India?")   # invoke the model
print(result.content)   # print the result

