from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
# Load the model
embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)

result = embedding.embed_query("What is the capital of France?") ## The contextual meaning of query is stored in the result

print(str(result))