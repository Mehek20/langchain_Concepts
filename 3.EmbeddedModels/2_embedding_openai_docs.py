from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
# Load the model
embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=32)

documents = {
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Mumbai is the capital of Maharashtra"
}

result = embedding.embed_documents(documents) ## The contextual meaning of query is stored in the result

print(str(result))