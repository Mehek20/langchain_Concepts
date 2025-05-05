from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large',dimensions=300)


documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = "tell me about bumrah"

doc_embeddings = embedding.embed_documents(documents) ## The contextual meaning of documents is stored in the result
query_embedding = embedding.embed_query(query) ## The contextual meaning of query is stored in the result

scores = cosine_similarity([query_embedding], doc_embeddings)[0] ## Cosine similarity between query and documents

index,score = sorted(list(enumerate(scores)),key = lambda x: x[1])[-1]## Sort the documents based on similarity score

print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score}")