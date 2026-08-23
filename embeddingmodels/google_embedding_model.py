from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# GoogleGenerativeAIEmbeddings is a LangChain class that lets your Python program use Google's embedding models.

load_dotenv()

# create an embedding model object, using this object we can send input to embedding model and get embeddings
embeddings = GoogleGenerativeAIEmbeddings(
	model="gemini-embedding-2-preview"
)

input= "LLM"
vector = embeddings.embed_query(input) # embed_query is a method which takes input and returns vector

print(f"Embedding dimensions: {len(vector)}")
print(vector[:10]) # Shows only 10 dimensions
