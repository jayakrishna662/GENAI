text = """
Python is a programming language.
It is widely used for web development, data science, and artificial intelligence.

RAG stands for Retrieval Augmented Generation.
It allows an LLM to retrieve relevant information from external documents.

LangChain is a framework for building applications with language models.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,   # try to make each chunk around 200 characters
    chunk_overlap=20  # neighboring chunks can share about 20 characters
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk)
    print("Length:", len(chunk))