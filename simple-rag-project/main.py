from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables from .env
load_dotenv()


# Configure the LLM 
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

# Configure the embedding model.
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-2-preview"
)


# 1) DATA INGESTION PHASE


# Create a loader for our company policy text file.
loader = TextLoader("documents/company_policy.txt")

# Load the text file into LangChain Document objects.
documents = loader.load()


# Create a text splitter to divide the document into smaller chunks.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Split the document into chunks.
chunks = text_splitter.split_documents(documents)


# Create the Chroma vector store.
# Chroma creates embeddings for the chunks
# and stores them along with the original text and metadata.
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)


# 2) RETRIEVAL PHASE



# Create a retriever from the vector store.
# It will return the top 2 relevant chunks.
retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)


# 3) AUGMENTATION PHASE



# Convert retrieved Document objects into one text string.
def format_docs(documents):
    return "\n\n".join(
        document.page_content for document in documents
    )


# Create a prompt template for our RAG application.
prompt_template = ChatPromptTemplate.from_template("""
Answer the question using only the information provided below.

Context:
{context}

Question:
{question}
""")



# 4) RAG CHAIN


# RunnableParallel recieves Question when we do rag_chain.invoke(question)
# Now the question goes into two places
# i) retriever --> The retriever searches Chroma and finds relevant chunks then we combine all chunks into one text string
# ii) RunnablePassthrough --> keep the question as it is
# So After RunnableParallel finishes, we get context and question
# We used RunnableParallel because , our prompt_template needs both context and question at a time

rag_chain = (
    RunnableParallel(
        context= retriever  | format_docs,
        question=RunnablePassthrough()  # it means Pass the original question through without changing it.
    )
    | prompt_template
    | llm
)



# 5) GENERATION PHASE


question = "How many days can employees work from home?"


# Run the complete RAG chain.
response = rag_chain.invoke(question)


# Display the final answer.
print("\n--- Final Answer ---")
print(response.content[0]["text"])