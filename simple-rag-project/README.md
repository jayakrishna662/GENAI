# Simple RAG Project

A basic Retrieval-Augmented Generation (RAG) application built with LangChain, Google Gemini, and Chroma.

The application loads a company policy document, splits it into smaller chunks, converts the chunks into embeddings, stores them in Chroma, retrieves relevant information for a user question, and uses Gemini to generate an answer based on the retrieved context.

## RAG Flow

```text
Document
   ↓
Document Loader
   ↓
Text Splitting
   ↓
Embeddings
   ↓
Chroma Vector Store
   ↓
Retriever
   ↓
Retrieved Context
   ↓
Prompt Template
   ↓
Gemini LLM
   ↓
Final Answer
```

## Technologies Used

- Python
- LangChain
- Google Gemini
- Gemini Embeddings
- Chroma

## Project Structure

```text
simple-rag-project/
│
├── documents/
│   └── company_policy.txt
│
├── main.py
├── requirements.txt
├── .env
└── .gitignore
```

## How It Works

### 1. Document Loading

`TextLoader` loads the company policy document into LangChain `Document` objects.

### 2. Text Splitting

`RecursiveCharacterTextSplitter` divides the document into smaller chunks.

### 3. Embeddings

Gemini Embeddings converts each chunk into a numerical vector that represents its meaning.

### 4. Vector Store

Chroma stores the generated vectors along with the original document chunks.

### 5. Retrieval

The retriever searches Chroma for the chunks most relevant to the user's question.

### 6. Augmentation

The retrieved chunks are added to a prompt along with the original question.

### 7. Generation

Gemini uses the question and retrieved context to generate the final answer.

## Setup

### 1. Clone the repository

```bash
git clone <repository-url>
cd simple-rag-project
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the API key

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_api_key_here
```

### 5. Run the application

```bash
python main.py
```

The application retrieves relevant information from the company policy document and generates an answer using Gemini.
