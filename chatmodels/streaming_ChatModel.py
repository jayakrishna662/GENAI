# With streaming, you receive the answer piece by piece(like token by token) as the LLM generates it.

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

user_input="Explain artificial intelligence in simple 2 sentences."

for chunk in model.stream(user_input):  # model.stream() calls the LLM and gives you the response chuck by chunk.
    print(chunk.content,end="")        # Chunk = a small piece of the LLM response received during streaming.