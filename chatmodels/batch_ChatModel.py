# batch() is designed for bulk processing of data.
# instead of calling model for each input multiple times, call the model only one time with multiple inputs at a time

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

questions = [
    "What is Python?",
    "What is Java?",
    "What is C++?"
]

responses = model.batch(questions)

for response in responses:
    print(response.content)