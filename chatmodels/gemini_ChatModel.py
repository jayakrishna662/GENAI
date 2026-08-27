from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "google_genai:gemini-3.5-flash-lite",
)

response = model.invoke("What is an Embedding Model?")

print(response.content[0]["text"]) # get the first item (dictionary) from the response.content, then get the value stored under the "text" key: