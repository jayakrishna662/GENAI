from dotenv import load_dotenv

load_dotenv()

# HuggingFaceEndpoint => Connects LangChain to a Hugging Face hosted API.
# ChatHuggingFace => Makes that Hugging Face model behave like a LangChain chat model.

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Set up the endpoint (connection to the model) 
# At this point, llm is just the low-level connection — it knows which model to call and how, but it doesnot behave like chat model.
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-4B-Instruct-2507",  # This tells Hugging Face: "I want to use this model."
    task="text-generation",                 # This tells Hugging Face:"I want this model to generate text."
    provider="auto",                        # This tells Hugging Face: "Find an available provider that can run this model."
    max_new_tokens=1024,                    # "Generate at most 1024 new tokens for the response."
)

model = ChatHuggingFace(llm=llm)  # Wrap llm so it behaves like  a chat model

response = model.invoke("Define photosynthesis in one sentence!")

print(response.content)