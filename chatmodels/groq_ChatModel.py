# From the dotenv package, import the load_dotenv function.
from dotenv import load_dotenv

# Execute the function.
# this function reads the .env file and makes available to the program.
# Groq automatically looks for the APIkey.
load_dotenv()


# init_chat_model is a LangChain function used to create a chat model.
from langchain.chat_models import init_chat_model

# configure the LLM that you're going to use.
# initialize a chat model and store that model object in the model variable.
model = init_chat_model(
    "groq:openai/gpt-oss-20b",   # This tells LangChain which model and provider to use.
    temperature=0.5,             # Here groq is provider, openai/gpt-oss-20b is model
)


for i in range(15):
    response = model.invoke("The sky is ____ in color")  # Use this model and generate a response for the given input.
    print(i + 1, ":", response.content)  # the model returns a AImessage object containing models reponse and additional info(like metadata,token usage,model info etc)
