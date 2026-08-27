# Conversation memory = storing previous HumanMessage and AIMessage objects and sending that history along with the new message.
# Here we are going to create a Simple Chat Bot with basic conversation memory.
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.messages import AIMessage, HumanMessage, SystemMessage

# HumanMessage => Represents a message from the user.
# AIMessage => Represents a message from the AI.
# SystemMessage => Represents instructions given to the AI.

model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0.5,             
)

# This list will store entire conversation.
messages=[
    SystemMessage(
    content="You are a Python tutor. Explain concepts in simple language with examples."
)
]

print("Welcome")
print("Type Exist to Stop.\n")

while True:
    prompt=input("YOU: ")

    if(prompt.lower()=="exist"):
        print("Good Bye!")
        break

    # Add user's message to conversation history
    # Create a HumanMessage object and put the value stored in prompt into its content field.
    messages.append(HumanMessage(content=prompt))

    # Send entire conversation to the model
    response = model.invoke(messages)

    # Add AI's response to conversation history
    # Create an AIMessage object and put the AI's response text inside its content field.
    messages.append(AIMessage(content=response.content))
    
    print("Chat Bot: "+ response.content)



""" NOTE: Technically the LLM itself isn't remembering anything here,
            Python program keeps the previous messages and sends them again with every new request.
    
    Limitation: As the messages list keeps growing, you end up sending more and more tokens to model,
                Eventually, you can run into the model's context window.
"""


