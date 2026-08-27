from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model(
    "groq:openai/gpt-oss-20b",
    temperature=0.5,             
)

print("Welcome")
print("Type Exist to Stop.\n")

while True:
    prompt=input("YOU: ")

    if(prompt.lower()=="exist"):
        print("Good Bye!")
        break

    response=model.invoke(prompt)
    print("Chat Bot: "+ response.content)