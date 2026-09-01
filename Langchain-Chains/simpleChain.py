# A chain that performs a single operation by connecting components is called simple chain
# such as connecting prompt and an LLM.


from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple words."
)

model = ChatGroq(
    model="openai/gpt-oss-20b"
)
""""
We ignore writing like this: 

final_prompt = prompt.invoke({
    "topic":"RAG",
})

response = model.invoke(final_prompt)

"""

# prompt | model means Connect prompt to model so that prompts's output flows into model.

chain = prompt | model 

response = chain.invoke({
    "topic": "RAG"
})

print(response.content)