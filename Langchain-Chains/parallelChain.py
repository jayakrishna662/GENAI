# In parallel chaining, multiple independent operations run at the same time.
"""
Example : Our application Customer Support Analysis needs to find three things independently:

Category 
Sentiment 
Summary

"""


from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# we will create three chains that can execute independently using same input

category_prompt = ChatPromptTemplate.from_template(
    "Identify the category of this customer message.\n\n{message}"
)

category_chain = category_prompt | model

sentiment_prompt = ChatPromptTemplate.from_template(
    "Identify the sentiment of this customer message.\n\n{message}"
)

sentiment_chain = sentiment_prompt | model


summary_prompt = ChatPromptTemplate.from_template(
    "Summarize this customer message in one sentence.\n\n{message}"
)

summary_chain = summary_prompt | model


# using RunnableParallel to run three independent chains parallely and getting response separately
# category, sentiment, summary represents the keys for final response
# bcz we get response in key value pairs

parallel_chain = RunnableParallel(
    category=category_chain,
    sentiment=sentiment_chain,
    summary=summary_chain
)

# That same input is given to all three chains.
response = parallel_chain.invoke({
    "message": "I was charged twice for my order and I want my money back."
})


# response contains key value pairs, where The actual values are AIMessage objects.

print(response["category"].content)
print(response["sentiment"].content)
print(response["summary"].content)