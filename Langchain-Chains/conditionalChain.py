# Conditional chaining means choosing which chain to execute based on a condition.
# we use RunnableBranch which is a LangChain class used to choose which chain to run based on a condition.

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# Create the refund prompt
refund_prompt = ChatPromptTemplate.from_template(
    """
    You are a refund specialist.

    Help the customer with their refund-related issue:

    {message}
    """
)

# Create the refund chain
refund_chain = refund_prompt | model

# Create the general support prompt
general_prompt = ChatPromptTemplate.from_template(
    """
    You are a general customer support assistant.

    Help the customer with their issue:

    {message}
    """
)

general_chain = general_prompt | model


# if condition is true, run refund_chain, general_chain is default chain
conditional_chain = RunnableBranch(
    (
        lambda x: "refund" in x["message"].lower(),    # x is variable that holds the input like x={"message": "..."}
        refund_chain
    ),
    (
        general_chain
    )
)

# Give input
response = conditional_chain.invoke({
    "message": "I want a replacement for my order."
})


# Print response
print(response.content)