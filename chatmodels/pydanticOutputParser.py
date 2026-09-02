from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

# 1. Create the model
model = ChatGroq(
    model="openai/gpt-oss-20b"
)


# 2. Define the expected output structure
class CustomerSupport(BaseModel):
    category: str
    sentiment: str
    summary: str


# 3. Create the Pydantic parser
# This is the parser that will take the LLM's structured output and convert it into a Pydantic object.
# We're telling the parser: Use CustomerSupport as the structure/schema , and we want response in that format
parser = PydanticOutputParser(
    pydantic_object=CustomerSupport
)


# 4. Create the prompt
prompt = ChatPromptTemplate.from_template("""
Analyze this customer support message.

Return the following information:
- category
- sentiment
- summary

Customer message:
{message}

{format_instructions}
""")


# 5. Add format instructions to the prompt
# parser.get_format_instructions() tells Instructions that should given to the LLM so that it produces the correct format
prompt = prompt.partial(
    format_instructions=parser.get_format_instructions()
)


# 6. Create the chain
chain = prompt | model | parser


# 7. Invoke the chain
response = chain.invoke({
    "message": "I was charged twice for my order and I want a refund."
})


# 8. Print the result
print(response)
