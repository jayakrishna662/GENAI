# jsonOutputParser takes JSON-formatted text produced by the model and parses it into a Python object, 
# usually a dictionary.

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

# 1. Create the model
model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# 2. Create the JSON parser
parser = JsonOutputParser()

# 3. Create the prompt
prompt = ChatPromptTemplate.from_template("""
Analyze the following customer support message.

Return the result ONLY as valid JSON.

The JSON must contain these fields:
- category
- sentiment
- summary

Customer message:
{message}
""")

# 4. Create the chain
chain = prompt | model | parser

# 5. Give input to the chain
response = chain.invoke({
    "message": "I was charged twice for my order and I want my money back."
})

print(type(response))
# 6. Print the result
print(response)

# if you want final response to be in json string you can use => json_string = json.dumps(response)