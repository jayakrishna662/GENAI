import json

from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# Define JSON schema
# person_schema is a Python dictionary that represents a JSON Schema.
person_schema = {
    # Metadata
    "title": "Person",    # Its just a title given to schema
    "description": "Information about a person",   # This describes what this schema is about
    "type": "object",     # Type of schema is object bcz we need response as json object

    "properties": {      # Tells What fields should this object contain
        "name": {
            "type": "string",
            "description": "Person's name"
        },
        "age": {
            "type": "integer",
            "description": "Person's age"
        },
        "email": {
            "type": "string",
             "minLength": 1,
            "description": "Person's email address, if email is not provided by user return null. dont guess any email"
        }
    },
    "required": ["name", "age", "email"]  # The final structured output must contain these fields
}


# Tell LangChain to return structured output
structured_model = model.with_structured_output(
    person_schema,
    method="json_schema"
)

# Call the model
response = structured_model.invoke(
    "John is 25 years old "
)

# Convert dictionary result to json

json_response=json.dumps(response)
print(json_response)