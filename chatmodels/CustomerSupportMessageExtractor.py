from dotenv import load_dotenv

load_dotenv()

from enum import Enum
from typing import Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel

# Create the LLM
# Configure the LLM that you're going to use.
model = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)



# The sentiment must be one of the values defined in Sentiment.
class Sentiment(str, Enum):
    POSITIVE = "Positive"
    NEUTRAL = "Neutral"
    NEGATIVE = "Negative"


# Define the Ouput structure we want
# Create SupportTicket class using Pydantic's BaseModel features.
# BaseModel is the parent class provided by Pydantic. It is a tool for data validation.
# Because of BaseModel, Pydantic can validate and manage the data according to the fields you define.

class SupportTicket(BaseModel):
    customer_name: Optional[str] = None  # Optional[str] means this field can be a string or None.
    order_id: Optional[str] = None  
    product: str
    issue_type: Optional[str] = None 
    issue_description: Optional[str] = None 
    requested_action: Optional[str] = None  
    sentiment: Sentiment

# Create Pydantic parser
# We are telling the parser to take the AI's response and convert it into a SupportTicket object.None
# Because LLM returns text, but we need response in SupportTicket structure format
parser = PydanticOutputParser(
    pydantic_object=SupportTicket
)


# Create prompt
# ChatPromptTemplate approach is very useful where the same prompt is used repeatedly.
prompt = ChatPromptTemplate.from_messages([
    (
        "system",       # System message i.e instructions to AI
        """
        Extract customer support information from the message.
        IMPORTANT RULES:
        - Only extract information that is explicitly stated in the customer message.
        - Do NOT infer, assume, or guess missing information.
        - If information is not mentioned, return None.
        - For issue_type, classify the customer's problem into a concise category such as:
            Damaged Product, Technical Issue, Billing Error, Delivery Delay, Wrong Item, etc.
        {format_instructions} # This is a placeholder for parser.get_format_instructions(), that generates instructions telling the AI how its output should be formatted according to your SupportTicket model.
        """
    ),
    (
        "human",
        "{message}"  # This is a placeholder for the customer message.
    )
])


# Get message from user
message = input("Enter customer support message: ")


# Create final prompt
# prompt.invoke() means Take my prompt template and replace the placeholders with these values.
final_prompt = prompt.invoke({
    "message": message,
    "format_instructions": parser.get_format_instructions()
})


# Send to LLM
response = model.invoke(final_prompt)


# Convert LLM response into SupportTicket object
# Take the unstructured/string response from the LLM and turn it into the structured object we defined with Pydantic.
ticket = parser.parse(response.content)


# Print result
output=ticket.model_dump_json() # converts it into a JSON string: since ticket  is a Pydantic object.
print(output)