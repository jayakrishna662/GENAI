# A chain that combines multiple operations(chains/tasks) sequentially, 
# where the output of one operation is passed to the next operation.

from dotenv import load_dotenv

load_dotenv()

from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

# Step 1
prompt1 = PromptTemplate.from_template(
    """
    Analyze this customer complaint.

    Complaint:
    {user_complaint}

    Identify:
    1. Customer name
    2. Order ID
    3. Product name
    4. Main problem
    5. Customer's emotion
    6. What the customer expects

    IMPORTANT:
    In your response, ALWAYS include the customer name,
    order ID, and product name along with your analysis.
    """
)

# Step 2
prompt2 = PromptTemplate.from_template(
     """
    Below is the complaint analysis:

    {prompt_1}

    Decide what the company should offer the customer.

    IMPORTANT:
    In your response, preserve and include:
    - Customer name
    - Order ID
    - Product name

    Then provide your recommended solution.
    """
)

# Step 3
prompt3 = PromptTemplate.from_template(
    """
    Below is the complaint analysis and recommended solution:

    {prompt_2}

    Write a professional customer support email.

    IMPORTANT:
    Use the customer name, order ID, and product name
    provided above in the email.

    The email should:
    - Address the customer by name
    - Mention the order ID
    - Mention the product
    - Apologize sincerely
    - Explain the solution
    - Be concise and professional

    Return ONLY the email.
    """
)

# Sequential chain
chain = prompt1 | model | prompt2 | model | prompt3 | model

response = chain.invoke({
    "user_complaint": """My name is Rahul Sharma and my Order ID is ORD-784521 for a Logitech Wireless Keyboard K380.
I placed the order on August 20, 2026, and it was expected to arrive on August 25, 2026.
The order arrived on August 30, five days late, and the keyboard was damaged with broken keys and a cracked body.
The keyboard does not work, and I am disappointed because I needed it for an important work project.
I would like either a replacement with expedited shipping or a full refund. Please resolve this issue as soon as possible."""
    })

print(response.content)