# JSONLoader uses the jq Python package to process the jq_schema 
# so you need to do pip install jq

from langchain_community.document_loaders import JSONLoader

loader = JSONLoader(
    file_path="customer_support.json",
    jq_schema=".[]",   # This tells the loader: Take each object inside the top-level JSON array.
    text_content=False
)

documents = loader.load()

print(len(documents))

for document in documents:
    print("-----")
    print(document.page_content)
    print(document.metadata)


# Note :  jq_schema=".[]" takes json array , selects each object and returns as an object/dictionary
# But JsonLoader expects text/string
# Since we want each JSON object to become a document, we add text_content=False to allow objects/dictionaries
# Now each object in json array becomes a document

