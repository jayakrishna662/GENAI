# CSVLoader can use Python's built-in CSV functionality, so it doesn't need a separate package like pypdf
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader("customer_support.csv")

# CSVLoader treats each CSV row as a document.
documents = loader.load()

print(len(documents))

for document in documents:
    print('---Document---')
    print(document.page_content)
    print(document.metadata)