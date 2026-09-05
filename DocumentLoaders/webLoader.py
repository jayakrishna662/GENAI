# WebBaseLoader needs something to parse the HTML of the webpage. It uses BeautifulSoup for that.
# so you need beautifulsoup4 library to parse and extract information from HTML web pages.
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    "https://www.geeksforgeeks.org/nlp/what-is-retrieval-augmented-generation-rag/"
)

documents = loader.load()

print(len(documents))

print(documents[0].page_content)

print(documents[0].metadata)