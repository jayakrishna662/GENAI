# Before running make sure to install pypdf library
# pypdf understands the PDF format and can extract text from PDF pages.
# Because pypdf is an optional dependency for the PDF loader. 
# langchain-community supports many different integrations/loaders, 
# but it doesn't necessarily install every library needed for every one of them.

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path of pdf")

documents = loader.load() # read data source and convert into LangChain document format

print(documents) # Prints the entire list, including the Document object wrapper itself.

print(documents[0].page_content) # from first Document object, print only the text content from that object

print(documents[0].metadata) # from first Document object, print only the metadata as a dictionary from that object

# PyPDFLoader creates a Document for each page.
print(len(documents)) # get length of documents
