from langchain_community.document_loaders import TextLoader

# Create a TextLoader pointing to the file path
# encoding="utf-8" ensures special characters (emojis, symbols, non-English text) 
# are read correctly — without it, Windows may default to 'cp1252' encoding 
# and throw a UnicodeDecodeError on such characters
loader = TextLoader("path of text file",  encoding="utf-8")

documents = loader.load() # read data source and convert into LangChain document format

print(documents) # Prints the entire list, including the Document object wrapper itself.

print(documents[0].page_content) # from first Document object, print only the text content from that object

print(documents[0].metadata) # from first Document object, print only the metadata as a dictionary from that object
