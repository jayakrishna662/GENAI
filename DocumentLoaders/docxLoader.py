# Word documents need a library that understands the DOCX format.
# So do pip install docx2txt
from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("sampleDocx.docx")

documents = loader.load()

# Docx2txtLoader treats the entire .docx file as one document. 
# It extracts all the text from the Word file and puts it into one LangChain Document.
# So we get only len=1
# it does not normally create one Document per Word page.
print(len(documents))

print(documents[0].page_content)

print(documents[0].metadata)