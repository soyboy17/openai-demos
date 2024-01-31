""""
WEB-READER CHATBOT

This code uses langchain and OpenAI to create a chatbot that can answer questions about a webpage provided by the user. It checks whether or not the URL is a PDF or a webpage, and then uses the appropriate loader to retrieve the data.

Test pages:
    https://en.wikipedia.org/wiki/Ensemble_learning
    https://arxiv.org/ftp/arxiv/papers/2310/2310.13702.pdf

"""

from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os

os.environ["OPENAI_API_KEY"] = "" # Enter your API key here


#Prompt user for URL to read
url = input("Enter a URL\n")  # test URL: https://en.wikipedia.org/wiki/Ensemble_learning

# Takes a URL, checks if it is a PDF or other file and loads it into a RetrievalQA object.
# todo: add support for other file types
if url.endswith(".pdf"):
    loader = PyPDFLoader(url)
else: 
    loader = WebBaseLoader(url)

pages = loader.load_and_split()
    
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(pages, embeddings)

qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())

while True:        
    query = input("Ask me a question!\n")
    #print(f"You asked: {query}")    # Debug statement
    response = qa.invoke(query)
    print(f"Response: {response['result']}")