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

def read_page(url):
    loader = WebBaseLoader(url)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_documents(texts, embeddings)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())

    return qa

def is_pdf(url):
    if url.endswith(".pdf"):
        return True
    
def read_pdf(url):
    # PyPDFLoader retrieves data from pdf file
    loader = PyPDFLoader(url)
    pages = loader.load_and_split()

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(pages, embeddings)

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=db.as_retriever())

    return qa

#Prompt user for URL to read
url = input("Enter a URL\n")  # test URL: https://en.wikipedia.org/wiki/Ensemble_learning

#Check if URL is a PDF, if so, read it with read_pdf
if is_pdf(url):
    qa = read_pdf(url)

#Otherwise, read it with read_page
else:
    qa = read_page(url)

while True:        
    query = input("Ask me a question!\n")
    #print(f"You asked: {query}")    # Debug statement
    response = qa.invoke(query)
    print(f"Response: {response['result']}")