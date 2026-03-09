import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

print("Loading and splitting the CV")
loader = PyPDFLoader("nahamcv.pdf")#cv
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")#converting text chunks into numerical vectors

print("saving the Vector DB(ChromaDB)")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("Database created")