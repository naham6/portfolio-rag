from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()
app = FastAPI(title="Portfolio RAG")

print("Loading DB")

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})#fetch the top 3 most relevant chunks for any question

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

system_prompt = (
    "You are an AI assistant representing the user's professional portfolio. "
    "Use the following pieces of retrieved context to answer the question about the user's experience, skills, or projects. "
    "If the answer is not in the context, say that you don't know based on the provided document. "
    "Keep the answer concise, professional, and highlight the user's strengths.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)#MCP Pipeline using LangChain

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
async def ask_portfolio(request: QueryRequest):
    #Take question - Search DB - Prompt - Gemini - Answer
    response = rag_chain.invoke({"input": request.question})
    return {
        "question": request.question, 
        "answer": response["answer"]
    }