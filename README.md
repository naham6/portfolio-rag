# Portfolio RAG API 

A RESTful API built with **FastAPI**, **LangChain**, and **ChromaDB** that utilizes a Retrieval-Augmented Generation (RAG) pipeline to allow users to contextually query professional portfolio and CV data.

## Overview
This project demonstrates the orchestration of Large Language Models (LLMs) with local vector databases. It processes a PDF document (like a CV), chunks the text, converts it into 768-dimensional mathematical vectors using Google's Embedding model, and stores it in a local ChromaDB instance. 

When queried, the API performs a semantic similarity search (Cosine Similarity) to retrieve the most relevant context, injects it into a strict system prompt, and leverages Gemini to generate a response.

## Tech Stack
* **Backend Framework:** FastAPI, Uvicorn
* **AI Orchestration (MCP):** LangChain (`langchain-classic`, `langchain-google-genai`)
* **Vector Database:** ChromaDB
* **Embeddings Model:** Google `gemini-embedding-001`
* **LLM:** Google `gemini-2.5-flash`
* **Document Processing:** PyPDF

## How It Works (The Pipeline)
1. **Document Ingestion:** Reads the target PDF and splits it into manageable chunks with controlled overlaps to preserve context.
2. **Vectorization:** Passes chunks through an embedding model to translate semantic meaning into vector coordinates.
3. **Retrieval:** When a `POST` request is received, the user's question is embedded and compared against the ChromaDB database to retrieve the top 3 most relevant context chunks.
4. **Generation:** The retrieved English text is injected into a custom prompt template and sent to the LLM to formulate a precise answer strictly grounded in the provided context.

## Local Setup & Installation
**Install requirements.txt**
**Clone the repository**
```bash
git clone https://github.com/naham6/portfolio-rag
cd portfolio-rag
```
**Create a .env file and insert**
```bash
GEMINI_API_KEY=your_api
```
