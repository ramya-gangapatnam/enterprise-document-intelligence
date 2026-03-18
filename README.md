# Enterprise Document Intelligence System

**Python | FastAPI | OpenAI | RAG | ChromaDB**

A document question-answering system that lets users query enterprise documents using natural language.  
Built with a focus on clean architecture, controlled retrieval, and production-minded design.

---

## Overview

This project ingests unstructured documents, converts them into embeddings, stores them in a vector database, and retrieves relevant context to generate grounded answers using an LLM.

It is designed not just to “work,” but to behave predictably, avoid common pitfalls (like duplicate embeddings and noisy retrieval), and expose useful runtime metrics.

---

## How It Works
User Question
↓
Convert to embedding
↓
Search vector database (ChromaDB)
↓
Retrieve relevant chunks
↓
Build prompt with context
↓
LLM generates answer
↓
Return answer + sources + metrics

---

## Features

### Document Ingestion
- Supports `.txt`, `.pdf`, `.docx`
- Extracts text and splits into chunks with overlap
- Generates embeddings and stores them in ChromaDB

---

### Document Replacement (No Duplicates)
- Re-uploading the same file replaces its previous embeddings
- Prevents duplicate vectors
- Keeps retrieval clean and consistent

---

### Hash-Based Change Detection
- Computes SHA256 hash of uploaded files
- Skips re-indexing if the file hasn’t changed
- Reduces unnecessary embedding calls and latency

---

### Scoped Retrieval (By File)
You can query a specific document:

```json
{
  "question": "How many remote work days are allowed?",
  "source": "sample_policy.txt"
}
This avoids mixing unrelated documents in results.

Global Retrieval:
-If no source is provided, the system searches across all indexed documents.

Validation and Error Handling:
-Rejects unsupported file types
-Rejects empty uploads
-Fails clearly if a requested document is not indexed
-Avoids silent failures

Observability (Latency + Tokens):
Each response includes:
-model used
-latency (ms)
-input tokens
-output tokens
-total tokens
This helps monitor performance and cost.

Tech Stack:
-FastAPI
-OpenAI API (LLM + embeddings)
-ChromaDB (persistent vector store)
-Pydantic (validation)
-Tenacity (retry logic)
-Python logging

API Endpoints:
--Health Check - GET /health
--Upload Document - POST /upload (Uploads and indexes a document)
--Ask Question - POST /ask

Request:
{
  "question": "How many remote work days are allowed?",
  "source": "sample_policy.txt"
}

Response:
{
  "question": "...",
  "answer": "...",
  "sources": ["sample_policy.txt"],
  "retrieved_chunks_count": 2,
  "status": "success",
  "model_used": "gpt-4.1-mini",
  "latency_ms": 820.5,
  "input_tokens": 210,
  "output_tokens": 24,
  "total_tokens": 234
}

Running Locally
1. Clone the repository:
    git clone <your-repo-url>
    cd enterprise-document-intelligence
2. Create virtual environment:
    python -m venv venv
    venv\Scripts\activate
3. Install dependencies:
    pip install -r requirements.txt
4. Set environment variables
    Create a .env file:
        OPENAI_API_KEY=your_api_key
5. Run the application:
    uvicorn app.main:app --reload

Open: http://127.0.0.1:8000/docs

Example Flow:
1.Upload a document using /upload

2.Ask a question using /ask

3.Get a grounded answer with sources and metrics

Design Decisions
Controlled Retrieval:
Instead of always searching all documents, queries can be scoped to a specific file using metadata. This improves answer quality and avoids irrelevant context.

Clean Document Lifecycle:
Re-uploading a file replaces only its own vectors, not the entire database. This avoids duplication and keeps results consistent.

Avoiding Unnecessary Work:
Hash-based checks prevent re-indexing unchanged files, reducing cost and latency.

Explicit Error Handling:
The system fails clearly when something is wrong (e.g., unknown source), rather than returning misleading results.

Observability Built In:
Latency and token usage are included in responses to help understand system behavior and cost.

## Future Improvements:
- Async ingestion pipeline to avoid blocking uploads during embedding
- Streaming responses for faster perceived latency
- Multi-tenant isolation (separate collections per user/workspace)
- Role-based access control for documents
- Dockerization and deployment setup
- Evaluation metrics for answer quality (faithfulness, relevance)