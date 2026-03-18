import logging
import os

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse

from app.chunker import chunk_text
from app.config import (
    CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    TOP_K_RESULTS,
    validate_config,
)
from app.embedder import get_embeddings
from app.ingest import load_document
from app.llm_service import generate_answer
from app.logging_config import setup_logging
from app.retriever import retrieve_context
from app.schemas import AskRequest, AskResponse, UploadResponse
from app.utils import deduplicate_sources, ensure_directory, sanitize_filename, compute_file_hash
from app.vector_store import store_chunks, delete_chunks_by_source, source_exists, get_existing_file_hash

# Configure logging once at startup.
setup_logging()
logger = logging.getLogger(__name__)

# Fail fast if required environment variables are missing.
validate_config()

app = FastAPI(
    title="Enterprise Document Intelligence System",
    version="1.0.0",
    description="RAG-powered enterprise document question answering with FastAPI, OpenAI, and ChromaDB.",
)

DOCUMENTS_DIR = "documents"
ensure_directory(DOCUMENTS_DIR)

ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx"}


@app.get("/health")
def health_check() -> dict:
    """
    Lightweight endpoint used to verify the API is running.
    """
    return {"status": "ok"}

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a document, extract text, chunk it, embed the chunks,
    and store them in the vector database.

    If a document with the same filename was indexed earlier, its old
    chunks are deleted first so the latest upload replaces prior vectors.
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

        filename = sanitize_filename(file.filename)
        extension = os.path.splitext(filename)[1].lower()

        if extension not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {extension}. Supported types are .txt, .pdf, .docx."
            )

        file_path = os.path.join(DOCUMENTS_DIR, filename)

        content = await file.read()
        file_hash = compute_file_hash(content)
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Save uploaded content locally so it can be processed by the ingestion pipeline.
        with open(file_path, "wb") as saved_file:
            saved_file.write(content)

        logger.info("Uploaded file saved: %s", file_path)

        text = load_document(file_path)

        chunks = chunk_text(
            text=text,
            source=filename,
            chunk_size=MAX_CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )

        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks were created from the uploaded document.")
        
        # Check if file already exists and is unchanged
        existing_hash = get_existing_file_hash(filename)

        if existing_hash and existing_hash == file_hash:
            logger.info("File %s unchanged. Skipping re-indexing.", filename)

            return UploadResponse(
                filename=filename,
                chunks_indexed=0,
                message="Document unchanged. Existing index reused.",
            )

        # Replace previously indexed vectors for the same filename.
        deleted_count = delete_chunks_by_source(filename)
        if deleted_count:
            logger.info("Deleted %s existing chunks for source %s", deleted_count, filename)

        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(chunk_texts)
        indexed_count = store_chunks(chunks, embeddings, file_hash)

        logger.info("Indexed %s chunks for %s", indexed_count, filename)

        return UploadResponse(
            filename=filename,
            chunks_indexed=indexed_count,
            message="Document indexed successfully.",
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Document validation failed")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail="Internal server error during document upload.")

@app.post("/ask", response_model=AskResponse)
def ask_question(request: AskRequest) -> AskResponse:
    """
    Retrieve relevant document chunks for the user's question and generate
    a grounded answer using the LLM.
    """
    try:
        logger.info("Received question: %s", request.question)

        # Validate that requested source exists in vector DB
        if request.source:
            if not source_exists(request.source):
                raise HTTPException(
                    status_code=400,
                    detail=f"Source '{request.source}' not found in indexed documents. Please upload it first."
                )

        retrieved_chunks = retrieve_context(
            question=request.question,
            top_k=TOP_K_RESULTS,
            source=request.source,
        )

        if not retrieved_chunks:
            return AskResponse(
                question=request.question,
                answer="The information is not available in the provided documents.",
                sources=[],
                retrieved_chunks_count=0,
            )

        answer = generate_answer(
            question=request.question,
            retrieved_chunks=retrieved_chunks,
        )

        sources = deduplicate_sources([chunk["source"] for chunk in retrieved_chunks])

        return AskResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            retrieved_chunks_count=len(retrieved_chunks),
        )

    except HTTPException:
        raise
    except ValueError as exc:
        logger.exception("Question validation failed")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        logger.exception("Question answering failed")
        raise HTTPException(status_code=500, detail="Internal server error during question answering.")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch any unhandled exceptions and return a consistent JSON error format.
    """
    logger.exception("Unhandled application error")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An unexpected internal server error occurred.",
        },
    )