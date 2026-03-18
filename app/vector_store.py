from typing import Dict, List
import chromadb

from app.config import CHROMA_PATH, COLLECTION_NAME

# Persistent client keeps vectors stored on disk, not just in memory.
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(chunks: List[Dict], embeddings: List[List[float]]) -> int:
    """
    Store chunk text, metadata, and embeddings in ChromaDB.
    """
    if not chunks:
        raise ValueError("No chunks provided for storage.")

    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch.")

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "chunk_index": chunk["chunk_index"],
        }
        for chunk in chunks
    ]

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    return len(chunks)


def query_chunks(query_embedding: List[float], top_k: int = 4, source: str | None = None) -> Dict:
    """
    Query the vector store using a query embedding.
    If source is provided, retrieval is scoped to that specific filename.
    """
    if not query_embedding:
        raise ValueError("Query embedding cannot be empty.")

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
    }

    if source:
        query_params["where"] = {"source": source}

    return collection.query(**query_params)


def reset_collection() -> None:
    """
    Reset the collection for local testing or re-indexing.
    Useful during development when you want a clean vector store.
    """
    global collection

    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    collection = client.get_or_create_collection(name=COLLECTION_NAME)


def delete_chunks_by_source(source: str) -> int:
    """
    Delete all chunks associated with the given source filename.

    This supports document-level replacement so re-uploading the same file
    does not create duplicate vectors in the collection.
    """
    if not source.strip():
        raise ValueError("Source filename cannot be empty.")

    existing = collection.get(
        where={"source": source},
        include=[]
    )

    ids = existing.get("ids", [])

    if not ids:
        return 0

    collection.delete(ids=ids)
    return len(ids)


def source_exists(source: str) -> bool:
    """
    Check whether any indexed chunks exist for the given source filename.
    """
    if not source or not source.strip():
        return False

    results = collection.get(where={"source": source.strip()})
    ids = results.get("ids", [])

    return bool(ids)
