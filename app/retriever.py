from typing import Dict, List

from app.embedder import get_embedding
from app.vector_store import query_chunks


def retrieve_context(question: str, top_k: int = 4) -> List[Dict]:
    """
    Convert the user question into an embedding and retrieve the
    most relevant chunks from the vector store.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    query_embedding = get_embedding(question)
    results = query_chunks(query_embedding=query_embedding, top_k=top_k)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    ids = results.get("ids", [[]])[0]

    retrieved_chunks = []

    for chunk_id, text, metadata in zip(ids, documents, metadatas):
        retrieved_chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
            }
        )

    return retrieved_chunks