from typing import List
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import OPENAI_API_KEY, EMBEDDING_MODEL

# Reuse one client instance across embedding calls.
client = OpenAI(api_key=OPENAI_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_embedding(text: str) -> List[float]:
    """
    Create an embedding for a single text input.
    This is used for query embedding during retrieval.
    """
    if not text.strip():
        raise ValueError("Cannot generate embedding for empty text.")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def get_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Create embeddings for multiple text chunks in one request.
    Batch embedding is more efficient than sending one request per chunk.
    """
    cleaned_texts = [text.strip() for text in texts if text.strip()]

    if not cleaned_texts:
        raise ValueError("Cannot generate embeddings for empty text list.")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=cleaned_texts
    )

    return [item.embedding for item in response.data]