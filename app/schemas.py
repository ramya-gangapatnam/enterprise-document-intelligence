from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """
    Request model for document question answering.
    """
    question: str = Field(
        ...,
        min_length=3,
        description="User question about uploaded enterprise documents"
    )
    source: Optional[str] = Field(
        default=None,
        description="Optional filename to scope retrieval to a specific uploaded document"
    )

class AskResponse(BaseModel):
    """
    Standard success response for the question-answering endpoint.
    """
    question: str
    answer: str
    sources: List[str]
    retrieved_chunks_count: int
    status: str = "success"
    model_used: Optional[str] = None


class UploadResponse(BaseModel):
    """
    Standard success response for the document upload endpoint.
    """
    filename: str
    chunks_indexed: int
    status: str = "success"
    message: str


class ErrorResponse(BaseModel):
    """
    Standard error response model for consistent API failures.
    """
    status: str = "error"
    message: str


class RetrievedChunk(BaseModel):
    """
    Internal representation of a retrieved chunk returned from vector search.
    """
    chunk_id: str
    text: str
    source: str
    chunk_index: int