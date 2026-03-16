from typing import List, Dict


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 700,
    chunk_overlap: int = 120,
) -> List[Dict]:
    """
    Split document text into overlapping chunks.

    Overlap helps preserve meaning across chunk boundaries, which improves
    semantic retrieval when relevant information spans adjacent chunks.
    """
    if not text.strip():
        raise ValueError("Cannot chunk empty text.")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    # Normalize whitespace so chunking is more consistent across file types.
    cleaned_text = " ".join(text.split())
    chunks = []

    start = 0
    chunk_index = 0
    text_length = len(cleaned_text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk_text_value = cleaned_text[start:end].strip()

        if chunk_text_value:
            chunks.append(
                {
                    "id": f"{source}_chunk_{chunk_index}",
                    "text": chunk_text_value,
                    "source": source,
                    "chunk_index": chunk_index,
                }
            )

        if end == text_length:
            break

        # Move forward with overlap so the next chunk retains nearby context.
        start = end - chunk_overlap
        chunk_index += 1

    return chunks