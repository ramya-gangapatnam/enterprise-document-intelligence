from typing import Dict, List


def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a grounded RAG prompt using the retrieved document chunks.

    The model is explicitly instructed to answer only from the provided
    context and to avoid inventing facts.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError("Retrieved chunks are required to build the prompt.")

    context_blocks = []

    for index, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            f"[Context {index} | Source: {chunk['source']} | Chunk: {chunk['chunk_index']}]\n"
            f"{chunk['text']}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
You are an enterprise document intelligence assistant.

Instructions:
- Answer only using the provided retrieved context.
- Do not use outside knowledge.
- Do not invent facts.
- If the answer is not available in the context, respond exactly with:
  "The information is not available in the provided documents."
- Keep the answer concise, clear, and professional.
- Prefer direct answers over long explanations.

User Question:
{question}

Retrieved Context:
{context_text}

Return only the final answer text.
""".strip()

    return prompt