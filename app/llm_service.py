import time
from typing import Dict, List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import OPENAI_API_KEY, CHAT_MODEL
from app.prompt_builder import build_prompt

# Reuse one OpenAI client instance across LLM calls.
client = OpenAI(api_key=OPENAI_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def generate_answer(question: str, retrieved_chunks: List[Dict]) -> Dict:
    """
    Generate the final grounded answer from retrieved document context.

    Returns answer text along with latency and token usage metadata
    for performance and cost monitoring.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError("Retrieved chunks cannot be empty.")

    prompt = build_prompt(question, retrieved_chunks)

    start_time = time.perf_counter()

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )

    end_time = time.perf_counter()
    latency_ms = round((end_time - start_time) * 1000, 2)

    answer = getattr(response, "output_text", "").strip()

    if not answer:
        raise ValueError("LLM returned an empty response.")

    usage = getattr(response, "usage", None)

    input_tokens = getattr(usage, "input_tokens", None) if usage else None
    output_tokens = getattr(usage, "output_tokens", None) if usage else None
    total_tokens = getattr(usage, "total_tokens", None) if usage else None

    return {
        "answer": answer,
        "model_used": CHAT_MODEL,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }