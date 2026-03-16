from typing import Dict, List

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import OPENAI_API_KEY, CHAT_MODEL
from app.prompt_builder import build_prompt

# Reuse one OpenAI client instance across LLM calls.
client = OpenAI(api_key=OPENAI_API_KEY)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8))
def generate_answer(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Generate the final grounded answer from retrieved document context.

    Retry logic helps recover from transient API failures and improves
    reliability for production-style usage.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if not retrieved_chunks:
        raise ValueError("Retrieved chunks cannot be empty.")

    prompt = build_prompt(question, retrieved_chunks)

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )

    answer = getattr(response, "output_text", "").strip()

    if not answer:
        raise ValueError("LLM returned an empty response.")

    return answer