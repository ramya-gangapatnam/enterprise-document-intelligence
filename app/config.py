import os
from dotenv import load_dotenv

# Load environment variables from the .env file at application startup.
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4.1-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "enterprise_docs")
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "4"))
MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "700"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def validate_config() -> None:
    """
    Fail fast if critical configuration is missing.
    This prevents confusing runtime errors later in the pipeline.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is missing. Add it to your .env file.")