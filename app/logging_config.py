import logging
from app.config import LOG_LEVEL


def setup_logging() -> None:
    """
    Configure application-wide logging format and level.
    Logging is important for debugging ingestion, retrieval, and API failures.
    """
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )