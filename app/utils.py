import os
import re
from typing import List


def sanitize_filename(filename: str) -> str:
    """
    Remove unsafe characters from uploaded filenames.
    This helps avoid path issues and keeps saved files predictable.
    """
    filename = os.path.basename(filename)
    return re.sub(r"[^a-zA-Z0-9._-]", "_", filename)


def ensure_directory(path: str) -> None:
    """
    Create a directory if it does not already exist.
    """
    os.makedirs(path, exist_ok=True)


def deduplicate_sources(sources: List[str]) -> List[str]:
    """
    Preserve order while removing duplicate source names.
    Useful when multiple retrieved chunks come from the same document.
    """
    seen = set()
    ordered = []

    for src in sources:
        if src not in seen:
            seen.add(src)
            ordered.append(src)

    return ordered