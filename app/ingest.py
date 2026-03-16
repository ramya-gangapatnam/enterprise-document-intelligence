import os
from typing import Callable
from docx import Document
from PyPDF2 import PdfReader


def load_txt(file_path: str) -> str:
    """
    Load plain text documents.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read().strip()


def load_pdf(file_path: str) -> str:
    """
    Extract text from each page of a PDF and combine it.
    """
    reader = PdfReader(file_path)
    pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        pages.append(page_text)

    return "\n".join(pages).strip()


def load_docx(file_path: str) -> str:
    """
    Extract text from DOCX paragraphs.
    """
    document = Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs).strip()


def load_document(file_path: str) -> str:
    """
    Route document loading based on file extension.
    Raises a clear error for unsupported or empty files.
    """
    extension = os.path.splitext(file_path)[1].lower()

    loaders: dict[str, Callable[[str], str]] = {
        ".txt": load_txt,
        ".pdf": load_pdf,
        ".docx": load_docx,
    }

    if extension not in loaders:
        raise ValueError(f"Unsupported file type: {extension}")

    text = loaders[extension](file_path)

    if not text.strip():
        raise ValueError("Document is empty after text extraction.")

    return text