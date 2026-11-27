"""PDF resume text extraction utilities."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

try:
    from PyPDF2 import PdfReader
except ImportError:  # pragma: no cover - fallback when pypdf is installed instead
    from pypdf import PdfReader


def extract_resume_text(source: bytes | str | Path | Any) -> str:
    """Extracts plain text from a PDF file path, bytes buffer, or uploaded file object."""
    if isinstance(source, (str, Path)):
        pdf_bytes = Path(source).read_bytes()
    elif isinstance(source, (bytes, bytearray)):
        pdf_bytes = bytes(source)
    elif hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        pdf_bytes = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
    else:
        raise TypeError("Resume source must be bytes, a file path, or a binary file-like object.")

    if not pdf_bytes:
        return ""

    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text: list[str] = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text and page_text.strip():
            pages_text.append(page_text.strip())

    return "\n\n".join(pages_text).strip()
