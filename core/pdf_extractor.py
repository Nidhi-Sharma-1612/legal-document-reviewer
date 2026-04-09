"""PDF text extraction using PyPDF2."""

import PyPDF2
from io import BytesIO
from dataclasses import dataclass
from typing import List


@dataclass
class ExtractionResult:
    text: str
    page_warnings: List[str]  # pages that could not be extracted


def extract_text_from_pdf(uploaded_file) -> ExtractionResult:
    """
    Extract text from an uploaded PDF file.

    Returns an ExtractionResult with the extracted text and any per-page warnings.
    Raises ValueError for user-facing errors (encrypted, empty, unreadable).
    Raises RuntimeError for unexpected low-level failures.
    """
    try:
        pdf_bytes = BytesIO(uploaded_file.read())
        reader = PyPDF2.PdfReader(pdf_bytes)
    except PyPDF2.errors.PdfReadError as e:
        raise ValueError(f"Failed to read PDF: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected error while opening PDF: {e}") from e

    if reader.is_encrypted:
        raise ValueError("This PDF is encrypted/password-protected and cannot be processed.")

    text = ""
    page_warnings: List[str] = []

    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
            else:
                page_warnings.append(
                    f"Page {page_num + 1} yielded no text — it may be a scanned image."
                )
        except Exception:
            page_warnings.append(
                f"Page {page_num + 1} could not be extracted and was skipped."
            )

    if not text.strip():
        raise ValueError(
            "No text could be extracted from this PDF. "
            "It may be a scanned (image-based) document. "
            "Please upload a text-based PDF."
        )

    return ExtractionResult(text=text, page_warnings=page_warnings)
