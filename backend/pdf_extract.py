# backend/pdf_extract.py

from typing import Union

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False
    from pypdf import PdfReader

def extract_text(file_path_or_bytes: Union[str, bytes]) -> str:
    """
    Extracts text from a PDF file (path or bytes).
    Uses PyMuPDF if available, else falls back to pypdf.
    Returns "" on failure.
    """
    try:
        if _HAS_FITZ:
            if isinstance(file_path_or_bytes, str):
                doc = fitz.open(file_path_or_bytes)
            else:
                doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        else:
            if isinstance(file_path_or_bytes, str):
                pdf = PdfReader(file_path_or_bytes)
            else:
                import io
                pdf = PdfReader(io.BytesIO(file_path_or_bytes))
            text = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
            return text
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return ""