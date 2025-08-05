# backend/pdf_extract.py

from typing import Union
import fitz  # PyMuPDF

def extract_text(file_path_or_bytes: Union[str, bytes]) -> str:
    """
    Extracts text from a PDF file (path or bytes). 
    Returns "" on failure.
    """
    try:
        if isinstance(file_path_or_bytes, str):
            doc = fitz.open(file_path_or_bytes)
        else:
            doc = fitz.open(stream=file_path_or_bytes, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return ""