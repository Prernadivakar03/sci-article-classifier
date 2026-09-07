"""
pdf_utils.py
-------------
Extracts raw text + page count from an uploaded PDF, used by the
"Analyze Article" dashboard page.
Requires: pip install pypdf
"""

from pypdf import PdfReader


def extract_text_from_pdf(file):
    """
    file: a file-like object (e.g. Streamlit's UploadedFile from st.file_uploader)
    Returns: (full_text: str, num_pages: int)
    """
    reader = PdfReader(file)
    num_pages = len(reader.pages)
    text_parts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
    full_text = "\n".join(text_parts)
    return full_text, num_pages