# src/pdf_processor.py
import os
import re
import pymupdf4llm

MEDIA_DIR = "./extracted_media"

def clean_markdown_text(text: str) -> str:
    """
    Cleans PDF extraction artifacts without destroying Markdown tables or image tags.
    """
    if not text:
        return ""

    # 1. Remove null bytes and non-printable control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)

    # 2. Rejoin words split across lines with hyphens (e.g., "docu-\nment" -> "document")
    text = re.sub(r'(\b[a-zA-Z]+)-\n([a-zA-Z]+\b)', r'\1\2', text)

    # 3. Clean trailing whitespaces from each line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)

    # 4. Collapse 3 or more consecutive empty lines into standard double newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def extract_and_clean_pdf(pdf_path: str) -> str:
    """
    Extracts high-fidelity tables, images, and text from a PDF,
    passes it through cleaning rules, and returns sanitized Markdown.
    """
    os.makedirs(MEDIA_DIR, exist_ok=True)
    
    print(f"[PDFProcessor] Extracting & cleaning PDF: {os.path.basename(pdf_path)}")
    
    # Extract raw layout-aware Markdown using PyMuPDF4LLM
    raw_markdown = pymupdf4llm.to_markdown(
        doc=pdf_path,
        write_images=True,
        image_path=MEDIA_DIR,
        image_format="png"
    )
    
    # Run through cleaning pipeline
    cleaned_markdown = clean_markdown_text(raw_markdown)
    return cleaned_markdown