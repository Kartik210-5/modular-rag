# src/chunker.py
import re
from typing import List
from src.models import generate_response

DOC_SUMMARY_PROMPT = """
Summarize the following document in 1-2 concise sentences. Mention the core topic, author/company, and purpose if present.
Do NOT include preamble.

Document Text:
{document_text}
"""

def generate_document_context(document_text: str) -> str:
    """Generates a SINGLE contextual summary for the entire PDF (1 LLM call per document)."""
    # Sample the first 2500 characters to keep it ultra-fast
    sample_text = document_text[:2500]
    prompt = DOC_SUMMARY_PROMPT.format(document_text=sample_text)
    return generate_response(prompt=prompt).strip()


def fast_bounded_chunking(text: str, min_chars: int = 600, max_chars: int = 1200) -> List[str]:
    """
    Fast structural text splitter. Uses double newlines (paragraphs) and sentence boundaries
    to group text into chunks strictly bounded between min_chars and max_chars.
    Takes milliseconds to execute.
    """
    if not text.strip():
        return []

    # Split into paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_len = len(para)

        # If adding paragraph exceeds max_chars, finalize current chunk
        if current_length + para_len > max_chars and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

        # If a single paragraph is larger than max_chars, split it by sentences
        if para_len > max_chars:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                s_len = len(sentence)
                if current_length + s_len > max_chars and current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                    current_chunk = []
                    current_length = 0
                current_chunk.append(sentence)
                current_length += s_len
        else:
            current_chunk.append(para)
            current_length += para_len

        # If we reached the minimum size, store chunk and reset
        if current_length >= min_chars:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = []
            current_length = 0

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return chunks


def split_into_child_chunks(text: str, child_size: int = 300, child_overlap: int = 30) -> List[str]:
    """Slices text into smaller child chunks."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + child_size
        chunks.append(text[start:end])
        start += child_size - child_overlap

    return chunks