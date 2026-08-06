# src/document_loader.py
import os
import re
from typing import List, Dict, Tuple
from pypdf import PdfReader
from config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """Cleans up formatting issues, line splits, and whitespace from PDF text."""
    text = text.replace('\x0c', ' ')
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Generic sliding window string splitter."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap

    return chunks

def load_and_process_pdfs(pdf_dir: str = "./pdfs") -> Tuple[Dict, List[Dict]]:
    """
    Scans PDF directory, extracts text, creates Parent-Child chunks.
    Returns:
      1. parent_store: Dict mapping {parent_id: {"text": str, "source": str}}
      2. child_chunks: List of Dicts [{id, text, metadata: {parent_id, source}}]
    """
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"[Loader] Created directory '{pdf_dir}'. Place your PDF files here!")
        return {}, []

    parent_store = {}
    child_chunks = []

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            print(f"[Loader] Processing PDF for Parent-Child indexing: {filename}...")
            
            try:
                reader = PdfReader(filepath)
                raw_text = " ".join([page.extract_text() or "" for page in reader.pages])
                cleaned = clean_text(raw_text)

                # 1. Create Parent Chunks
                parent_texts = split_text_into_chunks(cleaned, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)

                for p_idx, p_text in enumerate(parent_texts):
                    parent_id = f"{filename}_parent_{p_idx}"
                    parent_store[parent_id] = {
                        "text": p_text,
                        "source": filename
                    }

                    # 2. Create Child Chunks inside this Parent Chunk
                    child_texts = split_text_into_chunks(p_text, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP)
                    for c_idx, c_text in enumerate(child_texts):
                        child_id = f"{parent_id}_child_{c_idx}"
                        child_chunks.append({
                            "id": child_id,
                            "text": c_text,
                            "metadata": {
                                "parent_id": parent_id,
                                "source": filename
                            }
                        })

            except Exception as e:
                print(f"[Loader Error] Failed to process {filename}: {e}")

    print(f"[Loader] Generated {len(parent_store)} parent blocks and {len(child_chunks)} child vectors.")
    return parent_store, child_chunks