# src/document_loader.py
import os
from typing import List, Dict, Tuple
from config import PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
from src.pdf_processor import extract_and_clean_pdf

def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Sliding-window chunker preserving overlap."""
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
    1. Scans ./pdfs for PDF files.
    2. Extracts and cleans tables, text, and image links.
    3. Splits cleaned text into Parent and Child chunk hierarchies.
    """
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"[Loader] Created directory '{pdf_dir}'. Drop your PDFs here.")
        return {}, []

    parent_store = {}
    child_chunks = []

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            
            try:
                # Extract and clean text using PyMuPDF4LLM + clean_markdown_text
                cleaned_text = extract_and_clean_pdf(filepath)

                if not cleaned_text:
                    print(f"[Loader Warning] No readable content found in {filename}.")
                    continue

                # 1. Create Parent Chunks
                parent_texts = split_text_into_chunks(cleaned_text, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP)

                for p_idx, p_text in enumerate(parent_texts):
                    parent_id = f"{filename}_parent_{p_idx}"
                    parent_store[parent_id] = {
                        "text": p_text,
                        "source": filename
                    }

                    # 2. Create Child Chunks linked to this Parent ID
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

    print(f"[Loader] Successfully processed and cleaned {len(parent_store)} parent blocks and {len(child_chunks)} child vectors.")
    return parent_store, child_chunks