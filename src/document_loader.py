import os
import re
from typing import List, Dict
# pyrefly: ignore [missing-import]
from pypdf import PdfReader
from config import CHUNK_SIZE, CHUNK_OVERLAP

def clean_text(text: str) -> str:
    """
    Cleans raw text extracted from PDFs:
    - Removes form feed/page split markers
    - Recombines words split across lines (e.g., 'infor-\nmation' -> 'information')
    - Replaces newlines and tabs with single spaces
    - Collapses multiple spaces into one
    """
    # Remove form feed control characters
    text = text.replace('\x0c', ' ')
    
    # Fix words split by line breaks and hyphens
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Replace all newlines and carriage returns with spaces
    text = re.sub(r'[\r\n]+', ' ', text)
    
    # Collapse multiple whitespaces into a single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_text_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Splits a cleaned text string into overlapping chunks."""
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def load_and_process_pdfs(pdf_dir: str = "./pdfs") -> List[Dict]:
    """
    Scans the specified PDF folder, extracts and cleans text,
    and returns chunked objects ready for vector embedding.
    """
    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir)
        print(f"[Loader] Created directory '{pdf_dir}'. Place your PDF files here!")
        return []

    all_chunks = []

    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(pdf_dir, filename)
            print(f"[Loader] Extracting text from: {filename}...")
            
            try:
                reader = PdfReader(filepath)
                raw_full_text = ""
                
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        raw_full_text += f" {extracted}"
                
                # Clean the extracted text
                cleaned_text = clean_text(raw_full_text)
                
                # Split clean text into chunks
                chunks = split_text_into_chunks(cleaned_text)
                
                for idx, chunk_text in enumerate(chunks):
                    all_chunks.append({
                        "id": f"{filename}_chunk_{idx}",
                        "text": chunk_text,
                        "metadata": {
                            "source": filename,
                            "chunk_index": idx
                        }
                    })
            except Exception as e:
                print(f"[Loader Error] Failed to process {filename}: {e}")

    print(f"[Loader] Successfully processed {len(all_chunks)} chunks from '{pdf_dir}'.")
    return all_chunks