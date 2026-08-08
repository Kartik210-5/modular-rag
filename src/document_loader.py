# src/document_loader.py
import os
from typing import List, Dict, Tuple
from src.pdf_processor import extract_and_clean_pdf
from src.chunker import (
    fast_bounded_chunking, 
    split_into_child_chunks, 
    generate_document_context
)

def load_and_process_pdfs(pdf_dir: str = "./pdfs") -> Tuple[Dict, List[Dict]]:
    """
    Fast PDF ingestion pipeline:
    1. Extracts and cleans text/tables from PDF.
    2. Runs 1 single LLM call per PDF to get a global document summary.
    3. Uses fast structural chunking for parents (600-1200 chars).
    4. Attaches the document summary to child snippets for context.
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
                # 1. Clean Extraction
                cleaned_text = extract_and_clean_pdf(filepath)

                if not cleaned_text:
                    print(f"[Loader Warning] No readable content found in {filename}.")
                    continue

                # 2. Single LLM call for document-wide context
                print(f"[Loader] Generating 1-line document context summary for {filename}...")
                doc_context = generate_document_context(cleaned_text)

                # 3. Fast Parent Chunking (0 vector calls)
                parent_texts = fast_bounded_chunking(
                    text=cleaned_text,
                    min_chars=600,
                    max_chars=1200
                )

                # 4. Create Parents and Children
                for p_idx, p_text in enumerate(parent_texts):
                    parent_id = f"{filename}_parent_{p_idx}"
                    parent_store[parent_id] = {
                        "text": p_text,
                        "source": filename
                    }

                    raw_child_texts = split_into_child_chunks(p_text, child_size=300, child_overlap=30)
                    
                    for c_idx, child_raw in enumerate(raw_child_texts):
                        child_id = f"{parent_id}_child_{c_idx}"
                        
                        # Attach global doc context + parent source tag instantly
                        enriched_child_text = f"Document: {filename} | Summary: {doc_context}\nSnippet: {child_raw}"

                        child_chunks.append({
                            "id": child_id,
                            "text": enriched_child_text,
                            "metadata": {
                                "parent_id": parent_id,
                                "source": filename
                            }
                        })

            except Exception as e:
                print(f"[Loader Error] Failed to process {filename}: {e}")

    print(f"\n[Loader] Successfully ingested {len(parent_store)} parents and {len(child_chunks)} children in seconds.")
    return parent_store, child_chunks