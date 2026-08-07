# ingest.py
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.document_loader import load_and_process_pdfs
from src.vector_store import VectorStoreManager

def run_ingestion():
    print("--- [Starting PDF Ingestion into ChromaDB] ---")
    
    # 1. Initialize Vector Store (PersistentClient)
    vector_store = VectorStoreManager()
    
    # 2. Extract, Clean, and Chunk PDFs from ./pdfs
    parent_store, child_chunks = load_and_process_pdfs(pdf_dir="./pdfs")
    
    if not child_chunks:
        print("[Ingestion] No PDFs found in './pdfs' or files are empty.")
        return

    # 3. Embed & Save vectors to ./chroma_db and parents to ./chroma_db/parent_store.json
    vector_store.add_data(parent_store, child_chunks)
    
    print("\n Success! ChromaDB database saved to './chroma_db/'")

if __name__ == "__main__":
    run_ingestion()