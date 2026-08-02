from src.document_loader import load_and_process_pdfs
from src.vector_store import VectorStoreManager

if __name__ == "__main__":
    # 1. Load, extract & clean PDFs from ./pdfs folder
    chunks = load_and_process_pdfs(pdf_dir="./pdfs")
    
    # 2. Store in ChromaDB
    db = VectorStoreManager()
    db.add_chunks(chunks)
    
    # 3. Test Retrieval
    query = "What are the main findings discussed in the document?"
    print(f"\n[Search Query]: {query}\n")
    
    matches = db.search(query_text=query, top_k=2)
    for idx, match in enumerate(matches):
        print(f"--- Result {idx + 1} (Source: {match['metadata']['source']}) ---")
        print(f"{match['text']}\n")