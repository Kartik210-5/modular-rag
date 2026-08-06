# main.py
from src.document_loader import load_and_process_pdfs
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import generate_rag_answer

def setup_pipeline(pdf_folder: str = "./pdfs"):
    print("\n--- [Phase 1: Initializing Parent-Document RAG Pipeline] ---")
    
    vector_store = VectorStoreManager()
    
    if vector_store.collection.count() == 0:
        print("[Setup] Storage empty. Processing PDFs for Parent-Child indexing...")
        parent_store, child_chunks = load_and_process_pdfs(pdf_dir=pdf_folder)
        if child_chunks:
            vector_store.add_data(parent_store, child_chunks)
        else:
            print("[Warning] Place PDF files into './pdfs' to populate the database.")
    else:
        print(f"[Setup] Loaded existing database with {len(vector_store.parent_store)} parent blocks.")

    return Retriever(vector_store)

def main():
    retriever = setup_pipeline()
    
    print("\n--- [Phase 2: Parent-Document Interactive Terminal] ---")
    print("Ask any question (or type 'exit' or 'q' to quit):\n")

    while True:
        try:
            query = input("\nUser > ").strip()
            if not query or query.lower() in ["exit", "q", "quit"]:
                break

            print("\nSearching small child vectors -> Fetching full parent context...")
            context, sources = retriever.get_context_and_sources(query, top_k_parents=2)
            
            answer = generate_rag_answer(query=query, context=context)
            
            print("\n" + "="*50)
            print("ASSISTANT ANSWER:")
            print(answer)
            print("-" * 50)
            print(f"Sources Used: {', '.join(sources) if sources else 'None'}")
            print("="*50)

        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    main()