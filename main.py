# main.py
from src.document_loader import load_and_process_pdfs
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import generate_rag_answer

def setup_pipeline(pdf_folder: str = "./pdfs"):
    print("\n--- [Phase 1: Initializing RAG Pipeline] ---")
    
    # 1. Initialize Vector Store
    vector_store = VectorStoreManager()
    
    # Check if database already contains documents
    existing_count = vector_store.collection.count()
    
    if existing_count == 0:
        print("[Setup] Vector DB is empty. Loading and indexing PDFs...")
        chunks = load_and_process_pdfs(pdf_dir=pdf_folder)
        if chunks:
            vector_store.add_chunks(chunks)
        else:
            print("[Warning] No PDFs found in the directory! Please place PDFs in './pdfs'")
    else:
        print(f"[Setup] Loaded existing vector store with {existing_count} chunks.")

    return Retriever(vector_store)

def main():
    retriever = setup_pipeline()
    
    print("\n--- [Phase 2: Interactive RAG Terminal] ---")
    print("Type your question below (or type 'exit' or 'q' to quit):\n")

    while True:
        try:
            query = input("\nUser > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "q", "quit"]:
                print("Goodbye!")
                break

            print("\nSearching database and generating answer...")
            
            # 1. Retrieve relevant chunks
            context, sources = retriever.get_context_and_sources(query, top_k=3)
            
            # 2. Generate answer with LLM
            answer = generate_rag_answer(query=query, context=context)
            
            # 3. Print output with citations
            print("\n" + "="*50)
            print("ASSISTANT ANSWER:")
            print(answer)
            print("-" * 50)
            print(f"Sources Used: {', '.join(sources) if sources else 'None'}")
            print("="*50)

        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()