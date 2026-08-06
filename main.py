# main.py
from src.document_loader import load_and_process_pdfs
from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import generate_rag_answer
from src.memory import ConversationMemory

def setup_pipeline(pdf_folder: str = "./pdfs"):
    print("\n--- [Phase 1: Initializing RAG Pipeline] ---")
    vector_store = VectorStoreManager()
    
    if vector_store.collection.count() == 0:
        print("[Setup] Vector store empty. Processing PDFs...")
        parent_store, child_chunks = load_and_process_pdfs(pdf_dir=pdf_folder)
        if child_chunks:
            vector_store.add_data(parent_store, child_chunks)
    else:
        print(f"[Setup] Loaded existing database with {len(vector_store.parent_store)} parent blocks.")

    return Retriever(vector_store)

def main():
    retriever = setup_pipeline()
    memory = ConversationMemory(max_history_turns=3)
    
    print("\n--- [Phase 2: Conversational RAG Terminal] ---")
    print("Type your questions below (or 'exit' to quit):\n")

    while True:
        try:
            user_query = input("\nUser > ").strip()
            if not user_query or user_query.lower() in ["exit", "q", "quit"]:
                break

            # 1. Resolve pronouns/context using memory for vector search
            search_query = memory.contextualize_query(user_query)
            if search_query != user_query:
                print(f"[Memory] Search query rephrased to: '{search_query}'")

            # 2. Retrieve parent context blocks using the contextualized query
            context, sources = retriever.get_context_and_sources(search_query, top_k_parents=2)
            
            # 3. Pass original query, context, and chat history to the LLM
            history_str = memory.get_formatted_history()
            answer = generate_rag_answer(
                query=user_query, 
                context=context, 
                chat_history=history_str
            )
            
            # 4. Store turn in memory
            memory.add_user_message(user_query)
            memory.add_ai_message(answer)

            # 5. Output response
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