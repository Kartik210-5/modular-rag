# app.py
import streamlit as st
import logging

# Mute HTTP logs from Ollama/httpx
logging.getLogger("httpx").setLevel(logging.WARNING)

from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import generate_rag_answer
from src.memory import ConversationMemory

# 1. Page Configuration
st.set_page_config(page_title="Local RAG Assistant", page_icon="📚", layout="centered")
st.title("📚 Local Modular RAG Assistant")
st.caption("Powered by Llama 3.2, Nomic Embeddings, ChromaDB & FlashRank")

# 2. Initialize RAG Pipeline (Cached so it runs only once on startup)
@st.cache_resource
def init_rag_system():
    vector_store = VectorStoreManager()
    retriever = Retriever(vector_store)
    return retriever

retriever = init_rag_system()

# 3. Initialize Session Memory (Persists across user clicks)
if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_history_turns=3)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # [{role: "user"/"assistant", content: "..."}]

# 4. Render Existing Chat Messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")

# 5. Handle New User Input
if user_input := st.chat_input("Ask something about your PDFs..."):
    # Display user query
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Searching documents & generating answer..."):
            # A. Contextualize query with conversation memory
            search_query = st.session_state.memory.contextualize_query(user_input)
            
            # B. Retrieve and re-rank context
            context, sources = retriever.get_context_and_sources(search_query, top_k_parents=2)
            
            # C. Generate answer
            history_str = st.session_state.memory.get_formatted_history()
            answer = generate_rag_answer(
                query=user_input, 
                context=context, 
                chat_history=history_str
            )
            
            # D. Render response
            st.markdown(answer)
            if sources:
                with st.expander("View Retrieved Sources"):
                    for src in sources:
                        st.caption(f"• {src}")

            # E. Update conversation memory
            st.session_state.memory.add_user_message(user_input)
            st.session_state.memory.add_ai_message(answer)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources
            })