# app.py
import streamlit as st
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import generate_rag_answer
from src.memory import ConversationMemory

st.set_page_config(page_title="Local RAG Assistant", page_icon="📚", layout="centered")
st.title("📚 Local Modular RAG Assistant")
st.caption("Parent-Child RAG + Query Decomposition + FlashRank Re-ranking")

@st.cache_resource
def init_rag_system():
    vector_store = VectorStoreManager()
    return Retriever(vector_store)

retriever = init_rag_system()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory(max_history_turns=3)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sub_queries" in msg and len(msg["sub_queries"]) > 1:
            with st.expander("View Decomposed Sub-Queries"):
                for sq in msg["sub_queries"]:
                    st.caption(f"• {sq}")
        if "sources" in msg and msg["sources"]:
            with st.expander("View Retrieved Sources"):
                for src in msg["sources"]:
                    st.caption(f"• {src}")

if user_input := st.chat_input("Ask a question about your PDFs..."):
    st.chat_message("user").markdown(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Decomposing query, retrieving context & re-ranking..."):
            # 1. Resolve pronouns with memory
            search_query = st.session_state.memory.contextualize_query(user_input)
            
            # 2. Decompose, retrieve across sub-queries, and re-rank
            context, sources, sub_queries = retriever.get_context_and_sources(
                query=search_query, 
                top_k_parents=2
            )
            
            # 3. Display generated sub-queries in UI if query was split
            if len(sub_queries) > 1:
                with st.expander("View Decomposed Sub-Queries"):
                    for sq in sub_queries:
                        st.caption(f"• {sq}")

            # 4. Generate final answer
            history_str = st.session_state.memory.get_formatted_history()
            answer = generate_rag_answer(
                query=user_input, 
                context=context, 
                chat_history=history_str
            )
            
            st.markdown(answer)
            if sources:
                with st.expander("View Retrieved Sources"):
                    for src in sources:
                        st.caption(f"• {src}")

            st.session_state.memory.add_user_message(user_input)
            st.session_state.memory.add_ai_message(answer)
            
            st.session_state.chat_history.append({
                "role": "assistant", 
                "content": answer,
                "sources": sources,
                "sub_queries": sub_queries
            })