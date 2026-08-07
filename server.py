# server.py

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.vector_store import VectorStoreManager
from src.retriever import Retriever
from src.generator import generate_rag_answer
from src.memory import ConversationMemory


app = FastAPI()


# ---------------------------------------------------------
# Initialize RAG once
# ---------------------------------------------------------

vector_store = VectorStoreManager()
retriever = Retriever(vector_store)

memory = ConversationMemory(
    max_history_turns=3
)


# ---------------------------------------------------------
# Request model
# ---------------------------------------------------------

class ChatRequest(BaseModel):

    query: str

    chat_history: list = []


# ---------------------------------------------------------
# Chat endpoint
# ---------------------------------------------------------

@app.post("/api/chat")
async def chat(request: ChatRequest):

    user_input = request.query


    # ---------------------------------------------
    # Resolve conversation context
    # ---------------------------------------------

    search_query = (
        memory.contextualize_query(
            user_input
        )
    )


    # ---------------------------------------------
    # Retrieve + rerank
    # ---------------------------------------------

    (
        context,
        sources,
        sub_queries
    ) = retriever.get_context_and_sources(

        query=search_query,

        top_k_parents=2

    )


    # ---------------------------------------------
    # Generate answer
    # ---------------------------------------------

    history_str = (
        memory.get_formatted_history()
    )


    answer = generate_rag_answer(

        query=user_input,

        context=context,

        chat_history=history_str

    )


    # ---------------------------------------------
    # Update memory
    # ---------------------------------------------

    memory.add_user_message(
        user_input
    )

    memory.add_ai_message(
        answer
    )


    # ---------------------------------------------
    # Return JSON to frontend
    # ---------------------------------------------

    return {

        "answer": answer,

        "sources": sources,

        "sub_queries": sub_queries

    }


# ---------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------

app.mount(
    "/",
    StaticFiles(
        directory=".",
        html=True
    ),
    name="frontend"
)
