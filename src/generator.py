# src/generator.py
from src.models import generate_response

SYSTEM_PROMPT = """
You are a helpful, precise assistant answering questions based strictly on the provided context and conversation history.
Rules:
1. Rely ONLY on the clear facts directly mentioned in the context.
2. Do NOT assume or extrapolate beyond what is explicitly stated.
3. If the answer cannot be determined from the context, state: "I cannot find the answer to this question in the provided documents."
"""

def generate_rag_answer(query: str, context: str, chat_history: str = "") -> str:
    """
    Combines conversation history, retrieved context, and the user query into the final prompt.
    """
    user_prompt = f"""
--- CONVERSATION HISTORY ---
{chat_history if chat_history else "None"}

--- RETRIEVED CONTEXT ---
{context}

--- USER QUESTION ---
{query}

--- ANSWER ---
"""
    answer = generate_response(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    return answer