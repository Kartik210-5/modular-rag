# src/generator.py
from src.models import generate_response

SYSTEM_PROMPT = """
You are a helpful, precise assistant answering questions based strictly on the provided context.
Rules:
1. Rely ONLY on the clear facts directly mentioned in the context.
2. Do NOT assume or extrapolate beyond what is explicitly stated.
3. If the answer cannot be determined from the context, clearly state: "I cannot find the answer to this question in the provided documents."
"""

def generate_rag_answer(query: str, context: str) -> str:
    """
    Combines user query and retrieved context into an LLM prompt and returns the answer.
    """
    user_prompt = f"""
--- RETRIEVED CONTEXT ---
{context}

--- USER QUESTION ---
{query}

--- ANSWER ---
"""
    # Call our lightweight local LLM model via models.py
    answer = generate_response(prompt=user_prompt, system_prompt=SYSTEM_PROMPT)
    return answer