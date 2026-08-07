# src/decomposer.py
import json
from typing import List
from src.models import generate_response

DECOMPOSITION_PROMPT = """
You are an AI search assistant. Your task is to break down complex user questions into simpler, distinct sub-queries for document retrieval.

Instructions:
1. If the input question asks about multiple topics or compares two things, split it into 2 to 3 distinct, standalone sub-questions.
2. If the input question is already simple and direct, return ONLY the original question.
3. Respond STRICTLY in valid JSON format as a list of strings. Do NOT include any intro text, markdown formatting, or explanations.

Example Input:
"What is Parent-Document retrieval and how does it reduce memory compared to standard chunking?"

Example Output:
["What is Parent-Document retrieval?", "How does Parent-Document retrieval reduce memory compared to standard chunking?"]

User Question: {query}
"""

def decompose_query(query: str) -> List[str]:
    """
    Passes a question to llama3.2:3b to decompose it into sub-queries.
    Returns a list of string queries.
    """
    prompt = DECOMPOSITION_PROMPT.format(query=query)
    raw_response = generate_response(prompt=prompt).strip()

    # Clean markdown formatting if present (e.g. ```json ... ```)
    if "```" in raw_response:
        raw_response = raw_response.split("```")[1]
        if raw_response.startswith("json"):
            raw_response = raw_response[4:].strip()

    try:
        sub_queries = json.loads(raw_response)
        if isinstance(sub_queries, list) and len(sub_queries) > 0:
            return [str(q).strip() for q in sub_queries]
    except Exception as e:
        print(f"[Decomposer] Json parse fallback: using original query. ({e})")

    return [query]