# src/memory.py
from typing import List, Dict
from src.models import generate_response

class ConversationMemory:
    def __init__(self, max_history_turns: int = 3):
        """
        max_history_turns: Number of Q&A pairs to keep in RAM.
        3 turns is optimal for an 8GB Mac to prevent context window bloat.
        """
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, str]] = []

    def add_user_message(self, message: str) -> None:
        self.history.append({"role": "user", "content": message})
        self._trim()

    def add_ai_message(self, message: str) -> None:
        self.history.append({"role": "assistant", "content": message})
        self._trim()

    def _trim(self) -> None:
        """Keeps only the most recent (max_history_turns * 2) messages."""
        max_messages = self.max_history_turns * 2
        if len(self.history) > max_messages:
            self.history = self.history[-max_messages:]

    def get_formatted_history(self) -> str:
        """Returns the conversation formatted as a string for LLM prompts."""
        if not self.history:
            return "No previous context."
        
        formatted = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def contextualize_query(self, new_query: str) -> str:
        """
        If chat history exists, rewrites ambiguous follow-up questions 
        into standalone queries for vector search.
        """
        if not self.history:
            return new_query

        prompt = f"""Given the conversation history and a follow-up question, rephrase the follow-up question to be a standalone question that can be understood without the conversation history. Do NOT answer the question, only rephrase it if necessary. If it is already standalone, return it unchanged.

Chat History:
{self.get_formatted_history()}

Follow-up Question: {new_query}

Standalone Question:"""

        standalone_query = generate_response(prompt=prompt).strip()
        return standalone_query