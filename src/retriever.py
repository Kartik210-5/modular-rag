# src/retriever.py
from typing import Tuple
from src.vector_store import VectorStoreManager

class Retriever:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def get_context_and_sources(self, query: str, top_k_parents: int = 3) -> Tuple[str, list]:
        """
        Searches child vector embeddings and returns full parent context blocks.
        """
        parent_matches = self.vector_store.search_parents(
            query_text=query, 
            top_k_parents=top_k_parents
        )
        
        if not parent_matches:
            return "No relevant context found.", []

        formatted_context_blocks = []
        sources = []

        for match in parent_matches:
            source_file = match["source"]
            text = match["text"]
            
            block = f"[Source: {source_file}]\n{text}"
            formatted_context_blocks.append(block)
            sources.append(source_file)

        full_context = "\n\n".join(formatted_context_blocks)
        unique_sources = list(set(sources))

        return full_context, unique_sources