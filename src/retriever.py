# src/retriever.py
from typing import Tuple
from src.vector_store import VectorStoreManager

class Retriever:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store

    def get_context_and_sources(self, query: str, top_k: int = 3) -> Tuple[str, list]:
        """
        Searches ChromaDB for the query and returns:
        1. A single formatted context string ready for the LLM prompt.
        2. A list of source metadata for citations.
        """
        results = self.vector_store.search(query_text=query, top_k=top_k)
        
        if not results:
            return "No relevant context found.", []

        formatted_context_blocks = []
        sources = []

        for idx, item in enumerate(results):
            source_file = item["metadata"]["source"]
            chunk_idx = item["metadata"]["chunk_index"]
            text = item["text"]
            
            # Format block with clear source markers for the LLM
            block = f"[Document: {source_file} | Chunk: {chunk_idx}]\n{text}"
            formatted_context_blocks.append(block)
            sources.append(source_file)

        # Join all retrieved text blocks into a single context string
        full_context = "\n\n".join(formatted_context_blocks)
        
        # Deduplicate source file names
        unique_sources = list(set(sources))

        return full_context, unique_sources