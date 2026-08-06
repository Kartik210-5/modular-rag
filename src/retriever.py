# src/retriever.py
from typing import Tuple
from src.vector_store import VectorStoreManager
from src.reranker import FlashReranker

class Retriever:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.reranker = FlashReranker()

    def get_context_and_sources(self, query: str, top_k_parents: int = 2) -> Tuple[str, list]:
        """
        1. Over-samples candidate parent blocks from ChromaDB.
        2. Passes candidates through FlashRank Cross-Encoder.
        3. Returns top_k_parents context blocks.
        """
        # Step 1: Over-sample from vector database (retrieve 8 candidates)
        raw_candidates = self.vector_store.search_parents(
            query_text=query, 
            top_k_parents=8
        )
        
        if not raw_candidates:
            return "No relevant context found.", []

        # Step 2: Re-rank raw candidates down to the best 'top_k_parents'
        reranked_matches = self.reranker.rerank(
            query=query, 
            candidate_chunks=raw_candidates, 
            top_n=top_k_parents
        )

        formatted_context_blocks = []
        sources = []

        for match in reranked_matches:
            source_file = match["source"]
            text = match["text"]
            score = match["score"]
            
            block = f"[Source: {source_file} | Relevance Score: {score:.4f}]\n{text}"
            formatted_context_blocks.append(block)
            sources.append(source_file)

        full_context = "\n\n".join(formatted_context_blocks)
        unique_sources = list(set(sources))

        return full_context, unique_sources