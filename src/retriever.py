# src/retriever.py
from typing import Tuple, List
from src.vector_store import VectorStoreManager
from src.reranker import FlashReranker
from src.decomposer import decompose_query

class Retriever:
    def __init__(self, vector_store: VectorStoreManager):
        self.vector_store = vector_store
        self.reranker = FlashReranker()

    def get_context_and_sources(self, query: str, top_k_parents: int = 2) -> Tuple[str, List[str], List[str]]:
        """
        1. Decomposes input query into sub-queries.
        2. Executes Hybrid Search (Vector + BM25 + RRF) per sub-query.
        3. Aggregates and deduplicates parent candidate blocks (holding tables/media).
        4. Re-ranks candidates using FlashRank cross-encoder.
        """
        sub_queries = decompose_query(query)
        print(f"[Retriever] Decomposed into sub-queries: {sub_queries}")

        aggregated_candidates = []
        seen_parent_ids = set()

        for sub_q in sub_queries:
            hybrid_candidates = self.vector_store.search_parents_hybrid(
                query_text=sub_q, 
                top_k_parents=4
            )
            for item in hybrid_candidates:
                if item["parent_id"] not in seen_parent_ids:
                    seen_parent_ids.add(item["parent_id"])
                    aggregated_candidates.append(item)

        if not aggregated_candidates:
            return "No relevant context found.", [], sub_queries

        # Re-rank parent candidates containing formatted Markdown tables/media
        reranked_matches = self.reranker.rerank(
            query=query, 
            candidate_chunks=aggregated_candidates, 
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

        return full_context, unique_sources, sub_queries