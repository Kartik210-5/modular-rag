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
        1. Decomposes query into sub-queries.
        2. Retrieves vector candidates for each sub-query.
        3. Deduplicates candidate parent blocks.
        4. Re-ranks aggregated candidates using FlashRank against the original query.
        """
        # Step 1: Decompose the query into sub-queries
        sub_queries = decompose_query(query)
        print(f"[Retriever] Decomposed into {len(sub_queries)} sub-queries: {sub_queries}")

        # Step 2: Search vector store for each sub-query and aggregate candidates
        aggregated_candidates = []
        seen_parent_ids = set()

        for sub_q in sub_queries:
            raw_candidates = self.vector_store.search_parents(
                query_text=sub_q, 
                top_k_parents=4
            )
            for item in raw_candidates:
                if item["parent_id"] not in seen_parent_ids:
                    seen_parent_ids.add(item["parent_id"])
                    aggregated_candidates.append(item)

        if not aggregated_candidates:
            return "No relevant context found.", [], sub_queries

        # Step 3: Re-rank aggregated candidates against the full original query
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