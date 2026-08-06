# src/reranker.py
from typing import List, Dict
# pyrefly: ignore [missing-import]
from flashrank import Ranker, RerankRequest

class FlashReranker:
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initializes FlashRank reranker.
        'ms-marco-MiniLM-L-12-v2' is ~34MB, highly accurate, and runs 
        without PyTorch/CUDA dependencies on CPU.
        """
        self.ranker = Ranker(model_name=model_name)

    def rerank(self, query: str, candidate_chunks: List[Dict], top_n: int = 2) -> List[Dict]:
        """
        Re-ranks candidate chunks based on true Cross-Encoder relevance scores.
        
        candidate_chunks: List of Dicts [{"text": str, "source": str}]
        """
        if not candidate_chunks:
            return []

        # Convert candidates into FlashRank passage format
        passages = [
            {
                "id": idx,
                "text": chunk["text"],
                "meta": {"source": chunk.get("source", "Unknown")}
            }
            for idx, chunk in enumerate(candidate_chunks)
        ]

        # Execute re-ranking request
        rerank_request = RerankRequest(query=query, passages=passages)
        ranked_results = self.ranker.rerank(rerank_request)

        # Extract the top_n results
        final_reranked = []
        for item in ranked_results[:top_n]:
            final_reranked.append({
                "text": item["text"],
                "source": item["meta"]["source"],
                "score": item["score"]
            })

        return final_reranked