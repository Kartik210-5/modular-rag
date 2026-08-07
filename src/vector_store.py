# src/vector_store.py
import json
import os
import re
import chromadb
from typing import List, Dict
from rank_bm25 import BM25Okapi
from config import VECTOR_DB_DIR, PARENT_STORE_FILE
from src.models import get_embedding

def tokenize(text: str) -> List[str]:
    """Alphanumeric tokenization for BM25 search."""
    return re.findall(r'\w+', text.lower())

class VectorStoreManager:
    def __init__(self, collection_name: str = "hybrid_parent_child_collection"):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.parent_store_file = PARENT_STORE_FILE
        self.parent_store = self._load_parent_store()
        
        # BM25 Sparse Index State
        self.bm25 = None
        self.child_metadata_list = []
        self._build_bm25_index()

    def _load_parent_store(self) -> Dict:
        if os.path.exists(self.parent_store_file):
            with open(self.parent_store_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_parent_store(self) -> None:
        os.makedirs(os.path.dirname(self.parent_store_file), exist_ok=True)
        with open(self.parent_store_file, "w", encoding="utf-8") as f:
            json.dump(self.parent_store, f, indent=2)

    def _build_bm25_index(self) -> None:
        """Constructs in-memory BM25 index over all child chunks in ChromaDB."""
        if self.collection.count() == 0:
            return

        print("[VectorStore] Building BM25 index over child chunks...")
        existing_data = self.collection.get(include=["documents", "metadatas"])
        
        documents = existing_data["documents"]
        metadatas = existing_data["metadatas"]

        if documents:
            tokenized_corpus = [tokenize(doc) for doc in documents]
            self.bm25 = BM25Okapi(tokenized_corpus)
            self.child_metadata_list = metadatas
            print(f"[VectorStore] BM25 Index ready ({len(documents)} child chunks).")

    def add_data(self, parent_store: Dict, child_chunks: List[Dict]) -> None:
        """Persists parent texts and indexes child chunks into ChromaDB & BM25."""
        if not child_chunks:
            return

        self.parent_store.update(parent_store)
        self._save_parent_store()

        print(f"[VectorStore] Embedding {len(child_chunks)} child chunks...")

        ids, documents, embeddings, metadatas = [], [], [], []

        for item in child_chunks:
            vector = get_embedding(item["text"])
            ids.append(item["id"])
            documents.append(item["text"])
            embeddings.append(vector)
            metadatas.append(item["metadata"])

        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        
        # Rebuild BM25 index with newly added documents
        self._build_bm25_index()

    def search_parents_hybrid(self, query_text: str, top_k_parents: int = 4, rrf_k: int = 60) -> List[Dict]:
        """
        Runs Hybrid Retrieval:
        1. Dense Vector Search via ChromaDB
        2. Sparse Lexical Search via BM25
        3. Merges results using Reciprocal Rank Fusion (RRF)
        """
        candidate_parent_ranks = {}  # {parent_id: rrf_score}
        
        # --- 1. Dense Vector Search ---
        query_vector = get_embedding(query_text)
        vector_results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k_parents * 4
        )

        if vector_results and vector_results["metadatas"]:
            for rank, meta in enumerate(vector_results["metadatas"][0], start=1):
                parent_id = meta.get("parent_id")
                if parent_id:
                    score = 1.0 / (rrf_k + rank)
                    candidate_parent_ranks[parent_id] = candidate_parent_ranks.get(parent_id, 0.0) + score

        # --- 2. Sparse BM25 Keyword Search ---
        if self.bm25:
            tokenized_query = tokenize(query_text)
            bm25_scores = self.bm25.get_scores(tokenized_query)
            
            top_bm25_indices = sorted(
                range(len(bm25_scores)), 
                key=lambda i: bm25_scores[i], 
                reverse=True
            )[:top_k_parents * 4]

            for rank, idx in enumerate(top_bm25_indices, start=1):
                if bm25_scores[idx] > 0:
                    meta = self.child_metadata_list[idx]
                    parent_id = meta.get("parent_id")
                    if parent_id:
                        score = 1.0 / (rrf_k + rank)
                        candidate_parent_ranks[parent_id] = candidate_parent_ranks.get(parent_id, 0.0) + score

        # --- 3. Rank Parent Chunks by Combined RRF Score ---
        sorted_parent_ids = sorted(
            candidate_parent_ranks.items(), 
            key=lambda item: item[1], 
            reverse=True
        )

        retrieved_parents = []
        for parent_id, rrf_score in sorted_parent_ids[:top_k_parents]:
            parent_data = self.parent_store.get(parent_id)
            if parent_data:
                retrieved_parents.append({
                    "parent_id": parent_id,
                    "text": parent_data["text"],
                    "source": parent_data["source"],
                    "rrf_score": rrf_score
                })

        return retrieved_parents