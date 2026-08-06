# src/vector_store.py
import json
import os
import chromadb
from typing import List, Dict
from config import VECTOR_DB_DIR, PARENT_STORE_FILE
from src.models import get_embedding

class VectorStoreManager:
    def __init__(self, collection_name: str = "parent_child_collection"):
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.parent_store_file = PARENT_STORE_FILE
        self.parent_store = self._load_parent_store()

    def _load_parent_store(self) -> Dict:
        """Loads persistent JSON key-value store mapping parent_ids to parent texts."""
        if os.path.exists(self.parent_store_file):
            with open(self.parent_store_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_parent_store(self) -> None:
        """Persists the parent dictionary onto disk."""
        os.makedirs(os.path.dirname(self.parent_store_file), exist_ok=True)
        with open(self.parent_store_file, "w", encoding="utf-8") as f:
            json.dump(self.parent_store, f, indent=2)

    def add_data(self, parent_store: Dict, child_chunks: List[Dict]) -> None:
        """Saves parent mappings to disk and child embeddings to ChromaDB."""
        if not child_chunks:
            print("[VectorStore] No chunks provided to index.")
            return

        # 1. Update and persist parent store
        self.parent_store.update(parent_store)
        self._save_parent_store()

        print(f"[VectorStore] Embedding {len(child_chunks)} child vectors via Ollama...")

        ids, documents, embeddings, metadatas = [], [], [], []

        for item in child_chunks:
            vector = get_embedding(item["text"])
            ids.append(item["id"])
            documents.append(item["text"])
            embeddings.append(vector)
            metadatas.append(item["metadata"])

        # 2. Upsert child chunks to ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print("[VectorStore] Indexing complete!")

    def search_parents(self, query_text: str, top_k_parents: int = 3) -> List[Dict]:
        """
        Embeds search query, searches child vectors in ChromaDB,
        and retrieves their unique parent blocks.
        """
        query_vector = get_embedding(query_text)
        
        # Over-sample child vectors to account for multiple child matches belonging to the same parent
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k_parents * 4
        )
        
        retrieved_parents = []
        seen_parent_ids = set()

        if results and results["metadatas"]:
            for meta in results["metadatas"][0]:
                parent_id = meta.get("parent_id")

                if parent_id and parent_id not in seen_parent_ids:
                    seen_parent_ids.add(parent_id)
                    parent_data = self.parent_store.get(parent_id)
                    
                    if parent_data:
                        retrieved_parents.append({
                            "parent_id": parent_id,
                            "text": parent_data["text"],
                            "source": parent_data["source"]
                        })

                if len(retrieved_parents) == top_k_parents:
                    break

        return retrieved_parents