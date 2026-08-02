# pyrefly: ignore [missing-import]
import chromadb
from typing import List, Dict
from config import VECTOR_DB_DIR
from src.models import get_embedding

class VectorStoreManager:
    def __init__(self, collection_name: str = "pdf_rag_collection"):
        # Initialize persistent storage on your disk
        self.client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
        
        # Get existing collection or create a new one using cosine similarity
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Generates vector embeddings for each chunk using nomic-embed-text
        and stores them in ChromaDB.
        """
        if not chunks:
            print("[VectorStore] No chunks provided to index.")
            return

        print(f"[VectorStore] Generating embeddings & saving {len(chunks)} chunks into ChromaDB...")

        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for item in chunks:
            # Generate embedding via Ollama
            vector = get_embedding(item["text"])
            
            ids.append(item["id"])
            documents.append(item["text"])
            embeddings.append(vector)
            metadatas.append(item["metadata"])

        # Upsert into ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"[VectorStore] Successfully indexed all chunks into ChromaDB!")

    def search(self, query_text: str, top_k: int = 3) -> List[Dict]:
        """
        Embeds the search query and retrieves the top-K most similar text chunks.
        """
        query_vector = get_embedding(query_text)
        
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k
        )
        
        retrieved = []
        if results and results["documents"]:
            for i in range(len(results["documents"][0])):
                retrieved.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i] if "distances" in results else None
                })
        return retrieved