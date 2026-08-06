# config.py

# Model Selection
LLM_MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"

# Parent-Document Chunking Configurations
PARENT_CHUNK_SIZE = 1200      # Large context block sent to the LLM
PARENT_CHUNK_OVERLAP = 150

CHILD_CHUNK_SIZE = 300        # Small focused block converted into vector embeddings
CHILD_CHUNK_OVERLAP = 30

# Storage Locations
VECTOR_DB_DIR = "./chroma_db"
PARENT_STORE_FILE = "./chroma_db/parent_store.json"