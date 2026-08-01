import ollama

# 1. Generate an embedding for a query
embed_response = ollama.embed(model="nomic-embed-text", input="How does RAG work?")
query_vector = embed_response['embeddings'][0]

# 2. Decompose a query using the LLM
decomposition_prompt = "Break this complex query into 2 simpler sub-questions: 'What is RAG and how do I install ChromaDB?'"
response = ollama.generate(model="llama3.2:3b", prompt=decomposition_prompt)
print(response['response'])