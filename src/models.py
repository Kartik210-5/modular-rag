import ollama
from config import LLM_MODEL, EMBED_MODEL

def ensure_model_exists(model_name: str) -> None:
    """
    Checks if the specified model exists locally in Ollama.
    If missing, automatically pulls it once.
    """
    try:
        # Get list of locally available models
        list_response = ollama.list()
        local_models = [m.model for m in list_response.models]
        
        # Check if the requested model (or its tag) is already downloaded
        if not any(model_name in model for model in local_models):
            print(f"[Ollama] Model '{model_name}' not found locally. Pulling now (one-time setup)...")
            ollama.pull(model_name)
            print(f"[Ollama] Successfully pulled '{model_name}'.")
    except Exception as e:
        print(f"[Error] Failed to connect to Ollama service: {e}")
        print("Make sure Ollama app is installed and running on your Mac!")
        raise e

def get_embedding(text: str) -> list[float]:
    """
    Generates a numerical vector embedding for a given text string.
    """
    ensure_model_exists(EMBED_MODEL)
    response = ollama.embed(model=EMBED_MODEL, input=text)
    return response['embeddings'][0]

def generate_response(prompt: str, system_prompt: str = "") -> str:
    """
    Sends a prompt to the local LLM and returns its text response.
    """
    ensure_model_exists(LLM_MODEL)
    
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = ollama.chat(model=LLM_MODEL, messages=messages)
    return response['message']['content']