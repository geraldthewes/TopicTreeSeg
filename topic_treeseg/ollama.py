''' Ollama Embeddings '''

import asyncio
from ollama import AsyncClient

from .embeddings import Embeddings

_global_ollama_client = None

def get_ollama_async_client_instance(embeddings_endpoint):
    global _global_ollama_client
    if _global_ollama_client is None:
        _global_ollama_client = AsyncClient(host=embeddings_endpoint)
    return _global_ollama_client


async def ollama_embeddings(config: Embeddings, chunks: list) ->  list:
    """Retrieve embeddings using Ollama."""
    ollama_client = get_ollama_async_client_instance(config.endpoint)
    response = await ollama_client.embed(model=config.model, input=chunks)
    return response['embeddings']
