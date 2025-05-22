''' OpenAI embeddings '''
import aiohttp
import asyncio
import json
import structlog

from .embeddings import Embeddings

logger = structlog.get_logger()

async def openai_embeddings(config: Embeddings, chunks:list) -> list:
    """Retrieve embeddings using OpenAI."""
    task_payload = {"model": config.model, "input": chunks}
    async with aiohttp.ClientSession(headers=config.headers) as session:
        async with session.post(
            config.endpoint,
            json=task_payload,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as response:
            if response.status != 200:
                error_details = await response.text()
                logger.error("OpenAI embedding request failed", status=response.status, error_details=error_details)
                raise Exception(f"EmbeddingRequestFailed: status={response.status}, details={error_details}")
            obj = await response.json()
            return [entry["embedding"] for entry in obj["data"]]
