# src/get_embedding.py
import os
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from pinecone_text.sparse import BM25Encoder
import httpx

load_dotenv()

SILICONFLOW_URL_EMBEDDING = os.getenv("SILICONFLOW_URL_EMBEDDING")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY")
EMBED_DIM = int(os.getenv("EMBED_DIM")) if os.getenv("EMBED_DIM") else 1024

if not SILICONFLOW_URL_EMBEDDING:
    raise RuntimeError("SILICONFLOW_URL_EMBEDDING is not set")
if not SILICONFLOW_API_KEY:
    raise RuntimeError("SILICONFLOW_API_KEY is not set")

headers = {
    "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
    "Content-Type": "application/json",
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10), reraise=True)
async def get_dense_embeddings(text: str, dim_size: int = EMBED_DIM) -> list[float]:
    dim = dim_size or EMBED_DIM or 1024
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B",
        "input": text,
        "encoding_format": "float",
        "dimensions": dim,
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.post(
            SILICONFLOW_URL_EMBEDDING,
            json=payload,
            headers=headers,
            timeout=20,
        )
        
    response.raise_for_status()
    data = response.json()
    if "data" in data and data["data"]:
        return data["data"][0]["embedding"]
    raise ValueError("Invalid embedding response: missing 'data' or empty list")

def get_sparse_embeddings(text: str, bm25_model: BM25Encoder, query_type: str = "search"):
    if query_type == "upsert":
        return bm25_model.encode_documents(text)
    return bm25_model.encode_queries(text)
