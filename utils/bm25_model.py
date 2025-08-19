# utils/bm25_model.py
from pathlib import Path
from pinecone_text.sparse import BM25Encoder

BASE_DIR = Path(__file__).resolve().parents[1]
BM25_PATH = BASE_DIR / "model" / "bm25_params.json"

def load_bm25_model(strict: bool = False) -> BM25Encoder:
    bm25 = BM25Encoder(stem=False)

    if not BM25_PATH.exists():
        if strict:
            raise FileNotFoundError(f"BM25 params not found: {BM25_PATH}")
        return bm25

    try:
        bm25.load(str(BM25_PATH))
        return bm25
    except Exception as e:
        if strict:
            raise RuntimeError(f"Failed to load BM25 params at {BM25_PATH}") from e
        return bm25
