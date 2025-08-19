# app.py
import os
import json
import time
import threading
from typing import List, Literal, Optional, Tuple, Iterable, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, PlainTextResponse

from pydantic import BaseModel
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# --- Import your pipeline ---
# IMPORTANT: If your file is located at ./rag_pipeline.py, use this import:
from rag_pipeline import RAG_pipeline
# If your project layout uses a package like "utils.rag_pipeline", switch to:
# from utils.rag_pipeline import RAG_pipeline

# ---------------- Basic Config ----------------
API_TITLE = os.getenv("API_TITLE", "RAG Chatbot API")
API_VERSION = os.getenv("API_VERSION", "1.0.0")
ALLOW_ORIGINS = [o.strip() for o in os.getenv("ALLOW_ORIGINS", "*").split(",")]
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "4000"))
MAX_CONCURRENT_STREAMS_PER_USER = int(os.getenv("MAX_CONCURRENT_STREAMS_PER_USER", "2"))
PER_USER_RATE = os.getenv("PER_USER_RATE", "60/minute")  # SlowAPI format

# ---------------- App & CORS ----------------
def rl_key(request: Request) -> str:
    # Prefer stable user id or bearer, then fallback to IP
    return (
        request.headers.get("x-user-id")
        or request.headers.get("authorization")
        or get_remote_address(request)
    )

limiter = Limiter(key_func=rl_key, default_limits=[PER_USER_RATE])

app = FastAPI(title=API_TITLE, version=API_VERSION)
app.state.limiter = limiter

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS if ALLOW_ORIGINS != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Retry-After"],
)

# -------------- Models -----------------
class ChatInput(BaseModel):
    message: str
    history: List[Tuple[Literal["user","assistant"], str]] = []
    user_id: Optional[str] = None

# -------------- In-memory concurrency guard (per-process) -----------------
# Simple, deployable without Redis. For multiple workers/pods, limits apply per worker.
_sem_lock = threading.Lock()
_active_counts = {}  # {user_key: count}

def _acquire_user_slot(user_key: str, max_slots: int) -> bool:
    with _sem_lock:
        count = _active_counts.get(user_key, 0)
        if count >= max_slots:
            return False
        _active_counts[user_key] = count + 1
        return True

def _release_user_slot(user_key: str) -> None:
    with _sem_lock:
        if user_key in _active_counts:
            _active_counts[user_key] -= 1
            if _active_counts[user_key] <= 0:
                _active_counts.pop(user_key, None)

# -------------- Error handlers -----------------
@app.exception_handler(RateLimitExceeded)
def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    # Basic, per-process rate limiting. Add Retry-After so clients can back off.
    return JSONResponse(
        status_code=429,
        content={"error": "rate_limited", "detail": "Too many requests. Please retry later."},
        headers={"Retry-After": "30", "Cache-Control": "no-store"},
    )

@app.get("/health", response_class=PlainTextResponse)
def health():
    return "ok"

# -------------- Helpers -----------------
def _extract_chunk(token: Any) -> Optional[str]:
    # RAG_pipeline(..., streaming=True) may yield LangChain chunks (with .content),
    # dicts with "content", plain strings, etc.
    try:
        if token is None:
            return None
        if isinstance(token, str):
            return token
        # LangChain MessageChunk-like:
        content = getattr(token, "content", None)
        if content is not None:
            return str(content)
        # Fallback to dict with content
        if isinstance(token, dict) and "content" in token:
            return str(token["content"])
        # Last resort
        return str(token)
    except Exception:
        return None

# ----------------- /chat (SYNC) -----------------
@app.post("/chat")
@limiter.limit(PER_USER_RATE)  # per-user key via rl_key()
def chat(payload: ChatInput, request: Request):
    # Validate input size
    if not payload.message or not isinstance(payload.message, str):
        raise HTTPException(status_code=400, detail="message is required")
    if len(payload.message) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=413, detail=f"message too long (>{MAX_INPUT_CHARS} chars)")
    # Normalize history format just to be safe
    history: List[Tuple[str, str]] = []
    for turn in payload.history or []:
        try:
            role, text = turn
            if role not in ("user", "assistant"):
                continue
            history.append((role, str(text)))
        except Exception:
            continue

    # per-user concurrency gating
    user_key = payload.user_id or request.headers.get("x-user-id") or rl_key(request)
    if not _acquire_user_slot(user_key, MAX_CONCURRENT_STREAMS_PER_USER):
        return JSONResponse(
            status_code=429,
            content={
                "error": "too_many_streams",
                "detail": f"Too many concurrent chats for this user (max {MAX_CONCURRENT_STREAMS_PER_USER})."
            },
            headers={"Retry-After": "5"},
        )

    def token_iter() -> Iterable[str]:
        # Initial event (optional but helpful to clients)
        yield "event: start\ndata: {}\n\n"
        try:
            # Call your existing RAG pipeline in streaming mode
            stream = RAG_pipeline(payload.message, history, streaming=True)
            for token in stream:
                chunk = _extract_chunk(token)
                if chunk:
                    # SSE: each message must end with a blank line
                    yield f"data: {chunk}\n\n"
            # Normal end
            yield "event: end\ndata: {}\n\n"
        except Exception as e:
            # Send a structured error event down the SSE stream
            err = {"error": "server_error", "detail": str(e)}
            yield "event: error\n" + f"data: {json.dumps(err)}\n\n"
        finally:
            # Always release the slot
            _release_user_slot(user_key)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "Connection": "keep-alive",
        "Content-Type": "text/event-stream",
    }
    return StreamingResponse(token_iter(), headers=headers, media_type="text/event-stream")


if __name__ == "__main__":
    # Run with: uvicorn app:app --host 0.0.0.0 --port 8000
    # For basic production: gunicorn -k uvicorn.workers.UvicornWorker -w 2 app:app
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=False)