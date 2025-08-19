# app.py
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi.middleware import SlowAPIMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import json
from utils.rag_pipeline import RAG_pipeline

limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])
app = FastAPI(title="RAG Chatbot API", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, lambda r, e: JSONResponse(status_code=429, content={"detail": "rate limit"}))
app.add_middleware(SlowAPIMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # change in prod
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
@limiter.limit("20/minute")
async def chat(request: Request):
    body = await request.json()
    query = body.get("message")
    history = body.get("history", [])
    if not query or not isinstance(query, str):
        return JSONResponse(status_code=400, content={"detail": "message is required"})

    def token_iter():
        try:
            for token in RAG_pipeline(query, history):
                chunk = getattr(token, "content", None)
                if chunk:
                    yield f"data: {chunk}\n\n"
        except Exception as e:
            yield f"event: error\n" f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            yield "event: done\n" "data: {}\n\n"

    return StreamingResponse(token_iter(), media_type="text/event-stream")