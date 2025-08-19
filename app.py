# app.py
import asyncio
import json
import time
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from utils.rag_pipeline import RAG_pipeline_async
import math
from dataclasses import dataclass

app = FastAPI(title="RAG Chatbot API", version="2.0.0-async")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: restrict in production
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

MAX_CONCURRENT = int(32)
GATE = asyncio.Semaphore(MAX_CONCURRENT)

WINDOW_SEC = 1.0
RATE_PER_IP = 20.0 
BURST_PER_IP = 30.0 

@dataclass
class _Bucket:
    tokens: float
    last: float

_buckets: dict[str, _Bucket] = {}

def allow(ip: str) -> tuple[bool, float | None]:
    now = time.time()
    b = _buckets.get(ip)
    if b is None:
        b = _Bucket(tokens=BURST_PER_IP, last=now)
        _buckets[ip] = b
    elapsed = max(0.0, now - b.last)
    b.tokens = min(BURST_PER_IP, b.tokens + elapsed * RATE_PER_IP)
    b.last = now
    if b.tokens >= 1.0:
        b.tokens -= 1.0
        return True, None
    need = 1.0 - b.tokens
    retry_after = need / RATE_PER_IP if RATE_PER_IP > 0 else 1.0
    return False, max(0.0, retry_after)

@app.post("/chat")
async def chat(request: Request):
    body = await request.json()
    query = body.get("message")
    history = body.get("history", [])
    if not query or not isinstance(query, str):
        return JSONResponse(status_code=400, content={"detail": "message is required"})

    # rate-limit per IP
    ip = request.client.host if request.client else "unknown"
    allowed, retry_after = allow(ip)
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too Many Requests"},
            headers={"Retry-After": str(int(math.ceil(retry_after or 1)))}
        )

    async with GATE:  # global concurrency cap
        async def agen():
            q: asyncio.Queue[str | None] = asyncio.Queue()

            async def producer():
                try:
                    stream = await RAG_pipeline_async(query, history, streaming=True)
                    async for token in stream:
                        chunk = getattr(token, "content", None)
                        if chunk:
                            await q.put("data: " + json.dumps({"content": chunk}) + "\n\n")
                except Exception as e:
                    await q.put("event: error\n" + "data: " + json.dumps({"error": str(e)}) + "\n\n")
                finally:
                    await q.put("event: done\n" + "data: {}\n\n")
                    await q.put(None)

            async def heartbeat():
                try:
                    while True:
                        await asyncio.sleep(20)
                        await q.put(": ping\n\n")  # SSE comment line
                except asyncio.CancelledError:
                    pass

            prod_task = asyncio.create_task(producer())
            hb_task = asyncio.create_task(heartbeat())
            try:
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    yield item
            finally:
                hb_task.cancel()
                prod_task.cancel()

        return StreamingResponse(
            agen(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # important behind nginx
            },
        )