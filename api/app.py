# api/app.py
import asyncio
import json
import time
import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from src.rag_pipeline import RAG_pipeline_async
import math
from dataclasses import dataclass

from google.oauth2 import service_account # local/docker
from google.cloud import firestore
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

app = FastAPI(title="RAG Chatbot UPI")
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
GATE = asyncio.Semaphore(MAX_CONCURRENT) # set semaphore with 32 max concurrent

RATE_PER_HOUR = 3.0  # fill rate per hour
RATE_PER_SECOND = RATE_PER_HOUR/3600.0
BURST_PER_IP = 10.0  # maximum capacity of bucket
TZ_NAME = "UTC"
DAILY_LIMIT = 50
RETENTION_DAYS = 35

# # ------------- local only (docker)
# creds = service_account.Credentials.from_service_account_file(
#     os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
# )
# _db = firestore.Client(project=os.environ["GOOGLE_CLOUD_PROJECT"], credentials=creds)

# ------------- prod cloud-run (service account setting on security cloud run)
_db = firestore.Client()

# calculate timer for reset global daily limit
def _seconds_until_midnight(tz_name: str = TZ_NAME) -> int:
    tz = ZoneInfo(tz_name)
    now = datetime.now(tz)
    tomorrow = now.date() + timedelta(days=1)
    reset = datetime.combine(tomorrow, datetime.min.time(), tzinfo=tz)
    return int((reset - now).total_seconds())

# get timer for clean up data in firestore
def _ttl_timestamp(days: int = RETENTION_DAYS):
    return datetime.now(timezone.utc) + timedelta(days=days)

# check global daily limit
def _allow_daily_global_firestore_sync() -> tuple[bool, int, int]:
    # get timezone and today format
    tz = ZoneInfo(TZ_NAME)
    today_id = datetime.now(tz).date().isoformat()
    # create firestore collection
    doc = _db.collection("rate_global").document(today_id)

    @firestore.transactional
    def _tx(tx: firestore.Transaction):
        snap = doc.get(transaction=tx)
        # get data
        count = int(snap.get("count") or 0) if snap.exists else 0

        # check data (reach max)
        if count >= DAILY_LIMIT:
            return False, 0
        
        # create or update data
        if snap.exists:
            tx.update(doc, {"count": firestore.Increment(1)})
        else:
            tx.set(doc, {"count": 1, "ttl": _ttl_timestamp()})
        
        # check remaining
        remaining = max(0, DAILY_LIMIT - (count + 1))
        return True, remaining

    # update and get data
    allowed, remaining = _tx(_db.transaction())
    # calculate remaining timer until reset
    retry_after = _seconds_until_midnight()
    return allowed, remaining, retry_after

# move rate limit checking to different thread pool
async def allow_daily_global_firestore() -> tuple[bool, int, int]:
    return await asyncio.to_thread(_allow_daily_global_firestore_sync)

@dataclass
class _Bucket:
    # store number of token and last time check bucket
    tokens: float
    last: float

# store bucket per IP
_buckets: dict[str, _Bucket] = {}

# token bucket rate limit per IP
def allow(ip: str) -> tuple[bool, float | None]:
    now = time.monotonic()
    # check bucket for that ip
    b = _buckets.get(ip)
    # create bucket if needed
    if b is None:
        # create bucket class contain number of token and last check
        b = _Bucket(tokens=BURST_PER_IP, last=now)
        # store the bucket of corresponding IP
        _buckets[ip] = b

    # get total time passed (elapsed) between now-last check of that IP
    elapsed = max(0.0, now - b.last)

    # calculate total token based on total time passed
    total_token = elapsed * RATE_PER_SECOND
    # store number of token user uses but cannot exceed capacity/burst
    b.tokens = min(BURST_PER_IP, b.tokens + total_token)
    b.last = now

    # reduce token by 1 because user request
    if b.tokens >= 1.0:
        b.tokens -= 1.0
        return True, None
    
    # calculate retry after if number of token in the bucket not enough for new request
    need = 1.0 - b.tokens
    retry_after = need / RATE_PER_SECOND if RATE_PER_SECOND > 0 else 1.0
    return False, max(0.0, retry_after)

@app.post("/chat")
async def chat(request: Request):
    # get request
    body = await request.json()
    query = body.get("message")
    history = body.get("history", [])

    # check query availability
    if not query or not isinstance(query, str):
        return JSONResponse(status_code=400, content={"detail": "message is required"})

    # global daily rate limit
    allowed_daily, remaining, retry_after = await allow_daily_global_firestore()
    if not allowed_daily:
        return JSONResponse(
            status_code=429,
            content={"detail": "Daily cap reached"},
            headers={
                "Retry-After": str(int(math.ceil(retry_after))),
                "X-RateLimit-Limit-Day": str(DAILY_LIMIT),
                "X-RateLimit-Remaining-Day": str(remaining),
                "X-RateLimit-Reset-Seconds": str(int(retry_after)),
            },
        )

    # rate-limit per IP
    ip = request.client.host if request.client else "unknown"
    # check rate limit
    allowed, retry_after = allow(ip)
    # show error message because rate limit exceed
    if not allowed:
        return JSONResponse(
            status_code=429,
            content={"detail": "Too Many Requests"},
            headers={
                "Retry-After": str(int(math.ceil(retry_after or 1)))}
        )

    # 
    async with GATE:
        async def agen():
            # create queue to store the chunk response data
            q: asyncio.Queue[str | None] = asyncio.Queue()
            
            # get chunk data
            async def producer():
                try:
                    # get stream data from pipeline
                    stream = await RAG_pipeline_async(query, history, streaming=True)
                    
                    # iterate stream data
                    async for token in stream:
                        # get text data from "content"
                        chunk = getattr(token, "content", None)
                        if chunk:
                            # put data in queue if chunk is available
                            await q.put("data: " + json.dumps({"content": chunk}) + "\n\n")
                except Exception as e:
                    await q.put("event: error\n" + "data: " + json.dumps({"error": str(e)}) + "\n\n")
                finally:
                    await q.put("event: done\n" + "data: {}\n\n")
                    await q.put(None)

            # just ping response from server to client
            async def heartbeat():
                try:
                    while True:
                        await asyncio.sleep(20)
                        await q.put(": ping\n\n")
                except asyncio.CancelledError:
                    pass
            
            # create task
            prod_task = asyncio.create_task(producer())
            hb_task = asyncio.create_task(heartbeat())
            
            # return chunk data while in streaming
            try:
                while True:
                    # get new chunk data
                    item = await q.get()
                    if item is None:
                        break
                    # return data streaming
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
                "X-Accel-Buffering": "no",
            },
        )