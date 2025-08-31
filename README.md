# RAG Chatbot — UPI CS Knowledge

A lightweight Retrieval-Augmented Generation (RAG) chatbot whose knowledge is sourced from **[http://cs.upi.edu/v2](http://cs.upi.edu/v2)**.
Backend is **FastAPI** (SSE streaming) with per-IP token-bucket rate limiting. Frontend is **Streamlit**.
For a deeper dive into the RAG pipeline design, **see the reference repo**: [https://github.com/bwbayu/Demo\_RAG\_Chatbot\_UPI](https://github.com/bwbayu/Demo_RAG_Chatbot_UPI).

---

## Overview

This app answers user questions by retrieving relevant content from the UPI CS site and composing responses with an LLM:

1. **Retrieval**

   * **Sparse**: BM25 (pre-fit index shipped in `/model`).
   * **Dense**: Embeddings (see `src/get_embedding.py`).

2. **Augmentation**
   Retrieved passages are assembled into a context window.

3. **Generation (Streaming)**
   FastAPI serves a **Server-Sent Events** (`text/event-stream`) endpoint at `/chat` that streams tokens to the client.

4. **Safety & Ops**
   * **Token bucket per IP** rate limiter to prevent abuse and noisy neighbors.
   * **Global rate limiter with firestore** rate limiter to prevent abuse and noisy neighbors in global scale.
   * Load testing with **Locust** (`src/locust.py`).

---

## Tech Stack

* **Backend**: FastAPI, SSE, Python 3.10+
* **Database**: Firestore
* **Retrieval**: BM25 (sparse), dense embeddings (provider configurable)
* **RAG Orchestration**: `src/rag_pipeline.py`
* **Rate Limiting**: Token bucket per IP
* **Frontend**: Streamlit (`/streamlit/web_chatbot.py`)
* **Containers / Orchestration**: Docker, Docker Compose
* **Perf Testing**: Locust

---

## Repository Layout

```
.
├─ api/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ app.py               # FastAPI app; SSE /chat endpoint; per-IP token bucket
├─ model/
│  └─ bm25_params.json     # Pre-fit BM25 artifacts (sparse index)
├─ src/
│  ├─ bm25_model.py        # Load BM25 model
│  ├─ get_embedding.py     # Dense & sparse embedding utilities
│  ├─ locust.py            # Load testing for /chat
│  └─ rag_pipeline.py      # RAG orchestration
├─ streamlit/
│  ├─ Dockerfile
│  ├─ requirements.txt
│  └─ web_chatbot.py       # Streamlit UI
├─ .env.example            # Environment keys (copy to .env)
├─ docker-compose.yml      # Run FastAPI + Streamlit together
└─ README.md
```

---

## Environment Variables

Copy and edit:

```bash
cp .env.example .env
```

Common keys (check `.env.example` for the authoritative list and exact names):

* **LLM / Embeddings**: e.g., `OPENAI_API_KEY` / model IDs / provider flags
* **RAG**: paths for BM25 index (`model/`), source base URL (`http://cs.upi.edu/v2`)
* **Server**: `API_HOST`, `API_PORT` (FastAPI), CORS origins
* **Rate limiting**: e.g., tokens per minute, refill rate
* **UI**: `API_BASE_URL` for the Streamlit app to reach the API
* **Firestore credentials**: `GOOGLE_APPLICATION_CREDENTIALS` path to json file of keys that you can generate from service account with roles *Firebase Admin*, `GOOGLE_CLOUD_PROJECT` project name in firestore

> Be explicit in production: lock down CORS, set sensible rate limits, and never commit `.env`.

---

## How to Run

### 1) Run locally with Python

**Backend (FastAPI):**

```bash
cd api
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
# Option A: uvicorn directly
uvicorn app:app --host 0.0.0.0 --port 8000
# Option B: if app exposes factory or reload needed, adapt accordingly
```

**Frontend (Streamlit):**

```bash
cd streamlit
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run web_chatbot.py --server.port 8501
```

By default:

* API at `http://localhost:8000`
* UI at `http://localhost:8501` (configure `API_BASE_URL` in `.env` if needed)

---

### 2) Run with Docker Compose

```bash
# From repo root
# 1. UNCOMMENT FOR LOCAL/DOCKER in api/app.py
docker compose up --build
```

Compose will bring up **api** and **streamlit** services, network them together, and pass env vars.

---

## API — `/chat` (SSE)

* **Path**: `/chat`
* **Protocol**: **Server-Sent Events** (`Content-Type: text/event-stream`)
* **Purpose**: streams generated tokens as they’re produced by the LLM
* **Rate limiting**: per-IP token bucket (expect HTTP 429 when exceeded) and global daily limit

---

## RAG Pipeline (Quick Map)

* **`src/bm25_model.py`** — loads the pre-fit BM25 index from `/model`
* **`src/get_embedding.py`** — produces dense and sparse embeddings
* **`src/rag_pipeline.py`** — wires retrieval + augmentation + generation
* **`model/`** — pre-built BM25 artifacts (update/regenerate when the corpus changes)

For a **full pipeline explanation**, **read this repo**:
👉 [https://github.com/bwbayu/Demo\_RAG\_Chatbot\_UPI](https://github.com/bwbayu/Demo_RAG_Chatbot_UPI)

---

## Load Testing

Using **Locust** to stress `/chat`:

```bash
# Ensure locust is installed per requirements; otherwise:
pip install locust

# From repo root (or adjust path)
locust -f src/locust.py --headless -u 50 -r 5 -H http://localhost:8000
```