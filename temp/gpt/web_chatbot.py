# web_chatbot.py
import os
import uuid
import json
import time
import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
CHAT_URL = f"{API_BASE}/chat"
TIMEOUT = int(os.getenv("TIMEOUT", "180"))
MAX_TURNS = int(os.getenv("MAX_TURNS", "12"))

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")

# Stable per-user id for rate limiting (no auth yet)
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict]: {"role": "user"|"assistant", "content": str}

# Render history
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

prompt = st.chat_input("Type your question...")

def to_pairs(msgs):
    return [(m["role"], m["content"]) for m in msgs]

def sse_stream_request(message: str, history_pairs):
    headers = {
        "Accept": "text/event-stream",
        "Content-Type": "application/json",
        "x-user-id": st.session_state.user_id,
    }
    payload = {"message": message, "history": history_pairs, "user_id": st.session_state.user_id}
    resp = requests.post(CHAT_URL, headers=headers, data=json.dumps(payload), stream=True, timeout=TIMEOUT)

    # Immediate HTTP layer errors (including 429) handled here
    if resp.status_code == 429:
        retry_after = int(resp.headers.get("Retry-After", "15"))
        detail = None
        try:
            detail = resp.json().get("detail")
        except Exception:
            pass
        raise RuntimeError(f"Rate limited. Retry in {retry_after}s. {detail or ''}".strip())

    if resp.status_code >= 400:
        try:
            body = resp.json()
        except Exception:
            body = {"detail": resp.text}
        raise RuntimeError(f"HTTP {resp.status_code}: {body.get('detail', 'Unknown error.')}")

    return resp

if prompt:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Placeholder for the assistant response
    with st.chat_message("assistant"):
        response_container = st.empty()

    full_response = ""
    try:
        history_pairs = to_pairs(st.session_state.messages[:-1])[-MAX_TURNS:]
        resp = sse_stream_request(prompt, history_pairs)
        # Stream the SSE
        for raw_line in resp.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            # SSE format: "event: <name>" or "data: <payload>"
            if raw_line.startswith("event:"):
                current_event = raw_line.split("event:", 1)[1].strip()
                if current_event == "error":
                    # Next "data:" should contain JSON with the error
                    # We'll just display a generic error to the user and break
                    response_container.markdown(full_response or "_An error occurred._")
                    break
                # Ignore "start" and "end" here; we stop on "end" when we see it later
                continue
            if raw_line.startswith("data:"):
                payload = raw_line[5:].lstrip()
                # There may be end-of-stream signals; we keep it simple
                if payload == "{}":
                    # Might be part of start/end events; ignore
                    continue
                # Append token to the on-screen message
                full_response += payload
                response_container.markdown(full_response)
    except Exception as e:
        # Show a clean error to the user
        full_response = f"⚠️ {e}"
        response_container.markdown(full_response)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Trim history to last MAX_TURNS*2 messages (user+assistant)
    if len(st.session_state.messages) > MAX_TURNS * 2:
        st.session_state.messages = st.session_state.messages[-MAX_TURNS*2:]

    st.rerun()