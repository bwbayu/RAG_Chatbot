# streamlit_app.py
import os
import requests
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from PIL import Image
import json
import time

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
CHAT_URL = f"{API_BASE}/chat"
TIMEOUT = 90
MAX_TURNS = 10

im_path = "assets/logo.png"
if os.path.exists(im_path):
    im = Image.open(im_path)
    st.set_page_config(page_title="Chatbot CSE UPI", page_icon=im)
else:
    st.set_page_config(page_title="Chatbot CSE UPI")

st.header("Mari Tanyakan Berbagai Hal Terkait Departemen Pendidikan Ilmu Komputer")
st.markdown("<small>Jawaban dihasilkan otomatis dan mungkin tidak selalu benar. Verifikasi ke sumber resmi https://cs.upi.edu/.</small>", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "processing" not in st.session_state:
    st.session_state.processing = False

def to_history_payload(messages):
    payload = []
    for m in messages:
        if isinstance(m, HumanMessage):
            payload.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            payload.append({"role": "assistant", "content": m.content})
    return payload

def disable_chat_input():
    st.session_state.processing = True

def stream_chat(message_text: str, history_messages):
    history_payload = to_history_payload(history_messages)
    with requests.post(
        CHAT_URL,
        json={"message": message_text, "history": history_payload},
        stream=True,
        timeout=TIMEOUT,
    ) as r:
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            msg = "Server sedang ramai. "
            if retry_after:
                msg += f"Coba lagi dalam ~{retry_after} detik."
            else:
                msg += "Coba lagi sebentar lagi."
            raise requests.HTTPError(msg, response=r)

        r.raise_for_status()
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            if raw.startswith(":"):
                continue
            if raw.startswith("event:"):
                ev = raw[6:].strip()
                if ev == "done":
                    break
                if ev == "error":
                    continue
                continue
            if raw.startswith("data: "):
                data_str = raw[6:]
                try:
                    payload = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                piece = payload.get("content", "")
                if piece:
                    yield piece
        return


for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

user_question = st.chat_input(
    "Tanyakan apa saja tentang Ilmu Komputer UPI",
    disabled=st.session_state.processing,
    on_submit=disable_chat_input,
)

if user_question:
    with st.chat_message("user"):
        st.markdown(user_question)
    st.session_state.messages.append(HumanMessage(user_question))
    if len(st.session_state.messages) > MAX_TURNS:
        st.session_state.messages = st.session_state.messages[-MAX_TURNS:]

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        try:
            for token in stream_chat(user_question, st.session_state.messages):
                full_response += token or ""
                response_container.markdown(full_response)
        except requests.HTTPError as http_err:
            status = getattr(http_err.response, "status_code", None)
            if status == 429:
                retry_after = None
                if http_err.response is not None:
                    retry_after = http_err.response.headers.get("Retry-After")
                    try:
                        data = http_err.response.json()
                        retry_after = data.get("retry_after", retry_after)
                    except Exception:
                        pass
                msg = "Maaf, server sedang ramai. "
                if retry_after:
                    msg += f"Coba lagi dalam ~{retry_after} detik."
                else:
                    msg += "Coba lagi sebentar lagi."
                full_response = msg
            else:
                full_response = f"Maaf, server error: {http_err}"
            response_container.markdown(full_response)
        except requests.RequestException as req_err:
            full_response = f"Maaf, koneksi bermasalah: {req_err}"
            response_container.markdown(full_response)
        except Exception as e:
            full_response = f"Terjadi error: {e}"
            response_container.markdown(full_response)

    st.session_state.messages.append(AIMessage(full_response))
    if len(st.session_state.messages) > MAX_TURNS:
        st.session_state.messages = st.session_state.messages[-MAX_TURNS:]

    st.session_state.processing = False
    st.rerun()
