import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from search import RAG_pipeline
from PIL import Image
from datetime import datetime

# max history chat (5 bot, 5 human)
MAX_TURNS = 10
im = Image.open("assets/logo.png")
# init streamlit app
st.set_page_config(page_title="Chatbot CSE UPI", page_icon=im)
st.header("Mari Tanyakan Berbagai Hal Terkait Departemen Pendidikan Ilmu Komputer")
st.markdown("<small>Jawaban dihasilkan otomatis dan mungkin tidak selalu benar. Verifikasi ke sumber resmi https://cs.upi.edu/.</small>", unsafe_allow_html=True)

# initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# display chat messages from history on app rerun
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

def disable_chat_input():
    st.session_state.processing = True

# input chat
user_question = st.chat_input(
    "Tanyakan apa saja tentang Ilmu Komputer UPI",
    disabled=st.session_state.processing,
    on_submit=disable_chat_input
    )

if user_question:
    # get chat history (if available)
    chat_history = st.session_state.messages
    # add user message to chat history and display it
    with st.chat_message("user"):
        st.markdown(user_question)

        st.session_state.messages.append(HumanMessage(user_question))
        # cut history chat
        if len(st.session_state.messages) > MAX_TURNS:
            st.session_state.messages = st.session_state.messages[-MAX_TURNS:]

    with st.chat_message("assistant"):
        response_container = st.empty()
        full_response = ""
        
        # Call RAG_pipeline and get the streaming response
        stream = RAG_pipeline(query=user_question, chat_history=chat_history)
        for chunk in stream:
            delta = getattr(chunk, "content", None)
            if not delta:
                continue
            full_response += delta
            response_container.markdown(full_response)

        print("finish : ", datetime.now())
        st.session_state.messages.append(AIMessage(full_response))
        # cut history chat
        if len(st.session_state.messages) > MAX_TURNS:
            st.session_state.messages = st.session_state.messages[-MAX_TURNS:]
        
    st.session_state.processing = False
    st.rerun()