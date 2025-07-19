import streamlit as st
from datetime import datetime
import logging
import time
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated
from langchain_core.documents import Document
from main import (
    define_workflow,
    create_and_import_db
)

namespace = "laptop-index"
# Initialize Pinecone and vector store
vector_store = create_and_import_db(namespace)

# --- Page Config ---
st.set_page_config(
    page_title="💻 The Gioi Di Dong Recommender Chatbot",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CSS ---


def apply_modern_minimalist_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 100%); color: #e8e8e8; }
    .chat-header h1 { font-size: 2.25rem; font-weight: 600; color: #fff; }
    .chat-header p { font-size: 1.1rem; color: #a0a0a0; }
    .welcome-assistant { background: rgba(255,255,255,0.05); padding: 1.5rem; border-radius: 20px; }
    .stChatMessage[data-testid="chat-message-user"] { background: #1d4ed8; margin-left: 15%; border-radius: 20px; }
    .stChatMessage[data-testid="chat-message-assistant"] { background: rgba(255,255,255,0.05); margin-right: 15%; border-radius: 20px; }
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = []


class ChatState(TypedDict):
    conversation: Annotated[list, add_messages]
    retrieved_docs: list[Document]
    user_query: str
    last_filter: dict
    current_step: str


def handle_user_input(user_input: str, chat_app):
    if not user_input.strip():
        return
    st.session_state.conversation.append(HumanMessage(content=user_input))

    try:
        state: ChatState = {
            "conversation": st.session_state.conversation,
            "retrieved_docs": [],
            "user_query": "",
            "last_filter": {},
            "current_step": "extract_filter"
        }
        state = chat_app.invoke(state)
        st.session_state.conversation = state["conversation"]
    except Exception as e:
        st.session_state.conversation.append(
            AIMessage(content="⚠️ Sorry, something went wrong.")
        )


def display_chat_interface(chat_app):
    st.markdown("""
        <div class="chat-header">
            <h1>💻 The Gioi Di Dong</h1>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.conversation:
        st.markdown("""
            <div class="welcome-assistant">
                👋 Ask me anything about **Laptop** or **Cellphone** recommendations!<br>
                For example:
                <ul>
                    <li>What's the best laptop under $1000 with 16GB RAM?</li>
                    <li>I want a phone with good battery and camera, any suggestions?</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    for msg in st.session_state.conversation:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)

    user_input = st.chat_input("Type your anything needs...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        handle_user_input(user_input, chat_app)
        st.rerun()
# --- Main ---


def main():
    initialize_session_state()
    apply_modern_minimalist_css()
    chat_app = define_workflow()
    display_chat_interface(chat_app)


if __name__ == "__main__":
    main()
