import streamlit as st
from datetime import datetime
import logging
import time
from main import extract_filter_from_query, retrieve, question_answering, PineconeVectorStore, pc, embeddingModel, create_and_import_db  # Adjust as needed

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

# --- Init State ---


def initialize_session_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# --- Handle User Input ---


def handle_user_input(user_input, vector_store):
    if not user_input.strip():
        return

    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input,
        "timestamp": datetime.now()
    })

    with st.spinner("Thinking..."):
        try:
            filter_query = extract_filter_from_query(user_input)
            results = retrieve(vector_store, top_k=3, filter=filter_query)
            ai_response = question_answering(user_input, results)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_response,
                "timestamp": datetime.now()
            })
        except Exception as e:
            logger.error(str(e))
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "⚠️ Sorry, something went wrong while processing your request.",
                "timestamp": datetime.now()
            })

# --- Display Interface ---


def display_chat_interface():
    st.markdown("""
        <div class="chat-header">
            <h1>💻 The Gioi Di Dong Recommender</h1>
        </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
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

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="👨‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            st.caption(f"*{msg['timestamp'].strftime('%H:%M')}*")

    user_input = st.chat_input("Type your anything needs...")
    if user_input:
        with st.chat_message("user", avatar="👨‍💻"):
            st.markdown(user_input)
        handle_user_input(user_input, vector_store)
        st.rerun()

# --- Main ---


def main():
    initialize_session_state()
    apply_modern_minimalist_css()

    namespace = "laptop-index"
    vector_store = create_and_import_db(namespace)

    display_chat_interface()


if __name__ == "__main__":
    main()
