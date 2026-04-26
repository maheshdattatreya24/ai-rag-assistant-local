import streamlit as st
import os

from src.loader import load_documents
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vectorstore import create_vectorstore
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI RAG Assistant", layout="wide")

st.title("🤖 AI RAG Assistant")
st.caption("Local AI • Fast • Private")

# --------------------------
# SIDEBAR
# --------------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.write("LLM: qwen2.5:1.5b")
    st.write("Embeddings: nomic-embed-text")

    if st.button("🧹 Clear Chat"):
        st.session_state.chat_history = []

# --------------------------
# MULTI FILE UPLOAD
# --------------------------
uploaded_files = st.file_uploader(
    "📄 Upload PDF/TXT",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if uploaded_files:
    os.makedirs("uploads", exist_ok=True)

    for file in uploaded_files:
        with open(os.path.join("uploads", file.name), "wb") as f:
            f.write(file.read())

    st.success("✅ Files uploaded successfully!")

    documents = load_documents("uploads")
    chunks = split_documents(documents)
    embeddings = get_embeddings()

    vectorstore = create_vectorstore(chunks, embeddings)

    pipeline = RAGPipeline()
    pipeline.vectorstore = vectorstore

    st.session_state.pipeline = pipeline

    st.success(f"✅ Ready! {len(chunks)} chunks indexed")

# --------------------------
# CHAT MEMORY
# --------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

query = st.chat_input("Ask something from your document...")

if query:
    pipeline = st.session_state.get("pipeline")

    if pipeline is None:
        st.warning("⚠️ Upload document first")
    else:
        history_text = "\n".join(
            [f"{r}: {m}" for r, m in st.session_state.chat_history[-5:]]
        )

        # STREAMING EFFECT
        with st.chat_message("assistant"):
            placeholder = st.empty()
            full_response = ""

            response = pipeline.ask(query, history_text)

            for word in response.split():
                full_response += word + " "
                placeholder.markdown(full_response)

        st.session_state.chat_history.append(("User", query))
        st.session_state.chat_history.append(("Bot", response))

# --------------------------
# DISPLAY CHAT
# --------------------------
for role, msg in st.session_state.chat_history:
    with st.chat_message("user" if role == "User" else "assistant"):
        if "📚 Sources:" in msg:
            answer, sources = msg.split("📚 Sources:")
            st.write(answer)
            st.markdown("### 📚 Sources")
            st.code(sources)
        else:
            st.write(msg)