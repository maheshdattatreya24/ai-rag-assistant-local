# 🤖 AI RAG Assistant (Local LLM-Based Document QA System)

A complete **Retrieval-Augmented Generation (RAG)** system that enables users to upload documents (PDF/TXT) and ask questions with accurate, context-aware responses using a **local Large Language Model (LLM)**.

This project demonstrates how modern AI systems reduce hallucination by combining **information retrieval** with **text generation**.

---

## 📌 Overview

Traditional LLMs generate answers based only on their training data, which can lead to outdated or incorrect responses (hallucination).

This project solves that problem using **RAG (Retrieval-Augmented Generation)**:
- Retrieves relevant information from documents
- Uses that context to generate grounded answers

---

## 🚀 Key Features

- 📄 Upload and query PDF/TXT documents
- 🔍 Semantic search using embeddings
- 🧠 Context-aware answer generation
- 📚 Source attribution for transparency
- 💬 Chat memory support
- ⚡ Fully local (no API cost, privacy-friendly)
- 🖥 Interactive UI using Streamlit

---

## 🧠 System Architecture


User Query
↓
Embedding (nomic-embed-text)
↓
Vector Search (ChromaDB)
↓
Relevant Document Chunks
↓
Prompt Construction
↓
LLM (qwen2.5:1.5b via Ollama)
↓
Final Answer + Sources


---

## ⚙️ How It Works (Step-by-Step)

### 1. Document Loading
- User uploads PDF or TXT files
- Files are read and converted into text documents

---

### 2. Text Splitting
- Large documents are split into smaller chunks
- Helps improve retrieval accuracy and LLM efficiency

---

### 3. Embedding Generation
- Each chunk is converted into vector embeddings
- Uses **Ollama embedding model: `nomic-embed-text`**

---

### 4. Vector Database (ChromaDB)
- Stores embeddings for all document chunks
- Enables fast similarity-based retrieval

---

### 5. Query Processing
When a user asks a question:
- Query is converted into embedding
- Top relevant chunks are retrieved using similarity search

---

### 6. Response Generation
- Retrieved chunks are passed as context
- LLM (`qwen2.5:1.5b`) generates answer
- Model is instructed to:
  - Use only provided context
  - Avoid hallucination

---

### 7. Output Enhancement
- Answer is displayed in UI
- Source snippets are shown
- Chat history is maintained for context

---

## 🛠 Tech Stack

| Component        | Technology |
|----------------|-----------|
| Programming     | Python |
| Framework       | LangChain |
| Vector DB       | ChromaDB |
| LLM             | Ollama (qwen2.5:1.5b) |
| Embeddings      | nomic-embed-text |
| UI              | Streamlit |

---

## 📂 Project Structure


ai-rag-assistant-local/
│
├── app.py # Streamlit UI application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── src/
│ ├── rag_pipeline.py # Core RAG pipeline (retrieval + generation)
│ ├── loader.py # Loads PDF/TXT documents
│ ├── splitter.py # Splits documents into chunks
│ ├── embeddings.py # Embedding model setup
│ └── vectorstore.py # ChromaDB vector database logic
│
├── data/ # Optional sample documents
│
└── .gitignore # Ignored files (venv, db, uploads, etc.)


---

## ▶️ Installation & Setup

### 1. Clone Repository

```bash
git clone https://github.com/your-username/ai-rag-assistant-local.git
cd ai-rag-assistant-local
