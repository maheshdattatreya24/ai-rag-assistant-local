from src.loader import load_documents
from src.splitter import split_documents
from src.vectorstore import create_vectorstore, load_vectorstore
from src.embeddings import get_embeddings

from langchain_community.llms import Ollama


class RAGPipeline:
    def __init__(self):
        print("🚀 Initializing RAG Pipeline...")

        self.embedding = get_embeddings()
        self.llm = Ollama(model="qwen2.5:1.5b", temperature=0)

        self.vectorstore = self._setup_db()

    def _setup_db(self):
        db = load_vectorstore(self.embedding)

        if db:
            print("✅ Loaded existing DB")
            return db

        print("📄 Creating new DB...")

        documents = load_documents("data")

        if not documents:
            raise ValueError("❌ No documents found")

        chunks = split_documents(documents)

        db = create_vectorstore(chunks, self.embedding)

        print(f"✅ {len(chunks)} chunks indexed")

        return db

    # 🔥 MAIN FUNCTION
    def ask(self, query, history=""):
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 10}
        )

        docs = retriever.invoke(query)

        if not docs:
            return "⚠️ No relevant information found."

        # 🔥 RERANK (simple but effective)
        docs = sorted(docs, key=lambda x: len(x.page_content))[:3]

        context = "\n\n".join([
            doc.page_content[:300] for doc in docs
        ])

        prompt = f"""
You are an expert AI assistant.

STRICT RULES:
- Answer ONLY from context
- If not found → say "I don't know based on document"
- Be precise

Chat History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.llm.invoke(prompt)

        # ✅ SOURCES
        sources = "\n\n".join([
            f"Source {i+1}: {doc.page_content[:150]}"
            for i, doc in enumerate(docs)
        ])

        return f"{response}\n\n---\n📚 Sources:\n{sources}"