from langchain_community.vectorstores import Chroma
import os

DB_DIR = "db"


def create_vectorstore(chunks, embedding):
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=DB_DIR
    )
    db.persist()
    return db


def load_vectorstore(embedding):
    if os.path.exists(DB_DIR):
        return Chroma(
            persist_directory=DB_DIR,
            embedding_function=embedding
        )
    return None