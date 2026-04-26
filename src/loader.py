from langchain_community.document_loaders import TextLoader, PyPDFLoader
import os

def load_documents(upload_dir="uploads"):
    docs = []

    for file in os.listdir(upload_dir):
        path = os.path.join(upload_dir, file)

        if file.endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        else:
            continue

        docs.extend(loader.load())

    return docs