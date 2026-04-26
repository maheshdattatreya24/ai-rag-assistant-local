from langchain_community.llms import Ollama


def create_rag_chain():
    llm = Ollama(model="qwen2.5:1.5b", temperature=0)

    def run(query, context, history=""):
        prompt = f"""
You are a helpful AI assistant.

Use ONLY the given context.
If answer not found, say "I don't know."

Chat History:
{history}

Context:
{context}

Question:
{query}

Answer:
"""
        return llm.invoke(prompt)

    return run