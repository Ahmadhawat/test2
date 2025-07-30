import os
import re
import argparse
import requests
import json
from collections import Counter


def tokenize(text):
    return re.findall(r"\w+", text.lower())


def load_documents(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(".txt"):
            path = os.path.join(directory, filename)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            tokens = set(tokenize(content))
            documents.append({"path": path, "text": content, "tokens": tokens})
    return documents


def jaccard_similarity(a, b):
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0


def retrieve(query, documents, top_k=3):
    q_tokens = set(tokenize(query))
    scored = [
        (jaccard_similarity(doc["tokens"], q_tokens), doc) for doc in documents
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored[:top_k] if score > 0]


def build_prompt(question, docs):
    context = "\n\n".join(doc["text"] for doc in docs)
    prompt = (
        "Answer the question using the context below. "
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Simple RAG pipeline using Ollama")
    parser.add_argument("docs_dir", help="Directory with text documents")
    parser.add_argument("--question", "-q", required=True, help="Question to ask")
    parser.add_argument("--top_k", type=int, default=3, help="Number of docs to retrieve")
    args = parser.parse_args()

    documents = load_documents(args.docs_dir)
    retrieved = retrieve(args.question, documents, args.top_k)
    prompt = build_prompt(args.question, retrieved)

    data = {"model": "llama3.2", "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            "http://172.17.0.2:11434/api/generate",
            headers=headers,
            data=json.dumps(data),
        )
        if response.status_code == 200:
            print(response.json().get("response", ""))
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Fehler bei der Anfrage:", e)


if __name__ == "__main__":
    main()
