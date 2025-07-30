import argparse
import json
import os
import requests

from HTML_Datein_in_TXT_Format import process_htm_files
import simple_rag_ollama as rag


def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML to text and run the RAG pipeline"
    )
    parser.add_argument(
        "--html_dir",
        help="Directory with HTML files to convert to text (optional)",
    )
    parser.add_argument(
        "--txt_dir",
        required=True,
        help="Directory containing or receiving converted text files",
    )
    parser.add_argument(
        "--question",
        "-q",
        required=True,
        help="Question to ask",
    )
    parser.add_argument(
        "--top_k", type=int, default=3, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--model", default="llama3.2", help="Ollama model to use"
    )
    parser.add_argument(
        "--ollama_url",
        default="http://172.17.0.2:11434/api/generate",
        help="Ollama generate endpoint",
    )
    args = parser.parse_args()

    # Optionally convert HTML files to text
    if args.html_dir:
        process_htm_files(args.html_dir, args.txt_dir)

    documents = rag.load_documents(args.txt_dir)
    retrieved = rag.retrieve(args.question, documents, args.top_k)
    prompt = rag.build_prompt(args.question, retrieved)

    data = {"model": args.model, "prompt": prompt, "stream": False}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            args.ollama_url, headers=headers, data=json.dumps(data)
        )
        if response.status_code == 200:
            print(response.json().get("response", ""))
        else:
            print("Error:", response.status_code, response.text)
    except Exception as e:
        print("Fehler bei der Anfrage:", e)


if __name__ == "__main__":
    main()
