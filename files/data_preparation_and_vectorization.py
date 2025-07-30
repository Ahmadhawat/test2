import os
import json
import re

# Simple HTML tag stripper using regex
TAG_RE = re.compile(r'<[^>]+>')


def extract_text_simple(html: str) -> str:
    """Remove HTML tags and extra whitespace."""
    text = TAG_RE.sub(' ', html)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def html_to_json(input_dir: str, output_dir: str):
    """Convert HTML files in input_dir to JSON with minimal structure."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    for name in os.listdir(input_dir):
        if name.lower().endswith(('.html', '.htm')):
            path = os.path.join(input_dir, name)
            with open(path, 'r', encoding='utf-8') as f:
                html = f.read()
            text = extract_text_simple(html)
            data = {'source': path, 'text': text}
            json_path = os.path.join(output_dir, os.path.splitext(name)[0] + '.json')
            with open(json_path, 'w', encoding='utf-8') as out:
                json.dump(data, out, ensure_ascii=False, indent=2)
            results.append(data)
    return results


def build_vocabulary(docs):
    """Build a vocabulary mapping from word to index."""
    vocab = {}
    for doc in docs:
        for word in re.findall(r'\b\w+\b', doc['text'].lower()):
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


def vectorize(docs, vocab):
    """Create bag-of-words vectors for each document."""
    vectors = []
    for doc in docs:
        counts = [0] * len(vocab)
        for word in re.findall(r'\b\w+\b', doc['text'].lower()):
            idx = vocab[word]
            counts[idx] += 1
        vectors.append({'source': doc['source'], 'vector': counts})
    return vectors


def save_vectors(vectors, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(vectors, f, ensure_ascii=False, indent=2)


def main():
    input_dir = 'html_files'  # Folder containing raw HTML files
    json_dir = 'json_output'  # Where JSON files will be written
    vector_file = 'vectors.json'

    docs = html_to_json(input_dir, json_dir)
    vocab = build_vocabulary(docs)
    vectors = vectorize(docs, vocab)
    save_vectors(vectors, vector_file)


if __name__ == '__main__':
    main()
