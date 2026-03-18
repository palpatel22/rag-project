import os
import json
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from pathlib import Path
from groq import Groq
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from pypdf import PdfReader

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_txt(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def load_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def load_csv(path):
    df = pd.read_csv(path)
    return "\n".join(
        df.astype(str).apply(lambda r: " | ".join(r), axis=1)
    )

def load_documents(folder="data"):
    docs = []
    for file in Path(folder).iterdir():
        if file.suffix == ".txt":
            print(f"📄 Loading: {file.name}")
            docs.append(load_txt(file))
        elif file.suffix == ".pdf":
            print(f"📕 Loading: {file.name}")
            docs.append(load_pdf(file))
        elif file.suffix == ".csv":
            print(f"📊 Loading: {file.name}")
            docs.append(load_csv(file))
    print(f"\n✅ Total {len(docs)} documents loaded!\n")
    return docs

def chunk_text(text, chunk_size=150, overlap=30):
    words = text.split()
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks

def build_chunks(documents):
    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_text(doc))
    print(f"✅ Total {len(all_chunks)} chunks bane!\n")
    return all_chunks

def build_index(chunks):
    print("⏳ BM25 index ban raha hai...")
    tokenized = [c.lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print("⏳ Embeddings generate ho rahi hain...")
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    embeddings = np.array(embeddings).astype('float32')
    print("⏳ FAISS index ban raha hai...")
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(embeddings)
    print("✅ Index ready hai!\n")
    return bm25, faiss_index

def retrieve(query, chunks, bm25, faiss_index, top_k=5):
    tokens = query.lower().split()
    bm25_scores = np.array(bm25.get_scores(tokens))
    bm25_norm = (bm25_scores - bm25_scores.min()) / \
                (bm25_scores.max() - bm25_scores.min() + 1e-9)
    query_vec = embedder.encode([query]).astype('float32')
    distances, indices = faiss_index.search(query_vec, len(chunks))
    faiss_scores = np.zeros(len(chunks))
    for rank, idx in enumerate(indices[0]):
        faiss_scores[idx] = 1 / (1 + distances[0][rank])
    faiss_norm = (faiss_scores - faiss_scores.min()) / \
                 (faiss_scores.max() - faiss_scores.min() + 1e-9)
    hybrid = 0.5 * bm25_norm + 0.5 * faiss_norm
    top_indices = np.argsort(hybrid)[::-1][:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "chunk": chunks[idx],
            "score": round(float(hybrid[idx]), 4),
        })
    return results

def generate_answer(query, retrieved_chunks):
    context = "\n\n".join(
        f"[Part {i+1}]: {r['chunk']}"
        for i, r in enumerate(retrieved_chunks)
    )
    prompt = f"""You are a helpful assistant.
Answer the question using ONLY the context below.
If answer is not in context, say "I don't know based on provided documents."

Context:
{context}

Question: {query}

Answer:"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    return response.choices[0].message.content

def evaluate_answer(query, answer, retrieved_chunks):
    context = "\n".join(r['chunk'] for r in retrieved_chunks)
    eval_prompt = f"""Rate this RAG answer. Reply in JSON only, no extra text.

Question: {query}
Context: {context}
Answer: {answer}

Return exactly this JSON:
{{
  "faithfulness": <0-10>,
  "relevance": <0-10>,
  "completeness": <0-10>,
  "comment": "<one line feedback>"
}}"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": eval_prompt}],
        max_tokens=200
    )
    raw = response.choices[0].message.content
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except:
        return {
            "faithfulness": 0,
            "relevance": 0,
            "completeness": 0,
            "comment": "Evaluation parse nahi hua"
        }

def ask(query, chunks, bm25, faiss_index):
    print(f"\n{'='*60}")
    print(f"❓ Question: {query}")
    print('='*60)
    retrieved = retrieve(query, chunks, bm25, faiss_index, top_k=5)
    answer = generate_answer(query, retrieved)
    print(f"\n💬 Answer:\n{answer}")
    print("\n⏳ Evaluation ho rahi hai...")
    scores = evaluate_answer(query, answer, retrieved)
    print(f"\n📊 Evaluation Scores:")
    print(f"   Faithfulness  : {scores['faithfulness']}/10")
    print(f"   Relevance     : {scores['relevance']}/10")
    print(f"   Completeness  : {scores['completeness']}/10")
    print(f"   Comment       : {scores['comment']}")
    return answer, scores

if __name__ == "__main__":
    documents = load_documents("data")
    chunks = build_chunks(documents)
    bm25, faiss_index = build_index(chunks)
    while True:
        print("\n" + "─"*60)
        query = input("🙋 Apna question likho (ya 'exit' likho): ").strip()
        if query.lower() == 'exit':
            print("👋 Bye!")
            break
        if query:
            ask(query, chunks, bm25, faiss_index)