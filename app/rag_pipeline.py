# app/rag_pipeline.py

import os
import glob
from typing import List, Tuple, Dict, Optional

import numpy as np
import faiss
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# 1. Environment & OpenAI setup
# -----------------------------

# Load env variables from .env (if you use one)
load_dotenv()  # loading the .env that we have created

# retrieving the key from here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

# setting up the backend API for using OPENAI model
client = OpenAI(api_key=OPENAI_API_KEY)

# using the embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"  # or "gpt-4o-mini", etc.

# Global storage
faiss_index: Optional[faiss.IndexFlatL2] = None
doc_texts: List[str] = []   # all chunk texts
doc_meta: List[Dict] = []   # metadata aligned with doc_texts


# -----------------------------
# 2. Chunking + document loading
# -----------------------------

def chunk_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> List[str]:
    """
    Split a long string into smaller overlapping chunks.

    We use character length as a simple proxy for token length.
    In production you might use a tokenizer (tiktoken, etc.),
    but this is perfectly fine for a first RAG project.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []

    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += step

    return chunks


def load_documents_and_chunk(
    data_dir: str = "data/sample_docs",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> Tuple[List[str], List[Dict]]:
    """
    Load all .txt documents from a directory and chunk them.
    """
    doc_texts: List[str] = []
    doc_meta: List[Dict] = []

    pattern = os.path.join(data_dir, "*.txt")
    file_paths = glob.glob(pattern)

    for path in file_paths:
        with open(path, "r", encoding="utf-8") as f:
            full_text = f.read()

        chunks = chunk_text(
            full_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for i, ch in enumerate(chunks):
            doc_texts.append(ch)
            doc_meta.append(
                {
                    "source_path": os.path.basename(path),
                    "chunk_idx": i,
                }
            )

    return doc_texts, doc_meta


# -----------------------------
# 3. Embeddings + index building
# -----------------------------

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Compute embeddings for a list of texts using OpenAI.
    Returns a (n_texts, dim) float32 numpy array.
    """
    if not texts:
        return np.zeros((0, 0), dtype="float32")

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    vectors = [item.embedding for item in response.data]
    return np.array(vectors, dtype="float32")


def initialize_index(
    data_dir: str = "data/sample_docs",
    chunk_size: int = 800,
    chunk_overlap: int = 200,
) -> None:
    """
    Load documents, chunk them, embed them, and build a FAISS index.
    Populates global doc_texts, doc_meta, and faiss_index.
    """
    global doc_texts, doc_meta, faiss_index

    # 1) load + chunk
    doc_texts, doc_meta = load_documents_and_chunk(
        data_dir=data_dir,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not doc_texts:
        raise RuntimeError(f"No documents found in {data_dir}")

    # 2) embed
    embeddings = embed_texts(doc_texts)  # sending the text for embedding
    if embeddings.size == 0:
        raise RuntimeError("Embeddings are empty; something went wrong.")

    # 3) build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)  # L2 distance (Euclidean)

    index.add(embeddings)

    faiss_index = index
    print(f"[RAG] Initialized index with {len(doc_texts)} chunks, dim={dim}.")


# -----------------------------
# 4. Retrieval + LLM answering
# -----------------------------

def retrieve_chunks(
    query: str,
    top_k: int = 4,
) -> List[Dict]:
    """
    Given a user query, embed it and retrieve the top_k most similar chunks.
    """
    if faiss_index is None:
        raise RuntimeError("FAISS index is not initialized. Call initialize_index() first.")

    # 1) Embed the query into the same vector space as the chunks
    query_vec = embed_texts([query])  # shape: (1, dim)

    # 2) Ask FAISS: "Which stored vectors are closest to this query?"
    distances, indices = faiss_index.search(query_vec, top_k)

    results: List[Dict] = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "text": doc_texts[idx],
                "distance": float(dist),
                "metadata": doc_meta[idx],
            }
        )

    return results


def build_messages_from_context(
    query: str,
    contexts: List[Dict],
) -> List[Dict]:
    """
    Build the chat messages for the LLM using retrieved chunks as context.
    """
    context_blocks = []
    for i, c in enumerate(contexts, start=1):
        meta = c["metadata"]
        context_blocks.append(
            f"[Document {i} | {meta['source_path']} | chunk {meta['chunk_idx']}]\n"
            f"{c['text']}\n"
        )

    context_str = "\n\n".join(context_blocks)

    system_msg = (
        "You are an assistant that answers questions based ONLY on the provided context. "
        "If the answer is not in the context, say you don't know."
    )

    user_msg = (
        f"Context:\n{context_str}\n\n"
        f"Question: {query}\n\n"
        "Answer concisely and clearly, and indicate which document(s) you used."
    )

    return [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
    ]


def answer_query(
    query: str,
    top_k: int = 4,
) -> Dict:
    """
    Full RAG step:
      1) retrieve top_k chunks
      2) call LLM with those chunks as context
      3) return answer + sources
    """
    contexts = retrieve_chunks(query, top_k=top_k)

    if not contexts:
        return {
            "answer": "I couldn't find any relevant information in the indexed documents.",
            "sources": [],
            "num_context_chunks": 0,
        }

    messages = build_messages_from_context(query, contexts)

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
    )

    answer = response.choices[0].message.content

    sources = []
    for c in contexts:
        src = c["metadata"]["source_path"]
        if src not in sources:
            sources.append(src)

    return {
        "answer": answer,
        "sources": sources,
        "num_context_chunks": len(contexts),
    }


# -----------------------------
# 5. Local test block
# -----------------------------

if __name__ == "__main__":
    initialize_index()  # uses default data_dir="data/sample_docs"

    print(f"Index built. Chunks: {len(doc_texts)}")

    question = "What topics are discussed in these documents?"
    result = answer_query(question, top_k=4)

    print("\nQuestion:", question)
    print("\nAnswer:\n", result["answer"])
    print("\nSources:", result["sources"])
    print("Used chunks:", result["num_context_chunks"])
