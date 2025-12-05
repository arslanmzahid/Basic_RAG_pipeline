# RAG Knowledge Service 🚀

A minimal **Retrieval-Augmented Generation (RAG)** microservice built with **FastAPI**, **OpenAI embeddings** and **FAISS** for semantic search over local documents.

Designed as a learning and portfolio project to demonstrate end-to-end AI engineering skills (LLMs + retrieval + API).

---

## 🧱 Architecture

**Indexing phase (startup):**

1. Load `.txt` files from `data/sample_docs/`
2. Split each document into overlapping chunks
3. Compute embeddings for all chunks using OpenAI (`text-embedding-3-small`)
4. Build a FAISS `IndexFlatL2` index over all chunk embeddings

**Query phase (per request):**

1. Receive a natural-language question via the `/query` endpoint
2. Embed the question using the same embedding model
3. Use FAISS to retrieve the top-k most similar chunks
4. Build a context-aware prompt with those chunks
5. Call an OpenAI chat model (e.g. `gpt-4.1-mini`)
6. Return the final answer + the list of source documents

---

## 🛠 Tech Stack

* **Python**
* **FastAPI** – web framework
* **Uvicorn** – ASGI server
* **OpenAI API** – embeddings + chat model
* **FAISS** – vector similarity search
* **NumPy** – numerical operations
* **python-dotenv** – environment variable handling

---

## 📦 Installation

```bash
git clone <your-repo-url>
cd <your-repo-folder>

# (optional but recommended) create venv
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## 🔐 Configuration

Create a file named **`.env`** in the project root:

```bash
OPENAI_API_KEY="sk-...."
```

> ⚠️ Do **not** commit `.env` to Git – it is already ignored via `.gitignore`.

---

## 📂 Adding Documents

Place your `.txt` files in:

```text
data/sample_docs/
    doc1.txt
    doc2.txt
    ...
```

These will be indexed automatically when the API starts.

---

## 🚀 Running the API

Start the FastAPI app with Uvicorn:

```bash
uvicorn app.main:app --reload
```

The service will:

* Load and chunk documents
* Create embeddings
* Build the FAISS index

You should see a log message when the index is initialized.

---

## ✅ Health Check

Open:

* `http://127.0.0.1:8000/health`

Example response:

```json
{
  "status": "ok",
  "num_chunks": 42
}
```

---

## 💬 Querying the RAG Endpoint

Use the interactive docs:

* `http://127.0.0.1:8000/docs`

Or call it via `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What topics are discussed in these documents?",
        "top_k": 4
      }'
```

Example response:

```json
{
  "answer": "The documents mainly discuss ...",
  "sources": ["doc1.txt", "doc2.txt"],
  "num_context_chunks": 4
}
```

---

## 🧠 Possible Extensions

* Add support for PDFs (using a text extractor)
* Persist FAISS index to disk
* Add authentication on the API
* Add a simple frontend (e.g. Streamlit or React) on top of the `/query` endpoint
* Swap in different models (OpenAI, Azure, local LLMs, etc.)

---

## 📌 How this fits an AI Engineer role

This project demonstrates:

* Building an end-to-end **RAG pipeline**
* Working with **LLMs + embeddings**
* Vector search using **FAISS**
* Exposing models as a **FastAPI microservice**
* Good engineering practices (env vars, modular code, health checks)

