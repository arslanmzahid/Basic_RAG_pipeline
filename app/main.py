# app/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager

from .rag_pipeline import initialize_index, answer_query, doc_texts


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[API] Starting up... building FAISS index.")
    initialize_index()
    print(f"[API] Startup complete. Loaded {len(doc_texts)} chunks.")
    yield
    print("[API] Shutdown complete.")


app = FastAPI(
    title="RAG Knowledge Service",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str
    top_k: int = 4


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    num_context_chunks: int


@app.get("/health")
def health():
    return {
        "status": "ok",
        "num_chunks": len(doc_texts),
    }


@app.post("/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    result = answer_query(req.question, req.top_k)

    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        num_context_chunks=result["num_context_chunks"],
    )
