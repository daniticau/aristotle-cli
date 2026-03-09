"""Download source text and load chunks into ChromaDB."""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import (
    CHROMA_DIR,
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    GUTENBERG_URL,
    RAW_TEXT_PATH,
    TOP_K,
)
from src.chunker import process_text


# ── Download ───────────────────────────────────────────────────────────────

def download_text(url: str = GUTENBERG_URL, dest: Path = RAW_TEXT_PATH) -> Path:
    """Download source text from Project Gutenberg. Skip if already exists."""
    if dest.exists():
        print(f"Text already exists at {dest}, skipping download")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


# ── Embedding ──────────────────────────────────────────────────────────────

def get_embedding_function() -> SentenceTransformerEmbeddingFunction:
    """Create embedding function with auto-detected device (cuda/cpu)."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    print(f"Using embedding device: {device}")
    return SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device=device,
    )


# ── ChromaDB ──────────────────────────────────────────────────────────────

def init_chroma(reset: bool = False) -> chromadb.ClientAPI:
    """Initialize ChromaDB persistent client."""
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass

    return client


def ingest(reset: bool = False) -> None:
    """Full ingestion pipeline: download → chunk → embed → store in ChromaDB."""
    # Download
    download_text()

    # Chunk
    if not CHUNKS_PATH.exists() or reset:
        chunks = process_text()
    else:
        print(f"Loading existing chunks from {CHUNKS_PATH}")
        with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
            chunks = json.load(f)

    print(f"Total chunks to ingest: {len(chunks)}")

    # Init ChromaDB
    client = init_chroma(reset=reset)
    embed_fn = get_embedding_function()

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch add chunks (ChromaDB handles batching internally, but we chunk to avoid memory issues)
    BATCH_SIZE = 100
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]

        ids = [c["id"] for c in batch]
        documents = [c["text"] for c in batch]
        metadatas = [
            {
                "work": c["work"],
                "book": c["book"],
                "chapter": c["chapter"],
                "chunk_index": c["chunk_index"],
                "word_count": c["word_count"],
                "tags": ",".join(c["tags"]),  # ChromaDB metadata can't hold lists
            }
            for c in batch
        ]

        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  Added batch {i // BATCH_SIZE + 1} ({len(batch)} chunks)")

    print(f"Ingestion complete: {collection.count()} chunks in collection '{COLLECTION_NAME}'")


# ── Query ──────────────────────────────────────────────────────────────────

def query_collection(
    query: str,
    n_results: int = TOP_K,
    where: dict | None = None,
) -> dict:
    """Query the ChromaDB collection. Returns raw ChromaDB results dict."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    embed_fn = get_embedding_function()

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embed_fn,
    )

    kwargs = {"query_texts": [query], "n_results": n_results}
    if where:
        kwargs["where"] = where

    return collection.query(**kwargs)


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Nicomachean Ethics into ChromaDB")
    parser.add_argument("--reset", action="store_true", help="Reset collection before ingesting")
    parser.add_argument("--query", type=str, help="Run a test query after ingestion")
    args = parser.parse_args()

    ingest(reset=args.reset)

    if args.query:
        print(f"\n{'='*60}")
        print(f"Query: {args.query}")
        print(f"{'='*60}")
        results = query_collection(args.query)
        for i, (doc, meta, dist) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            score = 1 - dist  # cosine distance → similarity
            print(f"\n--- Result {i+1} (score: {score:.3f}) ---")
            print(f"Source: Book {meta['book']}, Chapter {meta['chapter']}")
            print(f"Tags: {meta['tags']}")
            print(f"Preview: {doc[:200]}...")


if __name__ == "__main__":
    main()
