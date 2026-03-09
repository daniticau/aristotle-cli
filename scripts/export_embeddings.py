"""Export chunk embeddings from ChromaDB to a standalone JSON file.

Produces data/processed/chunks_with_embeddings.json — a self-contained file
with text, metadata, and pre-computed 384-dim embedding vectors for each chunk.
This file ships with the npm package so the TS port needs no vector DB.
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb
from config import CHROMA_DIR, CHUNKS_PATH, COLLECTION_NAME, EMBEDDING_MODEL


def export() -> None:
    """Extract all chunks + embeddings from ChromaDB and write to JSON."""
    # Load original chunks for metadata
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks_meta = {c["id"]: c for c in json.load(f)}

    # Connect to ChromaDB and pull everything
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)

    # Get all documents with embeddings
    result = collection.get(include=["documents", "metadatas", "embeddings"])

    records = []
    for doc_id, text, meta, embedding in zip(
        result["ids"],
        result["documents"],
        result["metadatas"],
        result["embeddings"],
    ):
        # Prefer original chunk metadata (has structured tags as list)
        original = chunks_meta.get(doc_id, {})
        tags = original.get("tags", [])
        if not tags and meta.get("tags"):
            tags = meta["tags"].split(",")

        records.append({
            "id": doc_id,
            "text": text,
            "book": meta["book"],
            "chapter": meta["chapter"],
            "tags": tags,
            "embedding": [round(x, 6) for x in embedding],  # 6 decimal places
        })

    # Sort by ID for deterministic output
    records.sort(key=lambda r: r["id"])

    out_path = CHUNKS_PATH.parent / "chunks_with_embeddings.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)

    size_kb = out_path.stat().st_size / 1024
    print(f"Exported {len(records)} chunks to {out_path} ({size_kb:.0f} KB)")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Embedding dim: {len(records[0]['embedding'])}")


if __name__ == "__main__":
    export()
