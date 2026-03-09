"""Generate retrieval ground truth for validating the TypeScript port.

Embeds test queries with the same model, records top-5 results + scores.
Saves to tests/fixtures/retrieval_ground_truth.json.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

TEST_QUERIES = [
    "What is virtue?",
    "What is courage?",
    "What is the highest good?",
    "What is happiness?",
    "What is the mean?",
    "What is justice?",
    "What is friendship?",
    "What is practical wisdom?",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent

    # Load chunks with embeddings
    chunks_path = project_root / "data" / "processed" / "chunks_with_embeddings.json"
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    # Pre-compute chunk embedding matrix
    chunk_embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)

    # Load model and embed queries
    print(f"Loading model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    ground_truth = []
    for query in TEST_QUERIES:
        query_embedding = model.encode(query, normalize_embeddings=False)
        query_norm = np.linalg.norm(query_embedding)

        # Cosine similarity
        similarities = chunk_embeddings @ query_embedding / (chunk_norms * query_norm)
        top_indices = np.argsort(similarities)[::-1][:5]

        results = []
        for idx in top_indices:
            results.append({
                "id": chunks[idx]["id"],
                "score": round(float(similarities[idx]), 6),
                "book": chunks[idx]["book"],
                "chapter": chunks[idx]["chapter"],
            })

        ground_truth.append({
            "query": query,
            "top_5": results,
        })
        print(f"  {query}: top score = {results[0]['score']:.4f}")

    out_path = project_root / "tests" / "fixtures" / "retrieval_ground_truth.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(ground_truth)} queries to {out_path}")


if __name__ == "__main__":
    main()
