/**
 * Retriever — load pre-computed embeddings, embed queries at runtime,
 * brute-force cosine similarity over 187 chunks.
 */

import { readFileSync } from "node:fs";
import { CHUNKS_PATH, EMBEDDING_MODEL, TOP_K } from "./config.js";

// ── Types ────────────────────────────────────────────────────────────────

export interface Chunk {
  id: string;
  text: string;
  book: number;
  chapter: number;
  tags: string[];
  embedding: number[];
}

export interface RetrievedChunk {
  id: string;
  text: string;
  book: number;
  chapter: number;
  score: number;
  tags: string[];
}

// ── Chunk store (loaded once) ────────────────────────────────────────────

let chunks: Chunk[] | null = null;
let chunkNorms: Float64Array | null = null;

function loadChunks(): void {
  if (chunks !== null) return;

  const raw = readFileSync(CHUNKS_PATH, "utf-8");
  chunks = JSON.parse(raw) as Chunk[];

  // Pre-compute norms for cosine similarity
  chunkNorms = new Float64Array(chunks.length);
  for (let i = 0; i < chunks.length; i++) {
    let sumSq = 0;
    const emb = chunks[i].embedding;
    for (let j = 0; j < emb.length; j++) {
      sumSq += emb[j] * emb[j];
    }
    chunkNorms[i] = Math.sqrt(sumSq);
  }
}

// ── Embedding pipeline (lazy-init) ──────────────────────────────────────

type PipelineFn = (text: string, options: { pooling: string; normalize: boolean }) => Promise<{ data: Float32Array }>;

let embedPipeline: PipelineFn | null = null;

async function ensureEmbedder(): Promise<PipelineFn> {
  if (embedPipeline) return embedPipeline;

  // Dynamic import — @huggingface/transformers is ESM-only
  const { pipeline } = await import("@huggingface/transformers");
  embedPipeline = (await pipeline("feature-extraction", EMBEDDING_MODEL, {
    dtype: "fp32",
  })) as unknown as PipelineFn;

  return embedPipeline;
}

/**
 * Initialize the retriever: load chunks from disk and download the
 * embedding model (first run only). Call this eagerly at boot.
 */
export async function initRetriever(): Promise<void> {
  loadChunks();
  await ensureEmbedder();
}

/**
 * Embed a query string and return the raw float array.
 */
async function embedQuery(query: string): Promise<Float64Array> {
  const pipe = await ensureEmbedder();
  const output = await pipe(query, { pooling: "mean", normalize: false });
  return new Float64Array(output.data);
}

/**
 * Retrieve the top-K most similar chunks to a query.
 */
export async function retrieve(
  query: string,
  topK: number = TOP_K,
): Promise<RetrievedChunk[]> {
  loadChunks();
  const queryEmb = await embedQuery(query);

  // Query norm
  let queryNormSq = 0;
  for (let i = 0; i < queryEmb.length; i++) {
    queryNormSq += queryEmb[i] * queryEmb[i];
  }
  const queryNorm = Math.sqrt(queryNormSq);

  // Cosine similarity against all chunks
  const scores: { index: number; score: number }[] = [];
  for (let i = 0; i < chunks!.length; i++) {
    const emb = chunks![i].embedding;
    let dot = 0;
    for (let j = 0; j < emb.length; j++) {
      dot += emb[j] * queryEmb[j];
    }
    const score = dot / (chunkNorms![i] * queryNorm);
    scores.push({ index: i, score });
  }

  // Sort descending by score, take top-K
  scores.sort((a, b) => b.score - a.score);
  const topResults = scores.slice(0, topK);

  return topResults.map(({ index, score }) => {
    const chunk = chunks![index];
    return {
      id: chunk.id,
      text: chunk.text,
      book: chunk.book,
      chapter: chunk.chapter,
      score,
      tags: chunk.tags,
    };
  });
}
