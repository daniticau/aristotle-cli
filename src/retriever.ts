/**
 * Retriever — load pre-computed embeddings, embed queries at runtime,
 * brute-force cosine similarity over the full Aristotle corpus.
 */

import { readFileSync } from "node:fs";
import { CHUNKS_PATH, EMBEDDING_MODEL, TOP_K } from "./config.js";

// ── Types ────────────────────────────────────────────────────────────────

export interface Chunk {
  id: string;
  text: string;
  work: string;
  book: number;
  chapter: number;
  tags: string[];
  embedding: number[];
}

export interface RetrievedChunk {
  id: string;
  text: string;
  work: string;
  book: number;
  chapter: number;
  score: number;
  tags: string[];
}

interface CorpusStore {
  chunks: Chunk[];
  chunkNorms: Float64Array;
  embeddingDimensions: number;
}

// ── Embedding pipeline (lazy-init) ──────────────────────────────────────

type PipelineFn = (
  text: string,
  options: { pooling: string; normalize: boolean },
) => Promise<{ data: ArrayLike<number> }>;

let corpusStore: CorpusStore | null = null;
let embedPipeline: PipelineFn | null = null;
let embedPipelinePromise: Promise<PipelineFn> | null = null;

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function assertValidChunks(value: unknown): asserts value is Chunk[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error("Corpus file must contain a non-empty array of chunks.");
  }

  const firstEmbedding = value[0]?.embedding;
  if (!Array.isArray(firstEmbedding) || firstEmbedding.length === 0) {
    throw new Error("Corpus chunks must contain a non-empty embedding array.");
  }

  const embeddingDimensions = firstEmbedding.length;

  for (let i = 0; i < value.length; i++) {
    const chunk = value[i] as Partial<Chunk>;
    if (
      typeof chunk?.id !== "string" ||
      typeof chunk?.text !== "string" ||
      typeof chunk?.work !== "string" ||
      typeof chunk?.book !== "number" ||
      typeof chunk?.chapter !== "number" ||
      !Array.isArray(chunk?.tags) ||
      !chunk.tags.every((tag: unknown) => typeof tag === "string") ||
      !Array.isArray(chunk?.embedding) ||
      chunk.embedding.length !== embeddingDimensions ||
      !chunk.embedding.every(
        (entry: unknown) =>
          typeof entry === "number" && Number.isFinite(entry),
      )
    ) {
      throw new Error(`Invalid corpus chunk at index ${i}.`);
    }
  }
}

export function calculateVectorNorm(vector: ArrayLike<number>): number {
  let sumSq = 0;
  for (let i = 0; i < vector.length; i++) {
    const value = Number(vector[i]);
    sumSq += value * value;
  }
  return Math.sqrt(sumSq);
}

export function cosineSimilarity(
  lhs: ArrayLike<number>,
  rhs: ArrayLike<number>,
  lhsNorm: number = calculateVectorNorm(lhs),
  rhsNorm: number = calculateVectorNorm(rhs),
): number {
  if (lhs.length !== rhs.length) {
    throw new Error(
      `Embedding dimension mismatch: left=${lhs.length}, right=${rhs.length}.`,
    );
  }

  const denominator = lhsNorm * rhsNorm;
  if (!Number.isFinite(denominator) || denominator === 0) {
    return 0;
  }

  let dot = 0;
  for (let i = 0; i < lhs.length; i++) {
    dot += Number(lhs[i]) * Number(rhs[i]);
  }

  const score = dot / denominator;
  return Number.isFinite(score) ? score : 0;
}

function loadChunks(): CorpusStore {
  if (corpusStore !== null) {
    return corpusStore;
  }

  let parsed: unknown;
  try {
    const raw = readFileSync(CHUNKS_PATH, "utf-8");
    parsed = JSON.parse(raw);
  } catch (error) {
    throw new Error(
      `Failed to load Aristotle corpus from ${CHUNKS_PATH}: ${formatError(error)}`,
      { cause: error instanceof Error ? error : undefined },
    );
  }

  assertValidChunks(parsed);

  const chunkNorms = new Float64Array(parsed.length);
  for (let i = 0; i < parsed.length; i++) {
    chunkNorms[i] = calculateVectorNorm(parsed[i].embedding);
  }

  corpusStore = {
    chunks: parsed,
    chunkNorms,
    embeddingDimensions: parsed[0].embedding.length,
  };

  return corpusStore;
}

async function ensureEmbedder(): Promise<PipelineFn> {
  if (embedPipeline !== null) {
    return embedPipeline;
  }

  if (embedPipelinePromise === null) {
    embedPipelinePromise = (async () => {
      try {
        // Dynamic import — @huggingface/transformers is ESM-only
        const { pipeline } = await import("@huggingface/transformers");
        return (await pipeline("feature-extraction", EMBEDDING_MODEL, {
          dtype: "fp32",
        })) as unknown as PipelineFn;
      } catch (error) {
        embedPipelinePromise = null;
        throw new Error(
          `Failed to initialize embedding model "${EMBEDDING_MODEL}": ${formatError(error)}`,
          { cause: error instanceof Error ? error : undefined },
        );
      }
    })();
  }

  embedPipeline = await embedPipelinePromise;
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
  return Float64Array.from(output.data);
}

/**
 * Retrieve the top-K most similar chunks to a query.
 */
export async function retrieve(
  query: string,
  topK: number = TOP_K,
): Promise<RetrievedChunk[]> {
  const normalizedQuery = query.trim();
  if (!normalizedQuery) {
    return [];
  }

  const { chunks, chunkNorms, embeddingDimensions } = loadChunks();
  const limit = Math.min(
    Math.max(0, Number.isFinite(topK) ? Math.floor(topK) : TOP_K),
    chunks.length,
  );

  if (limit === 0) {
    return [];
  }

  const queryEmb = await embedQuery(normalizedQuery);
  if (queryEmb.length !== embeddingDimensions) {
    throw new Error(
      `Embedding dimension mismatch: corpus=${embeddingDimensions}, query=${queryEmb.length}.`,
    );
  }

  const queryNorm = calculateVectorNorm(queryEmb);
  if (queryNorm === 0) {
    return [];
  }

  const scores: Array<{ index: number; score: number }> = [];
  for (let i = 0; i < chunks.length; i++) {
    scores.push({
      index: i,
      score: cosineSimilarity(
        chunks[i].embedding,
        queryEmb,
        chunkNorms[i],
        queryNorm,
      ),
    });
  }

  scores.sort((a, b) => b.score - a.score);
  const topResults = scores.slice(0, limit);

  return topResults.map(({ index, score }) => {
    const chunk = chunks[index];
    return {
      id: chunk.id,
      text: chunk.text,
      work: chunk.work,
      book: chunk.book,
      chapter: chunk.chapter,
      score,
      tags: chunk.tags,
    };
  });
}
