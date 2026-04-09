/**
 * Centralized configuration — single source of truth for all parameters.
 */

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolve relative to package root (one level up from src/ or dist/)
const PACKAGE_ROOT = join(__dirname, "..");

function readStringEnv(name: string, fallback: string): string {
  const value = process.env[name]?.trim();
  return value ? value : fallback;
}

function readNumberEnv(
  name: string,
  fallback: number,
  validate: (value: number) => boolean,
  expectation: string,
): number {
  const raw = process.env[name]?.trim();
  if (!raw) return fallback;

  const value = Number(raw);
  if (!Number.isFinite(value) || !validate(value)) {
    throw new Error(
      `Invalid ${name}: expected ${expectation}, received "${raw}".`,
    );
  }

  return value;
}

function readIntegerEnv(
  name: string,
  fallback: number,
  validate: (value: number) => boolean,
  expectation: string,
): number {
  return readNumberEnv(
    name,
    fallback,
    (value) => Number.isInteger(value) && validate(value),
    expectation,
  );
}

// ── Paths ────────────────────────────────────────────────────────────────
export const DATA_DIR = join(PACKAGE_ROOT, "data");
export const CHUNKS_PATH = join(DATA_DIR, "chunks_with_embeddings.json");
export const SYSTEM_PROMPT_PATH = join(DATA_DIR, "system.txt");

// ── Embedding Model ─────────────────────────────────────────────────────
export const EMBEDDING_MODEL = readStringEnv(
  "ARISTOTLE_EMBEDDING_MODEL",
  "Xenova/all-MiniLM-L6-v2",
);

// ── Anthropic LLM ───────────────────────────────────────────────────────
export const ANTHROPIC_MODEL = readStringEnv(
  "ARISTOTLE_ANTHROPIC_MODEL",
  "claude-haiku-4-5-20251001",
);
export const MAX_TOKENS = readIntegerEnv(
  "ARISTOTLE_MAX_TOKENS",
  150,
  (value) => value > 0,
  "a positive integer",
);
export const TEMPERATURE = readNumberEnv(
  "ARISTOTLE_TEMPERATURE",
  0.3,
  (value) => value >= 0 && value <= 1,
  "a number between 0 and 1",
);

// ── RAG Parameters ──────────────────────────────────────────────────────
export const TOP_K = readIntegerEnv(
  "ARISTOTLE_TOP_K",
  5,
  (value) => value > 0,
  "a positive integer",
);
export const MIN_RELEVANCE_SCORE = readNumberEnv(
  "ARISTOTLE_MIN_RELEVANCE_SCORE",
  0.25,
  (value) => value >= 0 && value <= 1,
  "a number between 0 and 1",
);
