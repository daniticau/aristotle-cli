/**
 * Centralized configuration — single source of truth for all parameters.
 */

import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Resolve relative to package root (one level up from src/ or dist/)
const PACKAGE_ROOT = join(__dirname, "..");

// ── Paths ────────────────────────────────────────────────────────────────
export const DATA_DIR = join(PACKAGE_ROOT, "data");
export const CHUNKS_PATH = join(DATA_DIR, "chunks_with_embeddings.json");
export const SYSTEM_PROMPT_PATH = join(DATA_DIR, "system.txt");

// ── Embedding Model ─────────────────────────────────────────────────────
export const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";

// ── Anthropic LLM ───────────────────────────────────────────────────────
export const ANTHROPIC_MODEL = "claude-haiku-4-5-20251001";
export const MAX_TOKENS = 300;
export const TEMPERATURE = 0.3;

// ── RAG Parameters ──────────────────────────────────────────────────────
export const TOP_K = 5;
export const MIN_RELEVANCE_SCORE = 0.25;
