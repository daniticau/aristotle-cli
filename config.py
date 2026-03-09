"""Centralized configuration — single source of truth for all paths and parameters."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = DATA_DIR / "chroma_db"
SYSTEM_PROMPT_PATH = PROJECT_ROOT / "prompts" / "system.txt"

# ── Source Text ────────────────────────────────────────────────────────────
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/8438/pg8438.txt"
RAW_TEXT_PATH = RAW_DIR / "nicomachean_ethics.txt"
CHUNKS_PATH = PROCESSED_DIR / "chunks.json"

# ── Embedding Model ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ── Anthropic LLM ─────────────────────────────────────────────────────────
ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 1024
TEMPERATURE = 0.3

# ── RAG Parameters ────────────────────────────────────────────────────────
TOP_K = 5
MIN_RELEVANCE_SCORE = 0.25  # chunks below this are considered off-topic

# ── Chunking Parameters ───────────────────────────────────────────────────
CHUNK_MIN_WORDS = 200
CHUNK_MAX_WORDS = 600

# ── ChromaDB Collection ───────────────────────────────────────────────────
COLLECTION_NAME = "nicomachean_ethics"
