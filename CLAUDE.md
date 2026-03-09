# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A RAG-powered CLI agent that embodies Aristotle, grounded in the Nicomachean Ethics. Users ask philosophical questions; the agent retrieves relevant passages from a ChromaDB vector store and responds in Aristotle's voice via Claude API with streaming output.

## Commands

```bash
# Setup
pip install -r requirements.txt    # chromadb, sentence-transformers, anthropic, rich
# Set ANTHROPIC_API_KEY in .env

# Ingest source text (first run or reset)
python -m src.ingest --reset
python -m src.ingest --query "What is courage?"   # test retrieval after ingest

# Run the CLI
python main.py
python main.py --debug             # show retrieved chunks + assembled prompt
```

In-CLI commands: `/debug` (toggle), `/help`, `exit`/`quit`.

## Architecture

```
User query
  → src/guardrails.py    regex jailbreak filter (no API call if matched)
  → src/inference.py     ChromaDB retrieval → relevance score gate → prompt assembly → Claude streaming
  → src/ui.py            Rich CLI with animated spinners, streaming output
```

**Three-layer persona lock** (defense in depth, zero extra API cost):
1. **Keyword pre-filter** (`guardrails.py`) — regex catches obvious jailbreaks, returns canned in-character refusal, skips API entirely
2. **Retrieval score gating** (`inference.py`) — if best chunk scores below `MIN_RELEVANCE_SCORE`, query is off-topic; sends deflection to LLM without RAG context
3. **System prompt** (`prompts/system.txt`) — identity-lock section anchors Aristotle persona with few-shot refusal examples

**Key class**: `AristotleAgent` in `src/inference.py` — owns the full pipeline: `ask()` → `retrieve()` → `assemble_prompt()` → `generate()`. Lazy-inits ChromaDB on first query (or eagerly via `_ensure_retriever()` at boot).

**Data pipeline**: `src/ingest.py` downloads from Project Gutenberg → `src/chunker.py` parses book/chapter structure, chunks at paragraph boundaries (200-600 words), applies concept tags → embeds with SentenceTransformer → stores in ChromaDB.

## Config

All parameters live in `config.py` — model names, paths, RAG params, chunking params. Secrets go in `.env` (never committed). The system prompt is a plain text file at `prompts/system.txt`.

## Key Patterns

- **Streaming**: `ask(stream=True)` returns `(AgentResponse, Generator)` tuple; the UI consumes the generator token-by-token
- **Debug mode**: `--debug` flag or `/debug` command shows retrieved passages (scores, sources) and the assembled prompt before LLM call
- **Concept tagging**: Chunks get tagged with philosophical concepts (virtue, justice, telos, etc.) during ingestion via keyword matching in `chunker.py`
- **Animated spinners**: Boot loads embeddings with cycling philosophical phases; each query shows rotating "thinking verbs" via background threads

## Voice Rules (from system prompt)

Aristotle speaks in flowing prose — no markdown, no bullet points, no headers, no bold. Response length scales with question complexity. Never hedges without resolving. Never acknowledges being an AI. Defines terms before reasoning, grounds claims in telos.
