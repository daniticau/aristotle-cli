# CLAUDE.md

A RAG-powered CLI agent that embodies Aristotle, grounded in the Nicomachean Ethics. TypeScript/Node.js.

## Commands

```bash
pnpm install
pnpm build            # tsup → dist/
pnpm test             # vitest
node dist/index.js    # run (needs ANTHROPIC_API_KEY in .env)
node dist/index.js --debug
```

## Architecture

Three-layer persona defense (no ChromaDB — pure JS):

```
User query
  → guardrails.ts     regex jailbreak filter
  → retriever.ts      brute-force cosine similarity over pre-computed embeddings (187 chunks × 384 dims)
  → agent.ts          AristotleAgent: ask → retrieve → assemble → Claude streaming
  → ui.ts             chalk/ora CLI with spinners, streaming output
```

Embeddings are pre-computed and shipped as JSON. Query embedding uses `@huggingface/transformers` (ONNX, ~30MB first-run download).

## Data Files

- `data/chunks_with_embeddings.json` — 187 chunks with text, metadata, 384-dim vectors (1.2MB)
- `data/system.txt` — Aristotle persona system prompt

## Legacy Python Version

The original Python implementation lives in `legacy/`. It includes the full data pipeline (ingestion, chunking, concept tagging, ChromaDB) that the TS version replaces with pre-computed embeddings. See `legacy/CLAUDE.md` for Python-specific docs.
