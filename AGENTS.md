# AGENTS.md

A RAG-powered CLI agent that embodies Aristotle, grounded in his complete works. TypeScript/Node.js.

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
  → retriever.ts      brute-force cosine similarity over pre-computed embeddings (1777 chunks × 384 dims)
  → agent.ts          AristotleAgent: ask → retrieve → assemble → Anthropic streaming
  → ui.ts             chalk/ora CLI with spinners, streaming output
```

Embeddings are pre-computed and shipped as JSON. Query embedding uses `@huggingface/transformers` (ONNX, ~30MB first-run download).

## Data Files

- `data/chunks_with_embeddings.json` — 1777 chunks from 18 works with text, metadata, 384-dim vectors (~11.5MB)
- `data/system.txt` — Aristotle persona system prompt
- `scripts/build_corpus.py` — standalone Python script to rebuild the corpus (`uv run scripts/build_corpus.py`)

## Corpus

18 works: Nicomachean Ethics, Politics, Rhetoric, Poetics, Metaphysics, Physics, On the Soul, Categories, On Interpretation, Prior Analytics, Posterior Analytics, Topics, On Sophistical Refutations, On the Heavens, On Generation and Corruption, Meteorology, Athenian Constitution, On Memory and Reminiscence.

## Legacy Python Version

The original Python implementation lives in `legacy/`. It includes the full data pipeline (ingestion, chunking, concept tagging, ChromaDB) that the TS version replaces with pre-computed embeddings. See `legacy/AGENTS.md` for Python-specific docs.
