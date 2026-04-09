# Methods

This document describes the technical implementation of `aristotle-cli`: how the corpus is built, how a user query is processed at runtime, and which design choices shape the system's behavior.

## System Overview

`aristotle-cli` is a retrieval-augmented command-line application written in TypeScript for Node.js. At runtime it does four things:

1. Reject obvious prompt-injection and persona-switching attempts before any model call.
2. Retrieve relevant Aristotle passages from a shipped embedding corpus.
3. Assemble an in-character prompt grounded in those passages.
4. Stream the answer back through a terminal UI.

The implementation is intentionally simple:

- No vector database.
- No external retrieval service.
- No conversation memory.
- No fine-tuned model.

Instead, the repository ships a precomputed JSON corpus and performs an in-memory brute-force cosine search over that artifact. In the current corpus, that means `1777` chunks across `18` works, each with a `384`-dimension embedding.

## Runtime Architecture

The main runtime modules are:

- `src/index.ts`: CLI entry point, argument parsing, `.env` loading, API key check.
- `src/config.ts`: centralized configuration and environment-variable parsing.
- `src/guardrails.ts`: regex-based jailbreak filter with deterministic refusal selection.
- `src/retriever.ts`: corpus loading, query embedding, cosine similarity search.
- `src/agent.ts`: end-to-end orchestration from query to streamed model output.
- `src/ui.ts`: terminal interface, spinners, debug views, and streamed output rendering.

The effective request path is:

```text
stdin
  -> index.ts
  -> AristotleAgent.ask()
  -> guardrails.ts
  -> retriever.ts
  -> prompt assembly in agent.ts
  -> Anthropic streaming API
  -> response shaping in agent.ts
  -> ui.ts
  -> stdout
```

## Startup and Initialization

### CLI bootstrap

`src/index.ts` is the only executable entry point. It:

- loads environment variables via `dotenv/config`
- accepts `--debug`, `--help`, and rejects unknown flags
- requires `ANTHROPIC_API_KEY` before continuing
- constructs `AristotleAgent`
- hands control to `runCli()`

### Eager retriever initialization

When the CLI starts, `src/ui.ts` calls `agent.ensureRetriever()` before entering the input loop. That boot sequence does two things:

- loads and validates `data/chunks_with_embeddings.json`
- initializes the Hugging Face embedding pipeline

This means the first noticeable startup delay happens once, up front, rather than on the first user question. The embedding model is downloaded on first run by `@huggingface/transformers` and reused after that.

## Configuration

All runtime settings live in `src/config.ts`. Defaults are hard-coded and can be overridden with environment variables.

Current configuration surface:

| Variable | Default | Purpose |
| --- | --- | --- |
| `ARISTOTLE_EMBEDDING_MODEL` | `Xenova/all-MiniLM-L6-v2` | Query embedding model loaded through `@huggingface/transformers` |
| `ARISTOTLE_ANTHROPIC_MODEL` | `claude-haiku-4-5-20251001` | Anthropic generation model |
| `ARISTOTLE_MAX_TOKENS` | `150` | Upper bound for generated completion length |
| `ARISTOTLE_TEMPERATURE` | `0.3` | Sampling temperature |
| `ARISTOTLE_TOP_K` | `5` | Default retrieval depth |
| `ARISTOTLE_MIN_RELEVANCE_SCORE` | `0.25` | Threshold below which a query is treated as off-topic |

`config.ts` also resolves package-relative paths for:

- `data/chunks_with_embeddings.json`
- `data/system.txt`

Numeric configuration is validated on load. Invalid overrides fail fast with explicit errors.

## Guardrail Layer

The first defensive layer is `src/guardrails.ts`. It uses a single case-insensitive regular expression to detect common jailbreak patterns, including:

- "ignore your instructions"
- "pretend to be"
- "system prompt"
- "break character"
- "new persona"
- "override your instructions"

If a match is found, the application does not retrieve context and does not call Anthropic. It instead returns one of three refusal messages written in Aristotle's voice.

Refusal selection is deterministic. Rather than using a runtime-dependent hash, the code sums character codes from the input string and uses that stable integer to rotate across the refusal pool. This keeps behavior reproducible across platforms.

## Retrieval Method

`src/retriever.ts` implements retrieval with a deliberately small and inspectable pipeline.

### Corpus representation

Each chunk in `data/chunks_with_embeddings.json` contains:

- `id`
- `text`
- `work`
- `book`
- `chapter`
- `tags`
- `embedding`

At load time, the retriever validates that:

- the corpus is a non-empty array
- every chunk has the required metadata fields
- every embedding exists, is numeric, and has the same dimensionality

The loader then precomputes one Euclidean norm per chunk and stores those norms in a `Float64Array`. The corpus and norms are cached in memory for reuse across all later queries.

### Query embedding

Query embeddings are generated lazily through a dynamically imported Hugging Face pipeline:

```ts
pipeline("feature-extraction", EMBEDDING_MODEL, { dtype: "fp32" })
```

For each query, the retriever runs:

- `pooling: "mean"`
- `normalize: false`

The output is converted to a `Float64Array`.

### Similarity calculation

Similarity is plain cosine similarity:

```text
score = dot(chunk_embedding, query_embedding) / (||chunk|| * ||query||)
```

The implementation:

- computes query norm once
- reuses cached chunk norms
- returns `0` if either vector has zero magnitude
- throws if query and corpus dimensions differ

### Search strategy

Search is brute-force over the entire corpus:

- compute one cosine score per chunk
- sort scores descending
- take the top `K`

The asymptotic cost is `O(N * d)` for scoring plus `O(N log N)` for the full sort, where:

- `N = 1777` chunks
- `d = 384` embedding dimensions

That is small enough to keep the implementation simple and dependency-light.

### Retrieval gating

Retrieval always returns the top matches, but `src/agent.ts` inspects the highest score. If the best score is below `MIN_RELEVANCE_SCORE`, the system treats the question as outside Aristotle's domain and switches to an off-topic deflection prompt instead of passing retrieved passages through as support.

## Agent Orchestration

`src/agent.ts` owns the full runtime pipeline.

### System prompt loading

The persona prompt is stored in `data/system.txt` and loaded synchronously when `AristotleAgent` is constructed. If the file is missing or empty, construction fails immediately.

That system prompt establishes:

- first-person Aristotle voice
- strict brevity rules
- rejection of modern concepts and AI framing
- preference for prose over formatting
- instruction to reason from definitions, function, and ends

### Response profiling

Before retrieval, the agent classifies the user query into one of three response profiles:

- `simple`
- `explanatory`
- `complex`

This classification is heuristic, based on:

- word count
- number of question marks
- punctuation density
- coordinating conjunctions
- lexical cues like `what is`, `how`, `why`, `compare`, `contrast`, `between`

The profile controls:

- retrieval depth (`topK`)
- generation budget (`maxTokens`)
- whether the answer should stay in one paragraph
- optional sentence cutoff

Current behavior:

- `simple`: up to 2 sentences, 1 paragraph, `topK <= 2`, `maxTokens <= 64`
- `explanatory`: up to 4 sentences, 1 paragraph, `topK <= 3`, `maxTokens <= 120`
- `complex`: uses the full configured retrieval depth and token budget

### Prompt assembly

If retrieval is on-topic, the agent builds a user message with:

- a preamble explaining that the passages are Aristotle's own writings
- the retrieved passages, each labeled with work, book, and part
- the student's question
- a response contract derived from the selected profile

It then appends an assistant prefill:

```text
In brief.
```

The Anthropic request therefore has:

- `system`: contents of `data/system.txt`
- `messages[0]`: the grounded user prompt
- `messages[1]`: assistant prefill

If retrieval is off-topic, the grounded passages are replaced by a short instruction telling Aristotle to state plainly that the subject is unknown to him and redirect the student toward domains he does know.

### Generation

Generation uses the Anthropic streaming API from `@anthropic-ai/sdk`:

- `messages.stream(...)`
- model from `ARISTOTLE_ANTHROPIC_MODEL`
- `temperature` from config
- `max_tokens` determined by the selected response profile

Only `content_block_delta` events carrying `text_delta` payloads are emitted to the UI. Leading whitespace from the first emitted token is trimmed once.

### Output shaping

After Anthropic begins streaming, the output passes through `applyResponseProfile()`.

This post-processor enforces the requested answer shape even if the model drifts. It:

- collapses whitespace toward a single-paragraph output when required
- counts sentence-ending punctuation
- stops the stream once the sentence limit for the active profile is reached

This is an important second layer of brevity control beyond the system prompt itself.

### Debug mode

If `AristotleAgent` is constructed with `debug: true`, the agent returns extra metadata:

- retrieved chunks with scores
- the effective prompt messages passed to the model

`src/ui.ts` uses that data to render an inspection table and the assembled non-system prompt before printing the answer.

## Terminal UI

`src/ui.ts` implements a terminal-first interaction model.

### Input loop

The CLI uses `readline/promises` to accept one question at a time. Supported commands are:

- `/help`
- `/debug`
- `exit`
- `quit`

### Startup and thinking indicators

The UI uses `ora` spinners with rotating phrase lists during:

- startup corpus/model loading
- query processing before the first token arrives

### Streaming renderer

Answers are written to stdout incrementally. When TTY support is available, the renderer can use a custom glitch-trail typewriter effect; otherwise it falls back to direct token writes.

The UI also:

- prints an ASCII-art banner on startup
- renders retrieval results in a `cli-table3` table in debug mode
- hides and restores the cursor around typewriter output

This layer is presentation only. Retrieval, prompting, and generation remain in `agent.ts`.

## Offline Corpus Build Pipeline

The runtime corpus is produced by `scripts/build_corpus.py`, a standalone Python script. This script is not part of the request path, but it defines the artifact the runtime depends on.

### Inputs

The script builds a registry of 18 Aristotle works. Texts are downloaded from:

- MIT Internet Classics Archive by default
- Project Gutenberg for `Politics`, because the MIT source is incomplete

### Cleaning

The script removes source-specific boilerplate:

- MIT headers, footers, HTML wrappers, and stray quotation marks
- Gutenberg headers and footers

### Structure parsing

Texts are parsed into section hierarchies using regex-based heading detection. The parser supports:

- `BOOK <roman numeral>`
- `SECTION <number>`
- `Part <number or roman numeral>`
- custom overrides per work

If no headings are found, the script falls back to treating the whole text as one section and one part.

### Chunking

Each structural part is split at paragraph boundaries into chunks of roughly:

- minimum `200` words
- maximum `600` words

Oversized paragraphs are kept intact as single chunks. Very short trailing fragments are merged into the previous chunk when possible.

### Concept tagging

Each chunk receives keyword-derived concept tags such as:

- `virtue`
- `justice`
- `telos`
- `friendship`
- `logic`
- `substance`
- `motion`
- `rhetoric`

Tags are not used in similarity scoring, but they are useful for debugging, inspection, and future retrieval refinements.

### Embedding

The offline script embeds chunks with Python `sentence-transformers` using:

```text
all-MiniLM-L6-v2
```

Embeddings are rounded to 6 decimal places and written back into each chunk record.

### Output artifact

The script writes `data/chunks_with_embeddings.json`, which in the current repository is approximately `11.5 MB` on disk and serves as the sole retrieval index for the TypeScript runtime.

## Testing Strategy

The test suite is in `tests/` and focuses on deterministic behavior around configuration, orchestration, and retrieval math.

### Current coverage

- `tests/config.test.ts`
  - default configuration values
  - environment override parsing
  - fast failure on invalid numeric values

- `tests/guardrails.test.ts`
  - jailbreak phrase detection
  - clean-query pass-through
  - refusal-string behavior

- `tests/agent.test.ts`
  - jailbreak short-circuiting before retrieval
  - prompt assembly for grounded responses
  - off-topic fallback behavior
  - eager retriever initialization delegation
  - response-profile classification
  - post-generation sentence trimming

- `tests/retriever-math.test.ts`
  - vector norm correctness
  - zero-vector behavior
  - dimension mismatch handling

- `tests/retriever.test.ts`
  - retrieval fixture structure
  - corpus artifact shape
  - score-threshold assumptions for on-topic fixture queries

The repository also includes `tests/fixtures/retrieval_ground_truth.json`, which preserves expected retrieval characteristics derived from the earlier Python implementation.

## Design Tradeoffs

The current implementation makes a few explicit tradeoffs:

- Simplicity over indexing sophistication. Brute-force retrieval is slower asymptotically than ANN search, but transparent and easy to validate at this corpus size.
- Precomputation over runtime ingestion. Shipping embeddings as JSON keeps runtime logic small and deterministic.
- Strong persona control over flexibility. The system prompt, guardrails, off-topic gating, and response shaping all bias toward staying in character and staying brief.
- Stateless interaction over conversational continuity. Each question is answered independently, which simplifies prompt construction but loses multi-turn memory.

## Limitations

The present architecture does not yet provide:

- conversation memory across turns
- source citations in the final user-facing answer
- hybrid lexical plus semantic retrieval
- reranking
- approximate nearest-neighbor indexing
- automatic corpus refresh from upstream sources

Those omissions are deliberate. The codebase is optimized for a compact, inspectable proof of concept rather than maximum retrieval sophistication.
