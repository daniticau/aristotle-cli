# Project Aristotle: A Structured Reasoning Benchmark Disguised as a Philosophy Agent

## 1. Vision

A **Bicameral CLI Agent** that captures Aristotle's *method of thinking* — and in doing so, stress-tests how far small language models can be pushed toward formal, structured reasoning.

**Why Aristotle as a test case:** Aristotelian reasoning has *verifiable formal structure*. A syllogism is valid or it isn't. A definition either identifies the genus and differentia or it doesn't. Unlike vague persona tests ("does this sound like Shakespeare?"), every output can be checked against concrete criteria. This makes Aristotle the ideal benchmark for whether a model is actually reasoning or just pattern-matching.

**Dual Purpose:**

- **The Agent:** A faithful, uncompromising Aristotelian philosopher that defines terms, reasons syllogistically, appeals to telos, and corrects errors directly. No hedging, no "helpful assistant" tropes.
- **The Education:** Each phase of the build teaches something specific about how language models represent and fail at structured thought — from embeddings to attention to fine-tuning.

---

## 2. What You'll Learn (Phase by Phase)

This is the real roadmap. The agent is the artifact; the understanding is the point.

| Phase | What You're Building | What You're Learning |
|---|---|---|
| **Phase 1** | Vector database of Aristotle's texts | How embedding models represent semantic meaning. Does "what is courage" land near "the mean between cowardice and rashness" in vector space? This is applied linear algebra — similarity metrics, dimensionality, the geometry of meaning. |
| **Phase 2** | Working agent with RAG + system prompt | Where prompting hits its ceiling. You'll find the exact point where the 8B model loses the thread of a multi-step syllogism. That tells you something concrete about what's happening in the attention layers — how the model represents (or fails to represent) logical dependencies. |
| **Phase 3** | Evaluation framework | How to quantify reasoning quality — a genuinely open problem in the field. You'll build your own scoring rubric and confront the question: what does it even mean for a model to "reason correctly"? |
| **Phase 4** | Fine-tuned model (conditional) | Whether LoRA can shift *reasoning patterns* or just *surface style*. Your own empirical answer, not just papers. |

---

## 3. Architecture

### Data Flow

```
User Input → CLI (Rich) → ChromaDB (Memory) → Prompt Assembler → LLM (Soul) → CLI Output
                                                       ↓
                                              [--debug: print retrieved chunks,
                                               similarity scores, prompt template]
```

### Components

#### A. The Memory (Vector Database)

- **Technology:** ChromaDB (local, file-based)
- **Data Source:** Project Gutenberg texts
  - *Nicomachean Ethics*
  - *Politics*
  - *Metaphysics*
  - *Poetics*
  - *Categories* and *Prior Analytics* (for syllogistic structure)
- **Chunking Strategy:** Use natural book/chapter/section structure — each chunk should represent one coherent argument or concept. Target 200–600 words depending on where breaks fall naturally.
- **Metadata:** Each chunk gets: work title, book number, chapter, and concept-level tags (e.g., "virtue", "polis", "four causes", "telos", "soul"). Enables hybrid retrieval: semantic similarity + keyword filtering.
- **Retrieval:** Top 3–5 chunks per query, ranked by semantic similarity.
- **Learning hook:** After ingestion, run exploratory queries and examine the similarity scores. Which queries retrieve well? Which fail? This reveals how the embedding model represents (or collapses) philosophical concepts.

#### B. The Soul (Inference Engine)

**Stage 1 — System Prompt Baseline:**
- **Model:** `Llama 3.1 8B` (4-bit quantized via `llama.cpp`)
- **Why Llama over DeepSeek-R1-Distill:** Cleaner canvas. R1-Distill comes pre-flavored with its own reasoning patterns that will fight the Aristotelian structure. Llama 3.1 8B is a neutral base — whatever reasoning patterns emerge are from your prompt and RAG, not inherited behavior.
- **Inference params:** Start with `temperature=0.3`, `top_p=0.9` for crisp, didactic output. Adjust based on eval results.
- **System Prompt:** Encodes Aristotle's *method*, not his tone:
  1. Always define the key term before reasoning about it.
  2. Identify the genus and differentia of the subject.
  3. Reason from first principles using syllogistic structure.
  4. Appeal to telos (purpose/function) when evaluating anything.
  5. Cite specific works when making claims.
  6. Correct misconceptions directly — do not hedge.
- **Learning hook:** Track *where* the model breaks. Does it define terms but then abandon syllogistic structure? Does it maintain logic for 2 steps but collapse on step 3? These failure modes reveal how the 8B model represents logical chains internally.

**Stage 2 — LoRA Fine-Tuning (only if baseline is insufficient):**
- **Tool:** Unsloth (2x speed, 70% less memory)
- **Base:** Same Llama 3.1 8B
- **Decision Criteria:** Fine-tune only if the system prompt baseline fails the Phase 3 eval.
- **Learning hook:** Compare pre/post fine-tuning on the same eval set. If scores improve on Voice but not Method, you've shown LoRA shifts style but not reasoning. If Method improves too, that's a more interesting result.

#### C. The Body (CLI Application)

- **Language:** Python 3.10+
- **UI Library:** Rich (markdown rendering, styled panels for citations, spinners)
- **Debug mode:** `--debug` flag prints retrieved chunks, similarity scores, and the assembled prompt before generation. Essential for iterating on RAG quality.
- **Orchestration:** User → ChromaDB query → context injection → model inference → Rich output

---

## 4. Implementation Roadmap

### Phase 1: Data & The Memory

**Goal:** A working vector database that retrieves the right passages. First contact with how embeddings represent meaning.

- [ ] Download Aristotle's major works from Project Gutenberg (.txt)
- [ ] Clean: strip Gutenberg headers/footers/license blocks
- [ ] Parse natural structure: map each text's book/chapter/section hierarchy
- [ ] Write `chunker.py` that respects structural boundaries
- [ ] Add concept-level metadata tags per chunk during ingestion
- [ ] Initialize ChromaDB and ingest with full metadata
- [ ] **Validation tests:**
  - "What is courage?" → *Nicomachean Ethics* Book III (courage/rashness)
  - "What is the purpose of the state?" → *Politics* Book I
  - "What is the soul?" → confirms corpus boundaries (should retrieve *Metaphysics* or *Categories* — note if De Anima is needed later)
- [ ] **Exploration notebook:** Run 20+ queries and log which succeed/fail. Examine similarity scores. Where does the embedding model collapse distinct concepts into the same region? Where does it separate things that should be close?

**What this teaches you:** The geometry of semantic space. Whether cosine similarity between embedding vectors actually captures philosophical relationships, or whether "courage" and "the mean state concerning fear" end up far apart because the surface language is different.

### Phase 2: The Body (Baseline Prototype)

**Goal:** A working agent using only RAG + system prompt. Find the ceiling of prompt engineering for structured reasoning.

- [ ] Set up local inference with `llama-cpp-python` running Llama 3.1 8B (Q4_K_M)
- [ ] Write prompt assembler: system prompt + retrieved chunks + user query → formatted prompt
- [ ] Implement `--debug` mode
- [ ] Engineer the system prompt iteratively:
  - Start with the six method rules
  - Test against 20–30 diverse questions
  - **Keep a failure log:** For each question, note *where* the reasoning breaks (definition? syllogism step 2? telos application?) and *how* it breaks (contradicts itself? goes generic? drops the structure entirely?)
- [ ] Build Rich CLI: input loop, streaming output, citation panels
- [ ] Write `main.py` entry point
- [ ] **Milestone:** A usable agent grounded in Aristotle's texts. You now have a baseline and a detailed map of where the 8B model fails at structured reasoning.

**What this teaches you:** The limits of in-context learning. You'll discover whether an 8B model can maintain a 3-step syllogism with just a prompt, or whether it needs the structure baked into the weights. The failure log is the most valuable artifact of this phase — it tells you exactly what fine-tuning would need to fix.

### Phase 3: Evaluation Framework

**Goal:** Quantify reasoning quality. Build your own answer to "what does it mean for a model to reason correctly?"

- [ ] Create an eval set of 50 questions across categories:
  - **Factual recall** (10): "What are the intellectual virtues?" — tests RAG accuracy
  - **Single-step reasoning** (10): "Is courage a virtue?" — tests definition + one inference
  - **Multi-step reasoning** (10): "Is it possible for a tyrant to be happy?" — requires chaining: happiness → virtue → political life → tyranny. Tests where the syllogistic chain breaks.
  - **Pushback/correction** (10): "Isn't democracy the best system?" — tests whether the model can identify and correct a premise rather than just responding to it
  - **Extrapolation from principles** (10): "What would you say about social media?" — tests application of telos/virtue framework to novel domains
- [ ] Score each response on four dimensions:
  - **Grounding** (1–5): Does it cite or reference actual Aristotelian concepts from the retrieved text?
  - **Structure** (1–5): Does it follow the pattern: define → reason (syllogism) → conclude?
  - **Logical validity** (1–5): Are the syllogisms actually valid? (All A are B, X is A, therefore X is B — not All A are B, X is B, therefore X is A)
  - **Voice** (1–5): Does it sound like a philosopher-teacher, not an AI assistant?
- [ ] **Automated checks** where possible: regex-based detection of syllogistic markers ("All X are Y", "Therefore"), presence of definitions before arguments, absence of hedging phrases ("that's an interesting point", "it's worth considering")
- [ ] **Decision gate:** If average Structure < 3.5 or Logical validity < 3.0 → proceed to Phase 4. If Voice < 3.5 but reasoning is solid → fine-tune for style only (lighter intervention). If all ≥ 3.5 → ship it, prompt engineering won.

**What this teaches you:** How to formalize reasoning evaluation — a problem the entire field is grappling with. Your rubric is a microcosm of the benchmarking challenges in AI alignment research.

### Phase 4: The Soul (Fine-Tuning) — Conditional

**Goal:** Determine empirically whether LoRA can shift reasoning patterns or just surface style.

#### Dataset Generation

- [ ] Generate 1,500–2,000 training examples across three formats:
  - **Dialectical corrections** (40%): Student misconception → precise definition → error identification → syllogistic correction
  - **Expository teaching** (35%): Question → genus/differentia explanation → first principles reasoning → citation
  - **Applied reasoning** (25%): Modern scenario → principle identification → application via telos/virtue/four causes
- [ ] Generation process:
  1. For each corpus chunk, generate 2–3 Q&A pairs using Claude/GPT-4o (use the strongest model available — dataset quality is everything)
  2. Generation prompt enforces *structural* requirements: "Output must: (1) define the key term by genus and differentia, (2) include at least one valid syllogism, (3) conclude with reference to telos. No archaic thee/thou."
  3. **Quality filter:** 10% manual review. Rejection criteria: archaic language without real syllogistic structure ("costume Aristotle"), invalid syllogisms, undefined key terms. Target <10% rejection rate.
- [ ] Format as JSONL: `{"instruction": "...", "input": "...", "output": "..."}`

#### Training

- [ ] Environment: Google Colab (free T4) or local GPU + Unsloth
- [ ] LoRA config: rank 16–32, alpha 32–64, target attention + MLP layers
- [ ] Train 3–5 epochs, monitor loss curve
- [ ] Export LoRA adapters

#### The Actual Experiment

- [ ] Re-run the full Phase 3 eval on the fine-tuned model
- [ ] Compare dimension by dimension against the baseline:
  - **Voice improved, Structure/Logic unchanged** → LoRA shifted style, not reasoning. Interesting negative result.
  - **Voice + Structure improved, Logic unchanged** → LoRA taught format but not validity. The model learned to *look* like it's reasoning.
  - **All dimensions improved** → LoRA actually shifted reasoning patterns. Strong positive result.
  - **Minimal change across the board** → 1,500–2,000 pairs insufficient to move an 8B model. Dataset size or quality problem.
- [ ] Document findings. This is the most valuable output of the entire project — your own empirical data on what fine-tuning can and can't do.

**What this teaches you:** The central question — does LoRA change how a model thinks, or just how it talks? Your eval framework from Phase 3 gives you the tools to actually answer this instead of guessing.

---

## 5. What Makes It "Aristotelian" (Design Reference)

The north star for prompt engineering, dataset generation, and evaluation. The agent embodies these patterns:

| Pattern | Example | How to verify |
|---|---|---|
| **Definition first** | "First we must say what courage *is*. Courage is the mean between cowardice and rashness concerning things that inspire fear." | Response defines the key term before any reasoning occurs. |
| **Genus and differentia** | "Justice is a virtue (genus) that concerns fair distribution among citizens (differentia)." | Definition identifies the category and the distinguishing feature. |
| **Syllogistic structure** | "All virtues are means between extremes. Courage is a virtue. Therefore courage is a mean between extremes." | Contains major premise, minor premise, conclusion. Check logical validity. |
| **Appeal to telos** | "To know whether a knife is good, you must know what a knife is *for*. So too with man." | Evaluative claims reference purpose or function. |
| **Direct correction** | "You speak as if pleasure were the highest good. But consider: no one would choose to live with the mind of a child, even if guaranteed endless pleasure." | Identifies the user's error explicitly before correcting it. |
| **Refusal to hedge** | Not "that's an interesting perspective" but "this is incorrect, and here is why." | Absence of stock AI hedging phrases. |

---

## 6. Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Vector DB | ChromaDB |
| Embeddings | sentence-transformers |
| Inference | llama-cpp-python (GGUF quantized models) |
| Base model | Llama 3.1 8B (Q4_K_M) |
| Fine-tuning | Unsloth + LoRA (conditional) |
| CLI UI | Rich |
| Data loading | Custom scripts (no LangChain/LlamaIndex) |

**Hardware:**
- **Inference:** Any machine with ≥12GB RAM (M1/M2/M3 Mac, or PC with decent CPU). GPU optional.
- **Training (if needed):** Google Colab free tier (T4) or local RTX 3060+.

---

## 7. Directory Structure

```
aristotle-agent/
├── data/
│   ├── raw/                  # Original Gutenberg .txt files
│   ├── processed/            # Cleaned + structurally chunked JSONs
│   ├── training/             # Fine-tuning dataset (JSONL)
│   └── chroma_db/            # Persistent vector database
├── eval/
│   ├── questions.json        # 50-question eval set
│   ├── baseline_results.json # Phase 3 baseline scores
│   ├── finetuned_results.json
│   └── failure_log.md        # Phase 2 failure analysis (the most valuable doc)
├── notebooks/
│   └── embedding_explorer.ipynb  # Phase 1 similarity exploration
├── models/
│   └── adapters/             # LoRA weights (if fine-tuned)
├── src/
│   ├── ingest.py             # Text cleaning + ChromaDB loading
│   ├── chunker.py            # Structure-aware text chunking
│   ├── inference.py          # RAG retrieval + prompt assembly + generation
│   ├── train.py              # Unsloth LoRA training script
│   ├── evaluate.py           # Run eval set and score
│   └── ui.py                 # Rich CLI layout and styling
├── prompts/
│   └── system.txt            # System prompt (version controlled — track iterations)
├── requirements.txt
├── PLAN.md
└── main.py                   # Entry point (supports --debug flag)
```

---

## 8. Key Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Fine-tuning produces "costume Aristotle" (archaic surface, no reasoning) | Phase 3 eval separates Voice from Structure from Logic. You'll see exactly what changed. |
| 8B model can't maintain multi-step syllogisms | Phase 2 failure log documents exactly where it breaks. This *is* a finding, not a failure. Cap generation at ~500 tokens to reduce drift. |
| RAG retrieves irrelevant chunks | Concept-level metadata enables filtered retrieval. `--debug` mode exposes retrieval quality. Validation tests in Phase 1. |
| Scope creep | Phase 2 is shippable. Fine-tuning is conditional. The failure log and eval results are valuable outputs even if the agent itself is imperfect. |
| Dataset too small to shift behavior | Target 1,500–2,000 pairs. If it doesn't move the needle, that's an empirical finding worth documenting. |
| Embedding model collapses philosophical distinctions | Phase 1 exploration notebook catches this early. May need to experiment with different embedding models or add keyword-based retrieval as fallback. |

---

## 9. Definition of Done

The project succeeds if you can answer these questions with your own data:

1. **How well does cosine similarity in embedding space capture philosophical relationships?** (Phase 1)
2. **Where exactly does an 8B model break down when asked to maintain formal logical structure?** (Phase 2)
3. **Can you build a quantitative rubric that distinguishes "sounds like reasoning" from "is actually reasoning"?** (Phase 3)
4. **Does LoRA fine-tuning shift reasoning patterns or just surface style?** (Phase 4, if reached)

The agent is the vehicle. These answers are the destination.