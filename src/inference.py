"""Core RAG pipeline: retrieve → assemble prompt → generate response."""

from dataclasses import dataclass
from typing import Generator

import anthropic
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from config import (
    ANTHROPIC_MODEL,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_TOKENS,
    MIN_RELEVANCE_SCORE,
    SYSTEM_PROMPT_PATH,
    TEMPERATURE,
    TOP_K,
)
from src.guardrails import check_jailbreak


# ── Data Classes ───────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """A passage retrieved from the vector store."""
    id: str
    text: str
    book: int
    chapter: int
    score: float
    tags: list[str]


@dataclass
class DebugInfo:
    """Debug information from a query."""
    chunks: list[RetrievedChunk]
    prompt: list[dict]


@dataclass
class AgentResponse:
    """Full response from the agent pipeline."""
    text: str
    debug: DebugInfo | None = None


# ── Agent ──────────────────────────────────────────────────────────────────

class AristotleAgent:
    """RAG agent that retrieves passages and generates Aristotelian responses."""

    def __init__(self, debug: bool = False):
        self.debug = debug
        self.model = ANTHROPIC_MODEL
        self.client = anthropic.Anthropic()
        self._retriever_ready = False

        # Load system prompt
        with open(SYSTEM_PROMPT_PATH, "r", encoding="utf-8") as f:
            self.system_prompt = f.read().strip()

    def _ensure_retriever(self) -> None:
        """Lazy-init ChromaDB client and collection on first query."""
        if self._retriever_ready:
            return

        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

        embed_fn = SentenceTransformerEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            device=device,
        )
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = client.get_collection(
                name=COLLECTION_NAME,
                embedding_function=embed_fn,
            )
        except Exception:
            raise RuntimeError(
                f"Collection '{COLLECTION_NAME}' not found.\n"
                f"Run ingestion first: python -m src.ingest --reset"
            )
        self._retriever_ready = True

    def retrieve(self, query: str, n_results: int = TOP_K) -> list[RetrievedChunk]:
        """Query ChromaDB and return retrieved chunks with similarity scores."""
        self._ensure_retriever()
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1 - dist  # cosine distance → similarity
            tags = meta.get("tags", "").split(",") if meta.get("tags") else []

            chunks.append(RetrievedChunk(
                id=meta.get("id", ""),
                text=doc,
                book=meta["book"],
                chapter=meta["chapter"],
                score=score,
                tags=tags,
            ))

        return chunks

    def assemble_prompt(self, query: str, chunks: list[RetrievedChunk]) -> list[dict]:
        """Build the user message with RAG context. System prompt sent separately."""
        passages = []
        for i, chunk in enumerate(chunks, 1):
            source = f"Book {chunk.book}, Chapter {chunk.chapter}"
            passages.append(f"[Passage {i} — {source}]\n{chunk.text}")

        context_block = "\n\n".join(passages)

        user_message = (
            f"The following passages from your Nicomachean Ethics are provided for reference:\n\n"
            f"{context_block}\n\n"
            f"---\n\n"
            f"Question: {query}"
        )

        messages = [{"role": "user", "content": user_message}]
        return messages

    def generate(self, messages: list[dict], stream: bool = True):
        """Generate a response using the Anthropic API.

        If stream=True, returns a generator yielding token strings.
        If stream=False, returns the complete response string.
        """
        if stream:
            return self._generate_stream(messages)
        else:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                system=self.system_prompt,
                messages=messages,
            )
            return response.content[0].text

    def _generate_stream(self, messages: list[dict]) -> Generator[str, None, None]:
        """Stream tokens from the Anthropic API."""
        with self.client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            system=self.system_prompt,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                yield text

    def ask(self, query: str, stream: bool = True) -> AgentResponse:
        """Full pipeline: guardrail → retrieve → assemble → generate.

        If stream=True, the text field will be empty and you must consume
        the generator returned by generate() separately.
        """
        # Layer 3: keyword pre-filter (no API call)
        refusal = check_jailbreak(query)
        if refusal is not None:
            if stream:
                return AgentResponse(text=""), self._yield_text(refusal)
            return AgentResponse(text=refusal)

        # Layer 2: retrieval score gating
        chunks = self.retrieve(query)
        best_score = max(c.score for c in chunks) if chunks else 0.0

        if best_score < MIN_RELEVANCE_SCORE:
            # Off-topic — send deflection to LLM without RAG context
            deflection = (
                "The student has asked about something entirely outside your "
                "knowledge. You have never heard of such a thing. Tell them plainly "
                "that this matter is unknown to you, and invite them to ask about "
                "virtue, justice, the good life, or the other subjects you know.\n\n"
                f"Student's question: {query}"
            )
            messages = [{"role": "user", "content": deflection}]
        else:
            messages = self.assemble_prompt(query, chunks)

        debug_info = None
        if self.debug:
            debug_messages = [
                {"role": "system", "content": self.system_prompt},
                *messages,
            ]
            debug_info = DebugInfo(chunks=chunks, prompt=debug_messages)

        if stream:
            token_generator = self.generate(messages, stream=True)
            return AgentResponse(text="", debug=debug_info), token_generator
        else:
            text = self.generate(messages, stream=False)
            return AgentResponse(text=text, debug=debug_info)

    @staticmethod
    def _yield_text(text: str) -> Generator[str, None, None]:
        """Yield a complete string as a single token for streaming compatibility."""
        yield text
