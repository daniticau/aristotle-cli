/**
 * AristotleAgent — owns the full pipeline: ask → retrieve → assemble → generate.
 */

import { readFileSync } from "node:fs";
import Anthropic from "@anthropic-ai/sdk";
import {
  ANTHROPIC_MODEL,
  MAX_TOKENS,
  MIN_RELEVANCE_SCORE,
  SYSTEM_PROMPT_PATH,
  TEMPERATURE,
} from "./config.js";
import { checkJailbreak } from "./guardrails.js";
import { retrieve, initRetriever, type RetrievedChunk } from "./retriever.js";

// ── Types ────────────────────────────────────────────────────────────────

export interface DebugInfo {
  chunks: RetrievedChunk[];
  prompt: Array<{ role: string; content: string }>;
}

export interface AgentResponse {
  text: string;
  debug: DebugInfo | null;
}

// ── Agent ────────────────────────────────────────────────────────────────

export class AristotleAgent {
  debug: boolean;
  private model: string;
  private client: Anthropic;
  private systemPrompt: string;

  constructor(debug = false) {
    this.debug = debug;
    this.model = ANTHROPIC_MODEL;
    this.client = new Anthropic();
    this.systemPrompt = readFileSync(SYSTEM_PROMPT_PATH, "utf-8").trim();
  }

  /** Eagerly initialize the retriever (load chunks + download model). */
  async ensureRetriever(): Promise<void> {
    await initRetriever();
  }

  private static readonly PREFILL = "In brief:";

  /** Build the message list with RAG context + assistant prefill. */
  private assemblePrompt(
    query: string,
    chunks: RetrievedChunk[],
  ): Array<{ role: "user" | "assistant"; content: string }> {
    const passages = chunks.map((chunk, i) => {
      const source = `${chunk.work}, Book ${chunk.book}, Part ${chunk.chapter}`;
      return `[Passage ${i + 1} — ${source}]\n${chunk.text}`;
    });

    const contextBlock = passages.join("\n\n");
    const userMessage =
      `The following passages from your own writings are provided for reference:\n\n` +
      `${contextBlock}\n\n` +
      `---\n\n` +
      `Question: ${query}`;

    return [
      { role: "user", content: userMessage },
      { role: "assistant", content: AristotleAgent.PREFILL },
    ];
  }

  /** Stream tokens from the Anthropic API. */
  async *generateStream(
    messages: Array<{ role: "user" | "assistant"; content: string }>,
  ): AsyncGenerator<string> {
    const stream = this.client.messages.stream({
      model: this.model,
      max_tokens: MAX_TOKENS,
      temperature: TEMPERATURE,
      system: this.systemPrompt,
      messages,
    });

    let first = true;
    for await (const event of stream) {
      if (
        event.type === "content_block_delta" &&
        event.delta.type === "text_delta"
      ) {
        let text = event.delta.text;
        if (first) {
          text = text.trimStart();
          first = false;
          if (!text) continue;
        }
        yield text;
      }
    }
  }

  /**
   * Full pipeline: guardrail → retrieve → assemble → generate.
   * Returns [AgentResponse, AsyncGenerator<string>].
   */
  async ask(
    query: string,
  ): Promise<[AgentResponse, AsyncGenerator<string>]> {
    // Layer 1: keyword pre-filter (no API call)
    const refusal = checkJailbreak(query);
    if (refusal !== null) {
      return [
        { text: "", debug: null },
        (async function* () {
          yield refusal;
        })(),
      ];
    }

    // Layer 2: retrieval score gating
    const chunks = await retrieve(query);
    const bestScore = chunks.length > 0
      ? Math.max(...chunks.map((c) => c.score))
      : 0;

    let messages: Array<{ role: "user" | "assistant"; content: string }>;

    if (bestScore < MIN_RELEVANCE_SCORE) {
      // Off-topic — send deflection to LLM without RAG context
      const deflection =
        "The student has asked about something entirely outside your " +
        "knowledge. You have never heard of such a thing. Tell them plainly " +
        "that this matter is unknown to you, and invite them to ask about " +
        "virtue, justice, the good life, or the other subjects you know.\n\n" +
        `Student's question: ${query}`;
      messages = [{ role: "user", content: deflection }];
    } else {
      messages = this.assemblePrompt(query, chunks);
    }

    let debugInfo: DebugInfo | null = null;
    if (this.debug) {
      debugInfo = {
        chunks,
        prompt: [
          { role: "system", content: this.systemPrompt },
          ...messages,
        ],
      };
    }

    const tokenGenerator = this.generateStream(messages);
    return [{ text: "", debug: debugInfo }, tokenGenerator];
  }
}
