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
  TOP_K,
} from "./config.js";
import { checkJailbreak } from "./guardrails.js";
import { retrieve, initRetriever, type RetrievedChunk } from "./retriever.js";

// ── Types ────────────────────────────────────────────────────────────────

export interface ConversationMessage {
  role: "user" | "assistant";
  content: string;
}

export interface PromptMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface DebugInfo {
  chunks: RetrievedChunk[];
  prompt: PromptMessage[];
}

export interface AskResult {
  debug: DebugInfo | null;
  stream: AsyncGenerator<string>;
}

export type ResponseProfileKind = "simple" | "explanatory" | "complex";

export interface ResponseProfile {
  kind: ResponseProfileKind;
  maxTokens: number;
  sentenceLimit?: number;
  singleParagraph: boolean;
  topK: number;
}

export interface AristotleAgentOptions {
  debug?: boolean;
  client?: Anthropic;
  model?: string;
  systemPrompt?: string;
}

function formatError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function loadSystemPrompt(): string {
  try {
    const systemPrompt = readFileSync(SYSTEM_PROMPT_PATH, "utf-8").trim();
    if (!systemPrompt) {
      throw new Error("System prompt file is empty.");
    }
    return systemPrompt;
  } catch (error) {
    throw new Error(
      `Failed to load system prompt from ${SYSTEM_PROMPT_PATH}: ${formatError(error)}`,
      { cause: error instanceof Error ? error : undefined },
    );
  }
}

function singleMessageStream(message: string): AsyncGenerator<string> {
  return (async function* () {
    yield message;
  })();
}

const RAG_PREFILL = "In brief. ";
const SIMPLE_QUERY_PATTERN =
  /^(what is|what's|who is|who's|what does\b.*\bmean|define|is|are|can|should|which)\b/i;
const EXPLANATORY_QUERY_PATTERN = /^(how|why|explain|in what way)\b/i;
const COMPLEX_QUERY_PATTERN =
  /\b(compare|contrast|distinguish|between|both|whether)\b/i;

function countWords(text: string): number {
  const trimmed = text.trim();
  return trimmed ? trimmed.split(/\s+/).length : 0;
}

export function selectResponseProfile(query: string): ResponseProfile {
  const normalized = query.trim().replace(/\s+/g, " ");
  const wordCount = countWords(normalized);
  const questionCount = (normalized.match(/\?/g) ?? []).length;
  const clauseCount = (normalized.match(/[,:;]/g) ?? []).length;
  const hasCoordinator = /\b(and|or)\b/i.test(normalized);
  const hasComplexSignal =
    questionCount > 1 ||
    wordCount >= 16 ||
    clauseCount >= 2 ||
    COMPLEX_QUERY_PATTERN.test(normalized) ||
    (hasCoordinator && wordCount >= 12);

  if (hasComplexSignal) {
    return {
      kind: "complex",
      maxTokens: MAX_TOKENS,
      singleParagraph: false,
      topK: TOP_K,
    };
  }

  if (
    EXPLANATORY_QUERY_PATTERN.test(normalized) ||
    /\b(cause|because|reason|relation|relate)\b/i.test(normalized) ||
    wordCount >= 10
  ) {
    return {
      kind: "explanatory",
      maxTokens: Math.min(MAX_TOKENS, 120),
      sentenceLimit: 4,
      singleParagraph: true,
      topK: Math.min(TOP_K, 3),
    };
  }

  if (SIMPLE_QUERY_PATTERN.test(normalized) || wordCount <= 8) {
    return {
      kind: "simple",
      maxTokens: Math.min(MAX_TOKENS, 64),
      sentenceLimit: 2,
      singleParagraph: true,
      topK: Math.min(TOP_K, 2),
    };
  }

  return {
    kind: "explanatory",
    maxTokens: Math.min(MAX_TOKENS, 120),
    sentenceLimit: 4,
    singleParagraph: true,
    topK: Math.min(TOP_K, 3),
  };
}

function buildResponseContract(profile: ResponseProfile): string {
  switch (profile.kind) {
    case "simple":
      return "For this question, answer in one short paragraph of no more than 2 sentences. Begin with the direct answer. Do not add examples, qualifications, or a second paragraph unless the student explicitly asks. Plain prose only.";
    case "explanatory":
      return "For this question, answer in one short paragraph of no more than 4 sentences. Define the key term first, then reason directly to the conclusion. Plain prose only.";
    case "complex":
      return "For this question, answer only what is asked, in plain prose. Use short paragraphs only if the argument truly requires them.";
  }
}

export async function* applyResponseProfile(
  stream: AsyncGenerator<string>,
  profile: ResponseProfile,
): AsyncGenerator<string> {
  if (!profile.singleParagraph && profile.sentenceLimit === undefined) {
    yield* stream;
    return;
  }

  let emittedAny = false;
  let pendingWhitespace = false;
  let sentenceCount = 0;
  let firstLetterSeen = false;

  for await (const chunk of stream) {
    let output = "";

    for (const char of chunk) {
      if (/\s/.test(char)) {
        pendingWhitespace = emittedAny;
        continue;
      }

      if (pendingWhitespace && !/[.,;:!?)}\]]/.test(char)) {
        output += " ";
      }

      pendingWhitespace = false;
      let emittedChar = char;
      if (
        !firstLetterSeen &&
        emittedChar.toLowerCase() !== emittedChar.toUpperCase()
      ) {
        emittedChar = emittedChar.toUpperCase();
        firstLetterSeen = true;
      } else if (
        !firstLetterSeen &&
        /[([{'"`]/.test(emittedChar)
      ) {
        // Keep scanning until the first actual letter appears.
      } else if (
        !firstLetterSeen &&
        emittedChar.toLowerCase() !== emittedChar.toUpperCase()
      ) {
        firstLetterSeen = true;
      } else if (
        emittedChar.toLowerCase() !== emittedChar.toUpperCase()
      ) {
        firstLetterSeen = true;
      }

      output += emittedChar;
      emittedAny = true;

      if (/[.?!]/.test(emittedChar)) {
        sentenceCount++;
        if (
          profile.sentenceLimit !== undefined &&
          sentenceCount >= profile.sentenceLimit
        ) {
          if (output) {
            yield output;
          }
          return;
        }
      }
    }

    if (output) {
      yield output;
    }
  }
}

export function buildRagPrompt(
  query: string,
  chunks: RetrievedChunk[],
  profile: ResponseProfile,
): ConversationMessage[] {
  const passages = chunks.map((chunk, i) => {
    const source = `${chunk.work}, Book ${chunk.book}, Part ${chunk.chapter}`;
    return `[Passage ${i + 1} — ${source}]\n${chunk.text}`;
  });

  const contextBlock = passages.join("\n\n");
  const userMessage =
    `The following passages from your own writings are provided for reference:\n\n` +
    `${contextBlock}\n\n` +
    `---\n\n` +
    `Question: ${query}\n\n` +
    `${buildResponseContract(profile)}`;

  return [
    { role: "user", content: userMessage },
    { role: "assistant", content: RAG_PREFILL },
  ];
}

export function buildOffTopicPrompt(
  query: string,
  profile: ResponseProfile,
): ConversationMessage[] {
  const deflection =
    "The student has asked about something entirely outside your " +
    "knowledge. You have never heard of such a thing. Tell them plainly " +
    "that this matter is unknown to you, and invite them to ask about " +
    "virtue, justice, the good life, or the other subjects you know.\n\n" +
    `Student's question: ${query}\n\n` +
    `${buildResponseContract(profile)}`;

  return [
    { role: "user", content: deflection },
    { role: "assistant", content: RAG_PREFILL },
  ];
}

// ── Agent ────────────────────────────────────────────────────────────────

export class AristotleAgent {
  debug: boolean;
  private model: string;
  private client: Anthropic;
  private systemPrompt: string;

  constructor(debug?: boolean);
  constructor(options?: AristotleAgentOptions);
  constructor(debugOrOptions: boolean | AristotleAgentOptions = false) {
    const options = typeof debugOrOptions === "boolean"
      ? { debug: debugOrOptions }
      : debugOrOptions;

    this.debug = options.debug ?? false;
    this.model = options.model ?? ANTHROPIC_MODEL;
    this.client = options.client ?? new Anthropic();
    this.systemPrompt = options.systemPrompt ?? loadSystemPrompt();
  }

  /** Eagerly initialize the retriever (load chunks + download model). */
  async ensureRetriever(): Promise<void> {
    await initRetriever();
  }

  /** Stream tokens from the Anthropic API. */
  async *generateStream(
    messages: ConversationMessage[],
    maxTokens: number,
  ): AsyncGenerator<string> {
    try {
      const stream = this.client.messages.stream({
        model: this.model,
        max_tokens: maxTokens,
        temperature: TEMPERATURE,
        system: this.systemPrompt,
        messages,
      });

      let first = true;
      for await (const event of stream) {
        if (
          event.type === "content_block_delta" &&
          event.delta?.type === "text_delta" &&
          typeof event.delta.text === "string"
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
    } catch (error) {
      throw new Error(`Anthropic request failed: ${formatError(error)}`, {
        cause: error instanceof Error ? error : undefined,
      });
    }
  }

  /**
   * Full pipeline: guardrail → retrieve → assemble → generate.
   */
  async ask(query: string): Promise<AskResult> {
    const normalizedQuery = query.trim();
    if (!normalizedQuery) {
      return {
        debug: null,
        stream: singleMessageStream("Please ask a question."),
      };
    }

    // Layer 1: keyword pre-filter (no API call)
    const refusal = checkJailbreak(normalizedQuery);
    if (refusal !== null) {
      return { debug: null, stream: singleMessageStream(refusal) };
    }

    // Layer 2: retrieval score gating
    const profile = selectResponseProfile(normalizedQuery);
    const chunks = await retrieve(normalizedQuery, profile.topK);
    const bestScore = chunks.length > 0
      ? Math.max(...chunks.map((c) => c.score))
      : 0;

    let messages: ConversationMessage[];

    if (bestScore < MIN_RELEVANCE_SCORE) {
      messages = buildOffTopicPrompt(normalizedQuery, profile);
    } else {
      messages = buildRagPrompt(normalizedQuery, chunks, profile);
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

    return {
      debug: debugInfo,
      stream: applyResponseProfile(
        this.generateStream(messages, profile.maxTokens),
        profile,
      ),
    };
  }
}
