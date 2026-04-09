import { afterEach, describe, expect, it, vi } from "vitest";

const CONFIG_ENV_KEYS = [
  "ARISTOTLE_EMBEDDING_MODEL",
  "ARISTOTLE_ANTHROPIC_MODEL",
  "ARISTOTLE_MAX_TOKENS",
  "ARISTOTLE_TEMPERATURE",
  "ARISTOTLE_TOP_K",
  "ARISTOTLE_MIN_RELEVANCE_SCORE",
] as const;

const ORIGINAL_ENV = { ...process.env };

function resetConfigEnv(): void {
  for (const key of CONFIG_ENV_KEYS) {
    delete process.env[key];
  }
}

function restoreOriginalEnv(): void {
  for (const key of Object.keys(process.env)) {
    if (!(key in ORIGINAL_ENV)) {
      delete process.env[key];
    }
  }

  for (const [key, value] of Object.entries(ORIGINAL_ENV)) {
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
}

async function loadConfigModule() {
  vi.resetModules();
  return import("../src/config.js");
}

afterEach(() => {
  restoreOriginalEnv();
  vi.resetModules();
});

describe("config", () => {
  it("uses default values when overrides are not provided", async () => {
    resetConfigEnv();

    const config = await loadConfigModule();

    expect(config.EMBEDDING_MODEL).toBe("Xenova/all-MiniLM-L6-v2");
    expect(config.ANTHROPIC_MODEL).toBe("claude-haiku-4-5-20251001");
    expect(config.MAX_TOKENS).toBe(150);
    expect(config.TEMPERATURE).toBe(0.3);
    expect(config.TOP_K).toBe(5);
    expect(config.MIN_RELEVANCE_SCORE).toBe(0.25);
  });

  it("parses environment overrides", async () => {
    resetConfigEnv();
    process.env.ARISTOTLE_EMBEDDING_MODEL = "custom/embedder";
    process.env.ARISTOTLE_ANTHROPIC_MODEL = "custom-model";
    process.env.ARISTOTLE_MAX_TOKENS = "256";
    process.env.ARISTOTLE_TEMPERATURE = "0.6";
    process.env.ARISTOTLE_TOP_K = "7";
    process.env.ARISTOTLE_MIN_RELEVANCE_SCORE = "0.4";

    const config = await loadConfigModule();

    expect(config.EMBEDDING_MODEL).toBe("custom/embedder");
    expect(config.ANTHROPIC_MODEL).toBe("custom-model");
    expect(config.MAX_TOKENS).toBe(256);
    expect(config.TEMPERATURE).toBe(0.6);
    expect(config.TOP_K).toBe(7);
    expect(config.MIN_RELEVANCE_SCORE).toBe(0.4);
  });

  it("fails fast on invalid numeric overrides", async () => {
    resetConfigEnv();
    process.env.ARISTOTLE_MIN_RELEVANCE_SCORE = "2";

    await expect(loadConfigModule()).rejects.toThrow(
      /ARISTOTLE_MIN_RELEVANCE_SCORE/,
    );
  });
});
