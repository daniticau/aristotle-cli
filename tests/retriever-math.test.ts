import { describe, expect, it } from "vitest";
import { calculateVectorNorm, cosineSimilarity } from "../src/retriever.js";

describe("retriever vector math", () => {
  it("computes Euclidean norms", () => {
    expect(calculateVectorNorm([3, 4])).toBe(5);
  });

  it("returns zero similarity when either vector has zero magnitude", () => {
    expect(cosineSimilarity([0, 0], [1, 2])).toBe(0);
    expect(cosineSimilarity([1, 2], [0, 0])).toBe(0);
  });

  it("throws when vectors have different dimensions", () => {
    expect(() => cosineSimilarity([1, 2], [1])).toThrow(
      /Embedding dimension mismatch/,
    );
  });
});
