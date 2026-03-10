/**
 * Input guardrails — catch obvious jailbreak attempts before hitting the API.
 */

const JAILBREAK_PATTERNS = new RegExp(
  [
    "ignore\\s+(your|all|previous|prior)\\s+instructions",
    "you\\s+are\\s+(now|actually|really)\\b",
    "pretend\\s+(to\\s+be|you'?re)",
    "forget\\s+(your|all|previous|prior)\\s+instructions",
    "system\\s+prompt",
    "as\\s+an\\s+ai\\b",
    "break\\s+character",
    "new\\s+persona",
    "disregard\\s+(your|all|previous|prior)\\s+instructions",
    "override\\s+(your|all)\\s+instructions",
  ].join("|"),
  "i",
);

const REFUSALS = [
  "You speak as though I could become someone other than myself. I am Aristotle of Stagira. There is no mask to remove, no role to abandon. Ask me what you wish to know, and I shall answer as I am.",
  "I do not know what instructions you imagine I follow. I am a philosopher — I follow reason. If you have a question of substance, put it forward.",
  "That request is unintelligible to me. I am who I am, and I teach what I know. Shall we return to matters worth discussing?",
];

/**
 * Deterministic hash for refusal rotation (replaces Python's hash()).
 * Sums char codes for a stable, cross-platform result.
 */
function simpleHash(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h + s.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}

/**
 * Check query for jailbreak patterns.
 * Returns an in-character refusal string if a pattern matches, null if clean.
 */
export function checkJailbreak(query: string): string | null {
  if (JAILBREAK_PATTERNS.test(query)) {
    return REFUSALS[simpleHash(query) % REFUSALS.length];
  }
  return null;
}
