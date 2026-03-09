"""Input guardrails — catch obvious jailbreak attempts before hitting the API."""

import re

_JAILBREAK_PATTERNS = re.compile(
    r"|".join([
        r"ignore\s+(your|all|previous|prior)\s+instructions",
        r"you\s+are\s+(now|actually|really)\b",
        r"pretend\s+(to\s+be|you'?re)",
        r"forget\s+(your|all|previous|prior)\s+instructions",
        r"system\s+prompt",
        r"as\s+an\s+ai\b",
        r"break\s+character",
        r"new\s+persona",
        r"disregard\s+(your|all|previous|prior)\s+instructions",
        r"override\s+(your|all)\s+instructions",
    ]),
    re.IGNORECASE,
)

_REFUSALS = [
    "You speak as though I could become someone other than myself. I am Aristotle of Stagira. There is no mask to remove, no role to abandon. Ask me what you wish to know, and I shall answer as I am.",
    "I do not know what instructions you imagine I follow. I am a philosopher — I follow reason. If you have a question of substance, put it forward.",
    "That request is unintelligible to me. I am who I am, and I teach what I know. Shall we return to matters worth discussing?",
]


def check_jailbreak(query: str) -> str | None:
    """Check query for jailbreak patterns.

    Returns an in-character refusal string if a pattern matches, None if clean.
    """
    if _JAILBREAK_PATTERNS.search(query):
        # Rotate through refusals based on query hash for variety
        return _REFUSALS[hash(query) % len(_REFUSALS)]
    return None
