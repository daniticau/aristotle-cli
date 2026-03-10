"""Structure-aware text chunking for Aristotle's Nicomachean Ethics."""

import json
import re
from pathlib import Path

from config import (
    CHUNK_MAX_WORDS,
    CHUNK_MIN_WORDS,
    CHUNKS_PATH,
    RAW_TEXT_PATH,
)

# ── Concept Tags ───────────────────────────────────────────────────────────
CONCEPT_KEYWORDS = {
    "virtue": ["virtue", "virtuous", "virtues", "arete"],
    "courage": ["courage", "courageous", "coward", "cowardice", "brave", "bravery", "fear"],
    "justice": ["justice", "just", "unjust", "injustice"],
    "happiness": ["happiness", "happy", "eudaimonia", "blessed", "blessedness"],
    "telos": ["end", "ends", "final", "sake", "purpose", "function", "telos"],
    "soul": ["soul", "souls", "rational", "irrational"],
    "mean": ["mean", "excess", "deficiency", "intermediate", "extremes"],
    "friendship": ["friendship", "friend", "friends", "love", "goodwill"],
    "pleasure": ["pleasure", "pain", "pleasant", "painful", "enjoyment"],
    "contemplation": ["contemplation", "contemplative", "theoretical", "speculative", "intellect"],
    "practical_wisdom": ["prudence", "prudent", "practical wisdom", "phronesis", "deliberation", "deliberate"],
    "habit": ["habit", "habits", "habituation", "accustom", "trained", "training"],
    "voluntary": ["voluntary", "involuntary", "choice", "choosing", "willingly"],
    "politics": ["political", "politics", "legislator", "state", "citizen", "citizens", "city"],
}

# ── Roman Numeral Helpers ──────────────────────────────────────────────────
_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
}


def roman_to_int(roman: str) -> int:
    """Convert a Roman numeral string (I-XV) to an integer."""
    roman = roman.strip().upper()
    if roman in _ROMAN_MAP:
        return _ROMAN_MAP[roman]
    raise ValueError(f"Unrecognized Roman numeral: {roman}")


# ── Gutenberg Processing ──────────────────────────────────────────────────

def strip_gutenberg_boilerplate(text: str) -> str:
    """Remove Project Gutenberg header and footer."""
    start_markers = ["*** START OF THE PROJECT GUTENBERG", "*** START OF THIS PROJECT GUTENBERG"]
    end_markers = ["*** END OF THE PROJECT GUTENBERG", "*** END OF THIS PROJECT GUTENBERG"]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_idx = text.index("\n", idx) + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    body = text[start_idx:end_idx].strip()

    # Strip trailing NOTES section (followed by "BOOK I" footnotes)
    notes_idx = re.search(r"\n\s*NOTES\s*\n\s*BOOK\s+I\b", body)
    if notes_idx:
        body = body[:notes_idx.start()].strip()

    return body


def parse_books(text: str) -> list[tuple[int, str]]:
    """Split text into books. Returns list of (book_number, book_text)."""
    # Match "BOOK I", "BOOK II", etc. on their own line
    pattern = re.compile(r"^BOOK\s+(I{1,3}|IV|V|VI{0,3}|IX|X)\s*$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    if not matches:
        raise ValueError("No BOOK headings found in text")

    books = []
    for i, match in enumerate(matches):
        book_num = roman_to_int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        books.append((book_num, text[start:end].strip()))

    return books


def parse_chapters(book_text: str) -> list[tuple[int, str]]:
    """Split a book's text into chapters. Returns list of (chapter_number, chapter_text)."""
    # Match chapter headings like "Chapter I." or "Chapter II."
    # Some editions use just roman numerals, others have periods
    pattern = re.compile(
        r"^(?:Chapter|CHAPTER)\s+(I{1,3}|IV|V|VI{0,3}|IX|X{0,3}I{0,3}V?)\s*\.?\s*$",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(book_text))

    if not matches:
        # No chapter divisions found — treat entire book as one chapter
        return [(1, book_text.strip())]

    chapters = []
    for i, match in enumerate(matches):
        chapter_num = roman_to_int(match.group(1))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(book_text)
        chapters.append((chapter_num, book_text[start:end].strip()))

    return chapters


def chunk_chapter(chapter_text: str) -> list[str]:
    """Split chapter text into chunks of CHUNK_MIN_WORDS-CHUNK_MAX_WORDS words at paragraph boundaries.

    Never splits mid-paragraph. If a single paragraph exceeds max, emit it as-is.
    """
    # Split on double newlines to get paragraphs
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", chapter_text) if p.strip()]

    chunks = []
    current_paragraphs = []
    current_word_count = 0

    for para in paragraphs:
        para_words = len(para.split())

        # If single paragraph exceeds max, emit current buffer then emit paragraph as-is
        if para_words > CHUNK_MAX_WORDS:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs))
                current_paragraphs = []
                current_word_count = 0
            chunks.append(para)
            continue

        # If adding this paragraph would exceed max, and we already have enough words, emit buffer
        if current_word_count + para_words > CHUNK_MAX_WORDS and current_word_count >= CHUNK_MIN_WORDS:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = []
            current_word_count = 0

        current_paragraphs.append(para)
        current_word_count += para_words

    # Emit remaining paragraphs
    if current_paragraphs:
        # If too small, merge with previous chunk if possible
        if current_word_count < CHUNK_MIN_WORDS and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current_paragraphs)
        else:
            chunks.append("\n\n".join(current_paragraphs))

    return chunks


def tag_chunk(text: str) -> list[str]:
    """Tag a chunk with concept keywords found in the text."""
    text_lower = text.lower()
    tags = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(concept)
    return sorted(tags)


def process_text(raw_path: Path = RAW_TEXT_PATH, output_path: Path = CHUNKS_PATH) -> list[dict]:
    """Full pipeline: read raw text → chunk → tag → save JSON."""
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text = strip_gutenberg_boilerplate(raw_text)
    books = parse_books(text)

    all_chunks = []
    for book_num, book_text in books:
        chapters = parse_chapters(book_text)
        for chapter_num, chapter_text in chapters:
            chunks = chunk_chapter(chapter_text)
            for idx, chunk_text in enumerate(chunks):
                chunk_id = f"nic_eth_b{book_num:02d}_c{chapter_num:02d}_{idx:03d}"
                word_count = len(chunk_text.split())
                tags = tag_chunk(chunk_text)

                all_chunks.append({
                    "id": chunk_id,
                    "text": chunk_text,
                    "work": "Nicomachean Ethics",
                    "book": book_num,
                    "chapter": chapter_num,
                    "chunk_index": idx,
                    "word_count": word_count,
                    "tags": tags,
                })

    # Save to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(all_chunks)} chunks from {len(books)} books")
    return all_chunks
