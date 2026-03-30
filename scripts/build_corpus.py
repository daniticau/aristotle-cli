# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "sentence-transformers>=3.0",
#   "requests>=2.31",
# ]
# ///
"""
Build the complete Aristotle corpus for aristotle-cli.

Downloads texts from MIT Internet Classics Archive, chunks at paragraph
boundaries (200-600 words), embeds with all-MiniLM-L6-v2, and writes
data/chunks_with_embeddings.json.

Usage:
    uv run scripts/build_corpus.py
"""

import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────

CHUNK_MIN_WORDS = 200
CHUNK_MAX_WORDS = 600
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "chunks_with_embeddings.json"

MIT_BASE = "https://classics.mit.edu/Aristotle"


# ── Text Registry ────────────────────────────────────────────────────────

@dataclass
class WorkConfig:
    """Configuration for one Aristotle text."""
    slug: str             # Short ID used in chunk IDs
    title: str            # Human-readable title
    mit_file: str         # Filename on MIT Classics (e.g., "metaphysics.mb.txt")
    # Structure pattern overrides (auto-detected if None)
    book_pattern: str | None = None      # Regex for book headings
    part_pattern: str | None = None      # Regex for part/chapter headings
    # Alternative: use Project Gutenberg instead of MIT
    gutenberg_id: int | None = None      # Gutenberg ebook ID (overrides mit_file)


WORKS: list[WorkConfig] = [
    # ── Ethics & Politics ────────────────────────────────────────────────
    WorkConfig(
        slug="nic_eth",
        title="Nicomachean Ethics",
        mit_file="nicomachaen.mb.txt",
        part_pattern=r"^\s*(\d+)\s*$",  # Bare numbers: "1", "2", etc.
    ),
    WorkConfig(
        slug="politics",
        title="Politics",
        mit_file="politics.mb.txt",
        gutenberg_id=6762,  # MIT mb.txt is truncated; Gutenberg has all 8 books
        part_pattern=r"^CHAPTER\s+([IVXLC]+)\s*$",
    ),
    WorkConfig(slug="ath_const", title="Athenian Constitution", mit_file="athenian_const.mb.txt"),

    # ── Logic (Organon) ──────────────────────────────────────────────────
    WorkConfig(slug="categories", title="Categories", mit_file="categories.mb.txt"),
    WorkConfig(slug="on_interp", title="On Interpretation", mit_file="interpretation.mb.txt"),
    WorkConfig(slug="prior_analytics", title="Prior Analytics", mit_file="prior.mb.txt"),
    WorkConfig(slug="posterior_analytics", title="Posterior Analytics", mit_file="posterior.mb.txt"),
    WorkConfig(slug="topics", title="Topics", mit_file="topics.mb.txt"),
    WorkConfig(slug="soph_refut", title="On Sophistical Refutations", mit_file="sophist_refut.mb.txt"),

    # ── Metaphysics ──────────────────────────────────────────────────────
    WorkConfig(slug="metaphysics", title="Metaphysics", mit_file="metaphysics.mb.txt"),

    # ── Natural Philosophy ───────────────────────────────────────────────
    WorkConfig(slug="physics", title="Physics", mit_file="physics.mb.txt"),
    WorkConfig(slug="on_heavens", title="On the Heavens", mit_file="heavens.mb.txt"),
    WorkConfig(slug="gen_corrupt", title="On Generation and Corruption", mit_file="gener_corr.mb.txt"),
    WorkConfig(slug="meteorology", title="Meteorology", mit_file="meteorology.mb.txt"),

    # ── Psychology ───────────────────────────────────────────────────────
    WorkConfig(slug="on_soul", title="On the Soul", mit_file="soul.mb.txt"),
    WorkConfig(slug="on_memory", title="On Memory and Reminiscence", mit_file="memory.1b.txt"),

    # ── Rhetoric & Poetics ───────────────────────────────────────────────
    WorkConfig(slug="rhetoric", title="Rhetoric", mit_file="rhetoric.mb.txt"),
    WorkConfig(
        slug="poetics",
        title="Poetics",
        mit_file="poetics.mb.txt",
    ),
]


# ── Concept Tags ─────────────────────────────────────────────────────────

CONCEPT_KEYWORDS: dict[str, list[str]] = {
    # Original (Ethics)
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
    # Logic
    "logic": ["syllogism", "syllogisms", "premise", "premises", "conclusion", "predicate", "predication"],
    "definition": ["definition", "genus", "species", "differentia", "essence"],
    # Metaphysics
    "substance": ["substance", "substances", "ousia", "substratum"],
    "form_matter": ["form", "matter", "hylomorphism", "actuality", "potentiality", "potency"],
    "causation": ["cause", "causes", "efficient", "material cause", "formal cause", "final cause"],
    # Rhetoric & Poetics
    "rhetoric": ["rhetoric", "rhetorical", "persuasion", "persuade", "orator", "oratory"],
    "poetry": ["tragedy", "tragic", "comedy", "epic", "imitation", "mimesis", "catharsis", "plot"],
    # Natural philosophy
    "nature": ["nature", "natural", "physis", "organic"],
    "motion": ["motion", "movement", "change", "rest", "place", "void", "time", "continuous"],
    "generation": ["generation", "corruption", "coming-to-be", "passing-away", "destruction"],
}


# ── English number words (for Politics) ──────────────────────────────────

_ENGLISH_NUMS = {
    "ONE": 1, "TWO": 2, "THREE": 3, "FOUR": 4, "FIVE": 5,
    "SIX": 6, "SEVEN": 7, "EIGHT": 8, "NINE": 9, "TEN": 10,
    "ELEVEN": 11, "TWELVE": 12, "THIRTEEN": 13, "FOURTEEN": 14, "FIFTEEN": 15,
}

_ROMAN_MAP = {
    "I": 1, "II": 2, "III": 3, "IV": 4, "V": 5,
    "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10,
    "XI": 11, "XII": 12, "XIII": 13, "XIV": 14, "XV": 15,
    "XVI": 16, "XVII": 17, "XVIII": 18, "XIX": 19, "XX": 20,
    "XXI": 21, "XXII": 22, "XXIII": 23, "XXIV": 24, "XXV": 25,
    "XXVI": 26, "XXVII": 27, "XXVIII": 28, "XXIX": 29, "XXX": 30,
}


def parse_heading_number(text: str) -> int:
    """Parse a heading number from Roman numeral, English word, or Arabic numeral."""
    text = text.strip().upper()
    if text in _ROMAN_MAP:
        return _ROMAN_MAP[text]
    if text in _ENGLISH_NUMS:
        return _ENGLISH_NUMS[text]
    if text.isdigit():
        return int(text)
    raise ValueError(f"Cannot parse heading number: {text!r}")


# ── Text Download & Cleanup ──────────────────────────────────────────────

def download_text(work: WorkConfig) -> str:
    """Download a text from MIT Internet Classics Archive or Project Gutenberg."""
    if work.gutenberg_id:
        url = f"https://www.gutenberg.org/cache/epub/{work.gutenberg_id}/pg{work.gutenberg_id}.txt"
    else:
        url = f"{MIT_BASE}/{work.mit_file}"
    print(f"  Downloading {work.title}...", end=" ", flush=True)

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    text = resp.text
    print(f"({len(text):,} chars)")
    return text


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

    return text[start_idx:end_idx].strip()


def strip_mit_boilerplate(text: str) -> str:
    """Remove MIT Internet Classics Archive header and footer."""
    # Some .1b.txt files have HTML wrappers — strip them first
    text = re.sub(r"<[^>]+>", "", text)

    # Strip header: everything up to and including the dashed separator line
    # Header format: "Provided by...\n\nTitle\nBy Aristotle\n...\n------"
    dash_match = re.search(r"\n-{5,}\n", text)
    if dash_match:
        text = text[dash_match.end():]

    # Strip footer: "THE END" and everything after
    end_match = re.search(r"\n\s*THE\s+END\s*\n", text, re.IGNORECASE)
    if end_match:
        text = text[:end_match.start()]

    # Remove stray leading/trailing quotation marks that MIT adds around passages
    text = re.sub(r'^\s*"\s*', "", text)
    text = re.sub(r'\s*"\s*$', "", text)

    return text.strip()


def strip_boilerplate(text: str, work: WorkConfig) -> str:
    """Strip boilerplate based on the text source."""
    if work.gutenberg_id:
        return strip_gutenberg_boilerplate(text)
    return strip_mit_boilerplate(text)


# ── Structure Parsing ────────────────────────────────────────────────────

@dataclass
class Section:
    """A structural unit (book, section, or the whole text)."""
    number: int
    parts: list["Part"] = field(default_factory=list)


@dataclass
class Part:
    """A sub-unit (part, chapter) within a section."""
    number: int
    text: str = ""


def detect_and_parse_structure(text: str, work: WorkConfig) -> list[Section]:
    """Parse the text into a hierarchy of sections and parts."""
    # Step 1: Detect book-level structure
    book_re = work.book_pattern or r"^BOOK\s+([IVXLC]+)\s*$"
    book_pattern = re.compile(book_re, re.MULTILINE)
    book_matches = list(book_pattern.finditer(text))

    # Also try SECTION pattern if no books found
    if not book_matches:
        section_pattern = re.compile(r"^SECTION\s+(\d+)\s*$", re.MULTILINE)
        book_matches = list(section_pattern.finditer(text))

    # Step 2: Detect part-level structure
    part_re = work.part_pattern or r"^Part\s+(\d+|[IVXLC]+)\s*$"
    part_pattern = re.compile(part_re, re.MULTILINE)

    if book_matches:
        # Split into books/sections first
        sections = []
        for i, match in enumerate(book_matches):
            num = parse_heading_number(match.group(1))
            start = match.end()
            end = book_matches[i + 1].start() if i + 1 < len(book_matches) else len(text)
            book_text = text[start:end].strip()
            section = Section(number=num)

            # Find parts within this book
            part_matches = list(part_pattern.finditer(book_text))
            if part_matches:
                for j, pm in enumerate(part_matches):
                    pnum = parse_heading_number(pm.group(1))
                    pstart = pm.end()
                    pend = part_matches[j + 1].start() if j + 1 < len(part_matches) else len(book_text)
                    part_text = book_text[pstart:pend].strip()
                    # Clean quotation marks
                    part_text = re.sub(r'^\s*"\s*', "", part_text)
                    part_text = re.sub(r'\s*"\s*$', "", part_text)
                    if part_text:
                        section.parts.append(Part(number=pnum, text=part_text))
            else:
                # No parts — treat whole book as one part
                book_text = re.sub(r'^\s*"\s*', "", book_text)
                book_text = re.sub(r'\s*"\s*$', "", book_text)
                if book_text:
                    section.parts.append(Part(number=1, text=book_text))

            if section.parts:
                sections.append(section)
        return sections

    # No books or sections — look for parts only
    part_matches = list(part_pattern.finditer(text))
    if part_matches:
        section = Section(number=1)
        for j, pm in enumerate(part_matches):
            pnum = parse_heading_number(pm.group(1))
            pstart = pm.end()
            pend = part_matches[j + 1].start() if j + 1 < len(part_matches) else len(text)
            part_text = text[pstart:pend].strip()
            part_text = re.sub(r'^\s*"\s*', "", part_text)
            part_text = re.sub(r'\s*"\s*$', "", part_text)
            if part_text:
                section.parts.append(Part(number=pnum, text=part_text))
        return [section] if section.parts else []

    # Fallback: whole text is one section, one part
    if text.strip():
        return [Section(number=1, parts=[Part(number=1, text=text.strip())])]
    return []


# ── Chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str) -> list[str]:
    """Split text into chunks of CHUNK_MIN_WORDS-CHUNK_MAX_WORDS words at paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    chunks: list[str] = []
    current_paras: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())

        if para_words > CHUNK_MAX_WORDS:
            if current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_words = 0
            chunks.append(para)
            continue

        if current_words + para_words > CHUNK_MAX_WORDS and current_words >= CHUNK_MIN_WORDS:
            chunks.append("\n\n".join(current_paras))
            current_paras = []
            current_words = 0

        current_paras.append(para)
        current_words += para_words

    if current_paras:
        if current_words < CHUNK_MIN_WORDS and chunks:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current_paras)
        else:
            chunks.append("\n\n".join(current_paras))

    return chunks


def tag_chunk(text: str) -> list[str]:
    """Tag a chunk with concept keywords found in the text."""
    text_lower = text.lower()
    tags = []
    for concept, keywords in CONCEPT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            tags.append(concept)
    return sorted(tags)


# ── Main Pipeline ────────────────────────────────────────────────────────

def process_work(work: WorkConfig) -> list[dict]:
    """Download, parse, chunk, and tag one work. Returns chunk dicts (without embeddings)."""
    raw = download_text(work)
    text = strip_boilerplate(raw, work)
    sections = detect_and_parse_structure(text, work)

    if not sections:
        print(f"  WARNING: No structure found for {work.title}, skipping")
        return []

    chunks = []
    for section in sections:
        for part in section.parts:
            text_chunks = chunk_text(part.text)
            for idx, chunk_text_str in enumerate(text_chunks):
                # Build chunk ID
                if len(sections) > 1 or section.number > 1:
                    chunk_id = f"{work.slug}_b{section.number:02d}_p{part.number:02d}_{idx:03d}"
                else:
                    chunk_id = f"{work.slug}_p{part.number:02d}_{idx:03d}"

                chunks.append({
                    "id": chunk_id,
                    "text": chunk_text_str,
                    "work": work.title,
                    "book": section.number,
                    "chapter": part.number,
                    "tags": tag_chunk(chunk_text_str),
                })

    print(f"  -> {len(chunks)} chunks from {len(sections)} sections")
    return chunks


def main() -> None:
    print("=" * 60)
    print("Building Aristotle corpus")
    print("=" * 60)

    # Process all works
    all_chunks: list[dict] = []
    for work in WORKS:
        try:
            chunks = process_work(work)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  ERROR processing {work.title}: {e}")
            continue
        # Be polite to MIT servers
        time.sleep(0.5)

    print(f"\nTotal chunks: {len(all_chunks)}")

    # Embed all chunks
    print(f"\nLoading embedding model ({EMBEDDING_MODEL})...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["text"] for c in all_chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    # Attach embeddings (rounded to 6 decimal places)
    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = [round(float(v), 6) for v in emb]

    # Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)

    file_size_mb = OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"\nWrote {len(all_chunks)} chunks to {OUTPUT_PATH}")
    print(f"File size: {file_size_mb:.1f} MB")

    # Print summary by work
    print(f"\n{'Work':<35} {'Chunks':>6}")
    print("-" * 43)
    work_counts: dict[str, int] = {}
    for c in all_chunks:
        work_counts[c["work"]] = work_counts.get(c["work"], 0) + 1
    for title, count in sorted(work_counts.items()):
        print(f"  {title:<33} {count:>6}")
    print("-" * 43)
    print(f"  {'TOTAL':<33} {len(all_chunks):>6}")


if __name__ == "__main__":
    main()
