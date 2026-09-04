"""
Step 2 - Boilerplate Removal + Step 4 preprocessing helpers (clean.py)

Line-level blocklist filtering happens BEFORE question segmentation:
anything matching the blocklist below is dropped from the raw OCR text.
The patterns were tuned against all 10 papers (Marian College Kuttikkanam
question papers) and cover the repeated header / instruction lines such as:

  "MARIAN COLLEGE KUTTIKKANAM (AUTONOMOUS)"
  "252CCRUBC103 Operating Systems"
  "Time: 2 Hours Maximum Marks: 40"
  "Answer one question from each bunch"
  the trailing "CO1 ... CO5" course-outcome block, and OCR junk fragments.

clean.py also hosts the preprocessing used for matching: lowercasing,
stopword removal, punctuation stripping and NLTK lemmatization
(Step 4 of the pipeline).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Line-level blocklist  (tuned against the 10 scanned PDFs)
# ---------------------------------------------------------------------------
BLOCKLIST: list[tuple[str, str]] = [
    # (name, regex)
    ("page marker",        r"^<page \d+>\s*$"),
    ("exam paper code",    r"^\s*[A-Z]{2,4}\s*\d{4,5}\s*$"),          # SE8205
    ("course code line",   r"(?:CCRUBC|SECUBC|DSCUBC|CSCUBC|CRUBC)\s*\d{3,4}"),  # 252CCRUBC103 Operating Systems
    ("college name",       r"MARIAN\s+COLLEGE|COLLEGE\s+KUTTIKKANAM"),
    ("autonomous suffix",  r"\(?AUTONOMOUS\)?|AUTONOMOUS\s*$"),
    ("degree banner",      r"UG\s+PROGRAMME|PROGRAMME\s*\(?\s*HONORS|REGULAR\s+EXAMINATIONS|\(1\s+SEMESTER\s+UG\)"),
    ("semester line",      r"^\s*(?:FIRST|SECOND|THIRD|FOURTH)\s+SEMESTER\b"),
    ("time header",        r"^\s*TIME\s*[:.]\s*(?=.{0,45}$)"),   # "Time: 2 Hours Maximum Marks: 40"
    ("marks line",         r"\b(?:MAXIMUM|TOTAL)\s+MARKS\b|MINUTES\s+TOTAL"),
    ("instructions",       r"ANSWER\s+ONE\s+QUESTION\s+FROM\s+EACH|INSTRUCTIONS\s+TO\s+"),
    ("exam banner",        r"INITIAL\s+SUMMATIVE\s+ASSESSMENT|QUESTION\s+PAPER\s+[A-Z]\s*$|DEGREE\s+.*?EXAMINATION"),
    ("candidate fields",   r"REG\s*NO|ROLL\s*NO|COURSE\s+CODE|DATE\s*[:.]|DURATION\s*[:.]|BRANCH\s*[:.]"),

    # OCR junk: pure symbol runs, and very short non-word fragments ("Z", "cm", "ey,")
    ("symbol junk",        r"^[\s_\-\u2014\u2013=|\u00b7*\u2022~^<>'\"`\\/]+$"),
    ("short junk",         r"^\s*(?:[^A-Za-z0-9]*[A-Za-z]){0,2}[^A-Za-z0-9]*\s*$"),
]

_compiled = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in BLOCKLIST]


def is_boilerplate(line: str) -> bool:
    """True if the OCR line is boilerplate / noise and should be dropped."""
    return any(p.search(line) for _, p in _compiled)


def clean_lines(raw_text: str) -> tuple[list[str], list[str]]:
    """
    Filter raw OCR text into (kept_lines, filtered_lines).
    Also trims the trailing course-outcome block that every paper ends with.
    """
    kept, filtered = [], []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # a lone "OR" is the option separator the segmenter relies on - keep it
        if re.fullmatch(r"OR", line, re.IGNORECASE):
            kept.append(line)
            continue
        # never treat a marks tag line as boilerplate (e.g. "time. ( (CO2:5 Marks)")
        if re.search(r"CO\s*[\dOS]{1,2}\s*[:.)]?\s*\d+\s*MARKS", line, re.IGNORECASE):
            kept.append(line)
            continue
        if is_boilerplate(line):
            filtered.append(line)
            continue
        kept.append(line)

    # Trim the trailing course-outcome block that every paper ends with.
    #   * a "Course Outcomes" header (plain papers, e.g. DM ISA 2024) or a
    #     "COx | ..." outcome entry (bunch papers) starts the block -> cut there.
    #   * bunch papers always close their last question with a "(COx:N Marks)"
    #     tag, so if no header was found, everything after the LAST such tag is
    #     the outcome block and gets cut.
    outcome_start = None
    for i, line in enumerate(kept):
        if re.search(r"^\s*COURSE\s+OUTCOMES?\s*$", line, re.IGNORECASE):
            outcome_start = i
            break
        # outcome entry with a CO code + separator, but NOT a marks tag
        if re.search(r"^\s*(?:CO|Co|c0|COS)\s*\d?[OS]?\s*[|:\\]\s*", line) and not re.search(
            r"CO\s*\d\s*[:.)]?\s*\d+\s*MARKS?", line, re.IGNORECASE
        ):
            outcome_start = i
            break
    if outcome_start is not None:
        kept = kept[:outcome_start]
        return kept, filtered

    # fall back for bunch papers: drop everything after the last CO marks tag
    last_tag = -1
    for i, line in enumerate(kept):
        if re.search(r"CO\s*[\dOS]{1,2}\s*[:.)]?\s*\d+\s*MARKS?", line, re.IGNORECASE):
            last_tag = i
    if last_tag >= 0:
        kept = kept[: last_tag + 1]
    return kept, filtered


# ---------------------------------------------------------------------------
# 2. Preprocessing for matching (Step 4)
# ---------------------------------------------------------------------------
_NLTK_DATA = ROOT / "nltk_data"


def _ensure_nltk() -> None:
    import nltk

    nltk.data.path.insert(0, str(_NLTK_DATA))
    for corpus in ("stopwords", "wordnet", "punkt_tab", "punkt"):
        try:
            nltk.data.find(f"corpora/{corpus}")
        except LookupError:
            nltk.download(corpus, download_dir=str(_NLTK_DATA), quiet=True)


_ensure_nltk()

import nltk  # noqa: E402
from nltk.corpus import stopwords  # noqa: E402
from nltk.stem import WordNetLemmatizer  # noqa: E402

_STOP = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()


def preprocess_for_matching(text: str) -> str:
    """Lowercase -> tokenize -> strip non-alphanumerics -> stopwords -> lemmatize."""
    tokens = nltk.word_tokenize(text.lower())
    words = []
    for t in tokens:
        if not re.search(r"[a-z]", t):   # drop pure digits/symbols
            continue
        w = re.sub(r"[^a-z0-9]", "", t)
        if w and w not in _STOP and len(w) > 1:
            words.append(_LEMMATIZER.lemmatize(w))
    return " ".join(words)


# ---------------------------------------------------------------------------
# CLI (evidence for the report: real filtered lines)
# ---------------------------------------------------------------------------
def main() -> None:
    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    for subj in manifest["subjects"]:
        for paper in subj["papers"]:
            raw = ROOT / "processed" / "raw" / f"{subj['id']}_{paper['year']}.txt"
            if not raw.exists():
                print(f"[missing raw] {raw.name}")
                continue
            kept, filtered = clean_lines(raw.read_text(encoding="utf-8"))
            out = ROOT / "processed" / "clean" / f"{subj['id']}_{paper['year']}.txt"
            out.write_text("\n".join(kept), encoding="utf-8")
            samp = ROOT / "processed" / "clean" / f"{subj['id']}_{paper['year']}_filtered.txt"
            samp.write_text("\n".join(filtered), encoding="utf-8")
            print(f"{subj['id']}_{paper['year']:<32} kept={len(kept):3d} filtered={len(filtered):3d}")


if __name__ == "__main__":
    main()
