"""
Step 3 - Question Segmentation (segment.py)

Splits a cleaned paper into individual questions. Two formats occur:

  * "bunch" papers (DS, OS, DF, CPP, DM-2025): the paper is organised into
    BUNCH I..V; each bunch offers two options separated by a lone "OR" line,
    and every leaf question ends with a marks tag such as "(CO2:8 Marks)".
    -> boundary at every BUNCH / OR line and right after every CO..Marks tag.

  * plain papers (DM ISA 2024): numbered "1." / "2." top-level questions.
    -> boundary at every Arabic "N." / "N)" line.

OCR is noisy, so detection is deliberately lenient:
  * CO tags may read "(CO1: 5 Marks)", "COS:4 Marks" or "(11.8.3 ...";
  * bunches may read "BUNCH |", "| BUNCH II" or "— BUNCH |";
  * stray margin labels ("1.B.1", ".B.2", "IV.A") frequently land before or
    inside their question's body and are stripped as label noise.

Because scanned OCR is imperfect, the auto segmentation is hand-checked
once, and any remaining mistakes are patched through
processed/segment_overrides/<subject>_<year>.json (an explicit, versioned
manual-review step - see the report).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from clean import clean_lines, preprocess_for_matching  # noqa: E402

# leaf-end tag: "(CO2:8 Marks)", "COS:4 Marks)", "(CO1: 5 Marks)", "(CO5S:5"
CO_TAG = re.compile(r"CO\s*[\dOS]{1,2}\s*[:.)]?\s*\d+(?:\.\d+)?\s*MARKS?\s*\)?", re.IGNORECASE)
# "BUNCH I" / "BUNCH |" / "— BUNCH |" / "| BUNCH II"
BUNCH_HDR = re.compile(r"^\W*BUNCH\s*[IVX1l|\\]{1,4}\W*$", re.IGNORECASE)
# lone "OR" option separator
OR_LINE = re.compile(r"^\W*OR\W*$", re.IGNORECASE)
# plain Arabic question numbers: "1. Express", "2) Compute"
ARABIC_Q = re.compile(r"^\s*\d{1,3}\s*[.)]\s+\S")

# --- label noise -----------------------------------------------------------
# characters that may precede a label / pollute a chunk start
_JUNK_PREFIX = re.compile(r"^[\s_|\\/@©\u00bf\ufffd\u00bb\u00ab~\u2022\-\u2014\u2013\u00b7*+\u2019'\"=.]+\s*")
# dotted labels: "1.A", "1.A.1", "II.B.(a)", "I.B", "\\V.A.", "H1.B", "W.B.1"
# (roman/arabic/H/W prefix + separator + option letter)
_ROMANISH = "0-9IVXl1LHW|\\\\"
_LABEL_DOTTED = re.compile(
    rf"^[{_ROMANISH}]{{1,7}}\s*[.\u00b7\u2022]\s*[AB]"
    rf"(?:\s*[.\u00b7\u2022]\s*(?:[0-9]{{1,3}}|\(\s*[a-z]\s*\)))?"
    rf"\s*[.\u00b7\u2022]?\s*[-_=:»«]*\s*",
    re.IGNORECASE,
)
# letter-dot-digit labels from margin columns: "A.2", "B.1"
_LABEL_LETTER_DIGIT = re.compile(r"^[AB]\s*[.\u00b7\u2022]\s*[0-9]{1,3}\s*[-_=:»«]*\s*")
# labels whose separator vanished under OCR: "ILA", "lA", "VA", "vA", "IILA", "1A"
_LABEL_GLUED = re.compile(
    r"^(?:[IVXl1L|\\]{1,6}[ab]|[0-9]{1,3}[AB])\s*(?=[A-Za-z0-9]|\s*[A-Z])", re.IGNORECASE
)
# numeric chains with >=2 dots: "11.8.3", "(11.8.3", "1.8.1"
_LABEL_NUMCHAIN = re.compile(r"^\(?[0-9]{1,3}(?:\s*[.\u00b7\u2022]\s*[0-9]{1,3}){2,}\)?\s*")
# two-part OCR label with junk separator: "11.8»" / "1.8 (" / "18/" before a body word
_LABEL_NUM2JUNK = re.compile(r"^(?:[0-9]{1,3}\s*[.\u00b7\u2022]\s*[0-9]{1,3}\s*[»«\u00bf\ufffd:).+\-]?\s*\(?|\d{1,3}\s*[/|])\s*(?=[A-Z0-9])")
# roman run (>=2) + any capital letter: "IV.K", "II.A"
_LABEL_ROMAN_CAP = re.compile(
    r"^[IVXl1L]{2,6}\s*[.\u00b7\u2022]\s*[A-Z]"
    r"(?:\s*[.\u00b7\u2022]\s*(?:[0-9]{1,3}|\(\s*[a-z]\s*\)))?"
    r"(?=\s*[.\u00b7\u2022(]|\s*$|\s*[-_=:»«]*\s*[0-9]|\s+[A-Z][a-z]{2,}\b)"
)
# leading dot labels: ".B.1", ".A"
_LABEL_LEADING_DOT = re.compile(r"^[.\u00b7\u2022]\s*[AB](?:\s*[.\u00b7\u2022]\s*[0-9]{1,3})?\s*[-_=:»«]*\s*")
# single stray roman char glued to the next word: "LImagine" (V.A.LImagine)
_LABEL_GLUED_WORD = re.compile(r"^[IVXl1L]{1,2}(?=[A-Z][a-z]{2,})")

_LABEL_PATTERNS = (
    _LABEL_DOTTED,
    _LABEL_LEADING_DOT,
    _LABEL_LETTER_DIGIT,
    _LABEL_NUMCHAIN,
    _LABEL_NUM2JUNK,
    _LABEL_ROMAN_CAP,
    _LABEL_GLUED,
    _LABEL_GLUED_WORD,
)

# inline stray margin labels inside a chunk's text: " 1.B.1 ", " IV.A ", " 1.B.(a) "
_INLINE_LABEL = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[0-9IVXl1L|\\]{1,6}\s*[.\u00b7\u2022]\s*[AB]"
    r"|[IVXl1L|\\]{1,7}[AB])"
    r"(?:\s*[.\u00b7\u2022]\s*(?:[0-9]{1,3}|\(\s*[a-z]\s*\)))?"
    r"(?![A-Za-z0-9])",
    re.IGNORECASE,
)

_MARKS_AT_END = re.compile(
    r"\s*[\(\[]?\s*CO\s*[\dOS]{1,2}\s*[:.)]?\s*(\d+(?:\.\d+)?)\s*MARKS?\s*[\)\]]?\s*$",
    re.IGNORECASE,
)
_CO_TAG_ANYWHERE = re.compile(r"[\(\[]?\s*CO\s*[\dOS]{1,2}\s*[:.)]?\s*\d+(?:\.\d+)?\s*MARKS?\s*[\)\]]?", re.IGNORECASE)


def segment_paper(raw_text: str) -> list[str]:
    """Return the paper's questions as raw text chunks (labels still attached)."""
    kept, _ = clean_lines(raw_text)
    is_bunch = any(BUNCH_HDR.match(l) for l in kept)

    chunks: list[list[str]] = []
    buf: list[str] = []
    prev_was_tag = False

    def flush() -> None:
        if buf:
            chunks.append(list(buf))
        buf.clear()

    if is_bunch:
        for line in kept:
            if BUNCH_HDR.match(line) or OR_LINE.match(line):
                flush()                       # BUNCH / OR close the previous option
                prev_was_tag = False
                continue
            if prev_was_tag:
                flush()                       # new leaf starts right after a CO tag
                buf.append(line)
                prev_was_tag = False
            else:
                buf.append(line)
            if CO_TAG.search(line):
                prev_was_tag = True
        flush()
    else:
        for line in kept:
            if ARABIC_Q.match(line):
                flush()
            buf.append(line)
        flush()

    return ["\n".join(c) for c in chunks]


def _strip_leading_noise(text: str) -> str:
    """Repeatedly remove junk prefixes / OCR'd question labels from the start."""
    changed = True
    while changed and text:
        changed = False
        while text and ord(text[0]) > 127 and not text[0].isalnum():
            text = text[1:].strip()
            changed = True
        new = _JUNK_PREFIX.sub("", text)
        for pat in _LABEL_PATTERNS:
            m = pat.match(new)
            if m:
                new = new[m.end():]
                changed = True
                break
        if new != text:
            text = new.strip()
            changed = True
        if not changed:
            break
    return text.strip()


def _clean_chunk_text(chunk: str) -> str:
    text = " ".join(l.strip() for l in chunk.splitlines() if l.strip())
    text = _INLINE_LABEL.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return _strip_leading_noise(text)


def build_questions(raw_text: str) -> list[dict]:
    chunks = segment_paper(raw_text)
    out = []
    for i, chunk in enumerate(chunks, start=1):
        text = _clean_chunk_text(chunk)
        marks = None
        m = _MARKS_AT_END.search(text)
        if m:
            marks = float(m.group(1)) if "." in m.group(1) else int(m.group(1))
            text = text[: m.start()].strip()
        out.append(
            {
                "q_no": str(i),
                "marks": marks,
                "raw_text": text,
                "clean_text": preprocess_for_matching(text),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Manual-correction layer (one-time hand review of the auto segmentation)
# ---------------------------------------------------------------------------
def apply_overrides(questions: list[dict], overrides: list[dict]) -> list[dict]:
    """overrides: [{"action": "drop", "q": [..]} | {"action": "merge", "q": [..]}]"""
    if not overrides:
        return questions
    drop = {int(x) for o in overrides if o.get("action") == "drop" for x in o.get("q", [])}
    merge = {int(x) for o in overrides if o.get("action") == "merge" for x in o.get("q", [])}

    result: list[dict] = []
    for q in questions:
        n = int(q["q_no"])
        if n in drop:
            continue
        if n in merge and result:
            prev = result[-1]
            joined = " ".join(
                t for t in (_CO_TAG_ANYWHERE.sub(" ", prev["raw_text"]),
                             _CO_TAG_ANYWHERE.sub(" ", q["raw_text"])) if t.strip()
            )
            joined = re.sub(r"\s+", " ", joined).strip()
            prev["raw_text"] = joined
            prev["marks"] = prev["marks"] if prev["marks"] is not None else q["marks"]
            prev["clean_text"] = preprocess_for_matching(joined)
        else:
            result.append(dict(q))
    for i, q in enumerate(result, start=1):
        q["q_no"] = str(i)
    return result


def paper_json(subject: str, year: int, exam: str, display: str, file: str) -> dict:
    raw = (ROOT / "processed" / "raw" / f"{subject}_{year}.txt").read_text(encoding="utf-8")
    questions = build_questions(raw)
    ov_path = ROOT / "processed" / "segment_overrides" / f"{subject}_{year}.json"
    overrides = json.loads(ov_path.read_text(encoding="utf-8")) if ov_path.exists() else []
    questions = apply_overrides(questions, overrides)
    return {
        "subject": subject,
        "display_name": display,
        "year": year,
        "exam": exam,
        "source_file": file,
        "n_overrides": len(overrides),
        "questions": questions,
    }


def main() -> None:
    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    for subj in manifest["subjects"]:
        for paper in subj["papers"]:
            data = paper_json(subj["id"], paper["year"], paper.get("exam", ""),
                              subj["display_name"], paper["file"])
            dst = ROOT / "processed" / "segmented" / f"{subj['id']}_{paper['year']}.json"
            dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
            qs = data["questions"]
            print(f"\n=== {subj['display_name']} {paper['year']} -> {len(qs)} questions ===")
            for q in qs:
                flag = ""
                if len(q["clean_text"]) < 12:
                    flag = "  <-- SUSPICIOUS (too short)"
                print(f"  Q{q['q_no']:<3} [{str(q['marks']):<4}] {q['raw_text'][:100]}{flag}")


if __name__ == "__main__":
    main()
