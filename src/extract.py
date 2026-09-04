"""
Step 1 - Extraction (extract.py)

Hybrid strategy per page:
  1. Try PyMuPDF's embedded-text layer first (fast, lossless).
  2. If a page yields < MIN_TEXT_CHARS characters, treat it as a scan and
     OCR it with Tesseract (via pytesseract), rendering the page with
     PyMuPDF at 300 dpi -- no poppler/pdf2image needed.

Raw text of every paper is cached to processed/raw/<subject>_<year>.txt so
later runs skip re-OCR entirely. A tiny meta JSON records, per paper, which
pages were OCR'd vs. taken from the embedded text layer (used in the report
to justify the hybrid design).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_TEXT_CHARS = 20          # below this a page is assumed to be a scan
OCR_DPI = 300
TESSERACT_PATHS = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
]


def _tesseract_cmd() -> str:
    """Locate the tesseract binary (it is often missing from PATH on Windows)."""
    from_env = os.environ.get("TESSERACT_CMD")
    if from_env and os.path.exists(from_env):
        return from_env
    for p in TESSERACT_PATHS:
        if os.path.exists(p):
            return p
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, "tesseract.exe") if os.name == "nt" else os.path.join(p, "tesseract")
        if p and os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "Tesseract not found. Install it or point TESSERACT_CMD at tesseract.exe."
    )


def _ocr_page(page: fitz.Page) -> str:
    """Render one page at OCR_DPI with PyMuPDF and OCR it with Tesseract."""
    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = _tesseract_cmd()
    pix = page.get_pixmap(dpi=OCR_DPI, colorspace=fitz.csGRAY)
    from PIL import Image

    img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
    return pytesseract.image_to_string(img)


def extract_page(page: fitz.Page) -> tuple[str, str]:
    """Return (text, source) where source in {'text', 'ocr'}."""
    text = page.get_text().strip()
    if len(text) >= MIN_TEXT_CHARS:
        return text, "text"
    return _ocr_page(page), "ocr"


def extract_pdf(pdf_path: str | Path) -> tuple[str, dict]:
    """Extract all pages of a PDF. Returns (full_text, meta)."""
    doc = fitz.open(str(pdf_path))
    chunks: list[str] = []
    meta = {"pages": len(doc), "ocr_pages": [], "text_pages": [], "empty_pages": []}
    for i, page in enumerate(doc):
        text, source = extract_page(page)
        if len(text.strip()) < MIN_TEXT_CHARS:
            meta["empty_pages"].append(i + 1)  # truly blank / image-only page
        meta["ocr_pages" if source == "ocr" else "text_pages"].append(i + 1)
        chunks.append(f"<page {i + 1}>\n{text.strip()}")
    doc.close()
    return "\n\n".join(chunks), meta


def paper_key(paper: dict) -> str:
    exam = paper.get("exam", "")
    return f"{exam}_{paper['year']}" if exam else f"{paper['year']}"


def raw_path(paper: dict, subject_id: str) -> Path:
    return ROOT / "processed" / "raw" / f"{subject_id}_{paper['year']}.txt"


def main() -> None:
    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    overall = {"papers": [], "total_pages": 0, "ocr_pages": 0}
    for subj in manifest["subjects"]:
        for paper in subj["papers"]:
            dst = raw_path(paper, subj["id"])
            if dst.exists():
                print(f"[skip cached] {dst.name}")
                meta = json.loads(
                    (dst.with_suffix(".meta.json")).read_text(encoding="utf-8")
                ) if (dst.with_suffix(".meta.json")).exists() else {}
            else:
                print(f"[extract]    {subj['display_name']} {paper['year']} ...", flush=True)
                pdf = ROOT / "dataset" / paper["file"]
                text, meta = extract_pdf(pdf)
                dst.write_text(text, encoding="utf-8")
                meta["subject"] = subj["id"]
                meta["paper"] = paper["file"]
                (dst.with_suffix(".meta.json")).write_text(
                    json.dumps(meta, indent=2), encoding="utf-8"
                )
            overall["papers"].append({"subject": subj["id"], "file": paper["file"], **meta})
            overall["total_pages"] += meta.get("pages", 0)
            overall["ocr_pages"] += len(meta.get("ocr_pages", []))
    (ROOT / "processed" / "raw" / "_summary.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(
        f"\nDone: {overall['total_pages']} pages total, "
        f"{overall['ocr_pages']} OCR'd, "
        f"{overall['total_pages'] - overall['ocr_pages']} from embedded text."
    )


if __name__ == "__main__":
    main()
