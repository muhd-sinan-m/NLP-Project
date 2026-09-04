"""
Step 5 - Matching (match.py)

Compares the two papers of one subject (Year A vs Year B) ONLY - no global
N x N matrix across subjects. TF-IDF is fitted per subject pair so the
vocabulary stays topic-focused.

  corpus      = clean_text(questions_A) + clean_text(questions_B)
  tfidf       = TfidfVectorizer(ngram_range=(1, 2)).fit_transform(corpus)
  sim_matrix  = cosine_similarity(tfidf[:nA], tfidf[nA:])

Because the goal is *topic-level* relatedness (not near-duplicates), scores
are lower than exact-duplicate matching and the threshold sits around
0.3-0.4 (see report). We store the full nA x nB similarity matrix per
subject so the Streamlit app can re-threshold live with a slider, and
retrieval at any threshold = cells above it (capped at max_per_row per
Year A question to keep the view readable).

NOTE (report): TF-IDF captures lexical overlap. A genuinely same-topic pair
with zero shared vocabulary (e.g. "Explain ACID properties" vs "What ensures
transaction reliability?") scores ~0 - this is the documented limitation
that SBERT-style embeddings would address.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_THRESHOLD = 0.35


def load_paper_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pair_papers(subject: dict) -> tuple[dict, dict, dict]:
    """Return (year_a_meta, year_b_meta, sim_matrix rows) for one subject."""
    papers = sorted(subject["papers"], key=lambda p: p["year"])
    a = load_paper_json(ROOT / "processed" / "segmented" / f"{subject['id']}_{papers[0]['year']}.json")
    b = load_paper_json(ROOT / "processed" / "segmented" / f"{subject['id']}_{papers[1]['year']}.json")
    return a, b, subject


def build_sim_matrix(questions_a: list[dict], questions_b: list[dict]) -> np.ndarray:
    corpus = [q["clean_text"] for q in questions_a] + [q["clean_text"] for q in questions_b]
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    tfidf = vectorizer.fit_transform(corpus)
    n = len(questions_a)
    sim = cosine_similarity(tfidf[:n], tfidf[n:])
    return sim


def retrieve(questions_a, questions_b, sim, threshold=DEFAULT_THRESHOLD, max_per_row=3):
    """All (i, j) cells with sim >= threshold, capped at max_per_row per A row."""
    matches = []
    for i, row in enumerate(sim):
        order = np.argsort(-row)
        for j in order:
            if row[j] < threshold:
                break
            matches.append(
                {
                    "qa": questions_a[i]["q_no"],
                    "qb": questions_b[j]["q_no"],
                    "similarity": float(row[j]),
                }
            )
            if len([m for m in matches if m["qa"] == questions_a[i]["q_no"]]) >= max_per_row:
                break
    return sorted(matches, key=lambda m: -m["similarity"])


def main() -> None:
    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    for subj in manifest["subjects"]:
        a, b, _ = pair_papers(subj)
        sim = build_sim_matrix(a["questions"], b["questions"])
        out = {
            "subject": subj["id"],
            "display_name": subj["display_name"],
            "year_a": {"year": a["year"], "exam": a["exam"], "source_file": a["source_file"]},
            "year_b": {"year": b["year"], "exam": b["exam"], "source_file": b["source_file"]},
            "q_no_a": [q["q_no"] for q in a["questions"]],
            "q_no_b": [q["q_no"] for q in b["questions"]],
            "sim": sim.tolist(),
        }
        dst = ROOT / "processed" / "matches" / f"{subj['id']}.json"
        dst.write_text(json.dumps(out), encoding="utf-8")
        n = len(a["questions"]); m = len(b["questions"])
        matches = retrieve(a["questions"], b["questions"], sim, DEFAULT_THRESHOLD)
        print(f"{subj['display_name']:<40} {a['year']}({n}) vs {b['year']}({m})  "
              f"-> {len(matches)} pairs @ {DEFAULT_THRESHOLD}")
        for mt in matches[:8]:
            qa = a["questions"][int(mt["qa"]) - 1]
            qb = b["questions"][int(mt["qb"]) - 1]
            print(f"    {mt['similarity']:.2f}  A{mt['qa']} [{qa['raw_text'][:58]}]\n"
                  f"            B{mt['qb']} [{qb['raw_text'][:58]}]")


if __name__ == "__main__":
    main()
