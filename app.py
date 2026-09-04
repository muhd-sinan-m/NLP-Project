"""
PYQ Repeated Question Analyzer (Streamlit app)

Run with:  streamlit run app.py

Pipeline behind this UI:
  PDF -> OCR (Tesseract, hybrid w/ PyMuPDF) -> boilerplate removal ->
  question segmentation (with one-time manual corrections) -> preprocessing
  (NLTK) -> per-subject TF-IDF (1-2 grams) -> cosine similarity ->
  threshold slider (live precision/recall trade-off).

Two tabs:
  1. "Papers & Related Topics" — full side-by-side papers and the matched /
     related question pairs (the same topic tested in both years).
  2. "NLP Evaluation" — precision / recall / F1 / accuracy of the matcher
     against the hand-labeled gold pairs in labels/, recomputed live when the
     threshold slider moves, plus per-threshold tables and FP/FN examples.

All heavy computation is cached in processed/: the app only loads the
precomputed per-subject similarity matrices and re-thresholds instantly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

BANDS = [
    (0.35, "#1a7f37", "high"),   # strong lexical overlap (near topic match)
    (0.20, "#9a6700", "medium"), # related topic, different wording
    (0.0,  "#cf222e", "low"),    # weak / noisy overlap
]

THRESHOLDS = [0.15, 0.25, 0.35, 0.50]


@st.cache_data
def load_subject(subject_id: str) -> dict:
    m = json.loads((ROOT / "processed" / "matches" / f"{subject_id}.json").read_text(encoding="utf-8"))
    a = json.loads(
        (ROOT / "processed" / "segmented" / f"{subject_id}_{m['year_a']['year']}.json").read_text(encoding="utf-8")
    )
    b = json.loads(
        (ROOT / "processed" / "segmented" / f"{subject_id}_{m['year_b']['year']}.json").read_text(encoding="utf-8")
    )
    gold = json.loads((ROOT / "labels" / f"{subject_id}.json").read_text(encoding="utf-8"))
    return {
        "meta": m,
        "a": a,
        "b": b,
        "gold": {(int(pa), int(pb)) for pa, pb in gold["pairs"]},
    }


@st.cache_data
def subject_list() -> list[str]:
    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    return [s["id"] for s in manifest["subjects"]]


def paper_label(p: dict) -> str:
    return " ".join(str(x) for x in [p["exam"], p["year"]] if x).strip()


def band_color(sim: float) -> tuple[str, str]:
    for lo, color, name in BANDS:
        if sim >= lo:
            return color, name
    return BANDS[-1][1], BANDS[-1][2]


GENERIC_WORDS = {
    # exam-instruction verbs & connectors that carry no topic signal
    "explain", "demonstrate", "describe", "discuss", "show", "consider", "following",
    "given", "write", "list", "state", "mention", "using", "provide", "also", "find",
    "justify", "identify", "illustrate", "example", "question", "answer", "marks",
    "use", "used", "include", "including", "called", "known", "perform", "process",
}


def common_keywords(clean_a: str, clean_b: str, min_len: int = 3, limit: int = 12) -> list[str]:
    """Content words (stopword-free, lemmatized) appearing in BOTH questions' preprocessed text."""
    wa = set(w for w in clean_a.split() if len(w) >= min_len and not w.isdigit() and w not in GENERIC_WORDS)
    wb = set(w for w in clean_b.split() if len(w) >= min_len and not w.isdigit() and w not in GENERIC_WORDS)
    return sorted(wa & wb)[:limit]


def metrics_for(sim: list[list[float]], gold: set[tuple[int, int]], threshold: float) -> dict:
    """Same definition as src/evaluate.py: every matrix cell >= threshold is retrieved."""
    n_a, n_b = len(sim), len(sim[0]) if sim else 0
    retrieved = {
        (i + 1, j + 1)
        for i, row in enumerate(sim)
        for j, s in enumerate(row)
        if s >= threshold
    }
    tp = len(retrieved & gold)
    fp = len(retrieved - gold)
    fn = len(gold - retrieved)
    total = n_a * n_b
    tn = total - tp - fp - fn
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    acc = (tp + tn) / total if total else 0.0
    return {
        "threshold": threshold,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "accuracy": round(acc, 3),
        "retrieved": len(retrieved),
        "gold": len(gold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "retrieved_set": retrieved,
    }


def example_pairs(data: dict, r: dict, kind: str, limit: int = 5) -> list[tuple[int, int]]:
    gold = data["gold"]
    if kind == "fp":
        pairs = sorted(r["retrieved_set"] - gold)
    else:
        pairs = sorted(gold - r["retrieved_set"])
    return pairs[:limit]


st.set_page_config(page_title="PYQ Repeated Question Analyzer", layout="wide")

st.title("PYQ Repeated Question Analyzer")
st.caption("Finds questions from two years of the *same subject* that test the same topic, "
           "even when they are worded completely differently.")

subject_id = st.sidebar.selectbox("Select Subject", subject_list(), format_func=lambda s: s.replace("_", " ").title())
threshold = st.sidebar.slider(
    "Similarity threshold", 0.05, 0.60, 0.15, 0.01,
    help="Only pairs at or above this cosine similarity are shown. Lower = more matches "
         "(higher recall, lower precision); higher = only near-duplicates.",
)
max_per_row = st.sidebar.slider("Max matches per year-A question", 1, 5, 3)

data = load_subject(subject_id)
meta, A, B, gold = data["meta"], data["a"], data["b"], data["gold"]
label_a, label_b = paper_label(meta["year_a"]), paper_label(meta["year_b"])

sim = meta["sim"]
qa, qb = A["questions"], B["questions"]
name = meta["display_name"]

tab1, tab2 = st.tabs(["Related Questions (Both Years)", "NLP Evaluation"])

# ============================================================ TAB 1 — browse + matches
with tab1:
    matches = []
    for i, row in enumerate(sim):
        hits = sorted(range(len(row)), key=lambda j: -row[j])
        kept = 0
        for j in hits:
            if row[j] < threshold:
                break
            matches.append({"i": i, "j": j, "sim": row[j]})
            kept += 1
            if kept >= max_per_row:
                break
    matches.sort(key=lambda m: -m["sim"])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Subject", name)
    c2.metric(f"{label_a}", f"{len(qa)} questions")
    c3.metric(f"{label_b}", f"{len(qb)} questions")
    c4.metric("Matched pairs", f"{len(matches)} @ ≥ {threshold:.2f}")

    st.divider()

    st.subheader("Related Questions in Both Papers")
    st.markdown(
        "Only questions the matcher linked across the two years are shown below — questions "
        "with no related counterpart are hidden. Each row is a pair (one from each year) that "
        "tests the **same topic**, even though the wording differs. Color band = similarity: "
        "<span style='color:#1a7f37'>**high**</span> (strongly overlapping), "
        "<span style='color:#9a6700'>**medium**</span> (same topic, different wording), "
        "<span style='color:#cf222e'>**low**</span> (weak/noisy). Expand a row to read both "
        "questions in full and see the keywords they share.",
        unsafe_allow_html=True,
    )

    if not matches:
        st.info(f"No question pairs reach a similarity of {threshold:.2f} — try lowering the threshold.")
    else:
        for mt in matches:
            i, j, s = mt["i"], mt["j"], mt["sim"]
            color, band = band_color(s)
            qa_q, qb_q = qa[i], qb[j]
            marks = f" ({qa_q['marks']} vs {qb_q['marks']} marks)" if (qa_q["marks"] and qb_q["marks"]) else ""
            with st.expander(
                f"{band.upper()} · similarity {s:.2f} — {label_a} Q{qa_q['q_no']} ↔ {label_b} Q{qb_q['q_no']}{marks}"
            ):
                st.markdown(
                    f"<span style='color:{color}'><b>● similarity {s:.2f} ({band})</b></span>",
                    unsafe_allow_html=True,
                )
                cc1, cc2 = st.columns(2)
                cc1.markdown(f"**{label_a} — Q{qa_q['q_no']}:**\n\n{qa_q['raw_text']}")
                cc2.markdown(f"**{label_b} — Q{qb_q['q_no']}:**\n\n{qb_q['raw_text']}")
                kws = common_keywords(qa_q["clean_text"], qb_q["clean_text"])
                if kws:
                    st.markdown(f"**Common keywords in both questions:** `{', '.join(kws)}`")
                else:
                    st.caption("No shared content keywords (the questions test the same topic with completely different vocabulary).")

    st.divider()
    st.caption(
        f"**{len(matches)} matched pair(s)** shown out of "
        f"{len(qa)} / {len(qb)} questions ({name}) at threshold ≥ {threshold:.2f} "
        f"(capped at {max_per_row} per year-A question). "
        "Move the threshold slider in the sidebar to see the precision/recall trade-off live."
    )

# ============================================================ TAB 2 — NLP evaluation
with tab2:
    st.subheader(f"NLP Evaluation — {name} ({label_a} vs {label_b})")
    st.markdown(
        "How well does the TF-IDF matcher find the pairs a human would call *same topic*? "
        "The system is evaluated against **hand-labeled gold pairs** in `labels/` (built by "
        "reading every question of both years and marking pairs that test the same concept, "
        "per the project rubric). A pair is *retrieved* if its cosine similarity is ≥ the "
        "sidebar threshold — so **all numbers below update live as you move the slider**."
    )

    r = metrics_for(sim, gold, threshold)

    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Precision", f"{r['precision']:.2f}")
    mc2.metric("Recall", f"{r['recall']:.2f}")
    mc3.metric("F1 Score", f"{r['f1']:.2f}")
    mc4.metric("Accuracy", f"{r['accuracy']:.2f}")
    mc5.metric("Retrieved pairs", f"{r['retrieved']} @ ≥ {threshold:.2f}")

    mc6, mc7, mc8, mc9 = st.columns(4)
    mc6.metric("Gold (true) pairs", r["gold"])
    mc7.metric("True positives", r["tp"])
    mc8.metric("False positives", r["fp"])
    mc9.metric("False negatives", r["fn"])

    st.caption(
        "**Precision** = of the pairs the system retrieved, how many really are same-topic. "
        "**Recall** = of all true same-topic pairs, how many were retrieved. **F1** = harmonic "
        "mean of the two. **Accuracy** = fraction of *all* possible pairs (including the many "
        "non-matching ones) classified correctly — it looks high because most pairs are "
        "correctly rejected, so P/R/F1 are the meaningful numbers here. Retrieved counts every "
        "matrix cell ≥ threshold (same definition as `src/evaluate.py`)."
    )

    st.divider()

    st.markdown("### Metrics across thresholds")
    st.markdown("Same table as `processed/eval/results.md` — where precision and recall cross is the sweet spot.")
    rows = [metrics_for(sim, gold, t) for t in THRESHOLDS]
    df = pd.DataFrame(
        [
            {
                "threshold": x["threshold"],
                "precision": x["precision"],
                "recall": x["recall"],
                "F1": x["f1"],
                "accuracy": x["accuracy"],
                "retrieved": x["retrieved"],
                "TP": x["tp"],
                "FP": x["fp"],
                "FN": x["fn"],
                "gold": x["gold"],
            }
            for x in rows
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("### Overall — all 5 subjects pooled")
    st.markdown("Every subject's matrix at the same thresholds, summed. Shows how the matcher "
                "behaves on the whole dataset, not just this subject.")
    all_data = [(load_subject(s)["meta"]["sim"], load_subject(s)["gold"]) for s in subject_list()]
    pooled_rows = []
    for t in THRESHOLDS:
        agg = {"tp": 0, "fp": 0, "fn": 0, "retrieved": 0, "gold": 0, "total": 0}
        for sim_s, gold_s in all_data:
            rr = metrics_for(sim_s, gold_s, t)
            agg["tp"] += rr["tp"]; agg["fp"] += rr["fp"]; agg["fn"] += rr["fn"]
            agg["retrieved"] += rr["retrieved"]; agg["gold"] += rr["gold"]
            agg["total"] += len(sim_s) * (len(sim_s[0]) if sim_s else 0)
        prec = agg["tp"] / agg["retrieved"] if agg["retrieved"] else 0.0
        rec = agg["tp"] / agg["gold"] if agg["gold"] else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (agg["tp"] + (agg["total"] - agg["tp"] - agg["fp"] - agg["fn"])) / agg["total"] if agg["total"] else 0.0
        pooled_rows.append(
            {
                "threshold": t,
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "F1": round(f1, 3),
                "accuracy": round(acc, 3),
                "retrieved": agg["retrieved"],
                "gold": agg["gold"],
                "TP": agg["tp"], "FP": agg["fp"], "FN": agg["fn"],
            }
        )
    st.dataframe(pd.DataFrame(pooled_rows), use_container_width=True, hide_index=True)

    st.divider()

    st.markdown(f"### What goes wrong at threshold {threshold:.2f}")
    fps = example_pairs(data, r, "fp")
    fns = example_pairs(data, r, "fn")
    if not fps and not fns:
        st.success("No false positives or false negatives at this threshold — the matcher is "
                   "perfect for this subject here.")
    if fps:
        st.markdown(f"**False positives ({r['fp']})** — retrieved as related, but actually different topics:")
        for i, j in fps:
            with st.expander(f"Q{i} ↔ Q{j}"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**{label_a} — Q{qa[i-1]['q_no']}:**\n\n{qa[i-1]['raw_text']}")
                c2.markdown(f"**{label_b} — Q{qb[j-1]['q_no']}:**\n\n{qb[j-1]['raw_text']}")
    if fns:
        st.markdown(f"**False negatives ({r['fn']})** — truly same topic, but missed:")
        for i, j in fns:
            with st.expander(f"Q{i} ↔ Q{j}"):
                c1, c2 = st.columns(2)
                c1.markdown(f"**{label_a} — Q{qa[i-1]['q_no']}:**\n\n{qa[i-1]['raw_text']}")
                c2.markdown(f"**{label_b} — Q{qb[j-1]['q_no']}:**\n\n{qb[j-1]['raw_text']}")

    st.caption(
        "Gold labels live in `labels/<subject>.json`; the full runnable evaluation is "
        "`src/evaluate.py`, and its output is `processed/eval/results.md`. See "
        "`report/REPORT.md` for methodology and limitations."
    )