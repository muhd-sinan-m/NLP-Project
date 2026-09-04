"""
Step 7 - Evaluation (evaluate.py)

Compares the matcher's output against the hand-labeled gold pairs in
labels/<subject>.json (built by reading every segmented question of the two
years and marking pairs that test the same topic, per the rubric in the
report). For a given threshold t:

    retrieved(t) = {(i, j) | sim[i][j] >= t}      (all matrix cells)
    gold         = hand-labeled pairs
    TP = retrieved ∩ gold ; FP = retrieved \\ gold ; FN = gold \\ retrieved

Precision / Recall / F1 are computed at several thresholds; example false
positives and false negatives (with their texts) are printed and written to
processed/eval/results.md for the report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

THRESHOLDS = [0.15, 0.25, 0.35, 0.5]


def load_all() -> list[dict]:
    manifest = json.loads((ROOT / "papers.json").read_text(encoding="utf-8"))
    out = []
    for subj in manifest["subjects"]:
        m = json.loads((ROOT / "processed" / "matches" / f"{subj['id']}.json").read_text(encoding="utf-8"))
        gold = json.loads((ROOT / "labels" / f"{subj['id']}.json").read_text(encoding="utf-8"))
        a = json.loads((ROOT / "processed" / "segmented" / f"{subj['id']}_{m['year_a']['year']}.json").read_text(encoding="utf-8"))
        b = json.loads((ROOT / "processed" / "segmented" / f"{subj['id']}_{m['year_b']['year']}.json").read_text(encoding="utf-8"))
        gold_pairs = {(int(a), int(b)) for a, b in gold["pairs"]}
        out.append(
            {
                "subject": subj["id"],
                "display_name": subj["display_name"],
                "year_a": m["year_a"]["year"],
                "year_b": m["year_b"]["year"],
                "sim": m["sim"],
                "qa": a["questions"],
                "qb": b["questions"],
                "gold": gold_pairs,
            }
        )
    return out


def metrics_for(d: dict, threshold: float) -> dict:
    retrieved = {
        (i + 1, j + 1)
        for i, row in enumerate(d["sim"])
        for j, s in enumerate(row)
        if s >= threshold
    }
    tp = len(retrieved & d["gold"])
    fp = len(retrieved - d["gold"])
    fn = len(d["gold"] - retrieved)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {
        "threshold": threshold,
        "n_retrieved": len(retrieved),
        "n_gold": len(d["gold"]),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "retrieved": retrieved,
    }


def fmt_row(r: dict) -> str:
    return (f"| {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['n_retrieved']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['n_gold']} |")


def main() -> None:
    data = load_all()
    lines: list[str] = []
    lines.append("# Evaluation results (TF-IDF + cosine similarity vs hand-labeled gold pairs)\n")
    all_rows = {t: {k: 0 for k in ("tp", "fp", "fn", "n_retrieved")} for t in THRESHOLDS}
    for d in data:
        lines.append(f"\n## {d['display_name']} — {d['year_a']} vs {d['year_b']} ({len(d['gold'])} gold pairs)")
        lines.append("| threshold | precision | recall | F1 | retrieved | TP | FP | FN | gold |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for t in THRESHOLDS:
            r = metrics_for(d, t)
            lines.append(fmt_row(r))
            for k in ("tp", "fp", "fn", "n_retrieved"):
                all_rows[t][k] += r[k]
    lines.append("\n## Overall (all 5 subjects pooled)")
    lines.append("| threshold | precision | recall | F1 | retrieved | TP | FP | FN | gold |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    n_gold_all = sum(len(d["gold"]) for d in data)
    for t in THRESHOLDS:
        a = all_rows[t]
        prec = a["tp"] / a["n_retrieved"] if a["n_retrieved"] else 0.0
        rec = a["tp"] / n_gold_all
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        lines.append(f"| {t:.2f} | {prec:.3f} | {rec:.3f} | {f1:.3f} | {a['n_retrieved']} | "
                      f"{a['tp']} | {a['fp']} | {a['fn']} | {n_gold_all} |")

    # example FPs / FNs at the working threshold
    t_show = 0.25
    lines.append(f"\n## Example false positives & false negatives @ threshold = {t_show}\n")
    for d in data:
        r = metrics_for(d, t_show)
        fps = sorted(r["retrieved"] - d["gold"])
        fns = sorted(d["gold"] - r["retrieved"])
        def qtext(paper_qs, n):
            return paper_qs[n - 1]["raw_text"][:180]
        if fns:
            lines.append(f"\n**{d['display_name']} — false negatives ({len(fns)}):**")
            for (i, j) in fns[:6]:
                lines.append(f"- A{i} [{qtext(d['qa'], i)}]")
                lines.append(f"  vs B{j} [{qtext(d['qb'], j)}]")
        if fps:
            lines.append(f"\n**{d['display_name']} — false positives ({len(fps)}):**")
            for (i, j) in sorted(fps)[:6]:
                lines.append(f"- A{i} [{qtext(d['qa'], i)}]")
                lines.append(f"  vs B{j} [{qtext(d['qb'], j)}]")

    out = ROOT / "processed" / "eval" / "results.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(l for l in lines if not l.startswith("- ")))


if __name__ == "__main__":
    main()
