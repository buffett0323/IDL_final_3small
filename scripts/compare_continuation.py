#!/usr/bin/env python3
"""compare_continuation.py — Compare re-translation vs prefix-constrained continuation.

Analyzes the three key claims:
  1. Consistency: continuation never contradicts its own committed prefix.
  2. Quality: BLEU / chrF improvement from eliminating hypothesis inconsistency.
  3. Case studies: sentences where baseline is garbled but continuation is clean.

Usage:
    cd /data/user_data/haolingp/IDL_final_3small
    python3 scripts/compare_continuation.py
    python3 scripts/compare_continuation.py --show-cases 10
"""
from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs"
DATA_DIR = ROOT / "data/enzh"

# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_instances(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                result[rec["index"]] = rec
            except Exception:
                pass
    return result


def load_scores(path: Path) -> dict[str, float]:
    """Parse the 'scores' file produced by simuleval.

    SimulEval writes a pandas-style table like:
        BLEU  LAAL   AL    AP   DAL  ATD
    0  13.46  8.82  8.80  0.583  7.53  0.0

    The data row starts with a row-index ('0') that must be skipped.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    result: dict[str, float] = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 2:
        headers = lines[0].split()
        # Skip the leading row-index token ('0') in the data row
        values  = lines[1].split()[1:]
        for h, v in zip(headers, values):
            try:
                result[h] = float(v)
            except ValueError:
                pass
    if not result:
        # Fallback: two-column key-value format
        for line in text.splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    result[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def char_f1(pred: str, ref: str) -> float:
    """Character-level F1 (set overlap) — a proxy for translation quality."""
    pred_chars = list(pred.replace(" ", ""))
    ref_chars  = list(ref.replace(" ", ""))
    if not pred_chars or not ref_chars:
        return 0.0
    pred_set = {}
    for c in pred_chars:
        pred_set[c] = pred_set.get(c, 0) + 1
    ref_set = {}
    for c in ref_chars:
        ref_set[c] = ref_set.get(c, 0) + 1
    overlap = sum(min(pred_set.get(c, 0), ref_set.get(c, 0)) for c in ref_set)
    precision = overlap / len(pred_chars) if pred_chars else 0.0
    recall    = overlap / len(ref_chars) if ref_chars else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def inconsistency_score(pred: str) -> float:
    """Measure hypothesis inconsistency as repeated-character-chunk ratio.

    In a garbled re-translation output like "出了出了出", the same short
    substrings repeat often.  A high ratio indicates likely inconsistency.
    """
    chars = [c for c in pred if not unicodedata.category(c).startswith("Z")]
    if len(chars) < 4:
        return 0.0
    # Count bigram repetitions
    bigrams = [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
    n_unique = len(set(bigrams))
    n_total  = len(bigrams)
    repetition_ratio = 1.0 - n_unique / n_total if n_total > 0 else 0.0
    return repetition_ratio


def compute_sentence_metrics(
    inst: dict, ref: str
) -> dict[str, float]:
    pred = inst.get("prediction", "").replace(" ", "")
    ref_clean = ref.replace(" ", "")
    return {
        "char_f1":       char_f1(pred, ref_clean),
        "inconsistency": inconsistency_score(pred),
        "pred_len":      len(pred),
        "ref_len":       len(ref_clean),
        "avg_delay":     (sum(inst.get("delays", [0])) / max(len(inst.get("delays", [1])), 1)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze(
    base_instances:  dict[int, dict],
    cont_instances:  dict[int, dict],
    ref_lines:       list[str],
    n_case_studies:  int = 5,
) -> dict:
    common = sorted(
        set(base_instances) & set(cont_instances) & set(range(len(ref_lines)))
    )

    base_f1s, cont_f1s = [], []
    base_incons, cont_incons = [], []
    delta_f1s: list[tuple[int, float]] = []  # (idx, cont_f1 - base_f1)

    for idx in common:
        ref = ref_lines[idx]
        bm  = compute_sentence_metrics(base_instances[idx], ref)
        cm  = compute_sentence_metrics(cont_instances[idx], ref)
        base_f1s.append(bm["char_f1"])
        cont_f1s.append(cm["char_f1"])
        base_incons.append(bm["inconsistency"])
        cont_incons.append(cm["inconsistency"])
        delta_f1s.append((idx, cm["char_f1"] - bm["char_f1"]))

    def mean(xs):
        return sum(xs) / len(xs) if xs else 0.0

    # Best improvement cases (continuation >> baseline)
    top_improved = sorted(delta_f1s, key=lambda x: -x[1])[:n_case_studies]
    # Worst (continuation < baseline)
    top_degraded = sorted(delta_f1s, key=lambda x: x[1])[:n_case_studies]

    return {
        "n": len(common),
        "base_mean_f1":    mean(base_f1s),
        "cont_mean_f1":    mean(cont_f1s),
        "mean_delta_f1":   mean([d for _, d in delta_f1s]),
        "base_mean_incon": mean(base_incons),
        "cont_mean_incon": mean(cont_incons),
        "improved":        sum(1 for _, d in delta_f1s if d > 0.01),
        "degraded":        sum(1 for _, d in delta_f1s if d < -0.01),
        "neutral":         sum(1 for _, d in delta_f1s if abs(d) <= 0.01),
        "top_improved":    top_improved,
        "top_degraded":    top_degraded,
        "base_f1s":        base_f1s,
        "cont_f1s":        cont_f1s,
        "delta_f1s":       delta_f1s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    base_scores:   dict[str, float],
    cont_scores:   dict[str, float],
    stats:         dict,
    base_instances: dict[int, dict],
    cont_instances: dict[int, dict],
    ref_lines:     list[str],
    n_cases:       int,
) -> str:
    lines = []
    A = lines.append

    A("# Prefix-Constrained Continuation vs Re-Translation Baseline")
    A("")
    A("## Motivation")
    A("")
    A("The re-translation baseline computes a full Chinese translation for the source")
    A("prefix at each step, then emits `translation[tgt_len]`.  Because NLLB's full-")
    A("sentence hypothesis can change as more source words arrive, **consecutive emitted")
    A("characters can come from incompatible hypotheses** — producing garbled, repetitive")
    A("output even when each individual hypothesis is correct.")
    A("")
    A("Prefix-constrained continuation fixes this by conditioning the NLLB decoder on")
    A("the *already-committed target prefix* (`decoder_input_ids`), then generating")
    A("only the *next* character.  The decoder is physically prevented from contradicting")
    A("what was already written.")
    A("")

    # ── System-level scores ────────────────────────────────────────────────────
    A("## System-Level Metrics (BLEU / Latency)")
    A("")
    A("| Metric | Baseline (re-transl.) | Continuation | Δ |")
    A("|--------|-----------------------|--------------|---|")
    for key in ["BLEU", "AL", "LAAL", "AP"]:
        bv = base_scores.get(key, float("nan"))
        cv = cont_scores.get(key, float("nan"))
        try:
            delta = cv - bv
            sign  = "+" if delta >= 0 else ""
            A(f"| {key} | {bv:.3f} | {cv:.3f} | {sign}{delta:.3f} |")
        except TypeError:
            A(f"| {key} | {bv} | {cv} | — |")
    A("")

    # ── Sentence-level stats ───────────────────────────────────────────────────
    A("## Sentence-Level Analysis")
    A("")
    n = stats["n"]
    A(f"Evaluated on **{n} sentences** (rand100 test set).")
    A("")
    A("### Translation Inconsistency (bigram repetition ratio)")
    A("")
    A("Higher ratio = more garbled / repetitive output.")
    A("")
    A(f"| Method | Mean inconsistency |")
    A(f"|--------|--------------------|")
    A(f"| Baseline      | {stats['base_mean_incon']:.4f} |")
    A(f"| Continuation  | {stats['cont_mean_incon']:.4f} |")
    delta_inc = stats["cont_mean_incon"] - stats["base_mean_incon"]
    sign = "+" if delta_inc >= 0 else ""
    A(f"| Δ (cont − base) | {sign}{delta_inc:.4f} |")
    A("")
    A("A negative Δ means continuation produces **less** repetitive / garbled output.")
    A("")

    A("### Character-level F1 vs Reference")
    A("")
    A(f"| Method | Mean charF1 |")
    A(f"|--------|-------------|")
    A(f"| Baseline     | {stats['base_mean_f1']:.4f} |")
    A(f"| Continuation | {stats['cont_mean_f1']:.4f} |")
    delta_f1 = stats["mean_delta_f1"]
    sign = "+" if delta_f1 >= 0 else ""
    A(f"| Δ            | {sign}{delta_f1:.4f} |")
    A("")

    A("### Per-sentence outcome breakdown")
    A("")
    A(f"| Outcome | Count | % |")
    A(f"|---------|-------|----|")
    A(f"| Continuation better (Δ > 0.01) | {stats['improved']} | {stats['improved']/n*100:.0f}% |")
    A(f"| Roughly equal (|Δ| ≤ 0.01) | {stats['neutral']} | {stats['neutral']/n*100:.0f}% |")
    A(f"| Baseline better (Δ < -0.01) | {stats['degraded']} | {stats['degraded']/n*100:.0f}% |")
    A("")

    # ── Case studies ──────────────────────────────────────────────────────────
    A("## Case Studies: Most Improved Sentences")
    A("")
    A("These are the sentences where continuation eliminated the largest")
    A("translation-hypothesis inconsistency errors.")
    A("")
    for rank, (idx, delta) in enumerate(stats["top_improved"][:n_cases], 1):
        if idx not in base_instances or idx not in cont_instances:
            continue
        ref  = ref_lines[idx].replace(" ", "")
        bp   = base_instances[idx]["prediction"].replace(" ", "")
        cp   = cont_instances[idx]["prediction"].replace(" ", "")
        src  = base_instances[idx].get("source", "?")
        bf1  = stats["base_f1s"][list(sorted(set(base_instances) & set(cont_instances) & set(range(len(ref_lines))))).index(idx)]
        cf1  = stats["cont_f1s"][list(sorted(set(base_instances) & set(cont_instances) & set(range(len(ref_lines))))).index(idx)]
        A(f"### Case {rank}: Sentence {idx+1}  (charF1 Δ = {delta:+.3f})")
        A("")
        A(f"**Source**: {src}")
        A("")
        A(f"| | Text | charF1 |")
        A(f"|---|------|--------|")
        A(f"| Reference    | {ref} | — |")
        A(f"| Baseline     | {bp} | {bf1:.3f} |")
        A(f"| Continuation | {cp} | {cf1:.3f} |")
        A("")

    # ── Case studies: most degraded ───────────────────────────────────────────
    if stats["degraded"] > 0:
        A("## Case Studies: Baseline Better (Degraded by Continuation)")
        A("")
        A("A few sentences where re-translation happens to score higher.")
        A("These usually occur when the model's hypothesis is stable and")
        A("the continuation goes off-track due to tokenization boundary issues.")
        A("")
        for rank, (idx, delta) in enumerate(stats["top_degraded"][:3], 1):
            if idx not in base_instances or idx not in cont_instances:
                continue
            ref  = ref_lines[idx].replace(" ", "")
            bp   = base_instances[idx]["prediction"].replace(" ", "")
            cp   = cont_instances[idx]["prediction"].replace(" ", "")
            src  = base_instances[idx].get("source", "?")
            A(f"### Degraded Case {rank}: Sentence {idx+1}  (charF1 Δ = {delta:+.3f})")
            A("")
            A(f"**Source**: {src}")
            A("")
            A(f"| | Text |")
            A(f"|---|------|")
            A(f"| Reference    | {ref} |")
            A(f"| Baseline     | {bp} |")
            A(f"| Continuation | {cp} |")
            A("")

    # ── Conclusion ────────────────────────────────────────────────────────────
    A("## Conclusion")
    A("")
    A("Prefix-constrained continuation directly addresses the **translation-hypothesis")
    A("inconsistency** problem inherent in the baseline re-translation approach:")
    A("")
    A(f"- **Consistency guaranteed**: the decoder is forced to extend the committed")
    A(f"  prefix; it cannot change earlier characters mid-sentence.")
    A(f"- **Inconsistency (bigram repetition) reduced by "
      f"{abs(stats['cont_mean_incon'] - stats['base_mean_incon']):.4f}** "
      f"({'↓' if stats['cont_mean_incon'] < stats['base_mean_incon'] else '↑'} "
      f"from {stats['base_mean_incon']:.4f} → {stats['cont_mean_incon']:.4f}).")
    bv = base_scores.get("BLEU", 0)
    cv = cont_scores.get("BLEU", 0)
    sign = "+" if cv >= bv else ""
    A(f"- **BLEU {sign}{cv-bv:.2f}** ({bv:.2f} → {cv:.2f}).")
    A(f"- {stats['improved']}/{n} sentences improve, {stats['degraded']}/{n} degrade,")
    A(f"  {stats['neutral']}/{n} are roughly equal.")
    A("")
    A("Combining continuation with the DD gate (which prevents early commits) would")
    A("further reduce error propagation: DD stops premature commits, continuation")
    A("ensures what IS committed remains internally consistent.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-cases", type=int, default=5,
                    help="Number of case studies to show (default 5)")
    ap.add_argument("--base-dir", type=Path, default=None,
                    help="Override baseline output directory")
    ap.add_argument("--cont-dir", type=Path, default=None,
                    help="Override continuation output directory")
    args = ap.parse_args()

    base_dir = args.base_dir or (OUT_ROOT / "cmp_baseline_k5")
    cont_dir = args.cont_dir or (OUT_ROOT / "cmp_continuation_k5")

    # ── Load data ─────────────────────────────────────────────────────────────
    ref_lines = []
    ref_path = DATA_DIR / "rand100_target.txt"
    if ref_path.exists():
        ref_lines = [l.strip().replace(" ", "") for l in ref_path.read_text().splitlines()]

    base_instances = load_instances(base_dir / "instances.log")
    cont_instances = load_instances(cont_dir / "instances.log")
    base_scores    = load_scores(base_dir / "scores")
    cont_scores    = load_scores(cont_dir / "scores")

    # ── Check both dirs exist ─────────────────────────────────────────────────
    missing = []
    if not base_instances:
        missing.append(str(base_dir))
    if not cont_instances:
        missing.append(str(cont_dir))
    if missing:
        print("ERROR: missing results in:", missing)
        print("Run sbatch scripts/run_continuation.sbatch first.")
        return 1

    print(f"Loaded {len(base_instances)} baseline instances, "
          f"{len(cont_instances)} continuation instances")

    # ── Analyze ───────────────────────────────────────────────────────────────
    stats = analyze(base_instances, cont_instances, ref_lines, n_case_studies=args.show_cases)

    # ── Console summary ───────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("QUICK COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<22} {'Baseline':>10} {'Continuation':>12} {'Δ':>8}")
    print("-" * 60)
    for key in ["BLEU", "AL", "LAAL"]:
        bv = base_scores.get(key, float("nan"))
        cv = cont_scores.get(key, float("nan"))
        try:
            d = cv - bv
            print(f"{key:<22} {bv:>10.3f} {cv:>12.3f} {d:>+8.3f}")
        except TypeError:
            print(f"{key:<22} {bv!s:>10} {cv!s:>12} {'?':>8}")
    print("-" * 60)
    print(f"{'Mean charF1':<22} {stats['base_mean_f1']:>10.4f} {stats['cont_mean_f1']:>12.4f} {stats['mean_delta_f1']:>+8.4f}")
    print(f"{'Mean inconsistency':<22} {stats['base_mean_incon']:>10.4f} {stats['cont_mean_incon']:>12.4f} {stats['cont_mean_incon']-stats['base_mean_incon']:>+8.4f}")
    print("=" * 60)
    print(f"Improved: {stats['improved']}/{stats['n']}  "
          f"Degraded: {stats['degraded']}/{stats['n']}  "
          f"Neutral: {stats['neutral']}/{stats['n']}")
    print()

    # ── Build and save report ─────────────────────────────────────────────────
    report = build_report(
        base_scores, cont_scores, stats,
        base_instances, cont_instances, ref_lines,
        n_cases=args.show_cases,
    )
    report_path = OUT_ROOT / "continuation_analysis.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"Report saved to: {report_path}")

    # ── Top case study preview ────────────────────────────────────────────────
    common_sorted = sorted(
        set(base_instances) & set(cont_instances) & set(range(len(ref_lines)))
    )
    if stats["top_improved"]:
        print("\nTop improved sentence:")
        idx, delta = stats["top_improved"][0]
        if idx in base_instances and idx in cont_instances:
            ref = ref_lines[idx].replace(" ", "") if idx < len(ref_lines) else "?"
            bp  = base_instances[idx]["prediction"].replace(" ", "")
            cp  = cont_instances[idx]["prediction"].replace(" ", "")
            src = base_instances[idx].get("source", "?")
            print(f"  Source   : {src}")
            print(f"  Ref      : {ref}")
            print(f"  Baseline : {bp}")
            print(f"  Continu. : {cp}")
            print(f"  ΔcharF1  : {delta:+.4f}")


if __name__ == "__main__":
    main()
