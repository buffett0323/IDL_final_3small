#!/usr/bin/env python3
"""compare_three_methods.py — Three-way comparison of translation strategies.

Compares:
  1. NLLB baseline (re-translation, take t-th char)
  2. NLLB continuation (forced decoder prefix via decoder_input_ids)
  3. Qwen3-4B-Base continuation (causal LM, append committed to prompt)

Outputs:
  - Console table with key metrics
  - outputs/three_method_report.md  — full report with case studies
  - outputs/three_method_plot.png   — quality vs latency plot

Usage:
    cd /data/user_data/haolingp/IDL_final_3small
    python3 scripts/compare_three_methods.py
    python3 scripts/compare_three_methods.py --show-cases 8
"""
from __future__ import annotations

import argparse
import json
import unicodedata
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs"
DATA_DIR = ROOT / "data/enzh"

# ─────────────────────────────────────────────────────────────────────────────
# Loaders
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
    """Parse simuleval scores file.

    SimulEval writes a pandas-style table:
        BLEU  LAAL   AL    AP   DAL  ATD
    0  13.46  8.82  8.80  0.583  7.53  0.0

    The data row starts with a row-index token ('0') that must be skipped.
    """
    if not path.exists():
        return {}
    text = path.read_text()
    result: dict[str, float] = {}
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) >= 2:
        headers = lines[0].split()
        values  = lines[1].split()[1:]   # skip leading row-index token
        for h, v in zip(headers, values):
            try:
                result[h] = float(v)
            except ValueError:
                pass
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def char_f1(pred: str, ref: str) -> float:
    pred_c = list(pred.replace(" ", ""))
    ref_c  = list(ref.replace(" ", ""))
    if not pred_c or not ref_c:
        return 0.0
    ps, rs = {}, {}
    for c in pred_c: ps[c] = ps.get(c, 0) + 1
    for c in ref_c:  rs[c] = rs.get(c, 0) + 1
    overlap = sum(min(ps.get(c, 0), rs.get(c, 0)) for c in rs)
    prec = overlap / len(pred_c)
    rec  = overlap / len(ref_c)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def inconsistency(pred: str) -> float:
    """Bigram repetition ratio — proxy for garbled output."""
    chars = [c for c in pred if not unicodedata.category(c).startswith("Z")]
    if len(chars) < 4:
        return 0.0
    bigrams = [chars[i] + chars[i+1] for i in range(len(chars) - 1)]
    return 1.0 - len(set(bigrams)) / len(bigrams)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────────────────────────────────────

METHODS = [
    ("NLLB baseline",         "cmp_baseline_k5"),
    ("NLLB continuation k5",  "cmp_continuation_k5"),
    ("Qwen4B-Base cont k5",   "cmp_qwen_continuation_k5"),
    ("Qwen30B-Inst cont k5",  "cmp_qwen30b_cont_k5"),
    ("Qwen30B-Inst cont k7",  "cmp_qwen30b_cont_k7"),
    ("Qwen30B-Inst cont k9",  "cmp_qwen30b_cont_k9"),
]


def gather_all(ref_lines: list[str]):
    results = {}
    for label, dirname in METHODS:
        d = OUT_ROOT / dirname
        instances = load_instances(d / "instances.log")
        scores    = load_scores(d / "scores")
        f1s, incons = [], []
        for idx, inst in instances.items():
            if idx < len(ref_lines):
                ref = ref_lines[idx]
                pred = inst["prediction"].replace(" ", "")
                f1s.append(char_f1(pred, ref))
                incons.append(inconsistency(pred))
        results[label] = {
            "instances": instances,
            "scores":    scores,
            "mean_f1":   sum(f1s) / len(f1s) if f1s else 0.0,
            "mean_incon": sum(incons) / len(incons) if incons else 0.0,
            "f1s":       f1s,
        }
    return results


def rank_improvements(results: dict, ref_lines: list[str], top_n: int = 5):
    """Find sentences where Qwen cont >> NLLB baseline."""
    base_data  = results["NLLB baseline"]
    qwen_data  = results["Qwen continuation"]
    nllb_data  = results["NLLB continuation"]

    common = sorted(
        set(base_data["instances"]) &
        set(qwen_data["instances"]) &
        set(nllb_data["instances"]) &
        set(range(len(ref_lines)))
    )

    deltas = []
    for idx in common:
        ref  = ref_lines[idx]
        bf1  = char_f1(base_data["instances"][idx]["prediction"].replace(" ",""), ref)
        qf1  = char_f1(qwen_data["instances"][idx]["prediction"].replace(" ",""), ref)
        nf1  = char_f1(nllb_data["instances"][idx]["prediction"].replace(" ",""), ref)
        deltas.append({
            "idx":      idx,
            "base_f1":  bf1,
            "nllb_f1":  nf1,
            "qwen_f1":  qf1,
            "delta_qb": qf1 - bf1,   # qwen vs baseline
            "delta_nb": nf1 - bf1,   # nllb-cont vs baseline
        })

    top_improved = sorted(deltas, key=lambda x: -x["delta_qb"])[:top_n]
    top_degraded = sorted(deltas, key=lambda x: x["delta_qb"])[:3]
    return deltas, top_improved, top_degraded


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def build_report(
    results:     dict,
    ref_lines:   list[str],
    n_cases:     int,
) -> str:
    deltas, top_improved, top_degraded = rank_improvements(results, ref_lines, n_cases)
    lines = []
    A = lines.append

    A("# Three-Way Comparison: Translation Strategies for EN→ZH SiMT")
    A("")
    A("## Methods")
    A("")
    A("| # | Method | Core Mechanism |")
    A("|---|--------|----------------|")
    A("| 1 | **NLLB baseline** | Re-translate full source prefix at each step; emit `translation[tgt_len]`. Causes hypothesis inconsistency. |")
    A("| 2 | **NLLB continuation** | Force NLLB decoder with committed prefix via `decoder_input_ids`. Eliminates inconsistency but suffers tokenisation boundary mismatch. |")
    A("| 3 | **Qwen continuation** | Append committed Chinese to few-shot prompt; causal LM generates the next token naturally. No encoder-decoder mismatch. |")
    A("")

    # System-level metrics table
    A("## System-Level Scores")
    A("")
    A("| Method | BLEU | AL | LAAL | AP | Mean charF1 | Mean inconsistency |")
    A("|--------|------|----|------|----|--------------|--------------------|")
    for label, _ in METHODS:
        r  = results[label]
        sc = r["scores"]
        bleu = sc.get("BLEU", float("nan"))
        al   = sc.get("AL",   float("nan"))
        laal = sc.get("LAAL", float("nan"))
        ap   = sc.get("AP",   float("nan"))
        f1   = r["mean_f1"]
        inc  = r["mean_incon"]
        try:
            A(f"| {label} | {bleu:.2f} | {al:.2f} | {laal:.2f} | {ap:.3f} | {f1:.4f} | {inc:.4f} |")
        except TypeError:
            A(f"| {label} | — | — | — | — | {f1:.4f} | {inc:.4f} |")
    A("")

    # Outcome breakdown
    n = len(deltas)
    qb_imp = sum(1 for d in deltas if d["delta_qb"] >  0.01)
    qb_deg = sum(1 for d in deltas if d["delta_qb"] < -0.01)
    qb_neu = n - qb_imp - qb_deg
    nb_imp = sum(1 for d in deltas if d["delta_nb"] >  0.01)
    nb_deg = sum(1 for d in deltas if d["delta_nb"] < -0.01)
    nb_neu = n - nb_imp - nb_deg

    A("## Per-Sentence Outcome (vs NLLB Baseline)")
    A("")
    A(f"| Method | Better (Δ>0.01) | Equal | Worse (Δ<-0.01) |")
    A(f"|--------|-----------------|-------|-----------------|")
    A(f"| NLLB continuation | {nb_imp} ({nb_imp/n:.0%}) | {nb_neu} | {nb_deg} ({nb_deg/n:.0%}) |")
    A(f"| Qwen continuation | {qb_imp} ({qb_imp/n:.0%}) | {qb_neu} | {qb_deg} ({qb_deg/n:.0%}) |")
    A("")

    # Case studies
    A(f"## Top {n_cases} Sentences: Qwen Most Improved vs NLLB Baseline")
    A("")
    for rank, d in enumerate(top_improved, 1):
        idx = d["idx"]
        ref  = ref_lines[idx].replace(" ", "")
        src  = results["NLLB baseline"]["instances"][idx].get("source", "?")
        bpred = results["NLLB baseline"]["instances"][idx]["prediction"].replace(" ", "")
        npred = results["NLLB continuation"]["instances"][idx]["prediction"].replace(" ", "")
        qpred = results["Qwen continuation"]["instances"][idx]["prediction"].replace(" ", "")
        A(f"### Case {rank}: Sent {idx+1}  (Qwen charF1 Δ vs baseline = {d['delta_qb']:+.3f})")
        A("")
        A(f"**Source**: {src}")
        A("")
        A(f"| Method | Output | charF1 |")
        A(f"|--------|--------|--------|")
        A(f"| Reference       | {ref} | — |")
        A(f"| NLLB baseline   | {bpred} | {d['base_f1']:.3f} |")
        A(f"| NLLB cont.      | {npred} | {d['nllb_f1']:.3f} |")
        A(f"| Qwen cont.      | {qpred} | {d['qwen_f1']:.3f} |")
        A("")

    if top_degraded:
        A("## Sentences Where Qwen Continuation Degrades")
        A("")
        for rank, d in enumerate(top_degraded, 1):
            idx = d["idx"]
            ref   = ref_lines[idx].replace(" ", "")
            src   = results["NLLB baseline"]["instances"][idx].get("source", "?")
            bpred = results["NLLB baseline"]["instances"][idx]["prediction"].replace(" ", "")
            qpred = results["Qwen continuation"]["instances"][idx]["prediction"].replace(" ", "")
            A(f"### Degraded Case {rank}: Sent {idx+1}  (Δ = {d['delta_qb']:+.3f})")
            A("")
            A(f"**Source**: {src}")
            A(f"| | Output |")
            A(f"|---|--------|")
            A(f"| Reference      | {ref} |")
            A(f"| NLLB baseline  | {bpred} |")
            A(f"| Qwen cont.     | {qpred} |")
            A("")

    # Discussion
    A("## Discussion")
    A("")
    A("### Why NLLB continuation falls short")
    A("")
    A("NLLB uses a seq2seq architecture: the encoder processes the source, the decoder")
    A("generates the target starting from `[decoder_start, lang_token]`. When we force")
    A("`decoder_input_ids = [decoder_start, zho_Hans, *committed_tokens]`, the model")
    A("must continue from an ARBITRARY Chinese prefix that was generated from a SHORTER")
    A("source prefix. The decoder cannot resolve the ambiguity of a single character")
    A("prefix (e.g., \"美\" could be the start of \"美国\", \"美联储\", \"美好\", etc.).")
    A("Result: lower BLEU than baseline despite reduced inconsistency.")
    A("")
    A("### Why Qwen continuation is more robust")
    A("")
    A("A causal LM processes source AND committed target as a unified token sequence.")
    A("The prompt ending with the committed Chinese text provides unambiguous context:")
    A("the model has seen `English: ... Chinese: 美国的` and knows exactly what was")
    A("meant, then continues naturally. There is no encoder-decoder context gap.")
    A("")
    A("### Remaining limitations of Qwen base model")
    A("")
    A("1. **No fine-tuning**: Qwen3-4B-Base was not fine-tuned for SiMT. It may output")
    A("   mixed Simplified/Traditional Chinese.")
    A("2. **Early-commit still happens**: wait-k determines when to start writing;")
    A("   combining Qwen continuation with DD gate would further reduce early-commit risk.")
    A("3. **Speed**: each character requires one forward pass through 4B parameters.")
    A("   Caching the static prompt prefix with KV cache would reduce this overhead.")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def make_plot(results: dict):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Plot] matplotlib not available, skipping.")
        return

    METHOD_STYLE = {
        "NLLB baseline":     dict(color="#2196F3", marker="o", s=120),
        "NLLB continuation": dict(color="#FF9800", marker="D", s=120),
        "Qwen continuation": dict(color="#9C27B0", marker="*", s=200),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Translation Strategy Comparison  (wait-k=5, EN→ZH, 100 sentences)",
                 fontsize=12, fontweight="bold")

    for ax_idx, (x_key, xlabel) in enumerate([
        ("AL",   "Average Lagging (AL)  ← lower is better"),
        ("LAAL", "LAAL  ← lower is better"),
    ]):
        ax = axes[ax_idx]
        for label, _ in METHODS:
            sc = results[label]["scores"]
            x  = sc.get(x_key)
            y  = sc.get("BLEU")
            if x is None or y is None:
                continue
            style = METHOD_STYLE.get(label, dict(color="gray", marker="x", s=80))
            ax.scatter(float(x), float(y), label=label, zorder=5, **style)
            ax.annotate(
                label.replace(" continuation", "\ncont.").replace(" baseline", "\nbaseline"),
                (float(x), float(y)), textcoords="offset points",
                xytext=(6, 4), fontsize=8,
            )
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("BLEU  ↑ higher is better", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    plot_path = OUT_ROOT / "three_method_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[Plot] Saved → {plot_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show-cases", type=int, default=5)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    ref_path = DATA_DIR / "rand100_target.txt"
    ref_lines = []
    if ref_path.exists():
        ref_lines = [l.strip().replace(" ", "") for l in ref_path.read_text().splitlines()]

    # Check which experiments are available
    results = gather_all(ref_lines)
    missing = [label for label, _ in METHODS if not results[label]["instances"]]
    if missing:
        print("WARNING: missing results for:", missing)
        print("Run:  sbatch scripts/run_qwen_continuation.sbatch")
        if len(missing) == len(METHODS):
            return 1

    # ── Console table ──────────────────────────────────────────────────────────
    print()
    print("=" * 72)
    print("THREE-METHOD COMPARISON  (wait-k=5, EN→ZH, rand100)")
    print("=" * 72)
    print(f"{'Method':<22} {'BLEU':>6} {'AL':>6} {'LAAL':>6} {'AP':>6} {'charF1':>8} {'Incon':>7}")
    print("-" * 72)
    for label, _ in METHODS:
        r  = results[label]
        sc = r["scores"]
        bleu = sc.get("BLEU", float("nan"))
        al   = sc.get("AL",   float("nan"))
        laal = sc.get("LAAL", float("nan"))
        ap_v = sc.get("AP",   float("nan"))
        f1   = r["mean_f1"]
        inc  = r["mean_incon"]
        try:
            print(f"{label:<22} {bleu:>6.2f} {al:>6.2f} {laal:>6.2f} {ap_v:>6.3f} {f1:>8.4f} {inc:>7.4f}")
        except TypeError:
            print(f"{label:<22}  {'(no data)':>40}")
    print("=" * 72)

    # ── Report ─────────────────────────────────────────────────────────────────
    report = build_report(results, ref_lines, args.show_cases)
    rp = OUT_ROOT / "three_method_report.md"
    rp.write_text(report, encoding="utf-8")
    print(f"\nReport → {rp}")

    # ── Plot ───────────────────────────────────────────────────────────────────
    if not args.no_plot:
        make_plot(results)


if __name__ == "__main__":
    main()
