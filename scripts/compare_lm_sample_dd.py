#!/usr/bin/env python3
"""compare_lm_sample_dd.py — Oracle future DD vs LM-sampled future DD.

Ablation experiment comparing:
  1. NLLB baseline        (wait-k=5, no DD)
  2. DD oracle  tau=0.05  (truncation futures — upper bound)
  3. DD lm_sample tau=0.05 (Qwen3-4B generates K English futures)
  4. DD lm_sample tau=0.03 (tighter threshold)

Key question: does lm_sample DD approach oracle DD quality?
If yes → DD is deployable without oracle knowledge (real-time streaming).

Outputs:
  - Console table
  - outputs/lm_dd_comparison.md
  - outputs/lm_dd_plot.png

Usage:
    cd /data/user_data/haolingp/IDL_final_3small
    python3 scripts/compare_lm_sample_dd.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs"


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_scores(path: Path) -> dict[str, float]:
    """Parse simuleval scores file (handles leading Pandas row-index token)."""
    if not path.exists():
        return {}
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values  = lines[1].split()[1:]   # skip leading row-index token
    result: dict[str, float] = {}
    for h, v in zip(headers, values):
        try:
            result[h] = float(v)
        except ValueError:
            pass
    return result


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


def load_dd_trace(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS = [
    {
        "label":   "Baseline k=5",
        "dir":     "lmsdd_baseline_k5",
        "marker":  "o",
        "color":   "#555555",
        "linestyle": "--",
    },
    {
        "label":   "DD oracle τ=0.05",
        "dir":     "lmsdd_oracle_tau0.05",
        "marker":  "s",
        "color":   "#2196F3",
        "linestyle": "-",
    },
    {
        "label":   "DD oracle τ=0.03",
        "dir":     "lmsdd_oracle_tau0.03",
        "marker":  "s",
        "color":   "#0D47A1",
        "linestyle": "-",
    },
    {
        "label":   "DD lm_sample τ=0.05",
        "dir":     "lmsdd_lm_tau0.05",
        "marker":  "^",
        "color":   "#FF9800",
        "linestyle": "-",
    },
    {
        "label":   "DD lm_sample τ=0.03",
        "dir":     "lmsdd_lm_tau0.03",
        "marker":  "D",
        "color":   "#E91E63",
        "linestyle": "-",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Analysis helpers
# ─────────────────────────────────────────────────────────────────────────────

def analyse_futures(trace: list[dict]) -> dict:
    """Analyse properties of LM-sampled futures in the DD trace."""
    if not trace:
        return {}
    total = 0
    unique_totals = 0
    read_count = 0
    commit_count = 0
    for rec in trace:
        futures = rec.get("futures", [])
        if futures:
            total += 1
            unique_totals += len(set(futures))
            if rec.get("decision") == "READ":
                read_count += 1
            else:
                commit_count += 1
    if total == 0:
        return {}
    return {
        "total_decisions": total,
        "avg_unique_futures": unique_totals / total,
        "read_rate": read_count / total,
        "commit_rate": commit_count / total,
        "read_count": read_count,
        "commit_count": commit_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  Oracle DD vs LM-sampled DD  —  Ablation Comparison")
    print("=" * 70)

    rows = []
    for exp in EXPERIMENTS:
        exp_dir = OUT_ROOT / exp["dir"]
        scores  = load_scores(exp_dir / "scores")
        trace   = load_dd_trace(exp_dir / "dd_trace.jsonl")

        bleu  = scores.get("BLEU",  float("nan"))
        laal  = scores.get("LAAL",  float("nan"))
        al    = scores.get("AL",    float("nan"))
        ap    = scores.get("AP",    float("nan"))

        future_stats = analyse_futures(trace)
        read_rate    = future_stats.get("read_rate",          float("nan"))
        avg_unique   = future_stats.get("avg_unique_futures", float("nan"))

        status = "OK" if scores else "MISSING"
        rows.append({**exp, "bleu": bleu, "laal": laal, "al": al, "ap": ap,
                     "read_rate": read_rate, "avg_unique": avg_unique,
                     "status": status, "trace_n": len(trace)})

    # ── Console table ────────────────────────────────────────────────────────
    header = f"{'Method':<30} {'BLEU':>6} {'AL':>6} {'LAAL':>6} {'AP':>6}  {'ReadRate':>9} {'AvgUniqF':>9}"
    print("\n" + header)
    print("-" * len(header))
    baseline_bleu = next((r["bleu"] for r in rows if "Baseline" in r["label"]), None)
    for r in rows:
        delta = ""
        if baseline_bleu is not None and "Baseline" not in r["label"]:
            d = r["bleu"] - baseline_bleu
            delta = f"  ({'+' if d >= 0 else ''}{d:.2f})"
        rr = f"{r['read_rate']:.1%}" if not _isnan(r["read_rate"]) else "  —"
        au = f"{r['avg_unique']:.2f}"  if not _isnan(r["avg_unique"]) else "  —"
        status_tag = "" if r["status"] == "OK" else " [MISSING]"
        print(
            f"{r['label']:<30} {r['bleu']:>6.2f} {r['al']:>6.2f} {r['laal']:>6.2f} "
            f"{r['ap']:>6.3f}  {rr:>9} {au:>9}{delta}{status_tag}"
        )

    # ── Plot: quality vs latency ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Oracle DD vs LM-sampled Future DD — Quality & Latency", fontsize=13)

    ax_bl  = axes[0]   # BLEU vs AL
    ax_laal = axes[1]  # BLEU vs LAAL

    for r in rows:
        if _isnan(r["bleu"]):
            continue
        for ax, x_key in [(ax_bl, "al"), (ax_laal, "laal")]:
            if _isnan(r[x_key]):
                continue
            ax.scatter(r[x_key], r["bleu"], marker=r["marker"], color=r["color"],
                       s=120, zorder=5)
            ax.annotate(r["label"], (r[x_key], r["bleu"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

    ax_bl.set_xlabel("AL (Average Lagging, words)")
    ax_bl.set_ylabel("BLEU")
    ax_bl.set_title("BLEU vs AL")
    ax_bl.grid(True, alpha=0.3)

    ax_laal.set_xlabel("LAAL (Length-Adaptive AL, words)")
    ax_laal.set_ylabel("BLEU")
    ax_laal.set_title("BLEU vs LAAL")
    ax_laal.grid(True, alpha=0.3)

    # Legend patches
    patches = [mpatches.Patch(color=r["color"], label=r["label"]) for r in rows]
    fig.legend(handles=patches, loc="lower center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_plot = OUT_ROOT / "lm_dd_plot.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] Saved → {out_plot}")

    # ── Markdown report ───────────────────────────────────────────────────────
    lines = [
        "# Oracle DD vs LM-sampled Future DD — Ablation Report\n",
        "## Summary Table\n",
        "| Method | BLEU | AL | LAAL | AP | ReadRate | AvgUniqFutures |",
        "|--------|------|-----|------|-----|---------|----------------|",
    ]
    for r in rows:
        bleu_s = f"{r['bleu']:.2f}" if not _isnan(r["bleu"]) else "—"
        al_s   = f"{r['al']:.2f}"   if not _isnan(r["al"])   else "—"
        laal_s = f"{r['laal']:.2f}" if not _isnan(r["laal"]) else "—"
        ap_s   = f"{r['ap']:.3f}"   if not _isnan(r["ap"])   else "—"
        rr_s   = f"{r['read_rate']:.1%}" if not _isnan(r["read_rate"]) else "—"
        au_s   = f"{r['avg_unique']:.2f}" if not _isnan(r["avg_unique"]) else "—"
        lines.append(f"| {r['label']} | {bleu_s} | {al_s} | {laal_s} | {ap_s} | {rr_s} | {au_s} |")

    lines += [
        "\n## Key Questions\n",
        "### Does lm_sample DD approach oracle DD quality?",
        "- Oracle futures are an upper bound (know the real future).",
        "- LM-sampled futures are plausible but not ground-truth.",
        "- If BLEU gap is small (< 1.0), lm_sample is practically viable.",
        "- If BLEU gap is large, oracle futures provide better future diversity.\n",
        "### What does AvgUniqFutures tell us?",
        "- LM-sampled futures should be diverse (high temperature → unique futures).",
        "- If AvgUniqFutures ≈ K, futures are diverse (good).",
        "- If AvgUniqFutures < K, LM collapses futures → poor DD signal.\n",
        "### Read rate vs quality trade-off",
        "- Higher read rate = more conservative = better quality but higher latency.",
        "- If lm_sample has higher read rate than oracle but similar BLEU,",
        "  it may be over-reading due to noisier JS estimates.\n",
        "## Ablation Interpretation",
        "This ablation answers: can DD work in real-time without oracle futures?",
        "- DD oracle: best-case ceiling (uses actual future words).",
        "- DD lm_sample: realistic deployment (no oracle, LM guesses future).",
        "- The gap measures how much we lose by not knowing the real future.\n",
        f"![Quality-Latency Plot](lm_dd_plot.png)\n",
    ]

    out_md = OUT_ROOT / "lm_dd_comparison.md"
    out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Saved → {out_md}")


def _isnan(x):
    try:
        import math
        return math.isnan(x)
    except (TypeError, ValueError):
        return True


if __name__ == "__main__":
    main()
