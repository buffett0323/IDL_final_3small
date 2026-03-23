#!/usr/bin/env python3
"""compare_semantic_lcp.py — DD gate vs Semantic LCP comparison.

Compares:
  1. Baseline (NLLB wait-k=5)
  2. DD oracle  tau=0.05           (from previous experiments)
  3. DD lm_sample tau=0.05         (from previous experiments)
  4. Semantic LCP  K=4 futures     (Qwen30B translate + quorum LCP)
  5. Semantic LCP  K=6 futures

Usage:
    cd /data/user_data/haolingp/IDL_final_3small
    python3 scripts/compare_semantic_lcp.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT     = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs"


def load_scores(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    lines = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    if len(lines) < 2:
        return {}
    headers = lines[0].split()
    values  = lines[1].split()[1:]
    result: dict[str, float] = {}
    for h, v in zip(headers, values):
        try:
            result[h] = float(v)
        except ValueError:
            pass
    return result


def load_lcp_trace(path: Path) -> list[dict]:
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


def analyse_lcp_trace(trace: list[dict]) -> dict:
    if not trace:
        return {}
    total = len(trace)
    committed = sum(1 for r in trace if r.get("delta"))
    read_steps = total - committed
    avg_delta_len = (
        sum(len(r.get("delta", "")) for r in trace if r.get("delta")) / max(committed, 1)
    )
    avg_candidates = (
        sum(len(r.get("candidates", [])) for r in trace) / total
    )
    return {
        "total_steps": total,
        "commit_rate": committed / total,
        "read_rate": read_steps / total,
        "avg_delta_len": avg_delta_len,
        "avg_candidates": avg_candidates,
    }


EXPERIMENTS = [
    {
        "label":  "Baseline NLLB k=5",
        "dir":    "semlcp_baseline_k5",
        "marker": "o", "color": "#555555", "group": "baseline",
    },
    {
        "label":  "DD oracle τ=0.05",
        "dir":    "lmsdd_oracle_tau0.05",
        "marker": "s", "color": "#2196F3", "group": "dd",
    },
    {
        "label":  "DD lm_sample τ=0.05",
        "dir":    "lmsdd_lm_tau0.05",
        "marker": "^", "color": "#FF9800", "group": "dd",
    },
    {
        "label":  "Qwen30B direct k=5",   # fair baseline: same model, no futures
        "dir":    "semlcp_direct_k5",
        "marker": "X", "color": "#795548", "group": "qwen_baseline",
    },
    {
        "label":  "Qwen30B direct k=3",
        "dir":    "semlcp_direct_k3",
        "marker": "X", "color": "#A1887F", "group": "qwen_baseline",
    },
    {
        "label":  "Semantic LCP K=4",     # Qwen30B + future consensus
        "dir":    "semlcp_main_k5_f4",
        "marker": "D", "color": "#9C27B0", "group": "semlcp",
    },
    {
        "label":  "Semantic LCP K=6",
        "dir":    "semlcp_main_k5_f6",
        "marker": "P", "color": "#E91E63", "group": "semlcp",
    },
]


def _nan(x):
    try:
        return math.isnan(x)
    except (TypeError, ValueError):
        return True


def main():
    print("\n" + "=" * 72)
    print("  Baseline  vs  DD gate  vs  Semantic LCP  — Full Comparison")
    print("=" * 72)

    rows = []
    for exp in EXPERIMENTS:
        exp_dir = OUT_ROOT / exp["dir"]
        scores  = load_scores(exp_dir / "scores")
        trace   = load_lcp_trace(exp_dir / "lcp_trace.jsonl")
        lcp_stats = analyse_lcp_trace(trace)

        bleu = scores.get("BLEU", float("nan"))
        al   = scores.get("AL",   float("nan"))
        laal = scores.get("LAAL", float("nan"))
        ap   = scores.get("AP",   float("nan"))

        commit_rate = lcp_stats.get("commit_rate", float("nan"))
        avg_delta   = lcp_stats.get("avg_delta_len", float("nan"))

        rows.append({
            **exp,
            "bleu": bleu, "al": al, "laal": laal, "ap": ap,
            "commit_rate": commit_rate, "avg_delta": avg_delta,
            "status": "OK" if scores else "MISSING",
        })

    # Console table
    hdr = f"{'Method':<28} {'BLEU':>6} {'AL':>6} {'LAAL':>6} {'AP':>6}  {'CommitRate':>10} {'AvgDelta':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    baseline_bleu = next((r["bleu"] for r in rows if r["group"] == "baseline"), None)
    for r in rows:
        delta_str = ""
        if baseline_bleu and not _nan(r["bleu"]) and r["group"] != "baseline":
            d = r["bleu"] - baseline_bleu
            delta_str = f"  ({'+' if d >= 0 else ''}{d:.2f})"
        cr = f"{r['commit_rate']:.1%}" if not _nan(r["commit_rate"]) else "   —"
        ad = f"{r['avg_delta']:.1f}ch"  if not _nan(r["avg_delta"]) else "   —"
        miss = " [MISSING]" if r["status"] != "OK" else ""
        print(
            f"{r['label']:<28} {r['bleu']:>6.2f} {r['al']:>6.2f} {r['laal']:>6.2f} "
            f"{r['ap']:>6.3f}  {cr:>10} {ad:>9}{delta_str}{miss}"
        )

    # Quality-Latency plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Baseline vs DD gate vs Semantic LCP — Quality & Latency", fontsize=13
    )

    group_zorder = {"baseline": 3, "dd": 4, "semlcp": 5}
    for r in rows:
        if _nan(r["bleu"]):
            continue
        for ax, x_key in [(ax1, "al"), (ax2, "laal")]:
            if _nan(r[x_key]):
                continue
            ax.scatter(
                r[x_key], r["bleu"],
                marker=r["marker"], color=r["color"], s=130,
                zorder=group_zorder.get(r["group"], 4),
            )
            ax.annotate(
                r["label"], (r[x_key], r["bleu"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8,
            )

    for ax, xlabel in [
        (ax1, "AL (Average Lagging, words)"),
        (ax2, "LAAL (Length-Adaptive AL, words)"),
    ]:
        ax.set_xlabel(xlabel)
        ax.set_ylabel("BLEU")
        ax.grid(True, alpha=0.3)

    ax1.set_title("BLEU vs AL")
    ax2.set_title("BLEU vs LAAL")

    patches = [mpatches.Patch(color=r["color"], label=r["label"]) for r in rows]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))

    plt.tight_layout(rect=[0, 0.08, 1, 1])
    out_plot = OUT_ROOT / "semantic_lcp_plot.png"
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"\n[Plot] → {out_plot}")

    # Markdown report
    md = [
        "# Baseline vs DD gate vs Semantic LCP — Comparison Report\n",
        "## Methods\n",
        "| Method | Description |",
        "|--------|-------------|",
        "| Baseline NLLB k=5 | Pure wait-k, character-by-character NLLB translation |",
        "| DD oracle τ=0.05 | Distribution Divergence gate with oracle futures |",
        "| DD lm_sample τ=0.05 | DD gate with Qwen3-4B sampled futures (no oracle) |",
        "| Semantic LCP K=4 | Qwen30B translates 4 futures; quorum-60% LCP committed |",
        "| Semantic LCP K=6 | Same with 6 futures (higher consensus quality) |",
        "\n## Results\n",
        "| Method | BLEU | AL | LAAL | AP | CommitRate | AvgDelta |",
        "|--------|------|-----|------|-----|------------|---------|",
    ]
    for r in rows:
        b = f"{r['bleu']:.2f}" if not _nan(r["bleu"]) else "—"
        a = f"{r['al']:.2f}"   if not _nan(r["al"])   else "—"
        l = f"{r['laal']:.2f}" if not _nan(r["laal"]) else "—"
        p = f"{r['ap']:.3f}"   if not _nan(r["ap"])   else "—"
        c = f"{r['commit_rate']:.1%}" if not _nan(r["commit_rate"]) else "—"
        d = f"{r['avg_delta']:.1f}"   if not _nan(r["avg_delta"])  else "—"
        md.append(f"| {r['label']} | {b} | {a} | {l} | {p} | {c} | {d}ch |")

    md += [
        "\n## Key Interpretation\n",
        "### What Semantic LCP adds over DD\n",
        "- DD gate: decides READ vs COMMIT based on JS divergence signal.",
        "- Semantic LCP: decides WHAT to commit based on translation consensus.",
        "- Semantic LCP uses Qwen30B (much higher quality than NLLB) for translation.",
        "- If Semantic LCP BLEU >> DD BLEU at similar latency: quality gain from Qwen30B.",
        "- If Semantic LCP BLEU ≈ DD BLEU: the gating mechanism (not translation quality) is the bottleneck.\n",
        "### AvgDelta\n",
        "- Characters committed per step. Higher = more aggressive commits.",
        "- Very low AvgDelta may indicate over-conservatism (too many READ steps).\n",
        f"![Quality-Latency Plot](semantic_lcp_plot.png)\n",
    ]
    out_md = OUT_ROOT / "semantic_lcp_report.md"
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"[Report] → {out_md}")


if __name__ == "__main__":
    main()
