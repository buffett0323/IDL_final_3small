#!/usr/bin/env python3
"""compare_waitk_dd.py — Aggregate results and plot quality-latency curves.

Reads all outputs/cmp_* score files produced by run_comparison.sbatch and:
  1. Prints a formatted comparison table to stdout.
  2. Writes outputs/comparison_report.md with detailed table + notes.
  3. Saves outputs/comparison_plot.png — a quality (BLEU) vs latency (AL)
     scatter / line plot comparing:
       • baseline wait-k  (k = 3, 5, 7, 9)
       • DD full gate       (tau = 0.03, 0.05)
       • DD veto            (tau = 0.03, 0.05)

Usage:
    cd /data/user_data/haolingp/IDL_final_3small
    python scripts/compare_waitk_dd.py

    # Only print table, skip plot:
    python scripts/compare_waitk_dd.py --no-plot

    # Include additional output directories (glob):
    python scripts/compare_waitk_dd.py --extra-dirs outputs/dd_sweep_baseline
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs"


# ── Score parsing ─────────────────────────────────────────────────────────────

def parse_scores(path: Path) -> Optional[dict]:
    """Parse a SimulEval scores file into a dict of metric->float."""
    if not path.exists():
        return None
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if len(lines) < 2:
            return None
        header = lines[0].split()
        vals   = lines[1].split()[1:]   # skip row index
        result = {}
        for k, v in zip(header, vals):
            try:
                result[k] = float(v)
            except ValueError:
                result[k] = v
        return result
    except Exception as e:
        print(f"[Warning] Could not parse {path}: {e}")
        return None


def parse_dd_trace(path: Path) -> Optional[dict]:
    """Parse dd_trace.jsonl → read-rate and mean JS stats."""
    if not path.exists():
        return None
    records = []
    try:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return None
    if not records:
        return None

    n_total  = len(records)
    n_read   = sum(1 for r in records if r.get("decision") == "READ")
    js_vals  = [r.get("avg_js_firstN", 0.0) for r in records]
    from statistics import mean as _mean
    return {
        "n_gate_calls": n_total,
        "n_read": n_read,
        "read_rate": n_read / n_total if n_total else 0.0,
        "avg_js": _mean(js_vals) if js_vals else 0.0,
    }


# ── Experiment catalogue ──────────────────────────────────────────────────────

def collect_experiments(extra_dirs: list[Path]) -> list[dict]:
    """Collect all cmp_* directories (and any extras) into experiment records."""
    records = []

    # -- wait-k baselines --
    for k in [3, 5, 7, 9]:
        d = OUT_ROOT / f"cmp_baseline_k{k}"
        sc = parse_scores(d / "scores")
        records.append({
            "label": f"baseline k={k}",
            "method": "baseline",
            "param_k": k,
            "param_tau": None,
            "dir": d,
            "scores": sc,
            "dd_trace": None,
        })

    # -- DD full gate --
    for tau in [0.03, 0.05]:
        d = OUT_ROOT / f"cmp_dd_full_tau{tau}"
        sc = parse_scores(d / "scores")
        tr = parse_dd_trace(d / "dd_trace.jsonl")
        records.append({
            "label": f"DD full τ={tau}",
            "method": "dd_full",
            "param_k": 5,
            "param_tau": tau,
            "dir": d,
            "scores": sc,
            "dd_trace": tr,
        })

    # -- entropy gate only (control: no DD) --
    d = OUT_ROOT / "cmp_entropy_only"
    sc = parse_scores(d / "scores")
    records.append({
        "label": "entropy only",
        "method": "entropy_only",
        "param_k": 5,
        "param_tau": 3.0,
        "dir": d,
        "scores": sc,
        "dd_trace": None,
    })

    # -- DD veto --
    for tau in [0.03, 0.05]:
        d = OUT_ROOT / f"cmp_dd_veto_tau{tau}"
        sc = parse_scores(d / "scores")
        tr = parse_dd_trace(d / "dd_trace.jsonl")
        records.append({
            "label": f"DD veto τ={tau}",
            "method": "dd_veto",
            "param_k": 5,
            "param_tau": tau,
            "dir": d,
            "scores": sc,
            "dd_trace": tr,
        })

    # -- NLLB prefix-constrained continuation (wait-k=5) --
    d = OUT_ROOT / "cmp_continuation_k5"
    sc = parse_scores(d / "scores")
    records.append({
        "label": "NLLB continuation k=5",
        "method": "continuation",
        "param_k": 5,
        "param_tau": None,
        "dir": d,
        "scores": sc,
        "dd_trace": None,
    })

    # -- Qwen3-4B-Base causal-LM continuation (wait-k=5) --
    d = OUT_ROOT / "cmp_qwen_continuation_k5"
    sc = parse_scores(d / "scores")
    records.append({
        "label": "Qwen continuation k=5",
        "method": "qwen_cont",
        "param_k": 5,
        "param_tau": None,
        "dir": d,
        "scores": sc,
        "dd_trace": None,
    })

    # -- Extra directories provided by the user --
    for d in extra_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        sc = parse_scores(d / "scores")
        tr = parse_dd_trace(d / "dd_trace.jsonl")
        if sc is not None:
            records.append({
                "label": d.name,
                "method": "extra",
                "param_k": None,
                "param_tau": None,
                "dir": d,
                "scores": sc,
                "dd_trace": tr,
            })

    return records


# ── Report generation ─────────────────────────────────────────────────────────

def _fmt_val(v, fmt=".2f"):
    if v is None:
        return "N/A"
    try:
        return format(float(v), fmt)
    except Exception:
        return str(v)


def build_report(records: list[dict]) -> str:
    lines: list[str] = []
    a = lines.append

    a("# Wait-k vs DD Full Gate vs DD Veto — Comparison Report")
    a("")
    a("## Experiment Setup")
    a("")
    a("| Parameter       | Value                           |")
    a("|-----------------|----------------------------------|")
    a("| Dataset         | rand100 (100 WMT19 EN→ZH)        |")
    a("| Base model      | NLLB-200-distilled-600M          |")
    a("| DD futures K    | 4 (truncation mode)              |")
    a("| DD steps        | 3 (avg_js_first3 = policy score) |")
    a("| Baseline method | wait-k, entropy gate τ=999 (off) |")
    a("| Veto baseline   | wait-k + entropy gate τ=3.0      |")
    a("")
    a("### Methods")
    a("")
    a("- **baseline wait-k**: pure wait-k policy, no uncertainty gate, varying k.")
    a("- **DD full gate**: post-wait-k, DD is the sole commit/read arbiter.")
    a("  DD score (avg JS divergence) > τ → READ; else → COMMIT.")
    a("- **DD veto**: baseline (wait-k + entropy gate) decides first. Only when")
    a("  baseline would COMMIT does DD get to veto if divergence is high.")
    a("")

    # ── Main table ────────────────────────────────────────────────────────────
    a("## Results")
    a("")
    a("| Method             | BLEU  | AL    | LAAL  | AP    | DAL   | Read% |")
    a("|--------------------|-------|-------|-------|-------|-------|-------|")
    for r in records:
        sc = r["scores"]
        tr = r["dd_trace"]
        if sc is None:
            a(f"| {r['label']:<18} | N/A   | N/A   | N/A   | N/A   | N/A   | N/A   |")
            continue
        read_pct = f"{tr['read_rate']:.0%}" if tr else "—"
        a(
            f"| {r['label']:<18} "
            f"| {_fmt_val(sc.get('BLEU'))} "
            f"| {_fmt_val(sc.get('AL'))} "
            f"| {_fmt_val(sc.get('LAAL'))} "
            f"| {_fmt_val(sc.get('AP'), '.3f')} "
            f"| {_fmt_val(sc.get('DAL'))} "
            f"| {read_pct:>5} |"
        )
    a("")

    # ── DD signal table ───────────────────────────────────────────────────────
    dd_rows = [r for r in records if r["dd_trace"] is not None]
    if dd_rows:
        a("## DD Gate Signal")
        a("")
        a("| Method             | Gate calls | READ%  | Avg JS (firstN) |")
        a("|--------------------|------------|--------|-----------------|")
        for r in dd_rows:
            tr = r["dd_trace"]
            a(
                f"| {r['label']:<18} "
                f"| {tr['n_gate_calls']:>10} "
                f"| {tr['read_rate']:>6.1%} "
                f"| {tr['avg_js']:>15.4f} |"
            )
        a("")

    # ── Analysis ──────────────────────────────────────────────────────────────
    baseline_rows  = [r for r in records if r["method"] == "baseline" and r["scores"]]
    dd_full_rows   = [r for r in records if r["method"] == "dd_full"  and r["scores"]]
    dd_veto_rows   = [r for r in records if r["method"] == "dd_veto"  and r["scores"]]

    a("## Analysis")
    a("")

    if baseline_rows:
        best_bleu_b = max(baseline_rows, key=lambda r: r["scores"].get("BLEU", 0))
        best_al_b   = min(baseline_rows, key=lambda r: r["scores"].get("AL", 999))
        a("### Baseline sweep")
        a(f"- Best BLEU : **{r['label']}** "
          f"BLEU={_fmt_val(best_bleu_b['scores'].get('BLEU'))} "
          f"AL={_fmt_val(best_bleu_b['scores'].get('AL'))}")
        a(f"- Lowest AL : **{best_al_b['label']}** "
          f"AL={_fmt_val(best_al_b['scores'].get('AL'))} "
          f"BLEU={_fmt_val(best_al_b['scores'].get('BLEU'))}")
        a("")

    # Pick reference = baseline k=5 (most commonly used comparison point)
    ref = next(
        (r for r in baseline_rows if r.get("param_k") == 5),
        baseline_rows[0] if baseline_rows else None,
    )
    if ref and ref["scores"]:
        ref_bleu = ref["scores"].get("BLEU", 0.0)
        ref_al   = ref["scores"].get("AL",   0.0)
        a(f"Reference (baseline k=5): BLEU={_fmt_val(ref_bleu)}, AL={_fmt_val(ref_al)}")
        a("")

        def _compare(rows: list[dict], method_name: str):
            if not rows:
                a(f"### {method_name}: no completed runs yet.")
                a("")
                return
            a(f"### {method_name}")
            for r in rows:
                sc = r["scores"]
                dbleu = sc.get("BLEU", 0) - ref_bleu
                dal   = sc.get("AL",   0) - ref_al
                verdict = []
                if dbleu > 0.5:
                    verdict.append(f"BLEU ↑{dbleu:+.2f} (meaningful gain)")
                elif dbleu > 0:
                    verdict.append(f"BLEU ↑{dbleu:+.2f} (marginal)")
                else:
                    verdict.append(f"BLEU {dbleu:+.2f} (worse)")
                if dal < 0:
                    verdict.append(f"AL ↓{-dal:.2f} (lower latency)")
                else:
                    verdict.append(f"AL +{dal:.2f} (higher latency)")
                a(f"- **{r['label']}**: {', '.join(verdict)}")
            a("")

        _compare(dd_full_rows, "DD full gate")
        _compare(dd_veto_rows, "DD veto")

    a("## Plot")
    a("")
    a("See `outputs/comparison_plot.png` for the quality-latency scatter plot.")
    a("X-axis: Average Lagging (AL) — lower is better (less latency).")
    a("Y-axis: BLEU score — higher is better.")
    a("Each method is shown as a curve through its operating points.")
    a("")

    return "\n".join(lines)


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plot(records: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
    except ImportError:
        print("[Plot] matplotlib not available, skipping plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Wait-k vs DD Full Gate vs DD Veto\n(EN→ZH, NLLB, rand100)",
                 fontsize=13, fontweight="bold", y=1.01)

    METHOD_STYLE = {
        "baseline":     dict(color="#2196F3", marker="o", linestyle="-",  label="baseline wait-k"),
        "entropy_only": dict(color="#FF9800", marker="D", linestyle="--", label="entropy gate only"),
        "dd_full":      dict(color="#F44336", marker="s", linestyle="--", label="DD full gate"),
        "dd_veto":      dict(color="#4CAF50", marker="^", linestyle="-.", label="DD veto"),
        "continuation": dict(color="#9C27B0", marker="P", linestyle="-",  label="NLLB continuation"),
        "qwen_cont":    dict(color="#E91E63", marker="*", linestyle="-",  label="Qwen continuation"),
        "extra":        dict(color="#9E9E9E", marker="x", linestyle=":",  label="extra"),
    }

    for ax_idx, (x_metric, y_metric, xlabel) in enumerate([
        ("AL",   "BLEU", "Average Lagging (AL)  ← lower is better"),
        ("LAAL", "BLEU", "Length-Adaptive AL (LAAL)  ← lower is better"),
    ]):
        ax = axes[ax_idx]

        # Group by method
        by_method: dict[str, list] = {}
        for r in records:
            sc = r["scores"]
            if sc is None:
                continue
            x = sc.get(x_metric)
            y = sc.get(y_metric)
            if x is None or y is None:
                continue
            m = r["method"]
            by_method.setdefault(m, []).append((float(x), float(y), r["label"]))

        legend_handles = []
        for method, pts in by_method.items():
            style = METHOD_STYLE.get(method, METHOD_STYLE["extra"])
            pts_sorted = sorted(pts, key=lambda p: p[0])  # sort by latency
            xs = [p[0] for p in pts_sorted]
            ys = [p[1] for p in pts_sorted]
            labels = [p[2] for p in pts_sorted]

            line, = ax.plot(
                xs, ys,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.8,
                markersize=8,
                alpha=0.85,
            )

            for x, y, lbl in zip(xs, ys, labels):
                # Short label suffix: k=N or τ=N
                short = lbl.split()[-1]
                ax.annotate(
                    short,
                    (x, y),
                    textcoords="offset points",
                    xytext=(5, 4),
                    fontsize=7.5,
                    color=style["color"],
                )

            legend_handles.append(
                mlines.Line2D([], [],
                              color=style["color"],
                              marker=style["marker"],
                              linestyle=style["linestyle"],
                              label=style["label"])
            )

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(f"{y_metric}  ↑ higher is better", fontsize=10)
        ax.set_title(f"{y_metric} vs {x_metric}", fontsize=11)
        ax.legend(handles=legend_handles, fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-plot", action="store_true",
                   help="Skip matplotlib plot generation.")
    p.add_argument("--extra-dirs", nargs="*", default=[],
                   help="Additional output directories to include.")
    p.add_argument("--report-path", type=Path, default=None,
                   help="Custom path for the markdown report (default: outputs/comparison_report.md).")
    p.add_argument("--plot-path", type=Path, default=None,
                   help="Custom path for the PNG plot (default: outputs/comparison_plot.png).")
    args = p.parse_args()

    report_path = args.report_path or (OUT_ROOT / "comparison_report.md")
    plot_path   = args.plot_path   or (OUT_ROOT / "comparison_plot.png")

    records = collect_experiments([Path(d) for d in args.extra_dirs])

    # ── Console table ─────────────────────────────────────────────────────────
    done   = [r for r in records if r["scores"] is not None]
    pending = [r for r in records if r["scores"] is None]

    print()
    print("=" * 72)
    print("COMPARISON RESULTS")
    print("=" * 72)
    print(f"{'Method':<22} {'BLEU':>6}  {'AL':>6}  {'LAAL':>6}  {'AP':>5}  {'Read%':>6}")
    print("-" * 72)
    for r in records:
        sc = r["scores"]
        tr = r["dd_trace"]
        if sc is None:
            print(f"  {r['label']:<20} {'(not yet run)':>40}")
            continue
        read_pct = f"{tr['read_rate']:.0%}" if tr else "  —  "
        print(
            f"  {r['label']:<20} "
            f"{sc.get('BLEU', 0):>6.2f}  "
            f"{sc.get('AL', 0):>6.2f}  "
            f"{sc.get('LAAL', 0):>6.2f}  "
            f"{sc.get('AP', 0):>5.3f}  "
            f"{read_pct:>6}"
        )
    print("=" * 72)
    print(f"\n{len(done)} / {len(records)} experiments complete.")
    if pending:
        print("Pending:", ", ".join(r["label"] for r in pending))
    print()

    # ── Report ────────────────────────────────────────────────────────────────
    report = build_report(records)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"[Report] Written → {report_path}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    if not args.no_plot:
        make_plot(records, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
