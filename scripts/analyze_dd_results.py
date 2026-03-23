#!/usr/bin/env python3
"""Analyze DD gate sweep results and generate a comparison report.

Parses outputs/dd_sweep*/ score files, dd_trace.jsonl files, and produces:
  - outputs/dd_sweep_report.md: aggregate comparison table + analysis
  - Stdout: quick summary table

Usage:
    python scripts/analyze_dd_results.py
    python scripts/analyze_dd_results.py --suffix _qwen4b --taus 0.01 0.02 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent


# ── Parsing helpers ───────────────────────────────────────────────────────────

def parse_scores(path: Path) -> Optional[dict]:
    """Parse a SimulEval scores file.  Returns None if not found/parseable."""
    if not path.exists():
        return None
    try:
        lines = path.read_text().strip().splitlines()
        # Format: header line + data line (space-separated, pandas-style)
        # e.g.  "     BLEU   LAAL    AL     AP     DAL  ATD"
        #        "0  13.462  8.816   8.8  0.583  7.531  0.0"
        header = lines[0].split()
        vals   = lines[1].split()[1:]  # skip row index "0"
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
    """Parse dd_trace.jsonl and return aggregate statistics."""
    if not path.exists():
        return None
    records = []
    try:
        with path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return None
    if not records:
        return None

    js_vals    = [r.get("avg_js_firstN", 0.0) for r in records]
    js1_vals   = [r.get("avg_js_first1", 0.0) for r in records]
    js3_vals   = [r.get("avg_js_first3", r.get("avg_js_first1", 0.0)) for r in records]
    n_read     = sum(1 for r in records if r.get("decision") == "READ")
    n_commit   = sum(1 for r in records if r.get("decision") == "COMMIT")
    n_total    = len(records)
    # Per-sentence forced reads
    from collections import defaultdict
    by_sent: dict[int, list] = defaultdict(list)
    for r in records:
        by_sent[r["sentence_id"]].append(r.get("decision"))
    forced_reads_per_sent = [
        sum(1 for d in decs if d == "READ")
        for decs in by_sent.values()
    ]

    return {
        "n_gate_calls": n_total,
        "n_commit": n_commit,
        "n_read": n_read,
        "read_rate": n_read / n_total if n_total else 0.0,
        "avg_js_firstN": mean(js_vals) if js_vals else 0.0,
        "median_js_firstN": median(js_vals) if js_vals else 0.0,
        "avg_js_first1": mean(js1_vals) if js1_vals else 0.0,
        "avg_js_first3": mean(js3_vals) if js3_vals else 0.0,
        "avg_forced_reads_per_sent": mean(forced_reads_per_sent) if forced_reads_per_sent else 0.0,
        "n_sentences": len(by_sent),
        "_records": records,  # kept for qualitative analysis
    }


# ── Report builder ────────────────────────────────────────────────────────────

def build_report(suffix: str, taus: list[float], dd_steps: int, dd_k: int) -> str:
    baseline_dir = ROOT / f"outputs/dd_sweep{suffix}_baseline"
    baseline_scores = parse_scores(baseline_dir / "scores")

    lines = []
    a = lines.append

    a("# DD Gate Sweep — Comparison Report")
    a("")
    a("## Experiment Configuration")
    a("")
    a(f"| Parameter         | Value |")
    a(f"|-------------------|-------|")
    a(f"| dataset           | rand100 (100 random WMT19 EN→ZH sentences) |")
    a(f"| base model        | `{'Qwen3-4B-Base' if 'qwen4b' in suffix else 'NLLB-200-distilled-600M'}` |")
    a(f"| wait-k            | 5 |")
    a(f"| DD futures K      | {dd_k} |")
    a(f"| DD steps (n_steps)| {dd_steps} — avg_js over first {dd_steps} Chinese tokens |")
    a(f"| DD main score     | avg_js_firstN (avg JS over first {dd_steps} steps) |")
    a(f"| future mode       | truncation (deterministic, 1..K extra src words) |")
    a(f"| taus swept        | {', '.join(str(t) for t in taus)} |")
    a("")

    # ── Aggregate table ───────────────────────────────────────────────────────
    a("## Aggregate Results")
    a("")
    a("| Run             | BLEU  | AL    | AP    | DAL   | LAAL  | Read% | Avg JS (firstN) |")
    a("|-----------------|-------|-------|-------|-------|-------|-------|-----------------|")

    def fmt(sc: Optional[dict], trace: Optional[dict], label: str) -> str:
        if sc is None:
            return f"| {label:<15} | N/A   | N/A   | N/A   | N/A   | N/A   | N/A   | N/A             |"
        bleu = sc.get("BLEU", float("nan"))
        al   = sc.get("AL",   float("nan"))
        ap   = sc.get("AP",   float("nan"))
        dal  = sc.get("DAL",  float("nan"))
        laal = sc.get("LAAL", float("nan"))
        read_pct = f"{trace['read_rate']:.0%}" if trace else "—"
        avg_js   = f"{trace['avg_js_firstN']:.4f}" if trace else "—"
        return (
            f"| {label:<15} | {bleu:5.2f} | {al:5.2f} | {ap:5.3f} |"
            f" {dal:5.2f} | {laal:5.2f} | {read_pct:>5} | {avg_js:>15} |"
        )

    base_trace = parse_dd_trace(baseline_dir / "dd_trace.jsonl")
    a(fmt(baseline_scores, base_trace, "baseline"))

    tau_results = []
    for tau in taus:
        d = ROOT / f"outputs/dd_sweep{suffix}_tau{tau}"
        sc = parse_scores(d / "scores")
        tr = parse_dd_trace(d / "dd_trace.jsonl")
        a(fmt(sc, tr, f"dd tau={tau}"))
        if sc:
            tau_results.append({"tau": tau, "scores": sc, "trace": tr})
    a("")

    # ── Delta table (vs baseline) ─────────────────────────────────────────────
    if baseline_scores and tau_results:
        a("### Deltas vs Baseline (↑ good for BLEU/LAAL, ↓ good for AL)")
        a("")
        a("| Run             | ΔBLEU  | ΔAL    | ΔLAAL  |")
        a("|-----------------|--------|--------|--------|")
        for r in tau_results:
            sc = r["scores"]
            dbleu = sc.get("BLEU", 0) - baseline_scores.get("BLEU", 0)
            dal   = sc.get("AL", 0)   - baseline_scores.get("AL", 0)
            dlaal = sc.get("LAAL", 0) - baseline_scores.get("LAAL", 0)
            a(f"| dd tau={r['tau']:<7}  | {dbleu:+6.2f} | {dal:+6.2f} | {dlaal:+6.2f} |")
        a("")

    # ── Best threshold ────────────────────────────────────────────────────────
    a("## Best Threshold")
    a("")
    if tau_results and baseline_scores:
        # Best BLEU
        best_bleu = max(tau_results, key=lambda r: r["scores"].get("BLEU", 0))
        # Best LAAL (quality-latency tradeoff)
        best_laal = max(tau_results, key=lambda r: r["scores"].get("LAAL", 0))
        # Lowest AL (latency)
        best_al = min(tau_results, key=lambda r: r["scores"].get("AL", float("inf")))
        a(f"| Criterion          | Best tau | Value |")
        a(f"|--------------------|----------|-------|")
        a(f"| Highest BLEU       | tau={best_bleu['tau']:<5} | {best_bleu['scores'].get('BLEU', 0):.2f} |")
        a(f"| Highest LAAL       | tau={best_laal['tau']:<5} | {best_laal['scores'].get('LAAL', 0):.2f} |")
        a(f"| Lowest AL (latency)| tau={best_al['tau']:<5} | {best_al['scores'].get('AL', 0):.2f} |")
        a("")

        # Simple recommendation
        bleu_base = baseline_scores.get("BLEU", 0)
        al_base   = baseline_scores.get("AL", 0)
        best_any  = max(tau_results, key=lambda r: r["scores"].get("BLEU", 0))
        bleu_gain = best_any["scores"].get("BLEU", 0) - bleu_base
        al_gain   = al_base - best_any["scores"].get("AL", 0)

        a("**Recommendation:**")
        if bleu_gain > 0.5:
            a(f"✓ DD gate tau={best_any['tau']} improves BLEU by {bleu_gain:+.2f}. "
              f"Worth pursuing further.")
        elif bleu_gain > 0:
            a(f"○ DD gate shows small BLEU gain ({bleu_gain:+.2f}). "
              f"Effects are marginal; check if latency trade-off is acceptable.")
        else:
            a(f"✗ DD gate does NOT improve BLEU at any threshold tested. "
              f"Possible issues: truncation futures are not diverse enough, or the "
              f"base model is already insensitive to extra source context at this prefix length.")
        if al_gain > 0:
            a(f"  Latency (AL) improves by {al_gain:.2f} at best DD tau — DD forces more READs, "
              f"which may reduce premature commits.")
        else:
            a(f"  DD gate increases latency (AL rises by {-al_gain:.2f}) — forcing extra READs "
              f"delays translation without quality gain.")
    a("")

    # ── DD trace analysis ─────────────────────────────────────────────────────
    a("## DD Signal Analysis")
    a("")
    a("| Run            | Gate calls | READ% | Avg forced READ/sent | Avg JS (first1) | Avg JS (first3) | Avg JS (firstN) |")
    a("|----------------|------------|-------|----------------------|-----------------|-----------------|-----------------|")
    for tau_r in tau_results:
        tr = tau_r["trace"]
        if tr:
            a(f"| dd tau={tau_r['tau']:<6}  | {tr['n_gate_calls']:>10} |"
              f" {tr['read_rate']:>5.0%} | {tr['avg_forced_reads_per_sent']:>20.2f} |"
              f" {tr['avg_js_first1']:>15.4f} | {tr['avg_js_first3']:>15.4f} |"
              f" {tr['avg_js_firstN']:>15.4f} |")
        else:
            a(f"| dd tau={tau_r['tau']:<6}  | N/A        | N/A   | N/A                  | N/A             | N/A             | N/A             |")
    a("")
    a("*Gate calls* = number of times DD was evaluated (once per unique source prefix per sentence).")
    a("*READ%* = fraction of gate evaluations that forced READ (DD score > tau).")
    a("")

    # ── Qualitative examples from trace ──────────────────────────────────────
    if tau_results and any(r["trace"] for r in tau_results):
        # Pick the tau with highest read rate (most DD signal) for examples
        best_for_examples = max(
            tau_results,
            key=lambda r: r["trace"].get("read_rate", 0) if r["trace"] else 0,
        )
        trace_path = (
            ROOT / f"outputs/dd_sweep{suffix}_tau{best_for_examples['tau']}"
            / "dd_trace.jsonl"
        )
        example_tau = best_for_examples["tau"]
        records_ex: list[dict] = []
        if trace_path.exists():
            with trace_path.open() as f:
                for line in f:
                    try:
                        records_ex.append(json.loads(line))
                    except Exception:
                        pass

        if records_ex:
            reads_ex   = [r for r in records_ex if r.get("decision") == "READ"]
            commits_ex = [r for r in records_ex if r.get("decision") == "COMMIT"]

            a(f"## Verbose Example Gate Decisions (tau={example_tau})")
            a("")
            a("Each entry shows: sentence_id, step (src_len/tgt_len), observed source prefix,")
            a("the K sampled English futures, DD scores, gate decision, and what baseline would do.")
            a("")
            a("### 5 cases where DD forced READ (high uncertainty — most informative)")
            a("")
            for rec in sorted(reads_ex, key=lambda r: -r.get("avg_js_firstN", 0))[:5]:
                a(f"**sent={rec['sentence_id']}  src_len={rec['src_len']}  tgt_len={rec['tgt_len']}**")
                src_pfx = rec.get("src_prefix", "")
                a(f"  Observed prefix : `{src_pfx[:100]}`")
                for i, fut in enumerate(rec.get("futures", [])[:4]):
                    a(f"  future[{i+1}]      : `{fut[:100]}`")
                a(f"  avg_js_first1={rec.get('avg_js_first1',0):.4f}  "
                  f"avg_js_first3={rec.get('avg_js_first3',0):.4f}  "
                  f"avg_js_firstN={rec.get('avg_js_firstN',0):.4f}")
                a(f"  per_step_js    : {[round(v,4) for v in rec.get('per_step_js',[])]}")
                a(f"  DD threshold   : {rec.get('dd_tau')}  "
                  f"→ **{rec.get('decision')}**  (baseline would: {rec.get('baseline_decision','COMMIT')})")
                a("")
            a("### 5 cases where DD allowed COMMIT (low uncertainty)")
            a("")
            for rec in sorted(commits_ex, key=lambda r: -r.get("avg_js_firstN", 0))[:5]:
                a(f"**sent={rec['sentence_id']}  src_len={rec['src_len']}  tgt_len={rec['tgt_len']}**")
                src_pfx = rec.get("src_prefix", "")
                a(f"  Observed prefix : `{src_pfx[:100]}`")
                a(f"  avg_js_firstN={rec.get('avg_js_firstN',0):.4f}  "
                  f"avg_js_first1={rec.get('avg_js_first1',0):.4f}")
                a(f"  → **COMMIT**  (baseline would: COMMIT)  — both agree")
                a("")

    # ── Observations ──────────────────────────────────────────────────────────
    a("## Observations and Failure Modes")
    a("")
    a("1. **Future diversity**: Using truncation futures (nested prefix extensions) means")
    a("   DD measures *same-path continuation stability*, not true branching robustness.")
    a("   High JS in truncation mode means even prefix extensions disagree — a strong")
    a("   signal of genuine uncertainty.  Low JS means the model is stable regardless of")
    a("   how many extra source words it sees.")
    a("")
    a("2. **Threshold sensitivity**: Very low tau (0.01-0.02) forces too many READs,")
    a("   increasing latency.  Very high tau (0.10) rarely triggers, behaving like baseline.")
    a("   The sweet spot depends on the base model's sensitivity to extra context.")
    a("")
    a("3. **Expected failure modes**:")
    a("   - If JS is uniformly low, DD never fires and is equivalent to baseline.")
    a("   - If JS is uniformly high, DD always fires and causes max latency.")
    a("   - Truncation futures all agree when the source is unambiguous (e.g. short,")
    a("     simple sentences), and disagree when additional context changes translation.")
    a("")
    a("4. **Next steps**:")
    a("   - Try lm_sample futures for more genuine diversity.")
    a("   - Combine DD with the existing entropy gate (DD AND entropy → more selective).")
    a("   - Tune threshold on a held-out set; rand100 is too small for reliable tuning.")
    a("")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suffix", default="",
                   help="Experiment suffix, e.g. '_qwen4b'")
    p.add_argument("--taus", nargs="+", type=float,
                   default=[0.01, 0.02, 0.03, 0.05, 0.08, 0.10])
    p.add_argument("--dd-steps", type=int, default=3)
    p.add_argument("--dd-k", type=int, default=4)
    p.add_argument("--report-path", type=Path,
                   default=None)
    args = p.parse_args()

    suffix = args.suffix
    report_path = args.report_path or (
        ROOT / f"outputs/dd_sweep{suffix}_report.md"
    )

    print(f"[Analyze] Building report for suffix='{suffix}' taus={args.taus}")
    report = build_report(suffix, args.taus, args.dd_steps, args.dd_k)
    report_path.write_text(report, encoding="utf-8")
    print(f"[Analyze] Report written to {report_path}")

    # Quick table to stdout
    print("\n" + "=" * 60)
    print("QUICK SUMMARY")
    print("=" * 60)
    baseline_scores = parse_scores(ROOT / f"outputs/dd_sweep{suffix}_baseline" / "scores")
    if baseline_scores:
        print(f"{'Run':<18} {'BLEU':>6}  {'AL':>6}  {'LAAL':>6}")
        print(f"{'baseline':<18} {baseline_scores.get('BLEU',0):>6.2f}  "
              f"{baseline_scores.get('AL',0):>6.2f}  {baseline_scores.get('LAAL',0):>6.2f}")
        for tau in args.taus:
            d = ROOT / f"outputs/dd_sweep{suffix}_tau{tau}"
            sc = parse_scores(d / "scores")
            if sc:
                dbleu = sc.get("BLEU", 0) - baseline_scores.get("BLEU", 0)
                print(f"{'dd tau='+str(tau):<18} {sc.get('BLEU',0):>6.2f}  "
                      f"{sc.get('AL',0):>6.2f}  {sc.get('LAAL',0):>6.2f}  "
                      f"(ΔBLEU={dbleu:+.2f})")
    print("=" * 60)


if __name__ == "__main__":
    main()
