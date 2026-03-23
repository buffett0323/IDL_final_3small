#!/usr/bin/env python3
"""Bootstrap significance test: baseline vs DD gate.

Usage:
    python scripts/dd_significance_test.py
"""
from __future__ import annotations
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def load_instances(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return sorted(records, key=lambda r: r["index"])

def corpus_bleu_from_instances(records: list[dict]) -> float:
    """Compute corpus BLEU using sacrebleu from instances."""
    import sacrebleu
    hyps = [r["prediction"] for r in records]
    refs = [[r["reference"].strip() for r in records]]
    result = sacrebleu.corpus_bleu(hyps, refs, tokenize="char")
    return result.score

def bootstrap_bleu_diff(baseline_records: list[dict],
                         system_records: list[dict],
                         n_bootstrap: int = 10000,
                         seed: int = 42) -> dict:
    """
    Paired bootstrap test: does system significantly outperform baseline?
    Returns: dict with delta, p_value, ci_lower, ci_upper
    """
    import sacrebleu
    rng = random.Random(seed)
    n = len(baseline_records)
    assert n == len(system_records), "Record count mismatch"

    baseline_bleu = corpus_bleu_from_instances(baseline_records)
    system_bleu   = corpus_bleu_from_instances(system_records)
    observed_delta = system_bleu - baseline_bleu

    # Bootstrap resampling
    deltas = []
    indices = list(range(n))
    for _ in range(n_bootstrap):
        sample = [rng.randint(0, n - 1) for _ in range(n)]
        base_sample = [baseline_records[i] for i in sample]
        sys_sample  = [system_records[i]   for i in sample]
        d = corpus_bleu_from_instances(sys_sample) - corpus_bleu_from_instances(base_sample)
        deltas.append(d)

    # p-value: fraction of bootstrap samples where delta ≤ 0 (one-sided test: H1: system > baseline)
    p_value = sum(1 for d in deltas if d <= 0) / n_bootstrap
    deltas_sorted = sorted(deltas)
    ci_lower = deltas_sorted[int(0.025 * n_bootstrap)]
    ci_upper = deltas_sorted[int(0.975 * n_bootstrap)]

    return {
        "baseline_bleu": baseline_bleu,
        "system_bleu":   system_bleu,
        "observed_delta": observed_delta,
        "p_value": p_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }

def avg_lagging(records: list[dict]) -> float:
    """Compute Average Lagging from instances."""
    total_al = 0.0
    count = 0
    for r in records:
        delays = r.get("delays", [])
        src_len = r.get("source_length", len(r.get("source", "").split()))
        tgt_len = len(delays)
        if tgt_len == 0:
            continue
        # AL = (1/|Y|) * sum_t (d_t - (t-1))  where d_t = delay at step t, t is 1-indexed
        al = sum(delays[t] - t for t in range(tgt_len)) / tgt_len
        total_al += al
        count += 1
    return total_al / count if count else 0.0

def main():
    baseline_dir = ROOT / "outputs/dd_sweep_baseline"
    baseline_records = load_instances(baseline_dir / "instances.log")

    taus = [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

    print("=" * 72)
    print("BASELINE vs DD GATE — Bootstrap Significance Test (n_bootstrap=10000)")
    print("=" * 72)
    print(f"{'Run':<18} {'BLEU':>6}  {'ΔBLEU':>7}  {'AL':>6}  {'ΔAL':>6}  {'p-val':>7}  {'95% CI delta':>15}  {'sig?':>5}")
    print("-" * 72)

    base_bleu = corpus_bleu_from_instances(baseline_records)
    base_al   = avg_lagging(baseline_records)
    print(f"{'baseline':<18} {base_bleu:>6.2f}  {'—':>7}  {base_al:>6.2f}  {'—':>6}  {'—':>7}  {'—':>15}  {'—':>5}")

    for tau in taus:
        d = ROOT / f"outputs/dd_sweep_tau{tau}"
        inst_path = d / "instances.log"
        if not inst_path.exists():
            print(f"{'dd tau='+str(tau):<18} {'N/A':>6}")
            continue
        sys_records = load_instances(inst_path)
        result = bootstrap_bleu_diff(baseline_records, sys_records)
        sys_al = avg_lagging(sys_records)
        dal = sys_al - base_al
        ci_str = f"[{result['ci_lower']:+.2f}, {result['ci_upper']:+.2f}]"
        sig = "***" if result["significant_at_01"] else ("*" if result["significant_at_05"] else " ns")
        print(f"{'dd tau='+str(tau):<18} {result['system_bleu']:>6.2f}  "
              f"{result['observed_delta']:>+7.2f}  {sys_al:>6.2f}  {dal:>+6.2f}  "
              f"{result['p_value']:>7.4f}  {ci_str:>15}  {sig:>5}")

    print("=" * 72)
    print("p-val: one-sided bootstrap, H1: system BLEU > baseline BLEU")
    print("sig:  *** p<0.01  * p<0.05  ns not significant")
    print()
    print("Note: AL computed from instances delays; may differ slightly from SimulEval AL.")

if __name__ == "__main__":
    import sys
    # Write to both stdout and a file
    out_path = Path("/tmp/sig_test_result.txt")
    lines_out = []

    class Tee:
        def write(self, msg):
            sys.__stdout__.write(msg)
            lines_out.append(msg)
        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = Tee()
    main()
    sys.stdout = sys.__stdout__
    out_path.write_text("".join(lines_out))
    sys.__stdout__.write(f"\nWritten to {out_path}\n")
