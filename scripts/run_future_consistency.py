#!/usr/bin/env python3
"""Future-Consistency Experiment Runner on rand100.

Loads the existing rand100 split (data/enzh/rand100_source.txt +
rand100_target.txt + rand100_indices.json), runs both scoring methods
from agents/future_consistency.py on all 100 examples, saves per-example
JSONL, and produces a Markdown summary report.

Usage (local, single GPU):
    python scripts/run_future_consistency.py \\
        --model-name /data/user_data/haolingp/models/Qwen3-4B-Base \\
        --causal-lm \\
        --device cuda:0 \\
        --K 4 \\
        --prefix-words 5 \\
        --cont-len 8 \\
        --output outputs/rand100_future_consistency \\
        --verbose

Dry-run (first 3 examples only):
    python scripts/run_future_consistency.py --dry-run --verbose

Sharded run (used by sbatch):
    python scripts/run_future_consistency.py \\
        --shard-id 0 --num-shards 4 \\
        --device cuda:0 ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "agents"))

from future_consistency import (
    FutureConsistencyScorer,
    compute_future_diversity,
    decide_from_divergence,
    decide_from_lcp,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_rand100(data_dir: Path) -> tuple[list[str], list[str], list[int]]:
    """Load existing rand100 split from the project's data directory."""
    src_path = data_dir / "rand100_source.txt"
    tgt_path = data_dir / "rand100_target.txt"
    idx_path = data_dir / "rand100_indices.json"

    if not src_path.exists():
        raise FileNotFoundError(f"rand100 source not found: {src_path}")

    sources = [l.rstrip("\n") for l in src_path.open()]
    targets = [l.rstrip("\n") for l in tgt_path.open()] if tgt_path.exists() else [""] * len(sources)
    indices = json.loads(idx_path.read_text()) if idx_path.exists() else list(range(len(sources)))
    return sources, targets, indices


def log_verbose_example(
    ex: dict,
    file=sys.stdout,
):
    """Print a detailed human-readable log of one example."""
    print("=" * 70, file=file)
    print(f"EXAMPLE {ex['example_id']}  (WMT19 idx {ex['wmt19_index']})", file=file)
    print(f"Source:   {ex['source']}", file=file)
    print(f"Prefix:   '{ex['observed_prefix']}'  ({ex['prefix_words']} words)", file=file)
    if ex.get("reference"):
        print(f"Reference: {ex['reference'][:80]}", file=file)
    print("", file=file)

    dd = ex["distribution_divergence"]
    div_dd = dd.get("future_diversity", {})
    mode_label = dd.get("future_mode", "truncation")
    print(f"── Distribution Divergence  [future_mode={mode_label}] ──────", file=file)
    # Diversity warning
    if div_dd.get("is_nested_warning"):
        print(f"  ⚠ NESTED-PREFIX WARNING: nested_prefix_rate={div_dd['nested_prefix_rate']:.2f}"
              f" — these futures are truncation variants, NOT branching futures.", file=file)
        print(f"  ℹ Results measure same-path stability, NOT branching robustness.", file=file)
    print(f"  future_diversity  : nested_rate={div_dd.get('nested_prefix_rate',0):.2f}"
          f"  avg_edit={div_dd.get('avg_future_edit_dist',0):.3f}"
          f"  unique={div_dd.get('unique_future_ratio',1):.2f}", file=file)
    for k, fut in enumerate(dd["futures"]):
        print(f"  future[{k+1}]: {fut[:80]}", file=file)
    print(f"  avg_js_divergence : {dd['avg_js']:.6f}", file=file)
    print(f"  max_js_divergence : {dd['max_js']:.6f}", file=file)
    print(f"  topk_overlap      : {dd['topk_overlap']:.4f}", file=file)
    print(f"  consensus_score   : {dd['consensus_score']:.6f}", file=file)
    kl_str = ", ".join(f"{v:.4f}" for v in dd.get("kl_from_first", []))
    print(f"  kl_from_first     : [{kl_str}]", file=file)
    print(f"  → decision        : {dd['decision']}", file=file)
    print("", file=file)

    sl = ex["semantic_lcp"]
    div_sl = sl.get("future_diversity", {})
    print(f"── Semantic LCP  [future_mode={sl.get('future_mode', 'truncation')}] ───────────────", file=file)
    if div_sl.get("is_nested_warning"):
        print(f"  ⚠ NESTED-PREFIX WARNING: futures are truncation variants.", file=file)
    for k, (fut, cont) in enumerate(zip(sl["futures"], sl["continuations"])):
        print(f"  future[{k+1}]: {fut[:60]}", file=file)
        print(f"  zh_cont[{k+1}]: {cont[:40]}", file=file)
    print(f"  literal_lcp       : '{sl['literal_lcp']}'  (len={sl['literal_lcp_len']})", file=file)
    print(f"  avg_edit_distance : {sl['avg_edit_distance']:.4f}", file=file)
    print(f"  token_overlap     : {sl['token_overlap']:.4f}", file=file)
    print(f"  semantic_agreement: {sl['semantic_agreement']:.4f}", file=file)
    print(f"  safe_prefix       : '{sl['safe_prefix_candidate']}'", file=file)
    print(f"  → decision        : {sl['decision']}", file=file)
    print("=" * 70, file=file)
    print("", file=file)


# ── report generation ─────────────────────────────────────────────────────────

def build_report(
    examples: list[dict],
    args: argparse.Namespace,
    report_path: Path,
):
    """Write a Markdown summary report from the list of scored examples."""

    # Aggregate stats
    dd_all = [ex["distribution_divergence"] for ex in examples]
    sl_all = [ex["semantic_lcp"] for ex in examples]

    avg_js_list = [d["avg_js"] for d in dd_all]
    max_js_list = [d["max_js"] for d in dd_all]
    overlap_list = [d["topk_overlap"] for d in dd_all]

    lcp_len_list = [s["literal_lcp_len"] for s in sl_all]
    edit_list = [s["avg_edit_distance"] for s in sl_all]
    agr_list = [s["semantic_agreement"] for s in sl_all]

    dd_decisions = [d["decision"] for d in dd_all]
    sl_decisions = [s["decision"] for s in sl_all]

    def fmt_counts(decisions):
        c = {k: decisions.count(k) for k in ["COMMIT", "BORDERLINE", "READ"]}
        total = len(decisions)
        return (f"COMMIT={c['COMMIT']} ({c['COMMIT']/total:.0%})  "
                f"BORDERLINE={c['BORDERLINE']} ({c['BORDERLINE']/total:.0%})  "
                f"READ={c['READ']} ({c['READ']/total:.0%})")

    def safe_std(lst):
        return stdev(lst) if len(lst) > 1 else 0.0

    # Qualitative examples
    def pick_examples(key_fn, n, reverse=False):
        ranked = sorted(examples, key=key_fn, reverse=reverse)
        return ranked[:n]

    strong_agree = pick_examples(lambda e: e["distribution_divergence"]["avg_js"], 5)
    strong_disagree = pick_examples(lambda e: e["distribution_divergence"]["avg_js"], 5, reverse=True)
    borderline = [e for e in examples if e["distribution_divergence"]["decision"] == "BORDERLINE"][:5]

    # Diversity stats
    dd_div = [ex["distribution_divergence"].get("future_diversity", {}) for ex in examples]
    nested_rates = [d.get("nested_prefix_rate", 0) for d in dd_div]
    fut_edit_dists = [d.get("avg_future_edit_dist", 0) for d in dd_div]
    future_mode = examples[0]["distribution_divergence"].get("future_mode", "truncation") if examples else "truncation"

    lines = []
    a = lines.append

    a("# Future-Consistency Experiment Report")
    a("")
    # Mode banner
    if future_mode == "truncation":
        a("> **⚠ Future mode: TRUNCATION** — futures are nested prefix extensions of")
        a("> the observed prefix.  This measures *same-path prefix-extension stability*,")
        a("> **not** branching future robustness.  Do not interpret COMMIT as robust")
        a("> to genuinely different source continuations.")
    else:
        a("> **Future mode: LM_SAMPLE** — futures are sampled from the base LM with")
        a("> temperature sampling and nested-prefix rejection.  Results measure")
        a("> *branching future robustness*.")
    a("")
    a("## 1. Experiment Configuration")
    a("")
    a(f"| Parameter           | Value |")
    a(f"|---------------------|-------|")
    a(f"| base_model          | `{args.model_name}` |")
    a(f"| causal_lm           | `{args.causal_lm}` |")
    a(f"| device              | `{args.device}` |")
    a(f"| future_strategy     | truncation (reveal 1..K more source words) |")
    a(f"| K (num_futures)     | `{args.K}` |")
    a(f"| prefix_words        | `{args.prefix_words}` |")
    a(f"| cont_len            | `{args.cont_len}` |")
    a(f"| top_k_overlap       | `{args.top_k_overlap}` |")
    a(f"| dataset             | rand100 (100 random WMT19 EN→ZH sentences) |")
    a(f"| n_examples          | `{len(examples)}` |")
    a(f"| commit_js_threshold | `{args.commit_js}` |")
    a(f"| read_js_threshold   | `{args.read_js}` |")
    a(f"| commit_lcp_min      | `{args.commit_lcp}` |")
    a(f"| read_edit_threshold | `{args.read_edit}` |")
    a(f"| future_mode         | `{future_mode}` |")
    if future_mode == "lm_sample":
        a(f"| lm_temperature      | `{getattr(args, 'lm_temperature', 1.2)}` |")
        a(f"| lm_top_p            | `{getattr(args, 'lm_top_p', 0.9)}` |")
    a("")
    a("## 1b. Future Diversity Diagnostics")
    a("")
    a(f"| Metric                  | Mean   | Note |")
    a(f"|-------------------------|--------|------|")
    a(f"| nested_prefix_rate      | {mean(nested_rates):.3f}  | 1.0 = all pairs nested (truncation always gives 1.0) |")
    a(f"| avg_future_edit_dist    | {mean(fut_edit_dists):.3f}  | 0 = identical, 1 = maximally different |")
    a("")
    nested_warn_count = sum(1 for d in dd_div if d.get("is_nested_warning"))
    if future_mode == "truncation":
        a(f"**All {len(examples)} examples use nested truncation futures.**")
        a(f"Results measure *same-path continuation stability only*, NOT branching robustness.")
    else:
        a(f"{nested_warn_count}/{len(examples)} examples fell back to nested futures.")
        a(f"{len(examples)-nested_warn_count}/{len(examples)} examples have genuinely diverse branching futures.")
    a("")
    a("## 2. Distribution Divergence Results")
    a("")
    a("| Metric            | Mean   | Median | Std    |")
    a("|-------------------|--------|--------|--------|")
    a(f"| avg_js_divergence | {mean(avg_js_list):.4f} | {median(avg_js_list):.4f} | {safe_std(avg_js_list):.4f} |")
    a(f"| max_js_divergence | {mean(max_js_list):.4f} | {median(max_js_list):.4f} | {safe_std(max_js_list):.4f} |")
    a(f"| topk_overlap      | {mean(overlap_list):.4f} | {median(overlap_list):.4f} | {safe_std(overlap_list):.4f} |")
    a("")
    a(f"**Decisions**: {fmt_counts(dd_decisions)}")
    a("")
    a("## 3. Semantic LCP Results")
    a("")
    a("| Metric              | Mean   | Median | Std    |")
    a("|---------------------|--------|--------|--------|")
    a(f"| literal_lcp_len     | {mean(lcp_len_list):.2f} | {median(lcp_len_list):.2f} | {safe_std(lcp_len_list):.2f} |")
    a(f"| avg_edit_distance   | {mean(edit_list):.4f} | {median(edit_list):.4f} | {safe_std(edit_list):.4f} |")
    a(f"| semantic_agreement  | {mean(agr_list):.4f} | {median(agr_list):.4f} | {safe_std(agr_list):.4f} |")
    non_empty = sum(1 for s in sl_all if s["literal_lcp_len"] > 0)
    a(f"| non-empty safe prefix | {non_empty}/{len(sl_all)} ({non_empty/len(sl_all):.0%}) | — | — |")
    a("")
    a(f"**Decisions**: {fmt_counts(sl_decisions)}")
    a("")
    a("## 4. Qualitative Examples")
    a("")
    a("### 4a. Strong Agreement (low JS divergence → stable futures)")
    a("")
    for ex in strong_agree:
        dd = ex["distribution_divergence"]
        sl = ex["semantic_lcp"]
        a(f"- **Example {ex['example_id']}** (JS={dd['avg_js']:.4f}, LCP={sl['literal_lcp_len']})  ")
        a(f"  `{ex['source'][:80]}`  ")
        a(f"  LCP=`{sl['literal_lcp'][:20]}` | dd={dd['decision']} | lcp={sl['decision']}")
        a("")
    a("### 4b. Strong Disagreement (high JS divergence → risky futures)")
    a("")
    for ex in strong_disagree:
        dd = ex["distribution_divergence"]
        sl = ex["semantic_lcp"]
        a(f"- **Example {ex['example_id']}** (JS={dd['avg_js']:.4f}, LCP={sl['literal_lcp_len']})  ")
        a(f"  `{ex['source'][:80]}`  ")
        a(f"  dd={dd['decision']} | lcp={sl['decision']}")
        a("")
    a("### 4c. Borderline Examples")
    a("")
    for ex in (borderline or examples[:5]):
        dd = ex["distribution_divergence"]
        sl = ex["semantic_lcp"]
        a(f"- **Example {ex['example_id']}** (JS={dd['avg_js']:.4f}, overlap={dd['topk_overlap']:.2f}, LCP={sl['literal_lcp_len']})  ")
        a(f"  `{ex['source'][:80]}`  ")
        a(f"  dd={dd['decision']} | lcp={sl['decision']}")
        a("")
    a("## 5. Observations")
    a("")
    a("### Patterns")
    a("")
    a(f"- Distribution divergence: {mean(avg_js_list):.4f} mean JS. "
      f"{'Low — futures are broadly consistent.' if mean(avg_js_list) < 0.1 else 'Moderate to high — futures diverge noticeably.'}")
    a(f"- LCP: {mean(lcp_len_list):.1f} mean chars in common. "
      f"{'Non-trivial agreement on the near-term Chinese continuation.' if mean(lcp_len_list) > 1 else 'Poor literal agreement across futures.'}")
    a(f"- Top-k overlap: {mean(overlap_list):.2%} of top-{args.top_k_overlap} tokens shared — "
      f"{'good token agreement' if mean(overlap_list) > 0.5 else 'weak token agreement'}.")
    a("")
    a("### Likely Failure Modes")
    a("")
    a("- Truncation futures that differ only in trailing punctuation/articles may")
    a("  produce artificially low JS divergence; true ambiguity cases (pronoun")
    a("  resolution, named entity completion) would need longer continuations.")
    a("- LCP is strictly literal: semantically equivalent paraphrases (e.g. 病院 vs")
    a("  医院) score zero agreement even when meaning is identical.")
    a("- Qwen4B few-shot prompt may repeat the next few-shot line for very short")
    a("  inputs; continuation cleaning strips this but may lose valid content.")
    a("")
    a("### Next Steps")
    a("")
    a("1. Compare future-consistency signals to per-step entropy on the same examples.")
    a("2. Use future-consistency as the gating signal instead of entropy threshold.")
    a("3. Try LM-sampled English futures (top-p sampling) instead of truncation.")
    a("4. Add embedding-based semantic clustering for a stronger 'semantic LCP'.")
    a("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Written to {report_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run future-consistency scoring on rand100."
    )
    # Model
    p.add_argument("--model-name", default="facebook/nllb-200-distilled-600M",
                   help="HF model name or local path for the base MT model.")
    p.add_argument("--causal-lm", action="store_true",
                   help="Load as causal LM (Qwen4B etc.) instead of seq2seq (NLLB).")
    p.add_argument("--source-lang", default="eng_Latn")
    p.add_argument("--target-lang", default="zho_Hans")
    p.add_argument("--cache-dir", default="/data/user_data/haolingp/models")
    p.add_argument("--device", default="cuda:0",
                   help="PyTorch device for the base model.")
    # Data
    p.add_argument("--data-dir",
                   default=str(Path(__file__).resolve().parent.parent / "data" / "enzh"),
                   help="Directory containing rand100_source.txt etc.")
    p.add_argument("--output", default="outputs/rand100_future_consistency",
                   help="Output directory (relative to repo root or absolute).")
    # Scoring params
    p.add_argument("--K", type=int, default=4,
                   help="Number of futures to sample per example.")
    p.add_argument("--prefix-words", type=int, default=5,
                   help="Number of source words in the observed prefix (= wait-k).")
    p.add_argument("--cont-len", type=int, default=8,
                   help="Short continuation length in tokens (Method 2).")
    p.add_argument("--top-k-overlap", type=int, default=10,
                   help="Top-k tokens to use for overlap metric.")
    # Decision thresholds
    p.add_argument("--commit-js", type=float, default=0.05,
                   help="JS divergence below which COMMIT is issued.")
    p.add_argument("--read-js", type=float, default=0.20,
                   help="JS divergence above which READ is issued.")
    p.add_argument("--overlap-thresh", type=float, default=0.5,
                   help="Min top-k overlap required for COMMIT (distribution method).")
    p.add_argument("--commit-lcp", type=int, default=2,
                   help="Min LCP length (chars) for COMMIT (semantic LCP method).")
    p.add_argument("--read-edit", type=float, default=0.8,
                   help="Edit distance above which READ is issued (semantic LCP).")
    p.add_argument("--commit-agreement", type=float, default=0.5,
                   help="Min semantic_agreement score for COMMIT.")
    # Future sampling mode
    p.add_argument("--future-mode", default="truncation",
                   choices=["truncation", "lm_sample"],
                   help="'truncation': reveal 1..K more source words (nested, tests "
                        "same-path stability). 'lm_sample': generate diverse English "
                        "futures via temperature sampling (tests branching robustness).")
    p.add_argument("--lm-temperature", type=float, default=1.2,
                   help="Sampling temperature for lm_sample mode (>1 = more diverse).")
    p.add_argument("--lm-top-p", type=float, default=0.9,
                   help="Nucleus sampling p for lm_sample mode.")
    p.add_argument("--lm-max-new-words", type=int, default=12,
                   help="Max continuation words per LM sample in lm_sample mode.")
    # Comparison mode
    p.add_argument("--compare-modes", action="store_true",
                   help="Run BOTH truncation and lm_sample modes and write a "
                        "side-by-side comparison report.")
    # Run control
    p.add_argument("--dry-run", action="store_true",
                   help="Process only the first 3 examples for debugging.")
    p.add_argument("--verbose", action="store_true",
                   help="Print detailed per-example logs to stdout.")
    p.add_argument("--verbose-example-id", type=int, default=0,
                   help="Which example (0-indexed) to log verbosely to a file.")
    p.add_argument("--shard-id", type=int, default=0,
                   help="Shard index (0-based) for multi-GPU parallel runs.")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Total number of shards.")
    p.add_argument("--resume", action="store_true",
                   help="Skip examples whose output JSONL entry already exists.")
    return p.parse_args()


def main():
    args = parse_args()

    # ── resolve paths ─────────────────────────────────────────────────────────
    root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir)
    out_dir = (
        Path(args.output) if Path(args.output).is_absolute()
        else root / args.output
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    sources, targets, wmt_indices = load_rand100(data_dir)
    total = len(sources)
    print(f"[Data] Loaded {total} rand100 examples from {data_dir}")

    # Apply shard slicing
    all_ids = list(range(total))
    if args.dry_run:
        all_ids = all_ids[:3]
        print("[DryRun] Processing only first 3 examples.")

    shard_ids = [
        i for idx, i in enumerate(all_ids)
        if idx % args.num_shards == args.shard_id
    ]
    print(f"[Shard {args.shard_id}/{args.num_shards}] Processing {len(shard_ids)} examples.")

    # ── output files ──────────────────────────────────────────────────────────
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    jsonl_path = out_dir / f"results{shard_suffix}.jsonl"
    verbose_log_path = out_dir / "verbose_example.txt"
    report_path = out_dir / "report.md"

    # Load already-done IDs for resume
    done_ids: set[int] = set()
    if args.resume and jsonl_path.exists():
        with jsonl_path.open() as f:
            for line in f:
                try:
                    d = json.loads(line)
                    done_ids.add(d["example_id"])
                except Exception:
                    pass
        print(f"[Resume] {len(done_ids)} examples already done, skipping.")

    # --compare-modes: run both truncation and lm_sample then write comparison
    if args.compare_modes:
        _run_comparison(args, sources, targets, wmt_indices, shard_ids, out_dir)
        return

    # ── load model ────────────────────────────────────────────────────────────
    scorer = FutureConsistencyScorer(
        model_name=args.model_name,
        device=args.device,
        causal_lm=args.causal_lm,
        future_mode=args.future_mode,
        lm_temperature=args.lm_temperature,
        lm_top_p=args.lm_top_p,
        lm_max_new_words=args.lm_max_new_words,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        cache_dir=args.cache_dir,
    )

    # ── main loop ─────────────────────────────────────────────────────────────
    results: list[dict] = []
    verbose_written = False

    with jsonl_path.open("a", encoding="utf-8") as fout:
        for ex_id in shard_ids:
            if ex_id in done_ids:
                continue

            source = sources[ex_id]
            reference = targets[ex_id] if targets[ex_id] else ""
            wmt_idx = wmt_indices[ex_id] if ex_id < len(wmt_indices) else ex_id

            source_words = source.split()
            prefix_len = min(args.prefix_words, len(source_words))
            observed_prefix = " ".join(source_words[:prefix_len])

            t0 = time.time()

            # Distribution divergence
            dd = scorer.score_distribution_divergence(
                source_words, prefix_len, args.K, args.top_k_overlap
            )
            dd["decision"] = decide_from_divergence(
                dd["avg_js"], dd["topk_overlap"],
                commit_js_threshold=args.commit_js,
                read_js_threshold=args.read_js,
                overlap_threshold=args.overlap_thresh,
            )

            # Semantic LCP
            sl = scorer.score_semantic_lcp(
                source_words, prefix_len, args.K, args.cont_len
            )
            sl["decision"] = decide_from_lcp(
                sl["literal_lcp_len"], sl["avg_edit_distance"],
                sl["semantic_agreement"],
                commit_lcp_threshold=args.commit_lcp,
                read_edit_threshold=args.read_edit,
                commit_agreement_threshold=args.commit_agreement,
            )

            elapsed = time.time() - t0

            record: dict[str, Any] = {
                "example_id": ex_id,
                "wmt19_index": wmt_idx,
                "source": source,
                "reference": reference,
                "observed_prefix": observed_prefix,
                "prefix_words": prefix_len,
                "elapsed_s": round(elapsed, 2),
                "distribution_divergence": dd,
                "semantic_lcp": sl,
            }

            # Write JSONL
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(record)

            # Console summary
            print(
                f"[{ex_id:3d}] JS={dd['avg_js']:.4f} ({dd['decision']:<10})  "
                f"LCP={sl['literal_lcp_len']} ({sl['decision']:<10})  "
                f"{elapsed:.1f}s  |  {source[:55]}"
            )

            # Verbose logging
            if args.verbose:
                log_verbose_example(record)

            # Save one detailed verbose log file
            if ex_id == args.verbose_example_id and not verbose_written:
                with verbose_log_path.open("w", encoding="utf-8") as vf:
                    log_verbose_example(record, file=vf)
                verbose_written = True
                print(f"[Verbose] Saved example log to {verbose_log_path}")

    print(f"\n[Done] Shard {args.shard_id}: {len(results)} examples processed.")
    print(f"[Output] JSONL: {jsonl_path}")

    # ── merge shards and generate report (only on shard 0 or single run) ──────
    if args.shard_id == 0 or args.num_shards == 1:
        _wait_and_merge_shards(out_dir, args.num_shards, results, args, report_path)


def _run_single_mode(
    args: argparse.Namespace,
    sources, targets, wmt_indices,
    shard_ids: list[int],
    out_dir: Path,
    future_mode: str,
) -> list[dict]:
    """Run one future-mode pass and return results list."""
    mode_dir = out_dir / future_mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    # Use shard-specific filename so parallel shards don't overwrite each other.
    # Detect shard info from args (may not be present in all call sites).
    shard_id = getattr(args, "shard_id", 0)
    num_shards = getattr(args, "num_shards", 1)
    if num_shards > 1:
        jsonl_path = mode_dir / f"results_shard{shard_id}.jsonl"
    else:
        jsonl_path = mode_dir / "results.jsonl"
    verbose_path = mode_dir / "verbose_example.txt"

    scorer = FutureConsistencyScorer(
        model_name=args.model_name,
        device=args.device,
        causal_lm=args.causal_lm,
        future_mode=future_mode,
        lm_temperature=args.lm_temperature,
        lm_top_p=args.lm_top_p,
        lm_max_new_words=args.lm_max_new_words,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        cache_dir=args.cache_dir,
    )

    results: list[dict] = []
    verbose_written = False

    with jsonl_path.open("w", encoding="utf-8") as fout:
        for ex_id in shard_ids:
            source = sources[ex_id]
            reference = targets[ex_id] if targets[ex_id] else ""
            wmt_idx = wmt_indices[ex_id] if ex_id < len(wmt_indices) else ex_id
            source_words = source.split()
            prefix_len = min(args.prefix_words, len(source_words))
            observed_prefix = " ".join(source_words[:prefix_len])
            t0 = time.time()

            dd = scorer.score_distribution_divergence(
                source_words, prefix_len, args.K, args.top_k_overlap
            )
            dd["decision"] = decide_from_divergence(
                dd["avg_js"], dd["topk_overlap"],
                commit_js_threshold=args.commit_js,
                read_js_threshold=args.read_js,
                overlap_threshold=args.overlap_thresh,
            )
            sl = scorer.score_semantic_lcp(source_words, prefix_len, args.K, args.cont_len)
            sl["decision"] = decide_from_lcp(
                sl["literal_lcp_len"], sl["avg_edit_distance"],
                sl["semantic_agreement"],
                commit_lcp_threshold=args.commit_lcp,
                read_edit_threshold=args.read_edit,
                commit_agreement_threshold=args.commit_agreement,
            )

            record = {
                "example_id": ex_id, "wmt19_index": wmt_idx,
                "source": source, "reference": reference,
                "observed_prefix": observed_prefix, "prefix_words": prefix_len,
                "elapsed_s": round(time.time() - t0, 2),
                "distribution_divergence": dd, "semantic_lcp": sl,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(record)

            div = dd.get("future_diversity", {})
            print(
                f"[{future_mode:<12}][{ex_id:3d}] "
                f"nested={div.get('nested_prefix_rate',0):.2f}  "
                f"fut_edit={div.get('avg_future_edit_dist',0):.3f}  "
                f"JS={dd['avg_js']:.4f} ({dd['decision']:<10})  "
                f"LCP={sl['literal_lcp_len']} ({sl['decision']:<10})  "
                f"{source[:40]}"
            )
            if ex_id == args.verbose_example_id and not verbose_written:
                with verbose_path.open("w", encoding="utf-8") as vf:
                    log_verbose_example(record, file=vf)
                verbose_written = True

    build_report(results, args, mode_dir / "report.md")
    return results


def _run_comparison(
    args: argparse.Namespace,
    sources, targets, wmt_indices,
    shard_ids: list[int],
    out_dir: Path,
):
    """Run truncation and lm_sample modes, then write a side-by-side comparison."""
    print("\n" + "=" * 65)
    print("COMPARISON MODE: truncation vs lm_sample")
    print("=" * 65 + "\n")

    if not args.causal_lm:
        print("[Warning] lm_sample requires --causal-lm. Using truncation only.")
        results_t = _run_single_mode(args, sources, targets, wmt_indices,
                                     shard_ids, out_dir, "truncation")
        return

    # Override args temporarily
    import copy
    args_t = copy.copy(args); args_t.future_mode = "truncation"
    args_l = copy.copy(args); args_l.future_mode = "lm_sample"

    print("── Phase 1: truncation futures ──────────────────────────────")
    results_t = _run_single_mode(args_t, sources, targets, wmt_indices,
                                 shard_ids, out_dir, "truncation")
    print("\n── Phase 2: lm_sample futures ───────────────────────────────")
    results_l = _run_single_mode(args_l, sources, targets, wmt_indices,
                                 shard_ids, out_dir, "lm_sample")

    _write_comparison_report(results_t, results_l, out_dir / "comparison_report.md", args)


def _write_comparison_report(
    results_t: list[dict],
    results_l: list[dict],
    report_path: Path,
    args: argparse.Namespace,
):
    """Write side-by-side comparison report: truncation vs lm_sample."""
    from statistics import mean, median, stdev

    def safe_std(lst): return stdev(lst) if len(lst) > 1 else 0.0

    def extract(results, key_path):
        keys = key_path.split(".")
        vals = []
        for r in results:
            v = r
            for k in keys:
                v = v.get(k, {})
            vals.append(float(v) if isinstance(v, (int, float)) else 0.0)
        return vals

    def count_dec(results, section, dec):
        return sum(1 for r in results if r[section]["decision"] == dec)

    lines = []
    a = lines.append
    n = len(results_t)

    a("# Future-Consistency: Truncation vs LM-Sample Comparison")
    a("")
    a("This report explicitly distinguishes:")
    a("- **Same-path stability** (truncation futures, nested prefix extensions)")
    a("- **Branching robustness** (lm_sample futures, genuinely diverse continuations)")
    a("")
    a("Do NOT claim branching robustness based on truncation results.")
    a("")
    a(f"n={n} examples, K={args.K}, prefix={args.prefix_words} words, cont_len={args.cont_len}")
    a("")

    # Future diversity comparison
    a("## 1. Future Diversity")
    a("")
    a("| Metric                    | truncation | lm_sample |")
    a("|---------------------------|------------|-----------|")
    nested_t = extract(results_t, "distribution_divergence.future_diversity.nested_prefix_rate")
    nested_l = extract(results_l, "distribution_divergence.future_diversity.nested_prefix_rate")
    edit_t   = extract(results_t, "distribution_divergence.future_diversity.avg_future_edit_dist")
    edit_l   = extract(results_l, "distribution_divergence.future_diversity.avg_future_edit_dist")
    uniq_t   = extract(results_t, "distribution_divergence.future_diversity.unique_future_ratio")
    uniq_l   = extract(results_l, "distribution_divergence.future_diversity.unique_future_ratio")
    a(f"| nested_prefix_rate (mean) | {mean(nested_t):.3f}       | {mean(nested_l):.3f}      |")
    a(f"| avg_future_edit_dist      | {mean(edit_t):.3f}       | {mean(edit_l):.3f}      |")
    a(f"| unique_future_ratio       | {mean(uniq_t):.3f}       | {mean(uniq_l):.3f}      |")
    a("")
    nested_warn_l = sum(1 for r in results_l
                        if r["distribution_divergence"].get("future_diversity",{}).get("is_nested_warning"))
    a(f"lm_sample fell back to nested futures in {nested_warn_l}/{n} examples "
      f"({nested_warn_l/n:.0%}).")
    a("")

    # JS divergence comparison
    a("## 2. Distribution Divergence (JS)")
    a("")
    a("| Metric             | truncation | lm_sample | Δ (lm−trunc) |")
    a("|--------------------|------------|-----------|--------------|")
    js_t = extract(results_t, "distribution_divergence.avg_js")
    js_l = extract(results_l, "distribution_divergence.avg_js")
    ov_t = extract(results_t, "distribution_divergence.topk_overlap")
    ov_l = extract(results_l, "distribution_divergence.topk_overlap")
    a(f"| avg_js  mean       | {mean(js_t):.4f}     | {mean(js_l):.4f}    | {mean(js_l)-mean(js_t):+.4f}       |")
    a(f"| avg_js  median     | {median(js_t):.4f}     | {median(js_l):.4f}    | {median(js_l)-median(js_t):+.4f}       |")
    a(f"| topk_overlap mean  | {mean(ov_t):.4f}     | {mean(ov_l):.4f}    | {mean(ov_l)-mean(ov_t):+.4f}       |")
    a("")
    for dec in ["COMMIT","BORDERLINE","READ"]:
        ct = count_dec(results_t, "distribution_divergence", dec)
        cl = count_dec(results_l, "distribution_divergence", dec)
        a(f"- DD {dec}: truncation={ct} ({ct/n:.0%})  lm_sample={cl} ({cl/n:.0%})")
    a("")
    if mean(js_l) > mean(js_t) * 1.5:
        a("✓ lm_sample futures produce **higher JS divergence** — diverse futures")
        a("  expose more genuine uncertainty than truncation futures.")
    else:
        a("ℹ JS divergence is similar under both modes — the model is robust")
        a("  to both prefix extensions and genuinely different continuations.")
    a("")

    # Semantic LCP comparison
    a("## 3. Semantic LCP")
    a("")
    a("| Metric                | truncation | lm_sample | Δ |")
    a("|-----------------------|------------|-----------|---|")
    lcp_t = extract(results_t, "semantic_lcp.literal_lcp_len")
    lcp_l = extract(results_l, "semantic_lcp.literal_lcp_len")
    ed_t  = extract(results_t, "semantic_lcp.avg_edit_distance")
    ed_l  = extract(results_l, "semantic_lcp.avg_edit_distance")
    agr_t = extract(results_t, "semantic_lcp.semantic_agreement")
    agr_l = extract(results_l, "semantic_lcp.semantic_agreement")
    a(f"| literal_lcp_len mean  | {mean(lcp_t):.2f}       | {mean(lcp_l):.2f}      | {mean(lcp_l)-mean(lcp_t):+.2f} |")
    a(f"| avg_edit_dist mean    | {mean(ed_t):.4f}     | {mean(ed_l):.4f}    | {mean(ed_l)-mean(ed_t):+.4f} |")
    a(f"| semantic_agreement    | {mean(agr_t):.4f}     | {mean(agr_l):.4f}    | {mean(agr_l)-mean(agr_t):+.4f} |")
    a("")
    for dec in ["COMMIT","BORDERLINE","READ"]:
        ct = count_dec(results_t, "semantic_lcp", dec)
        cl = count_dec(results_l, "semantic_lcp", dec)
        a(f"- LCP {dec}: truncation={ct} ({ct/n:.0%})  lm_sample={cl} ({cl/n:.0%})")
    a("")

    # Per-example flip analysis
    flipped = [(r_t, r_l) for r_t, r_l in zip(results_t, results_l)
               if r_t["distribution_divergence"]["decision"] !=
                  r_l["distribution_divergence"]["decision"]]
    a(f"## 4. Decision Changes (truncation→lm_sample)")
    a("")
    a(f"{len(flipped)}/{n} examples changed DD decision when switching to lm_sample futures:")
    a("")
    for r_t, r_l in flipped[:10]:
        a(f"- Example {r_t['example_id']}: "
          f"{r_t['distribution_divergence']['decision']} → {r_l['distribution_divergence']['decision']}  "
          f"| `{r_t['source'][:60]}`")
    if len(flipped) > 10:
        a(f"- ... and {len(flipped)-10} more (see results JSONL files)")
    a("")

    a("## 5. Interpretation")
    a("")
    a("| Claim | Supported by | Valid? |")
    a("|-------|-------------|--------|")
    trunc_commit = count_dec(results_t, "distribution_divergence", "COMMIT")
    lm_commit    = count_dec(results_l, "distribution_divergence", "COMMIT")
    a(f"| 'Same-path stable' (truncation COMMIT={trunc_commit/n:.0%}) | truncation mode | ✓ Valid |")
    if lm_commit/n > 0.5:
        a(f"| 'Branching robust' (lm_sample COMMIT={lm_commit/n:.0%}) | lm_sample mode | ✓ Valid |")
    else:
        a(f"| 'Branching robust' (lm_sample COMMIT={lm_commit/n:.0%}) | lm_sample mode | ⚠ Weak — "
          f"only {lm_commit/n:.0%} COMMIT under diverse futures |")
    a("")
    a("**Key takeaway**: Use truncation results to justify same-path stability only.")
    a("Use lm_sample results to justify branching future robustness.")
    a("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Comparison Report] Written to {report_path}")


def _wait_and_merge_shards(
    out_dir: Path,
    num_shards: int,
    own_results: list[dict],
    args: argparse.Namespace,
    report_path: Path,
):
    """Merge all shard JSONL files and write the final report."""
    all_results: list[dict] = []

    if num_shards == 1:
        all_results = own_results
    else:
        print(f"[Merge] Waiting for all {num_shards} shard files ...")
        for s in range(num_shards):
            p = out_dir / f"results_shard{s}.jsonl"
            if not p.exists():
                print(f"  [Warning] Shard {s} output not found: {p}")
                continue
            with p.open() as f:
                for line in f:
                    try:
                        all_results.append(json.loads(line))
                    except Exception:
                        pass

    if not all_results:
        print("[Report] No results to summarize.")
        return

    # Write merged JSONL (shard 0 only, avoids duplication)
    merged_path = out_dir / "results_all.jsonl"
    with merged_path.open("w", encoding="utf-8") as f:
        for r in sorted(all_results, key=lambda x: x["example_id"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[Merge] Merged {len(all_results)} examples → {merged_path}")

    build_report(all_results, args, report_path)


if __name__ == "__main__":
    main()
