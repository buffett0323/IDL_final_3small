#!/usr/bin/env python3
"""10-case diagnostic: Distribution Divergence vs Semantic LCP step-by-step.

Runs both future-consistency methods on 10 rand100 examples (configurable)
and prints a detailed step-by-step log showing exactly how data changes at
each step of the pipeline.

Usage:
    python scripts/run_10case_diagnostic.py \\
        --model-name /data/user_data/haolingp/models/Qwen3-4B-Base \\
        --causal-lm --device cuda:0 \\
        --future-mode lm_sample --K 4 \\
        --n-cases 10 --prefix-words 5 \\
        --out outputs/rand100_future_consistency/diagnostic_10case.txt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "agents"))
sys.path.insert(0, str(ROOT / "scripts"))

from future_consistency import (
    FutureConsistencyScorer,
    compute_future_diversity,
    decide_from_divergence,
    decide_from_lcp,
)


# ── data loader ─────────────────────────────────────────────────────────────

def load_rand100(data_dir: Path):
    src = (data_dir / "rand100_source.txt").read_text().splitlines()
    tgt = (data_dir / "rand100_target.txt").read_text().splitlines()
    idx = json.loads((data_dir / "rand100_indices.json").read_text())
    return src, tgt, idx


# ── step-by-step verbose printer ────────────────────────────────────────────

SEP  = "=" * 72
SEP2 = "-" * 60
SEP3 = "·" * 48


def print_case(i: int, ex_id: int, src: str, ref: str,
               prefix: str, dd: dict, sl: dict,
               args: argparse.Namespace, file=sys.stdout):
    p = lambda *a, **k: print(*a, **k, file=file)

    p(SEP)
    p(f"CASE {i+1:02d}  (rand100 idx={ex_id})")
    p(SEP)

    # ── source / prefix ────────────────────────────────────────────────────
    p(f"SOURCE   : {src}")
    p(f"PREFIX   : '{prefix}'  ({args.prefix_words} words)")
    p(f"REF      : {ref[:80]}{'…' if len(ref) > 80 else ''}")
    p("")

    # ── futures ────────────────────────────────────────────────────────────
    futures = dd["futures"]
    div = dd["future_diversity"]
    p(f"STEP 1 — Sample K={args.K} English futures  [mode={args.future_mode}]")
    p(SEP2)
    for k, f in enumerate(futures):
        p(f"  future[{k+1}] : {f[:90]}")
    p("")
    p(f"  Diversity diagnostics:")
    p(f"    nested_prefix_rate : {div['nested_prefix_rate']:.3f}"
      f"  {'⚠ NESTED (truncation)' if div['is_nested_warning'] else '✓ not nested'}")
    p(f"    avg_future_edit_dist: {div['avg_future_edit_dist']:.3f}"
      f"  (0=identical  1=maximally different)")
    p(f"    unique_future_ratio : {div['unique_future_ratio']:.3f}")
    p("")

    # ── distribution divergence ────────────────────────────────────────────
    p(f"STEP 2 — Distribution Divergence")
    p(SEP2)
    p(f"  For each future, run base MT model → get next Chinese token distribution.")
    p(f"  Compare distributions with Jensen-Shannon divergence.")
    p("")
    pairwise = dd["pairwise_js"]
    p(f"  Pairwise JS matrix (K×K):")
    for row in pairwise:
        p("    " + "  ".join(f"{v:.4f}" for v in row))
    p("")
    p(f"  avg_js_divergence : {dd['avg_js']:.6f}"
      f"  (commit_thresh={args.commit_js}  read_thresh={args.read_js})")
    p(f"  max_js_divergence : {dd['max_js']:.6f}")
    p(f"  top-{args.top_k_overlap} overlap    : {dd['topk_overlap']:.4f}"
      f"  (how many of top-{args.top_k_overlap} tokens appear in ALL futures' top-k)")
    p(f"  consensus_score   : {dd['consensus_score']:.6f}"
      f"  (max_v  min_k p_k(v) — highest guaranteed probability for any token)")
    kl = dd.get("kl_from_first", [])
    if kl:
        kl_str = "  ".join(f"{v:.4f}" for v in kl)
        p(f"  KL(p1‖pk) k=2..K  : {kl_str}")
    p("")
    p(f"  → DD decision     : {dd['decision']}")
    p("")

    # ── semantic lcp ───────────────────────────────────────────────────────
    p(f"STEP 3 — Semantic LCP")
    p(SEP2)
    p(f"  For each future, generate {args.cont_len}-token Chinese continuation.")
    p(f"  Measure character-level LCP + edit distance + token overlap.")
    p("")
    sl_futures  = sl["futures"]
    sl_conts    = sl["continuations"]
    for k, (f, c) in enumerate(zip(sl_futures, sl_conts)):
        p(f"  future[{k+1}] : {f[:70]}")
        p(f"  zh_cont[{k+1}]: {c}")
        p(SEP3)
    p("")
    p(f"  literal_lcp        : '{sl['literal_lcp']}'  (len={sl['literal_lcp_len']})")
    p(f"  avg_edit_distance  : {sl['avg_edit_distance']:.4f}"
      f"  (0=identical  1=maximally different)")
    p(f"  token_set_overlap  : {sl['token_overlap']:.4f}"
      f"  (char unigram Jaccard, mean pairwise)")
    p(f"  semantic_agreement : {sl['semantic_agreement']:.4f}"
      f"  (combined: LCP_norm + 1-edit_dist + overlap, /3)")
    p(f"  safe_prefix        : '{sl['safe_prefix_candidate']}'")
    p("")
    p(f"  → LCP decision     : {sl['decision']}")
    p("")

    # ── combined summary ───────────────────────────────────────────────────
    p(f"SUMMARY")
    p(SEP2)
    p(f"  DD  : JS={dd['avg_js']:.4f}  overlap={dd['topk_overlap']:.3f}"
      f"  → {dd['decision']}")
    p(f"  LCP : len={sl['literal_lcp_len']}  edit={sl['avg_edit_distance']:.3f}"
      f"  agree={sl['semantic_agreement']:.3f}  → {sl['decision']}")
    if dd["decision"] == sl["decision"]:
        p(f"  Both agree: {dd['decision']}")
    else:
        p(f"  Disagreement: DD={dd['decision']}  LCP={sl['decision']}")
    p("")


# ── aggregate stats ──────────────────────────────────────────────────────────

def print_aggregate(cases: list[dict], args: argparse.Namespace, file=sys.stdout):
    p = lambda *a, **k: print(*a, **k, file=file)

    p("")
    p(SEP)
    p(f"AGGREGATE  ({len(cases)} cases)  [future_mode={args.future_mode}  K={args.K}]")
    p(SEP)

    js_vals     = [c["dd"]["avg_js"] for c in cases]
    nested_vals = [c["dd"]["future_diversity"]["nested_prefix_rate"] for c in cases]
    edit_vals   = [c["dd"]["future_diversity"]["avg_future_edit_dist"] for c in cases]
    lcp_lens    = [c["sl"]["literal_lcp_len"] for c in cases]
    sl_ed       = [c["sl"]["avg_edit_distance"] for c in cases]
    agr_vals    = [c["sl"]["semantic_agreement"] for c in cases]

    dd_decisions = [c["dd"]["decision"] for c in cases]
    sl_decisions = [c["sl"]["decision"] for c in cases]

    p(f"Future diversity:")
    p(f"  nested_prefix_rate   mean={mean(nested_vals):.3f}  "
      f"({'all nested = truncation mode' if mean(nested_vals) > 0.8 else 'mostly branching = lm_sample mode'})")
    p(f"  avg_future_edit_dist mean={mean(edit_vals):.3f}  "
      f"median={median(edit_vals):.3f}")
    p("")
    p(f"Distribution Divergence:")
    p(f"  avg_js   mean={mean(js_vals):.4f}  median={median(js_vals):.4f}")
    p(f"  COMMIT={dd_decisions.count('COMMIT')}  "
      f"BORDERLINE={dd_decisions.count('BORDERLINE')}  "
      f"READ={dd_decisions.count('READ')}")
    p("")
    p(f"Semantic LCP:")
    p(f"  literal_lcp_len  mean={mean(lcp_lens):.2f}  median={median(lcp_lens):.2f}")
    p(f"  avg_edit_dist    mean={mean(sl_ed):.4f}  median={median(sl_ed):.4f}")
    p(f"  semantic_agree   mean={mean(agr_vals):.4f}  median={median(agr_vals):.4f}")
    p(f"  COMMIT={sl_decisions.count('COMMIT')}  "
      f"BORDERLINE={sl_decisions.count('BORDERLINE')}  "
      f"READ={sl_decisions.count('READ')}")
    p("")
    both_agree = sum(1 for c in cases if c["dd"]["decision"] == c["sl"]["decision"])
    p(f"DD/LCP agreement: {both_agree}/{len(cases)} cases agree on same decision")
    p("")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-name", default="/data/user_data/haolingp/models/Qwen3-4B-Base")
    p.add_argument("--causal-lm", action="store_true", default=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--future-mode", default="lm_sample",
                   choices=["truncation", "lm_sample"])
    p.add_argument("--lm-temperature", type=float, default=1.2)
    p.add_argument("--lm-top-p", type=float, default=0.9)
    p.add_argument("--lm-max-new-words", type=int, default=12)
    p.add_argument("--K", type=int, default=4, help="Number of futures")
    p.add_argument("--prefix-words", type=int, default=5)
    p.add_argument("--cont-len", type=int, default=8)
    p.add_argument("--top-k-overlap", type=int, default=10)
    p.add_argument("--n-cases", type=int, default=10,
                   help="How many rand100 examples to run")
    p.add_argument("--start-idx", type=int, default=0,
                   help="Start from this rand100 index")
    p.add_argument("--commit-js", type=float, default=0.05)
    p.add_argument("--read-js", type=float, default=0.20)
    p.add_argument("--overlap-thresh", type=float, default=0.5)
    p.add_argument("--commit-lcp", type=int, default=2)
    p.add_argument("--read-edit", type=float, default=0.80)
    p.add_argument("--commit-agreement", type=float, default=0.50)
    p.add_argument("--data-dir", type=Path,
                   default=ROOT / "data" / "enzh")
    p.add_argument("--out", type=Path,
                   default=ROOT / "outputs" / "rand100_future_consistency"
                              / "diagnostic_10case.txt")
    p.add_argument("--source-lang", default="eng_Latn")
    p.add_argument("--target-lang", default="zho_Hans")
    p.add_argument("--cache-dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()

    # Load data
    sources, targets, wmt_indices = load_rand100(args.data_dir)
    end_idx = min(args.start_idx + args.n_cases, len(sources))
    ex_ids = list(range(args.start_idx, end_idx))
    print(f"[Diagnostic] Running {len(ex_ids)} cases: {ex_ids}")

    # Load scorer
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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_file = args.out.open("w", encoding="utf-8")

    cases = []
    try:
        for i, ex_id in enumerate(ex_ids):
            src    = sources[ex_id]
            ref    = targets[ex_id]
            words  = src.split()
            prefix_len = min(args.prefix_words, len(words))
            prefix = " ".join(words[:prefix_len])

            print(f"[{i+1:2d}/{len(ex_ids)}] example {ex_id}: {src[:50]}…")

            dd = scorer.score_distribution_divergence(
                words, prefix_len, args.K, args.top_k_overlap)
            dd["decision"] = decide_from_divergence(
                dd["avg_js"], dd["topk_overlap"],
                commit_js_threshold=args.commit_js,
                read_js_threshold=args.read_js,
                overlap_threshold=args.overlap_thresh,
            )
            sl = scorer.score_semantic_lcp(words, prefix_len, args.K, args.cont_len)
            sl["decision"] = decide_from_lcp(
                sl["literal_lcp_len"], sl["avg_edit_distance"],
                sl["semantic_agreement"],
                commit_lcp_threshold=args.commit_lcp,
                read_edit_threshold=args.read_edit,
                commit_agreement_threshold=args.commit_agreement,
            )

            # Print to stdout (brief) and file (verbose)
            print(
                f"       DD={dd['decision']:<10} JS={dd['avg_js']:.4f}  "
                f"LCP={sl['decision']:<10} lcp_len={sl['literal_lcp_len']}"
                f"  nested={dd['future_diversity']['nested_prefix_rate']:.2f}"
                f"  fut_edit={dd['future_diversity']['avg_future_edit_dist']:.3f}"
            )

            print_case(i, ex_id, src, ref, prefix, dd, sl, args, file=out_file)
            out_file.flush()

            cases.append({"ex_id": ex_id, "dd": dd, "sl": sl})

        # Aggregate summary
        print_aggregate(cases, args, file=out_file)
        print_aggregate(cases, args, file=sys.stdout)

    finally:
        out_file.close()

    print(f"\n[Diagnostic] Full step-by-step log written to: {args.out}")


if __name__ == "__main__":
    main()
