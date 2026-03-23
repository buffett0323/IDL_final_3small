#!/usr/bin/env python3
"""
Reference-free COMET-QE evaluation for all test100 experiments.

Usage:
    python scripts/eval_comet_qe.py [--output-dir outputs/] [--pattern test100_*]

COMET-QE scores translations WITHOUT needing a reference translation,
so it is fairer to Qwen outputs that use valid paraphrases.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path


def load_instances(log_path: Path):
    srcs, hyps, refs = [], [], []
    for line in open(log_path, encoding="utf-8"):
        r = json.loads(line)
        src = r.get("source", "")
        hyp = r.get("prediction", "").replace(" ", "")
        ref = r.get("reference", "").replace(" ", "")
        srcs.append(src)
        hyps.append(hyp)
        refs.append(ref)
    return srcs, hyps, refs


def bleu_score(hyps, refs):
    import sacrebleu
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="char").score


def chrf_score(hyps, refs):
    """ChrF++ — character-level F-score, more robust to paraphrases than BLEU."""
    import sacrebleu
    return sacrebleu.corpus_chrf(hyps, [refs]).score


def comet_qe_score(srcs, hyps, model_name="Unbabel/wmt22-cometkiwi-da-xl",
                   batch_size=8, gpus=1):
    """Reference-free COMET-QE scoring."""
    try:
        from comet import download_model, load_from_checkpoint
    except ImportError:
        print("  [!] unbabel-comet not installed. Run: pip install unbabel-comet")
        return None

    print(f"  Loading COMET-QE model: {model_name} ...")
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    data = [{"src": s, "mt": h} for s, h in zip(srcs, hyps)]
    output = model.predict(data, batch_size=batch_size, gpus=gpus)
    return float(output.system_score)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", default="outputs", type=str)
    parser.add_argument("--pattern", default="test100_", type=str,
                        help="Only evaluate dirs matching this prefix")
    parser.add_argument("--no-comet", action="store_true",
                        help="Skip COMET-QE (just BLEU + ChrF)")
    parser.add_argument("--comet-model", default="Unbabel/wmt22-cometkiwi-da-xl",
                        help="COMET-QE model to use")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    base = Path(args.outputs_dir)
    dirs = sorted([d for d in base.iterdir()
                   if d.is_dir() and args.pattern in d.name])

    print(f"\n{'Experiment':<50}  {'BLEU':>6}  {'ChrF':>6}  {'COMET-QE':>9}")
    print("-" * 80)

    all_results = []
    for exp_dir in dirs:
        log = exp_dir / "instances.log"
        if not log.exists():
            continue

        srcs, hyps, refs = load_instances(log)
        if not hyps:
            continue

        bleu = bleu_score(hyps, refs)
        chrf = chrf_score(hyps, refs)

        comet = None
        if not args.no_comet:
            print(f"  [{exp_dir.name}] computing COMET-QE ...")
            comet = comet_qe_score(srcs, hyps,
                                   model_name=args.comet_model,
                                   gpus=1 if args.gpu >= 0 else 0)

        comet_str = f"{comet:.4f}" if comet is not None else "  N/A  "
        print(f"{exp_dir.name:<50}  {bleu:>6.2f}  {chrf:>6.2f}  {comet_str:>9}")
        all_results.append((exp_dir.name, bleu, chrf, comet))

    print()
    if all_results:
        best_bleu = max(all_results, key=lambda x: x[1])
        best_chrf = max(all_results, key=lambda x: x[2])
        print(f"Best BLEU:   {best_bleu[0]}  ({best_bleu[1]:.2f})")
        print(f"Best ChrF:   {best_chrf[0]}  ({best_chrf[2]:.2f})")
        if any(r[3] for r in all_results):
            best_qe = max(all_results, key=lambda x: x[3] or -999)
            print(f"Best COMET-QE: {best_qe[0]}  ({best_qe[3]:.4f})")


if __name__ == "__main__":
    main()
