"""
COMET evaluation for SimulEval instances.log files.

Usage:
    python scripts/eval_comet.py outputs/rand100_baseline outputs/rand100_sttr_qwen
    python scripts/eval_comet.py outputs/rand100_baseline  # single dir
    python scripts/eval_comet.py --all-rand100             # all rand100_* dirs
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_instances(log_path: str) -> tuple[list[str], list[str], list[str]]:
    """Load (sources, hypotheses, references) from instances.log."""
    sources, hyps, refs = [], [], []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            src = d.get("source", "")
            hyp = d.get("prediction", "")
            ref = d.get("reference", "")

            # Strip space-tokenized Chinese back to surface form for fairer comparison
            hyp = hyp.replace(" ", "")
            ref = ref.replace(" ", "").strip()

            sources.append(src)
            hyps.append(hyp)
            refs.append(ref)
    return sources, hyps, refs


def read_bleu_al(scores_path: str) -> tuple[float, float]:
    """Read BLEU and AL from SimulEval scores file."""
    try:
        import re
        txt = open(scores_path).read()
        nums = re.findall(r"[\d]+\.[\d]+", txt)
        bleu = float(nums[0]) if nums else float("nan")
        al = float(nums[2]) if len(nums) > 2 else float("nan")
        return bleu, al
    except Exception:
        return float("nan"), float("nan")


def eval_dir(
    exp_dir: str,
    model,
    batch_size: int = 64,
    gpus: int = 1,
) -> dict:
    """Evaluate one experiment directory with COMET."""
    log_path = os.path.join(exp_dir, "instances.log")
    if not os.path.exists(log_path):
        return {"error": "no instances.log"}

    sources, hyps, refs = load_instances(log_path)
    if not sources:
        return {"error": "empty log"}

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hyps, refs)
    ]

    output = model.predict(data, batch_size=batch_size, gpus=gpus, progress_bar=False)
    scores = output.scores
    system_score = output.system_score

    bleu, al = float("nan"), float("nan")
    scores_path = os.path.join(exp_dir, "scores")
    if os.path.exists(scores_path):
        bleu, al = read_bleu_al(scores_path)

    return {
        "name": os.path.basename(exp_dir),
        "n_sentences": len(sources),
        "comet": round(system_score * 100, 2),
        "bleu": bleu,
        "al": al,
        "sentence_scores": scores,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="*", help="Experiment directories to evaluate")
    parser.add_argument(
        "--all-rand100", action="store_true",
        help="Auto-find all outputs/rand100_* directories",
    )
    parser.add_argument(
        "--model", default="Unbabel/wmt22-comet-da",
        help="COMET model name (default: wmt22-comet-da)",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results to JSON file",
    )
    args = parser.parse_args()

    dirs = list(args.dirs)
    if args.all_rand100:
        outputs = Path("outputs")
        dirs += sorted(str(d) for d in outputs.glob("rand100_*") if d.is_dir())

    if not dirs:
        parser.print_help()
        sys.exit(1)

    print(f"[COMET] Loading model: {args.model}")
    from comet import download_model, load_from_checkpoint
    model_path = download_model(args.model)
    model = load_from_checkpoint(model_path)

    results = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"[SKIP] {d}: not a directory")
            continue
        print(f"\n[Eval] {d} ...", end=" ", flush=True)
        r = eval_dir(d, model, batch_size=args.batch_size, gpus=args.gpus)
        results.append(r)
        if "error" in r:
            print(r["error"])
        else:
            print(f"COMET={r['comet']:.2f}  BLEU={r['bleu']:.2f}  AL={r['al']:.2f}")

    print("\n" + "=" * 65)
    print(f"{'Experiment':<45} {'COMET':>7} {'BLEU':>7} {'AL':>6}")
    print("-" * 65)
    for r in results:
        if "error" in r:
            print(f"{r.get('name', '?'):<45} {'ERROR':>7}")
        else:
            print(
                f"{r['name']:<45} {r['comet']:>7.2f} {r['bleu']:>7.2f} {r['al']:>6.2f}"
            )
    print("=" * 65)

    if args.output:
        clean = [{k: v for k, v in r.items() if k != "sentence_scores"} for r in results]
        with open(args.output, "w") as f:
            json.dump(clean, f, indent=2)
        print(f"\n[Saved] {args.output}")

    return results


if __name__ == "__main__":
    main()
