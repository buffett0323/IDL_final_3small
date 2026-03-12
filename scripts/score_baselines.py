"""
Score baseline results from instances.log files.
Computes BLEU and AL directly, bypassing SimulEval's buggy scorers.

Usage:
    python scripts/score_baselines.py
    python scripts/score_baselines.py --output-dirs outputs/baseline_k3 outputs/baseline_k5
"""

import argparse
import json
from pathlib import Path

import sacrebleu


def compute_al(delays, source_length, target_length):
    """Compute Average Lagging (AL) for one sentence."""
    if target_length == 0 or source_length == 0 or not delays:
        return 0.0
    gamma = target_length / source_length
    al = 0.0
    tau = 0
    for t, d in enumerate(delays):
        if d <= source_length:
            al += d - t / gamma
            tau = t + 1
            if d == source_length:
                break
    if tau == 0:
        return float(source_length)
    return al / tau


def score_directory(output_dir):
    """Score one output directory."""
    log_path = Path(output_dir) / "instances.log"
    if not log_path.exists():
        print(f"  {output_dir}: no instances.log found, skipping")
        return None

    predictions = []
    references = []
    als = []

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line.strip())
            pred = rec["prediction"]
            ref = rec["reference"].strip()
            delays = rec["delays"]
            pred_len = rec["prediction_length"]

            predictions.append(pred)
            references.append(ref)

            # Compute source length from max delay (approximation)
            src_len = max(delays) if delays else 0
            als.append(compute_al(delays, src_len, pred_len))

    # Compute BLEU
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    avg_al = sum(als) / len(als) if als else 0.0

    return {
        "num_sentences": len(predictions),
        "BLEU": bleu.score,
        "AL": avg_al,
    }


def main():
    parser = argparse.ArgumentParser(description="Score baseline results")
    parser.add_argument(
        "--output-dirs",
        nargs="*",
        default=None,
        help="Output directories to score (default: auto-detect outputs/baseline_k*)",
    )
    args = parser.parse_args()

    if args.output_dirs:
        dirs = args.output_dirs
    else:
        dirs = sorted(Path("outputs").glob("baseline_k*"))
        if not dirs:
            print("No baseline output directories found in outputs/")
            return

    print(f"{'Directory':<30} {'Sentences':>10} {'BLEU':>8} {'AL':>8}")
    print("-" * 60)

    for d in dirs:
        result = score_directory(d)
        if result:
            print(
                f"{str(d):<30} {result['num_sentences']:>10} "
                f"{result['BLEU']:>8.2f} {result['AL']:>8.2f}"
            )

    # Write scores files
    for d in dirs:
        result = score_directory(d)
        if result:
            scores_path = Path(d) / "scores"
            with open(scores_path, "w") as f:
                f.write(f"BLEU: {result['BLEU']:.2f}\n")
                f.write(f"AL: {result['AL']:.2f}\n")
                f.write(f"Sentences: {result['num_sentences']}\n")


if __name__ == "__main__":
    main()
