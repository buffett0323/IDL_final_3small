"""
Download CoVoST2 En→De dataset from HuggingFace.

Prerequisites:
    1. pip install datasets huggingface_hub
    2. Accept the dataset license at https://huggingface.co/datasets/facebook/covost2
    3. Log in: huggingface-cli login

Usage:
    python scripts/download_covost2.py
    python scripts/download_covost2.py --splits test validation
    python scripts/download_covost2.py --output_dir data/covost2
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset


def download_and_save(splits, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        print(f"\n--- Downloading '{split}' split ---")
        dataset = load_dataset(
            "facebook/covost2", "en_de", split=split,
            trust_remote_code=True,
        )
        print(f"  {len(dataset)} samples")

        # Save text data (source sentences + translations) as JSON
        records = []
        for i, example in enumerate(dataset):
            records.append({
                "id": example.get("id", i),
                "sentence": example["sentence"],         # English transcript
                "translation": example["translation"],   # German reference
            })

        json_path = output_dir / f"{split}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Saved text to {json_path}")

        # Save SimulEval-format files (one sentence per line)
        src_path = output_dir / f"{split}_source.txt"
        tgt_path = output_dir / f"{split}_target.txt"
        with open(src_path, "w", encoding="utf-8") as src_f, \
             open(tgt_path, "w", encoding="utf-8") as tgt_f:
            for rec in records:
                src_f.write(rec["sentence"].strip() + "\n")
                tgt_f.write(rec["translation"].strip() + "\n")
        print(f"  Saved SimulEval source → {src_path}")
        print(f"  Saved SimulEval target → {tgt_path}")

        # Print a few examples
        print(f"\n  Example samples from '{split}':")
        for rec in records[:3]:
            print(f"    EN: {rec['sentence']}")
            print(f"    DE: {rec['translation']}")
            print()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download CoVoST2 En→De")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["test", "validation"],
        help="Which splits to download (default: test validation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/covost2",
        help="Output directory (default: data/covost2)",
    )
    args = parser.parse_args()
    download_and_save(args.splits, args.output_dir)
