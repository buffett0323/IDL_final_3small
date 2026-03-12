"""
Download WMT En-De test sets via sacrebleu.

No authentication or manual downloads required.

Usage:
    python scripts/download_wmt.py
    python scripts/download_wmt.py --test-sets wmt14 wmt19
    python scripts/download_wmt.py --output_dir data/wmt
"""

import argparse
import json
from pathlib import Path

import sacrebleu


def download_and_save(test_sets, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for ts in test_sets:
        print(f"\n--- Downloading '{ts}' En-De ---")

        # sacrebleu downloads the data automatically
        sources = sacrebleu.get_source_file(ts, "en-de")
        refs = sacrebleu.get_reference_files(ts, "en-de")

        # Read lines
        with open(sources, encoding="utf-8") as f:
            src_lines = [line.strip() for line in f if line.strip()]
        with open(refs[0], encoding="utf-8") as f:
            tgt_lines = [line.strip() for line in f if line.strip()]

        assert len(src_lines) == len(tgt_lines), (
            f"Mismatch: {len(src_lines)} sources vs {len(tgt_lines)} targets"
        )
        print(f"  {len(src_lines)} sentence pairs")

        # Save JSON
        records = []
        for i, (src, tgt) in enumerate(zip(src_lines, tgt_lines)):
            records.append({"id": i, "sentence": src, "translation": tgt})

        json_path = output_dir / f"{ts}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print(f"  Saved JSON -> {json_path}")

        # Save SimulEval-format files (one sentence per line)
        src_path = output_dir / f"{ts}_source.txt"
        tgt_path = output_dir / f"{ts}_target.txt"
        with open(src_path, "w", encoding="utf-8") as sf, \
             open(tgt_path, "w", encoding="utf-8") as tf:
            for rec in records:
                sf.write(rec["sentence"] + "\n")
                tf.write(rec["translation"] + "\n")
        print(f"  Saved SimulEval source -> {src_path}")
        print(f"  Saved SimulEval target -> {tgt_path}")

        # Print examples
        print(f"\n  Example pairs from '{ts}':")
        for rec in records[:3]:
            print(f"    EN: {rec['sentence']}")
            print(f"    DE: {rec['translation']}")
            print()

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download WMT En-De test sets")
    parser.add_argument(
        "--test-sets",
        nargs="+",
        default=["wmt14"],
        help="WMT test sets to download (default: wmt14). Options: wmt14, wmt19, wmt20, etc.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/wmt",
        help="Output directory (default: data/wmt)",
    )
    args = parser.parse_args()
    download_and_save(args.test_sets, args.output_dir)
