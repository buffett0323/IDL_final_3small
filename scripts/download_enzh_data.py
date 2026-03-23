#!/usr/bin/env python3
"""Download WMT19 En-Zh test set and prepare character-segmented files for SimulEval.

Usage:
    python scripts/download_enzh_data.py [--max-sentences N]
"""

import argparse
from pathlib import Path

import sacrebleu

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "enzh"


def segment_chinese(text: str) -> str:
    """Insert spaces between CJK characters for SimulEval word counting."""
    out = []
    buf = []
    for ch in text:
        cp = ord(ch)
        is_cjk = (
            (0x4E00 <= cp <= 0x9FFF)
            or (0x3400 <= cp <= 0x4DBF)
            or (0x20000 <= cp <= 0x2A6DF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x3000 <= cp <= 0x303F)
            or (0xFF00 <= cp <= 0xFFEF)
        )
        if is_cjk:
            if buf:
                out.append("".join(buf))
                buf = []
            out.append(ch)
        elif ch in (" ", "\t", "\n"):
            if buf:
                out.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return " ".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-sentences", type=int, default=0,
                        help="Limit to first N sentences (0 = all)")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # WMT19 En-Zh test set (1997 sentences)
    dataset = sacrebleu.DATASETS["wmt19"]
    src_path = sacrebleu.get_source_file("wmt19", "en-zh")
    ref_path = sacrebleu.get_reference_files("wmt19", "en-zh")[0]

    with open(src_path) as f:
        src_lines = [line.strip() for line in f if line.strip()]
    with open(ref_path) as f:
        ref_lines = [line.strip() for line in f if line.strip()]

    if args.max_sentences > 0:
        src_lines = src_lines[: args.max_sentences]
        ref_lines = ref_lines[: args.max_sentences]

    # Write source (English, one sentence per line)
    src_out = DATA_DIR / "wmt19_source.txt"
    src_out.write_text("\n".join(src_lines) + "\n", encoding="utf-8")

    # Write target (Chinese, character-segmented for SimulEval)
    ref_segmented = [segment_chinese(line) for line in ref_lines]
    tgt_out = DATA_DIR / "wmt19_target.txt"
    tgt_out.write_text("\n".join(ref_segmented) + "\n", encoding="utf-8")

    print(f"Wrote {len(src_lines)} sentences")
    print(f"  Source: {src_out}")
    print(f"  Target: {tgt_out}")


if __name__ == "__main__":
    main()
