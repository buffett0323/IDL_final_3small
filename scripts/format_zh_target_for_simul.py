#!/usr/bin/env python3
"""Convert compact Chinese reference lines to SimulEval / wmt19_target-style spacing.

Use when BLEU ~0 with sttr_enzh_agent: hypotheses are char-emits joined by spaces;
references must match (or pass simuleval --sacrebleu-tokenizer zh).

  python scripts/format_zh_target_for_simul.py \\
    --in data/covost_enzh/subset100_target_zh.txt \\
    --out data/covost_enzh/subset100_target_zh_simul.txt
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


def format_zh_reference_for_simul(text: str) -> str:
    text = text.strip().replace("\n", " ")
    tokens: list[str] = []
    for m in re.finditer(
        r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|[^\sA-Za-z0-9\u4e00-\u9fff]",
        text,
    ):
        tokens.append(m.group())
    return " ".join(tokens)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--out", dest="out", type=Path, required=True)
    args = ap.parse_args()
    lines = args.inp.read_text(encoding="utf-8").splitlines()
    out_lines = [format_zh_reference_for_simul(L) if L.strip() else "" for L in lines]
    args.out.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[write] {args.out} ({len(out_lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
