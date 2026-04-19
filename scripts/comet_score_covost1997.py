#!/usr/bin/env python3
"""Score COMET for all covost1997/* experiment directories."""
from __future__ import annotations
import argparse
import json
import sys
import unicodedata
from pathlib import Path

REPO = Path("/data/user_data/haolingp/IDL_final_3small")


def is_cjk(ch: str) -> bool:
    cp = ord(ch)
    return (0x4E00 <= cp <= 0x9FFF or
            0x3400 <= cp <= 0x4DBF or
            0x20000 <= cp <= 0x2A6DF or
            0x2A700 <= cp <= 0x2B73F or
            0xF900 <= cp <= 0xFAFF or
            0x3000 <= cp <= 0x303F or
            0xFF00 <= cp <= 0xFFEF)


def dejoin_zh(text: str) -> str:
    chars = list(text)
    out = []
    i = 0
    while i < len(chars):
        if chars[i] == ' ' and 0 < i < len(chars) - 1:
            pr, nx = chars[i - 1], chars[i + 1]
            pc = len(pr.strip()) == 1 and is_cjk(pr)
            nc = len(nx.strip()) == 1 and is_cjk(nx)
            pp = unicodedata.category(pr).startswith('P')
            np_ = unicodedata.category(nx).startswith('P')
            if (pc and nc) or (pc and np_) or (pp and nc):
                i += 1
                continue
        out.append(chars[i])
        i += 1
    return "".join(out).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--gpus", type=int, default=1)
    args = ap.parse_args()

    src_path = REPO / "data/covost_enzh/subset1997_source_asr.txt"
    tgt_raw  = REPO / "data/covost_enzh/subset1997_target_zh.txt"
    srcs = [l.strip() for l in open(src_path)]
    refs = [dejoin_zh(l) for l in open(tgt_raw)]
    print(f"Loaded {len(srcs)} src / {len(refs)} ref")

    exps = sorted((REPO / "outputs/covost1997").glob("*"))
    exps = [e for e in exps if e.is_dir() and (e / "instances.log").exists() and (e / "scores").exists()]
    print(f"Found {len(exps)} experiments")

    model = None
    results = []
    for exp in exps:
        cache = exp / "comet_score.txt"
        if cache.exists() and not args.force:
            val = float(cache.read_text().strip())
            print(f"[CACHED] {exp.name}: COMET={val:.4f}")
            results.append((exp.name, val))
            continue

        recs = []
        with open(exp / "instances.log") as f:
            for ln in f:
                if ln.strip():
                    recs.append(json.loads(ln))
        recs.sort(key=lambda r: r["index"])
        hyps = [dejoin_zh(r["prediction"]) for r in recs]

        if len(hyps) != len(refs):
            print(f"[SKIP] {exp.name}: hyp={len(hyps)} != ref={len(refs)}")
            continue

        if model is None:
            from comet import download_model, load_from_checkpoint
            mp = download_model("Unbabel/wmt22-comet-da")
            model = load_from_checkpoint(mp)

        data = [{"src": s, "mt": h, "ref": r} for s, h, r in zip(srcs, hyps, refs)]
        out = model.predict(data, batch_size=args.batch_size, gpus=args.gpus)
        sys_score = out[1] if isinstance(out, tuple) else out.system_score
        cache.write_text(f"{sys_score:.6f}\n")
        results.append((exp.name, float(sys_score)))
        print(f"[DONE] {exp.name}: COMET={sys_score:.4f}")

    print("\n=== CoVoST-1997 COMET summary ===")
    for name, sc in sorted(results):
        print(f"  {name}: {sc:.4f}")


if __name__ == "__main__":
    main()
