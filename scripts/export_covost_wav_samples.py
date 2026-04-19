#!/usr/bin/env python3
"""Export a few CoVoST clips for demo/pipeline.html (optional <audio>).

Uses ``Audio(decode=False)`` + soundfile so **torchcodec is not required** (avoids
CUDA/FFmpeg issues on some cluster login nodes).

  pip install datasets soundfile
  python scripts/export_covost_wav_samples.py --indices 0,1,2
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import soundfile as sf


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--indices",
        default="0,1,2",
        help="Comma-separated row indices in test split.",
    )
    p.add_argument(
        "--split",
        default="test[:500]",
        help="datasets split (must cover max index).",
    )
    p.add_argument(
        "--dataset",
        default="AudioLLMs/covost2_en_zh_test",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Default: <repo>/demo/audio",
    )
    args = p.parse_args()
    root = Path(__file__).resolve().parent.parent
    out_dir = args.out_dir or (root / "demo" / "audio")
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_list = [int(x.strip()) for x in args.indices.split(",") if x.strip()]

    from datasets import Audio, load_dataset

    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.cast_column("context", Audio(decode=False))
    for i in idx_list:
        row = ds[i]
        ctx = row["context"]
        if not isinstance(ctx, dict) or not ctx.get("bytes"):
            raise RuntimeError(f"idx={i}: expected context.bytes (decode=False)")
        wav, sr = sf.read(io.BytesIO(ctx["bytes"]), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        dur = len(wav) / float(sr)
        path = out_dir / f"sample_{i}.wav"
        sf.write(str(path), wav, int(sr))
        print(f"[write] {path}  ({dur:.2f}s @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
