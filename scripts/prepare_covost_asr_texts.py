#!/usr/bin/env python3
"""
Export CoVoST-style EN→ZH test audio to SimulEval text files via off-the-shelf ASR.

Dataset: Hugging Face ``AudioLLMs/covost2_en_zh_test`` (parquet; works with recent
``datasets``). Each row: speech in ``context``, Chinese reference in ``answer``.
Official ``facebook/covost2`` still uses a deprecated dataset script — use this mirror.

Dependencies (conda env with simuleval + transformers):
  pip install datasets soundfile
  (Audio is loaded with ``decode=False`` + WAV bytes — no torchcodec.)

Usage:
  cd /data/user_data/haolingp/IDL_final_3small
  python scripts/prepare_covost_asr_texts.py --hf-split 'test[:100]' --prefix subset100

Chinese references are written in **SimulEval / wmt19-style spacing** by default
(``--target-format simul``) so BLEU matches ``sttr_enzh_agent`` char-wise emits.
To fix an existing compact ``*_target_zh.txt`` without re-ASR, run
``scripts/format_zh_target_for_simul.py``.
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from pathlib import Path

import soundfile as sf
import torch
from datasets import Audio, load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def format_zh_reference_for_simul(text: str) -> str:
    """Match ``data/enzh/wmt19_target.txt`` style: tokens separated by spaces.

    SimulEval concatenates each ``WriteAction`` with spaces; Chinese references
    must use the same spacing or default sacrebleu tokenizer (``13a``) yields ~0 BLEU.
    """
    text = text.strip().replace("\n", " ")
    tokens: list[str] = []
    for m in re.finditer(
        r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*|[\u4e00-\u9fff]|[^\sA-Za-z0-9\u4e00-\u9fff]",
        text,
    ):
        tokens.append(m.group())
    return " ".join(tokens)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CoVoST EN→ZH: ASR → source_asr.txt + target_zh.txt")
    p.add_argument(
        "--dataset",
        default="AudioLLMs/covost2_en_zh_test",
        help="HF dataset id (parquet CoVoST EN→ZH test mirror).",
    )
    p.add_argument(
        "--hf-split",
        default="test[:100]",
        help="datasets split slice, e.g. test[:500] or test (full ~15k).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <repo>/data/covost_enzh).",
    )
    p.add_argument(
        "--prefix",
        default="subset100",
        help="Basename prefix for *_source_asr.txt / *_target_zh.txt / *_manifest.jsonl.",
    )
    p.add_argument(
        "--whisper-model",
        default="openai/whisper-small",
        help="HF model id for English ASR (speech → English text).",
    )
    p.add_argument("--device", default=None, help="cuda:0 or cpu (default: auto).")
    p.add_argument(
        "--dtype",
        choices=("float16", "float32", "bfloat16"),
        default="float16",
        help="Model dtype on GPU (float32 on CPU).",
    )
    p.add_argument(
        "--target-format",
        choices=("simul", "raw"),
        default="simul",
        help="simul: space-separated tokens like wmt19_target.txt (default). raw: dataset text.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir or (_repo_root() / "data" / "covost_enzh")
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        torch_dtype = torch.float32
    else:
        torch_dtype = getattr(torch, args.dtype)

    print(f"[load] dataset={args.dataset!r} split={args.hf_split!r}", flush=True)
    ds = load_dataset(args.dataset, split=args.hf_split)
    ds = ds.cast_column("context", Audio(decode=False))
    n = len(ds)
    print(f"[load] {n} examples", flush=True)

    print(f"[asr] loading {args.whisper_model!r} on {device} ({torch_dtype}) ...", flush=True)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        args.whisper_model,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    model.to(device)
    processor = AutoProcessor.from_pretrained(args.whisper_model)
    dev_arg: int | str = -1
    if device.startswith("cuda"):
        if ":" in device:
            dev_arg = int(device.split(":")[-1])
        else:
            dev_arg = 0

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=dev_arg,
    )

    src_lines: list[str] = []
    tgt_lines: list[str] = []
    manifest_path = out_dir / f"{args.prefix}_manifest.jsonl"

    gen_kw = {"language": "english", "task": "transcribe"}

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i in range(n):
            row = ds[i]
            ctx = row["context"]
            if not isinstance(ctx, dict) or not ctx.get("bytes"):
                print(f"[warn] idx={i} missing audio bytes; skip", file=sys.stderr)
                continue
            wav, sr = sf.read(io.BytesIO(ctx["bytes"]), dtype="float32")
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            sr = int(sr)
            dur_s = float(len(wav) / sr)
            if sr != 16000:
                print(f"[warn] idx={i} sample_rate={sr} expected 16000", file=sys.stderr)

            # Whisper requires return_timestamps=True for audio > 30s
            pipe_kw = {"generate_kwargs": gen_kw}
            if dur_s > 30.0:
                pipe_kw["return_timestamps"] = True
            out = pipe(
                {"array": wav, "sampling_rate": sr},
                **pipe_kw,
            )
            en = (out.get("text") or "").strip()
            if not en:
                en = "."
            zh = (row.get("answer") or "").strip().replace("\n", " ")
            if not zh:
                print(f"[warn] idx={i} empty Chinese reference; skipping", file=sys.stderr)
                continue
            if args.target_format == "simul":
                zh = format_zh_reference_for_simul(zh)

            src_lines.append(en)
            tgt_lines.append(zh)
            rec = {
                "idx": i,
                "duration_s": dur_s,
                "asr_en": en,
                "ref_zh": zh,
            }
            mf.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if (i + 1) % 10 == 0 or i + 1 == n:
                print(f"[asr] {i + 1}/{n}", flush=True)

    src_path = out_dir / f"{args.prefix}_source_asr.txt"
    tgt_path = out_dir / f"{args.prefix}_target_zh.txt"
    src_path.write_text("\n".join(src_lines) + "\n", encoding="utf-8")
    tgt_path.write_text("\n".join(tgt_lines) + "\n", encoding="utf-8")

    print(f"[write] {src_path} ({len(src_lines)} lines)", flush=True)
    print(f"[write] {tgt_path} ({len(tgt_lines)} lines)", flush=True)
    print(f"[write] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
