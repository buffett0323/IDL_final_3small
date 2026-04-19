#!/usr/bin/env python3
"""
COMET scoring for all SimulEval experiment outputs.

For each experiment directory that has both `scores` and `instances.log`,
extracts hypotheses, de-spaces Chinese text, and runs COMET scoring.
Results are cached in each experiment dir as `comet_score.txt`.

Usage:
    python scripts/comet_score_all.py --repo-root . --search-root outputs/
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

# ── Chinese text utilities ───────────────────────────────────────────

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
    """Remove spaces between CJK characters / CJK+punctuation pairs."""
    chars = list(text)
    result = []
    i = 0
    while i < len(chars):
        if chars[i] == ' ' and i > 0 and i < len(chars) - 1:
            prev_ch = chars[i - 1]
            next_ch = chars[i + 1]
            prev_cjk = len(prev_ch.strip()) == 1 and is_cjk(prev_ch)
            next_cjk = len(next_ch.strip()) == 1 and is_cjk(next_ch)
            prev_punct = unicodedata.category(prev_ch).startswith('P')
            next_punct = unicodedata.category(next_ch).startswith('P')

            if (prev_cjk and next_cjk) or \
               (prev_cjk and next_punct) or \
               (prev_punct and next_cjk):
                i += 1  # skip space
                continue
        result.append(chars[i])
        i += 1
    return "".join(result).strip()


# ── Extract hypotheses from instances.log ────────────────────────────

def extract_hypotheses(instances_log: Path) -> list[str]:
    records = []
    with open(instances_log) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            records.append(d)

    records.sort(key=lambda r: r["index"])
    return [dejoin_zh(r["prediction"]) for r in records]


def prepare_reference(ref_path: Path) -> list[str]:
    lines = []
    with open(ref_path) as f:
        for line in f:
            lines.append(dejoin_zh(line))
    return lines


# ── Map experiment dirs to data slices ───────────────────────────────

DATA_SLICES = {
    "wmt19":   {"src": "data/enzh/wmt19_source.txt",   "tgt": "data/enzh/wmt19_target.txt"},
    "wmt500":  {"src": "data/enzh/wmt500_source.txt",  "tgt": "data/enzh/wmt500_target.txt"},
    "rand100": {"src": "data/enzh/rand100_source.txt",  "tgt": "data/enzh/rand100_target.txt"},
}


def detect_slice(exp_dir: Path, repo_root: Path) -> str | None:
    """Heuristic: detect data slice from experiment path."""
    rel = str(exp_dir.relative_to(repo_root / "outputs"))
    name = exp_dir.name
    parent = exp_dir.parent.name

    # CoVoST — skip for now
    if "covost" in rel:
        return None

    # Explicit wmt19 runs
    if "wmt19_qwen" in rel:
        return "wmt19"

    # NLLB full runs on wmt19 (no 'wmt500' in name, under nllb_support or full/)
    if parent == "nllb_support":
        return "wmt19"
    if parent == "full" and "wmt500" not in name and not name.startswith("qwen") and not name.startswith("semlcp"):
        return "wmt19"

    # Explicit wmt500 in name
    if "wmt500" in name:
        return "wmt500"

    # Qwen runs under full/ or qwen_main/ or qwen_bridge/ → wmt500
    if parent in ("qwen_main", "qwen_bridge"):
        return "wmt500"
    if parent == "full" and (name.startswith("qwen") or name.startswith("semlcp")):
        return "wmt500"

    # NLLB full runs with wmt500 explicit
    if parent == "full" and "nllb_lcp" in name:
        return "wmt500"

    # Fair comparison, ablation, rand100 runs
    if name.startswith("fair_") or parent.startswith("fair_"):
        return "rand100"
    if "tc_ablation" in rel or "rand100" in rel:
        return "rand100"

    # Rerun NLLB support
    if "rerun_" in rel and "nllb" in rel:
        return "wmt19"

    # Fallback: count lines in instances.log to match
    inst = exp_dir / "instances.log"
    if inst.exists():
        n = sum(1 for l in open(inst) if l.strip())
        if n == 1997:
            return "wmt19"
        elif n == 500:
            return "wmt500"
        elif n == 100:
            return "rand100"

    return None


# ── Main scoring loop ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--search-root", type=Path, default=Path("outputs"))
    parser.add_argument("--gpus", type=int, default=0)
    parser.add_argument("--model", default="Unbabel/wmt22-comet-da")
    parser.add_argument("--force", action="store_true", help="Re-score even if cached")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    search = args.search_root.resolve()

    # Prepare de-spaced references
    ref_cache: dict[str, list[str]] = {}
    src_cache: dict[str, list[str]] = {}
    for slice_name, paths in DATA_SLICES.items():
        src_path = repo / paths["src"]
        tgt_path = repo / paths["tgt"]
        if src_path.exists() and tgt_path.exists():
            src_cache[slice_name] = [l.strip() for l in open(src_path)]
            ref_cache[slice_name] = prepare_reference(tgt_path)
            print(f"[COMET] Loaded {slice_name}: {len(src_cache[slice_name])} sentences")

    # Find experiment dirs
    experiments = []
    for inst_log in sorted(search.rglob("instances.log")):
        exp_dir = inst_log.parent
        if not (exp_dir / "scores").exists():
            continue
        experiments.append(exp_dir)

    print(f"\n[COMET] Found {len(experiments)} completed experiments\n")

    # Score each
    results = []
    for exp_dir in experiments:
        slice_name = detect_slice(exp_dir, repo)
        if slice_name is None:
            rel = exp_dir.relative_to(repo / "outputs")
            print(f"[SKIP] {rel} — cannot detect data slice")
            continue

        if slice_name not in ref_cache:
            print(f"[SKIP] {exp_dir.name} — no reference for {slice_name}")
            continue

        rel = str(exp_dir.relative_to(repo / "outputs"))
        comet_cache = exp_dir / "comet_score.txt"

        if comet_cache.exists() and not args.force:
            score = float(comet_cache.read_text().strip())
            n = len(ref_cache[slice_name])
            results.append((rel, slice_name, n, score))
            print(f"[CACHED] {rel}: COMET={score:.4f}")
            continue

        # Extract hypotheses
        hyps = extract_hypotheses(exp_dir / "instances.log")
        refs = ref_cache[slice_name]
        srcs = src_cache[slice_name]

        if len(hyps) != len(refs):
            print(f"[SKIP] {rel} — line mismatch: hyp={len(hyps)} ref={len(refs)}")
            continue

        print(f"[COMET] Scoring {rel} ({len(hyps)} sentences, {slice_name}) ...")

        try:
            from comet import download_model, load_from_checkpoint

            model_path = download_model(args.model)
            model = load_from_checkpoint(model_path)

            data = [{"src": s, "mt": h, "ref": r}
                    for s, h, r in zip(srcs, hyps, refs)]

            output = model.predict(data, batch_size=64, gpus=args.gpus)
            # COMET returns (scores_per_sentence, system_score)
            if isinstance(output, tuple):
                system_score = output[1]
            else:
                system_score = output.system_score

            comet_cache.write_text(f"{system_score:.6f}\n")
            results.append((rel, slice_name, len(hyps), system_score))
            print(f"[DONE] {rel}: COMET={system_score:.4f}")

        except Exception as e:
            print(f"[FAIL] {rel}: {e}")
            continue

    # ── Write summary ────────────────────────────────────────────────
    out_md = repo / "outputs" / "comet_scores.md"
    out_csv = repo / "outputs" / "comet_scores.csv"

    results.sort(key=lambda r: (r[1], r[0]))

    with open(out_md, "w") as f:
        f.write("# COMET Scores\n\n")
        f.write(f"Model: `{args.model}`\n\n")
        f.write("| Experiment | Data | N | COMET |\n")
        f.write("|:--|:--|--:|--:|\n")
        for rel, sl, n, sc in results:
            f.write(f"| {rel} | {sl} | {n} | {sc:.4f} |\n")

    with open(out_csv, "w") as f:
        f.write("experiment,data_slice,n,comet_score\n")
        for rel, sl, n, sc in results:
            f.write(f"{rel},{sl},{n},{sc:.6f}\n")

    print(f"\n{'='*60}")
    print(f"Results: {out_md}")
    print(f"CSV:     {out_csv}")
    print(f"{'='*60}\n")

    # Print summary table
    with open(out_md) as f:
        print(f.read())


if __name__ == "__main__":
    main()
