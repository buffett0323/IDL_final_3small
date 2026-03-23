#!/usr/bin/env python3
"""verbose_trace.py — Generate step-by-step verbose policy trace for a sentence.

Produces two side-by-side log files (baseline + DD) in the style of
verbose_AUD*.log, showing at each source step:
  - Source prefix seen so far
  - Chinese committed so far
  - [DD only] K sampled futures + JS divergence score
  - Policy decision: READ or WRITE + what character is written
  - Running translation buffer

Usage:
    cd /data/user_data/haolingp/IDL_final_3small
    python3 scripts/verbose_trace.py --sent 76
    python3 scripts/verbose_trace.py --sent 76 --mode dd_full --tau 0.05
    python3 scripts/verbose_trace.py --list       # list available sentences with veto activity
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT     = Path(__file__).resolve().parent.parent
OUT_ROOT = ROOT / "outputs"
DATA_DIR = ROOT / "data/enzh"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_instances(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    result = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                result[rec["index"]] = rec
            except Exception:
                pass
    return result


def load_dd_trace(path: Path, sent_id: int) -> list[dict]:
    if not path.exists():
        return []
    recs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                if r["sentence_id"] == sent_id:
                    recs.append(r)
            except Exception:
                pass
    return sorted(recs, key=lambda r: r["src_len"])


def load_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [l.rstrip("\n") for l in path.read_text(encoding="utf-8").splitlines()]


def reconstruct_timeline(chars: list[str], delays: list[int], max_src: int) -> dict[int, list[str]]:
    """Map src_len → list of chars written at that step."""
    timeline: dict[int, list[str]] = {}
    for char, delay in zip(chars, delays):
        timeline.setdefault(delay, []).append(char)
    return timeline


# ── Verbose log generators ────────────────────────────────────────────────────

def generate_baseline_log(
    sent_id: int,
    src_words: list[str],
    instance: dict,
    wait_k: int = 5,
) -> str:
    lines = []
    a = lines.append

    pred_chars = instance["prediction"].split()
    delays     = instance["delays"]
    reference  = instance["reference"].strip()
    source     = instance["source"]

    timeline = reconstruct_timeline(pred_chars, delays, len(src_words))

    a("=" * 70)
    a(f"# Sentence {sent_id}  [BASELINE wait-k={wait_k}]")
    a(f"# Source    : {source}")
    a(f"# Reference : {reference}")
    a(f"# Source len: {len(src_words)} words")
    a(f"# Policy    : read until src_len - tgt_len >= {wait_k}, then write 1 char per step")
    a("=" * 70)
    a("")

    tgt_ptr = 0
    committed = []

    for src_len in range(1, len(src_words) + 1):
        src_prefix = " ".join(src_words[:src_len])
        tgt_len    = tgt_ptr
        gap        = src_len - tgt_len
        chars_now  = timeline.get(src_len, [])

        a("─" * 70)
        a(f"Step {src_len}/{len(src_words)}")
        a(f"  Source prefix  : \"{src_prefix}\"")
        a(f"  Committed ZH   : \"{' '.join(committed)}\"  ({tgt_len} chars)")
        a(f"  Gap (src-tgt)  : {gap}  (need ≥ {wait_k} to write)")
        a("")

        if gap < wait_k and not (src_len == len(src_words)):
            a(f"  ► DECISION  : READ")
            a(f"    Reason    : gap={gap} < wait_k={wait_k} — need more source context")
        else:
            if chars_now:
                committed.extend(chars_now)
                tgt_ptr += len(chars_now)
                a(f"  ► DECISION  : WRITE")
                a(f"    Written   : \"{' '.join(chars_now)}\"")
                a(f"    Committed : \"{' '.join(committed)}\"")
            else:
                a(f"  ► DECISION  : READ")
                a(f"    Reason    : no chars ready at this step")
        a("")

    a("=" * 70)
    a("FINAL OUTPUT")
    a(f"  Reference : {reference}")
    a(f"  Prediction: {instance['prediction']}")
    a("=" * 70)

    return "\n".join(lines)


def generate_dd_log(
    sent_id: int,
    src_words: list[str],
    instance: dict,
    trace: list[dict],
    wait_k: int = 5,
    mode: str = "dd_full",
    tau: float = 0.05,
    baseline_instance: dict | None = None,
) -> str:
    lines = []
    a = lines.append

    pred_chars = instance["prediction"].split()
    delays     = instance["delays"]
    reference  = instance["reference"].strip()
    source     = instance["source"]

    timeline   = reconstruct_timeline(pred_chars, delays, len(src_words))
    trace_map  = {r["src_len"]: r for r in trace}

    mode_label = "DD Full Gate" if mode == "dd_full" else "DD Veto"

    a("=" * 70)
    a(f"# Sentence {sent_id}  [{mode_label}  τ={tau}  wait-k={wait_k}]")
    a(f"# Source    : {source}")
    a(f"# Reference : {reference}")
    a(f"# Source len: {len(src_words)} words")
    if mode == "dd_full":
        a(f"# Policy    : wait-k={wait_k}, then DD gate (JS > τ → READ, JS ≤ τ → WRITE)")
    else:
        a(f"# Policy    : wait-k={wait_k}, entropy gate first, then DD veto on commits")
    if baseline_instance:
        a(f"# Baseline  : {baseline_instance['prediction']}")
    a("=" * 70)
    a("")

    tgt_ptr  = 0
    committed = []

    for src_len in range(1, len(src_words) + 1):
        src_prefix = " ".join(src_words[:src_len])
        tgt_len    = tgt_ptr
        gap        = src_len - tgt_len
        chars_now  = timeline.get(src_len, [])
        tr         = trace_map.get(src_len)

        a("─" * 70)
        a(f"Step {src_len}/{len(src_words)}")
        a(f"  Source prefix  : \"{src_prefix}\"")
        a(f"  Committed ZH   : \"{' '.join(committed)}\"  ({tgt_len} chars)")
        a(f"  Gap (src-tgt)  : {gap}")
        a("")

        if gap < wait_k and tr is None:
            # Pure wait-k phase
            a(f"  ► DECISION  : READ")
            a(f"    Reason    : wait-k — gap={gap} < {wait_k}")

        elif tr is not None:
            # DD was evaluated at this step
            js   = tr["avg_js_firstN"]
            js1  = tr["avg_js_first1"]
            js3  = tr["avg_js_first3"]
            futures = tr.get("futures", [])
            K    = tr.get("K", len(futures))
            n_steps = tr.get("n_steps", 3)

            a(f"  [DD Gate]")
            a(f"    K futures (truncation, oracle source):")
            for i, fut in enumerate(futures[:K]):
                a(f"      future[{i+1}]: \"{fut}\"")
            a(f"    JS divergence:")
            a(f"      avg_js_first1 = {js1:.4f}  (1-step proxy)")
            a(f"      avg_js_first3 = {js3:.4f}  (3-step average)")
            a(f"      avg_js_firstN = {js:.4f}  (policy score, N={n_steps})")
            a(f"    Threshold τ = {tau}")

            if tr["decision"] == "READ":
                a(f"")
                a(f"  ► DECISION  : READ  🚫 (early-commit risk detected)")
                a(f"    Reason    : JS={js:.4f} > τ={tau} — future context is informative")
                a(f"                committing now risks being wrong when more source arrives")
            else:
                if chars_now:
                    committed.extend(chars_now)
                    tgt_ptr += len(chars_now)
                    a(f"")
                    a(f"  ► DECISION  : WRITE  ✅ (safe to commit)")
                    a(f"    Reason    : JS={js:.4f} ≤ τ={tau} — distributions stable across futures")
                    a(f"    Written   : \"{' '.join(chars_now)}\"")
                    a(f"    Committed : \"{' '.join(committed)}\"")
                else:
                    a(f"")
                    a(f"  ► DECISION  : READ")
                    a(f"    Reason    : JS={js:.4f} ≤ τ={tau} — DD says OK, but no char ready yet")

        else:
            # Source finished or beyond trace — flush
            if chars_now:
                committed.extend(chars_now)
                tgt_ptr += len(chars_now)
                a(f"  ► DECISION  : WRITE  (source finished — flush)")
                a(f"    Written   : \"{' '.join(chars_now)}\"")
                a(f"    Committed : \"{' '.join(committed)}\"")
            else:
                a(f"  ► DECISION  : READ  (no chars ready)")

        a("")

    a("=" * 70)
    a("FINAL OUTPUT")
    a(f"  Reference : {reference}")
    if baseline_instance:
        a(f"  Baseline  : {baseline_instance['prediction']}")
    a(f"  DD output : {instance['prediction']}")
    a("=" * 70)

    return "\n".join(lines)


# ── Sentence listing helper ───────────────────────────────────────────────────

def list_sentences(mode: str, tau: float) -> None:
    tau_str  = str(tau)
    trace_path = OUT_ROOT / f"cmp_{mode}_tau{tau_str}" / "dd_trace.jsonl"
    src_lines  = load_lines(DATA_DIR / "rand100_source.txt")

    by_sent: dict[int, list] = {}
    with trace_path.open() as f:
        for line in f:
            r = json.loads(line)
            by_sent.setdefault(r["sentence_id"], []).append(r)

    rows = []
    for sid, recs in by_sent.items():
        n_reads = sum(1 for r in recs if r["decision"] == "READ")
        max_js  = max(r["avg_js_firstN"] for r in recs)
        consec  = 0
        best    = 0
        for r in sorted(recs, key=lambda r: r["src_len"]):
            if r["decision"] == "READ":
                consec += 1
                best = max(best, consec)
            else:
                consec = 0
        rows.append((sid, n_reads, best, max_js))

    rows.sort(key=lambda r: -r[2])
    print(f"\n{'Sent':>5}  {'Vetos':>5}  {'MaxConsec':>9}  {'MaxJS':>7}  Source")
    print("-" * 72)
    for sid, nv, mc, mjs in rows[:20]:
        src = src_lines[sid - 1][:60] if sid <= len(src_lines) else ""
        print(f"{sid:>5}  {nv:>5}  {mc:>9}  {mjs:>7.4f}  {src}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sent", type=int, default=76,
                   help="Sentence ID (1-based, default 76)")
    p.add_argument("--mode", default="dd_full",
                   choices=["dd_full", "dd_veto"],
                   help="DD experiment mode")
    p.add_argument("--tau", type=float, default=0.03,
                   help="DD tau (default 0.03)")
    p.add_argument("--wait-k", type=int, default=5,
                   help="Wait-k value (default 5)")
    p.add_argument("--list", action="store_true",
                   help="List sentences ranked by veto activity then exit")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Directory for output log files (default: outputs/verbose_traces/)")
    args = p.parse_args()

    tau_str = str(args.tau)

    if args.list:
        list_sentences(args.mode, args.tau)
        return

    out_dir = args.out_dir or (OUT_ROOT / "verbose_traces")
    out_dir.mkdir(parents=True, exist_ok=True)

    sent_id = args.sent
    idx     = sent_id - 1

    # Load data
    src_lines  = load_lines(DATA_DIR / "rand100_source.txt")
    src_words  = src_lines[idx].split() if idx < len(src_lines) else []

    base_instances = load_instances(OUT_ROOT / "cmp_baseline_k5" / "instances.log")
    dd_dir         = OUT_ROOT / f"cmp_{args.mode}_tau{tau_str}"
    dd_instances   = load_instances(dd_dir / "instances.log")
    trace          = load_dd_trace(dd_dir / "dd_trace.jsonl", sent_id)

    base_inst = base_instances.get(idx)
    dd_inst   = dd_instances.get(idx)

    if base_inst is None:
        print(f"[Error] No baseline instance for sent {sent_id} (idx {idx})")
        return
    if dd_inst is None:
        print(f"[Error] No DD instance for sent {sent_id} (idx {idx})")
        return

    # ── Generate baseline log ──
    baseline_log = generate_baseline_log(
        sent_id, src_words, base_inst, wait_k=args.wait_k
    )
    base_path = out_dir / f"verbose_baseline_k{args.wait_k}_sent{sent_id}.log"
    base_path.write_text(baseline_log, encoding="utf-8")
    print(f"[Baseline log] → {base_path}")

    # ── Generate DD log ──
    dd_log = generate_dd_log(
        sent_id, src_words, dd_inst, trace,
        wait_k=args.wait_k, mode=args.mode, tau=args.tau,
        baseline_instance=base_inst,
    )
    dd_path = out_dir / f"verbose_{args.mode}_tau{tau_str}_sent{sent_id}.log"
    dd_path.write_text(dd_log, encoding="utf-8")
    print(f"[DD log]       → {dd_path}")

    # ── Quick console preview ──
    print()
    print("=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    print(f"Source    : {base_inst['source']}")
    print(f"Reference : {base_inst['reference'].strip()}")
    print(f"Baseline  : {base_inst['prediction']}")
    print(f"DD output : {dd_inst['prediction']}")
    if trace:
        n_read = sum(1 for r in trace if r["decision"] == "READ")
        print(f"DD blocked: {n_read} / {len(trace)} commits  "
              f"({n_read/len(trace):.0%} of post-wait-k steps)")
    print("=" * 70)
    print()
    print(f"Open the log files for the full step-by-step trace:")
    print(f"  {base_path}")
    print(f"  {dd_path}")


if __name__ == "__main__":
    main()
