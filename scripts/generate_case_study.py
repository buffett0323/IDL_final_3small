#!/usr/bin/env python3
"""
Generate a trace-based case study figure showing how each of the three
future-aware methods handles the same sentence at one commit point.

The figure shows:
  Row 1: English source prefix + sampled futures
  Row 2: DD + JS Gate decision
  Row 3: Future Literal LCP decision
  Row 4: Token Consensus Decoding decision

Usage:
    python scripts/generate_case_study.py                    # default SID=5
    python scripts/generate_case_study.py --sentence-id 3    # pick another
    python scripts/generate_case_study.py --list              # list options
"""
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.font_manager as fm

# CJK font setup: DroidSansFallbackFull only has CJK glyphs (no ASCII!),
# so we add it to the fallback chain after DejaVu Sans which covers ASCII.
_CJK_FONT_PATH = "/usr/share/fonts/google-droid-sans-fonts/DroidSansFallbackFull.ttf"
_DROID_REGULAR = "/usr/share/fonts/google-droid-sans-fonts/DroidSans.ttf"
for _p in [_CJK_FONT_PATH, _DROID_REGULAR]:
    if Path(_p).exists():
        fm.fontManager.addfont(_p)
# Droid Sans has ASCII; Droid Sans Fallback has CJK.
# matplotlib walks the list and picks the first font that has each glyph.
plt.rcParams["font.sans-serif"] = [
    "Droid Sans", "Droid Sans Fallback", "DejaVu Sans",
]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
# Delete stale font cache so the new fonts are discovered
_cache = Path(matplotlib.get_cachedir())
for _f in _cache.glob("fontlist-*.json"):
    _f.unlink(missing_ok=True)

# FontProperties for the two font files
_FP_CJK = fm.FontProperties(fname=_CJK_FONT_PATH) if Path(_CJK_FONT_PATH).exists() else None
_FP_ASCII = fm.FontProperties(fname=_DROID_REGULAR) if Path(_DROID_REGULAR).exists() else None

import re as _re
_CJK_RE = _re.compile(r'[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef\u3400-\u4dbf]')


def _has_cjk(s: str) -> bool:
    return bool(_CJK_RE.search(s))


def _txt(ax, x, y, s, *, fontsize=11, color="#333", **kw):
    """ax.text() — picks CJK or ASCII font based on text content.

    matplotlib cannot do per-glyph fallback, so we choose the font
    that covers the majority of glyphs in the string.  For mixed
    text, CJK font wins (it has basic ASCII punctuation / digits).
    """
    kw.setdefault("transform", ax.transAxes)
    kw.setdefault("va", "top")
    fp = _FP_CJK if (_has_cjk(s) and _FP_CJK) else _FP_ASCII
    if fp is not None:
        kw["fontproperties"] = fp
    return ax.text(x, y, s, fontsize=fontsize, color=color, **kw)


REPO = Path(__file__).resolve().parent.parent

TRACE_PATHS = {
    "dd_js":     REPO / "outputs/fair_js_gate_k5_f4/lcp_trace.jsonl",
    "semlcp":    REPO / "outputs/fair_semlcp_k5_f4/lcp_trace.jsonl",
    "consensus": REPO / "outputs/fair_consensus_k5/token_consensus_trace.jsonl",
    "direct":    REPO / "outputs/fair_direct_k5/lcp_trace.jsonl",
}

# Color scheme
C_SRC     = "#2196F3"   # blue  — source text
C_FUTURE  = "#90CAF9"   # light blue — futures
C_COMMIT  = "#4CAF50"   # green — committed text
C_BLOCK   = "#F44336"   # red   — blocked / READ
C_PARTIAL = "#FF9800"   # orange — partial commit
C_BG      = "#FAFAFA"
C_HEADER  = "#37474F"


def load_traces(method: str) -> dict[int, list[dict]]:
    """Load all traces for a method, grouped by sentence_id."""
    traces: dict[int, list[dict]] = {}
    path = TRACE_PATHS[method]
    if not path.exists():
        return traces
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            sid = d.get("sentence_id", -1)
            traces.setdefault(sid, []).append(d)
    return traces


def find_matching_step(traces: list[dict], src_prefix: str) -> dict | None:
    """Find the trace step closest to the given source prefix."""
    best = None
    for t in traces:
        src = t.get("src_text", "")
        if src == src_prefix:
            return t
        if src_prefix.startswith(src) or src.startswith(src_prefix):
            if best is None or abs(len(src) - len(src_prefix)) < abs(len(best.get("src_text", "")) - len(src_prefix)):
                best = t
    return best or (traces[0] if traces else None)


def wrap(text: str, width: int = 55) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def draw_case_study(sid: int, step_idx: int = 0, out_path: Path | None = None):
    """Draw a 4-panel case study figure."""

    all_traces = {m: load_traces(m) for m in ["dd_js", "semlcp", "consensus"]}

    # Get the target step
    dd_steps = all_traces["dd_js"].get(sid, [])
    lcp_steps = all_traces["semlcp"].get(sid, [])
    tc_steps = all_traces["consensus"].get(sid, [])

    if not dd_steps or not lcp_steps or not tc_steps:
        print(f"[ERROR] SID {sid}: missing traces (DD={len(dd_steps)}, LCP={len(lcp_steps)}, TC={len(tc_steps)})")
        return

    # Use the step_idx-th commit step for each method
    dd = dd_steps[min(step_idx, len(dd_steps) - 1)]
    lcp = lcp_steps[min(step_idx, len(lcp_steps) - 1)]

    # For TC, find the step with similar source prefix
    src_text = dd.get("src_text", lcp.get("src_text", ""))
    tc = find_matching_step(tc_steps, src_text) or tc_steps[0]

    # ── Figure layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 12), facecolor="white")
    gs = GridSpec(4, 1, height_ratios=[1.2, 1, 1, 1.2], hspace=0.35)

    # ── Panel 0: Source + Futures ────────────────────────────────────
    ax0 = fig.add_subplot(gs[0])
    ax0.set_xlim(0, 10)
    ax0.set_ylim(0, 10)
    ax0.axis("off")

    _txt(ax0, 0.5, 0.95, f"Case Study: Sentence #{sid}",
         fontsize=16, fontweight="bold", color=C_HEADER, ha="center")

    _txt(ax0, 0.05, 0.85, "Source prefix:", fontsize=11, fontweight="bold", color=C_HEADER)
    _txt(ax0, 0.05, 0.75, f'"{src_text}"',
         fontsize=12, color=C_SRC, style="italic",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", alpha=0.8))

    futures = dd.get("futures", lcp.get("futures", tc.get("futures", [])))[:4]
    _txt(ax0, 0.05, 0.58, f"Sampled futures ({len(futures)} shown):",
         fontsize=10, fontweight="bold", color=C_HEADER)

    for i, fut in enumerate(futures):
        if fut.startswith(src_text):
            cont = fut[len(src_text):].strip()
            display = f'{src_text} + "{cont[:60]}..."' if len(cont) > 60 else f'{src_text} + "{cont}"'
        else:
            display = f'"{fut[:70]}..."' if len(fut) > 70 else f'"{fut}"'
        _txt(ax0, 0.08, 0.48 - i * 0.11, f"F{i+1}: {display}",
             fontsize=8.5, color="#546E7A")

    # ── Panel 1: DD + JS Gate ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1])
    ax1.axis("off")

    js_score = dd.get("gate_js_score", 0)
    gate_tau = dd.get("gate_tau", 0.15)
    gate_decision = dd.get("gate_decision", "?")
    dd_delta = dd.get("delta", "")

    is_read = gate_decision == "READ"
    decision_color = C_BLOCK if is_read else C_COMMIT
    decision_icon = "WAIT" if is_read else "COMMIT"

    ax1.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.88, boxstyle="round,pad=0.02",
        facecolor="#FFF3E0" if is_read else "#E8F5E9", edgecolor=decision_color,
        linewidth=2, transform=ax1.transAxes))

    _txt(ax1, 0.05, 0.85, "Method 1: DD + JS Gate  (when to wait)",
         fontsize=12, fontweight="bold", color=C_HEADER)
    _txt(ax1, 0.05, 0.6, f"JS divergence = {js_score:.3f}   (threshold \u03c4 = {gate_tau})")
    _txt(ax1, 0.05, 0.35, f"Decision: {decision_icon}",
         fontsize=13, fontweight="bold", color=decision_color)

    if dd_delta:
        _txt(ax1, 0.4, 0.35, f'Output: "{dd_delta}"', fontsize=11, color=C_COMMIT)
    elif is_read:
        _txt(ax1, 0.4, 0.35, "\u2192 Read more source (no output)",
             fontsize=11, color=C_BLOCK, style="italic")

    # ── Panel 2: Future Literal LCP ──────────────────────────────────
    ax2 = fig.add_subplot(gs[2])
    ax2.axis("off")

    lcp_delta = lcp.get("delta", "")
    lcp_candidates = lcp.get("candidates", [])[:4]
    has_lcp = bool(lcp_delta)
    lcp_color = C_COMMIT if has_lcp else C_BLOCK

    ax2.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.88, boxstyle="round,pad=0.02",
        facecolor="#E8F5E9" if has_lcp else "#FFEBEE",
        edgecolor=lcp_color, linewidth=2, transform=ax2.transAxes))

    _txt(ax2, 0.05, 0.85, "Method 2: Future Literal LCP  (what chars are safe)",
         fontsize=12, fontweight="bold", color=C_HEADER)

    for i, cand in enumerate(lcp_candidates[:3]):
        _txt(ax2, 0.05, 0.65 - i * 0.13,
             f'T{i+1}: "{cand[:40]}{"..." if len(cand) > 40 else ""}"',
             fontsize=9, color="#546E7A")

    if has_lcp:
        _txt(ax2, 0.05, 0.2, f'LCP commit: "{lcp_delta}"',
             fontsize=12, fontweight="bold", color=C_COMMIT)
    else:
        _txt(ax2, 0.05, 0.2, "LCP = empty (candidates diverge at first character)",
             fontsize=11, fontweight="bold", color=C_BLOCK)

    # ── Panel 3: Token Consensus ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[3])
    ax3.axis("off")

    tc_delta = tc.get("delta", "")
    tc_steps_data = tc.get("steps", [])
    has_tc = bool(tc_delta)
    tc_color = C_COMMIT if has_tc else C_BLOCK

    ax3.add_patch(mpatches.FancyBboxPatch(
        (0.02, 0.05), 0.96, 0.88, boxstyle="round,pad=0.02",
        facecolor="#E8F5E9" if has_tc else "#FFEBEE",
        edgecolor=tc_color, linewidth=2, transform=ax3.transAxes))

    _txt(ax3, 0.05, 0.88, "Method 3: Token Consensus  (what tokens are safe, robustly)",
         fontsize=12, fontweight="bold", color=C_HEADER)

    step_strs = []
    for s in tc_steps_data[:6]:
        tok_text = s.get("selected_text", "?")
        avg_prob = s.get("selected_avg_prob", 0)
        decision = s.get("decision", "?")
        if decision == "APPEND":
            step_strs.append(f'"{tok_text}" (p={avg_prob:.2f})')
        else:
            step_strs.append(f"[{decision}]")

    if step_strs:
        chain = " \u2192 ".join(step_strs)
        _txt(ax3, 0.05, 0.68, f"Consensus chain: {chain}",
             fontsize=9.5, wrap=True)

    if has_tc:
        _txt(ax3, 0.05, 0.35, f'Token Consensus commit: "{tc_delta}"',
             fontsize=12, fontweight="bold", color=C_COMMIT)
        stop = tc.get("stop_reason", "?")
        _txt(ax3, 0.05, 0.15,
             f"Stopped: {stop} (committed {len(tc_steps_data)} tokens before disagreement)",
             fontsize=9.5, color="#777", style="italic")
    else:
        _txt(ax3, 0.05, 0.35, "No consensus found",
             fontsize=12, fontweight="bold", color=C_BLOCK)

    # ── Save ─────────────────────────────────────────────────────────
    if out_path is None:
        out_path = REPO / f"outputs/case_study_sid{sid}.pdf"

    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"[DONE] Saved: {out_path}")
    print(f"[DONE] Saved: {out_path.with_suffix('.png')}")


def list_candidates():
    """Show all sentence IDs with traces in all 3 methods."""
    all_traces = {m: load_traces(m) for m in ["dd_js", "semlcp", "consensus"]}
    common = (set(all_traces["dd_js"].keys()) &
              set(all_traces["semlcp"].keys()) &
              set(all_traces["consensus"].keys()))

    # Also load source lines for display
    src_lines = {}
    rand100_src = REPO / "data/enzh/rand100_source.txt"
    if rand100_src.exists():
        with open(rand100_src) as f:
            for i, line in enumerate(f):
                src_lines[i] = line.strip()

    print(f"\n{'SID':>4}  {'DD steps':>8}  {'LCP steps':>9}  {'TC steps':>8}  Source prefix (first step)")
    print("-" * 100)
    for sid in sorted(common):
        dd_steps = all_traces["dd_js"][sid]
        lcp_steps = all_traces["semlcp"][sid]
        tc_steps = all_traces["consensus"][sid]

        dd0 = dd_steps[0]
        src = dd0.get("src_text", src_lines.get(sid, "?"))[:60]
        dd_delta = dd0.get("delta", "")
        lcp_delta = lcp_steps[0].get("delta", "") if lcp_steps else ""
        tc_delta = tc_steps[0].get("delta", "") if tc_steps else ""
        js = dd0.get("gate_js_score", -1)

        # Mark interesting cases
        marker = ""
        if dd0.get("gate_decision") == "READ" and not lcp_delta and tc_delta:
            marker = " ★★★"  # DD=READ, LCP=empty, TC commits → best case
        elif dd0.get("gate_decision") == "READ" and lcp_delta:
            marker = " ★★"
        elif not lcp_delta and tc_delta:
            marker = " ★"

        print(f"{sid:>4}  {len(dd_steps):>8}  {len(lcp_steps):>9}  {len(tc_steps):>8}  "
              f"{src}{marker}")

    print(f"\n★★★ = DD:READ + LCP:empty + TC:commits (best showcase)")
    print(f"★★  = DD:READ + LCP:commits")
    print(f"★   = LCP:empty + TC:commits\n")


def main():
    parser = argparse.ArgumentParser(description="Generate case study figure")
    parser.add_argument("--sentence-id", type=int, default=5,
                        help="Sentence ID to visualize (default: 5)")
    parser.add_argument("--step", type=int, default=0,
                        help="Which commit step to show (0=first)")
    parser.add_argument("--list", action="store_true",
                        help="List candidate sentences with trace coverage")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output file path (default: outputs/case_study_sid{N}.pdf)")
    parser.add_argument("--all-interesting", action="store_true",
                        help="Generate figures for all ★★★ sentences")
    args = parser.parse_args()

    if args.list:
        list_candidates()
        return

    if args.all_interesting:
        all_traces = {m: load_traces(m) for m in ["dd_js", "semlcp", "consensus"]}
        common = (set(all_traces["dd_js"].keys()) &
                  set(all_traces["semlcp"].keys()) &
                  set(all_traces["consensus"].keys()))

        for sid in sorted(common):
            dd0 = all_traces["dd_js"][sid][0]
            lcp0 = all_traces["semlcp"][sid][0]
            tc0 = all_traces["consensus"][sid][0]
            if (dd0.get("gate_decision") == "READ" and
                    not lcp0.get("delta", "") and
                    tc0.get("delta", "")):
                print(f"\n=== Generating case study for SID {sid} ===")
                draw_case_study(sid, step_idx=0)
        return

    draw_case_study(args.sentence_id, step_idx=args.step, out_path=args.output)


if __name__ == "__main__":
    main()
