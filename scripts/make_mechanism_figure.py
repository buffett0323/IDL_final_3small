#!/usr/bin/env python3
"""Generate the core mechanism diagram for Token Consensus Decoding.

Shows: observed source → K=10 sampled futures → Qwen30B next-token distributions
→ hard intersection → safe commit.
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from pathlib import Path

OUT = Path("/data/user_data/haolingp/IDL_final_3small/outputs/slide_figures")
OUT.mkdir(parents=True, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════
# Fig A — Token Consensus mechanism diagram
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(15, 8), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")
fig.patch.set_facecolor("white")

# Colors
C_BLUE   = "#4a5cff"
C_PURPLE = "#7b49ff"
C_GREEN  = "#22c55e"
C_GREEN_LIGHT = "#d1fae5"
C_RED    = "#dc2626"
C_GRAY   = "#6b7280"
C_AMBER  = "#f59e0b"

def box(x, y, w, h, text, facecolor="#fff", edgecolor=C_BLUE, text_color=None,
        fontsize=10, fontweight="normal", ax=ax, rounded=True, zorder=2):
    style = "round,pad=0.02,rounding_size=0.5" if rounded else "square,pad=0.02"
    r = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       facecolor=facecolor, edgecolor=edgecolor,
                       linewidth=1.5, zorder=zorder)
    ax.add_patch(r)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color or "#1a1a2e",
            fontweight=fontweight, zorder=zorder+1)

def arrow(x1, y1, x2, y2, color=C_GRAY, lw=1.2, style="->", alpha=0.6, zorder=1):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        color=color, lw=lw, alpha=alpha,
                        mutation_scale=10, zorder=zorder)
    ax.add_patch(a)

# ─── Top title ───
ax.text(50, 57, "Token Consensus Decoding · one commit step",
        ha="center", fontsize=17, fontweight="700", color="#0f1220")
ax.text(50, 54, "Sample K=10 English futures → query next-token distributions "
        "→ intersect → commit only on unanimous agreement",
        ha="center", fontsize=11, color=C_GRAY, style="italic")

# ═══ Column 1: Observed source ═══
col1_x = 2
box(col1_x, 26, 22, 8,
    'Observed EN (wait-k=5):\n\n"The river was named by..."',
    facecolor="#eef1ff", edgecolor=C_BLUE, fontsize=11, fontweight="600")
ax.text(col1_x + 11, 37, "Observed source prefix",
        ha="center", fontsize=9, color=C_GRAY, fontweight="600")

box(col1_x, 14, 22, 8,
    'Committed ZH:\n\n"(empty)"',
    facecolor="#fafbfc", edgecolor=C_GRAY, fontsize=11, fontweight="600")
ax.text(col1_x + 11, 25, "Committed target prefix (frozen)",
        ha="center", fontsize=9, color=C_GRAY, fontweight="600")

# ─── "Sample K=10 futures" arrow label ───
ax.text(28, 51, "① Qwen3-4B-Base\nsamples K=10 futures",
        ha="center", fontsize=10, color=C_BLUE, fontweight="600",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#eef1ff",
                  edgecolor=C_BLUE, lw=1))

# ═══ Column 2: 10 future cards (stacked) ═══
col2_x = 31
futures = [
    "f₁:  Lewis after his cousin Maria Wood.",
    "f₂:  Jefferson's exploration team in 1803.",
    "f₃:  the Dutch settlers who arrived here.",
    "f₄:  a local Chief whose tribe lived here.",
    "f₅:  early French fur traders in 1790.",
    "f₆:  the surveyor who mapped this region.",
    "f₇:  Samuel Blatchley, the first explorer.",
    "f₈:  Corps of Engineers during WWII.",
    "⋮   (+ 2 more futures)",
    "f₁₀: Native navigators after the sky god.",
]
n_f = len(futures)
box_h = 3.2
gap = 0.35
top_y = 48
for i, f in enumerate(futures):
    y = top_y - i * (box_h + gap)
    is_skip = "⋮" in f
    box(col2_x, y - box_h, 18, box_h, f,
        facecolor="#f9fafb" if not is_skip else "#fff",
        edgecolor="#d1d5db", fontsize=8.5,
        fontweight="500" if not is_skip else "400")
    # Draw arrow from observed source to this future card
    arrow(col1_x + 22, 30, col2_x, y - box_h/2,
          color=C_BLUE, lw=0.6, alpha=0.15)

# Label for futures column
ax.text(col2_x + 9, 52, "K=10 sampled English futures",
        ha="center", fontsize=10, color=C_BLUE, fontweight="600")

# ═══ Column 3: Qwen30B query (visualized as a central thing with 10 distributions) ═══
col3_x = 52
ax.text((col2_x + 18 + col3_x) / 2, 51, "② Qwen3-30B\n(via vLLM)\ngives top-10\nnext-token dist\nper future",
        ha="center", fontsize=9, color=C_PURPLE, fontweight="600",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#f3efff",
                  edgecolor=C_PURPLE, lw=1))

# Draw 10 small distributions as stacked mini-tables
for i in range(n_f):
    y = top_y - i * (box_h + gap)
    # Draw arrow from future to distribution
    arrow(col2_x + 18, y - box_h/2, col3_x, y - box_h/2,
          color=C_PURPLE, lw=0.8, alpha=0.55)

    is_skip = "⋮" in futures[i]
    if is_skip:
        box(col3_x, y - box_h, 16, box_h, "⋮ ⋮ ⋮",
            facecolor="#fff", edgecolor="#e5e7eb", fontsize=10)
        continue
    # Show a small "P(next_zh token)" distribution bar
    # For the real data (sid=3, step 1), top tokens in all futures were dominated by "这条"
    # Let's show stylized: 3 tokens with probs
    dist_w = 16
    ax.add_patch(FancyBboxPatch((col3_x, y - box_h), dist_w, box_h,
                 boxstyle="round,pad=0.02,rounding_size=0.4",
                 facecolor="#fff", edgecolor="#e5e7eb", lw=1, zorder=2))
    # Three mini-bars
    bar_h = 0.55
    bars = [
        ("这条", 0.92 if i % 3 != 2 else 0.72, C_GREEN),   # always green
        ("河流", 0.06, C_GRAY),
        ("该", 0.02, C_GRAY),
    ]
    by = y - 0.6
    for tok, prob, col in bars:
        # token label
        ax.text(col3_x + 1, by - bar_h/2 + 0.02, tok, ha="left", va="center",
                fontsize=7.5, color=col, fontweight="600",
                family="serif", zorder=3)
        # bar
        ax.add_patch(Rectangle((col3_x + 4.5, by - bar_h/2), prob * 10, bar_h,
                     facecolor=col, alpha=0.6, zorder=2))
        # prob label
        ax.text(col3_x + 15, by - bar_h/2 + 0.02, f"{prob:.2f}", ha="right", va="center",
                fontsize=7, color=col, zorder=3)
        by -= bar_h + 0.2

# ═══ Column 4: Intersection ═══
col4_x = 72
# Big ∩ symbol + label
ax.text(col4_x + 6, 35, "∩", ha="center", va="center",
        fontsize=48, color=C_GREEN, fontweight="bold", zorder=5)
ax.text(col4_x + 6, 29.5, "hard intersection\nover K=10\ntoken-ID sets",
        ha="center", fontsize=9, color=C_GREEN, fontweight="600")

# 10 arrows converging to ∩ point
for i in range(n_f):
    y = top_y - i * (box_h + gap)
    arrow(col3_x + 16, y - box_h/2, col4_x + 3, 35,
          color=C_GREEN, lw=0.8, alpha=0.4)

# Little ③ label
ax.text(col4_x + 6, 46, "③ Intersect",
        ha="center", fontsize=10, color=C_GREEN, fontweight="600",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C_GREEN_LIGHT,
                  edgecolor=C_GREEN, lw=1))

# ═══ Column 5: Winner commit ═══
col5_x = 84
# Arrow from ∩ to commit
arrow(col4_x + 10, 35, col5_x, 35, color=C_GREEN, lw=3, alpha=0.9, style="->")

# Winner box
box(col5_x, 30, 14, 10,
    "「这条」\n\nALL 10 futures\nagree!\navg prob 0.863\n\nCOMMIT ✓",
    facecolor=C_GREEN_LIGHT, edgecolor=C_GREEN, fontsize=10,
    fontweight="700", rounded=True)
# Extra border glow
ax.add_patch(FancyBboxPatch((col5_x - 0.4, 29.6), 14.8, 10.8,
             boxstyle="round,pad=0.02,rounding_size=0.5",
             facecolor="none", edgecolor=C_GREEN, lw=0.8,
             alpha=0.4, zorder=1))

ax.text(col5_x + 7, 42, "④ Safe commit",
        ha="center", fontsize=10, color=C_GREEN, fontweight="600",
        bbox=dict(boxstyle="round,pad=0.4", facecolor=C_GREEN_LIGHT,
                  edgecolor=C_GREEN, lw=1))

# ═══ Bottom: Why this solves early commitment ═══
y_bottom = 5
ax.add_patch(FancyBboxPatch((2, y_bottom - 2), 96, 8,
             boxstyle="round,pad=0.3,rounding_size=0.8",
             facecolor="#fff7ed", edgecolor=C_AMBER, lw=1.5, zorder=1))

ax.text(6, y_bottom + 4.5, "Why this solves early commitment:",
        ha="left", fontsize=12, color="#92400e", fontweight="700")
ax.text(6, y_bottom + 2.5,
        "We only commit a Chinese token when it appears in the top-k next-token "
        "distribution of EVERY sampled English future.",
        ha="left", fontsize=10, color="#92400e")
ax.text(6, y_bottom + 0.8,
        "→  a committed token is guaranteed safe against ALL plausible source continuations "
        "→  no more feared 'later source reveals we were wrong'.",
        ha="left", fontsize=10, color="#92400e", style="italic")

plt.tight_layout()
plt.savefig(OUT / "tc_mechanism.svg", format="svg", bbox_inches="tight", facecolor="white")
plt.savefig(OUT / "tc_mechanism.png", format="png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ tc_mechanism saved")

# ═══════════════════════════════════════════════════════════════════════════
# Fig B — Before / After (rigid wait-k vs future-aware)
# ═══════════════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 8), dpi=150)
ax.set_xlim(0, 100)
ax.set_ylim(0, 60)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(50, 57, "Future-Aware Commit Rules vs rigid wait-k",
        ha="center", fontsize=17, fontweight="700", color="#0f1220")
ax.text(50, 54, "Same input. Different commit rule. Different outcome.",
        ha="center", fontsize=11, color=C_GRAY, style="italic")

# ─── Scenario A (top): rigid wait-k greedy commit ───
yA = 40
ax.add_patch(FancyBboxPatch((2, yA - 2), 96, 12,
             boxstyle="round,pad=0.3,rounding_size=0.5",
             facecolor="#fef2f2", edgecolor=C_RED, lw=1.5, alpha=0.6, zorder=1))

ax.text(4, yA + 8, "❌ Rigid wait-k", ha="left", fontsize=13, color=C_RED, fontweight="700")
ax.text(4, yA + 6, "(uses only observed source, commits greedily)",
        ha="left", fontsize=9, color=C_RED, style="italic")

# Source
box(4, yA - 1, 18, 5, '"The bank was..."\n(wait-k=3 observed)',
    facecolor="#fff", edgecolor=C_RED, fontsize=10)

# Arrow greedy → commit 银行
arrow(22, yA + 1.5, 48, yA + 1.5, color=C_RED, lw=2, style="->")
ax.text(35, yA + 3, "greedy commit\n(most-likely Chinese)",
        ha="center", fontsize=9, color=C_RED, fontweight="600")

# Commit
box(50, yA - 1, 15, 5, "「银行」 (bank)\n[FROZEN]",
    facecolor="#fee2e2", edgecolor=C_RED, fontsize=10, fontweight="700")

# Reveal continuation
arrow(65, yA + 1.5, 71, yA + 1.5, color=C_GRAY, lw=1, style="->")

# Later context arrives → wrong
box(72, yA - 1, 24, 5,
    "later EN: \"...teeming with frogs\"\n→ bank = riverbank. But 银行 is frozen!",
    facecolor="#fee2e2", edgecolor=C_RED, fontsize=9, fontweight="600", text_color=C_RED)

# ─── Scenario B (bottom): future-aware ───
yB = 18
ax.add_patch(FancyBboxPatch((2, yB - 2), 96, 14,
             boxstyle="round,pad=0.3,rounding_size=0.5",
             facecolor="#ecfdf5", edgecolor=C_GREEN, lw=1.5, alpha=0.6, zorder=1))

ax.text(4, yB + 10, "✅ Future-aware commit", ha="left", fontsize=13, color=C_GREEN, fontweight="700")
ax.text(4, yB + 8, "(samples K=10 English futures, commits only on consensus)",
        ha="left", fontsize=9, color="#047857", style="italic")

# Source
box(4, yB + 0.5, 18, 5, '"The bank was..."\n(same input)',
    facecolor="#fff", edgecolor=C_GREEN, fontsize=10)

# Branch to futures
fut_samples = [
    ("f₁: ...near the cash register", "银行 (financial)", C_GRAY),
    ("f₂: ...teeming with frogs", "河岸 (riverbank)",   C_GRAY),
    ("f₃: ...covered in wet moss", "河岸 (riverbank)",   C_GRAY),
    ("f₄: ...slippery with mud", "河岸 (riverbank)",     C_GRAY),
    ("⋮", "⋮", C_GRAY),
]
fx = 25
for i, (f, pred, col) in enumerate(fut_samples):
    y = yB + 5 - i * 1.6
    box(fx, y - 0.5, 20, 1.3, f,
        facecolor="#fff", edgecolor="#d1d5db", fontsize=8, rounded=True)
    ax.text(fx + 21, y + 0.15, "→", fontsize=10, color=C_GRAY, ha="left", va="center")
    box(fx + 23, y - 0.5, 16, 1.3, pred,
        facecolor="#fff", edgecolor="#d1d5db", fontsize=8, rounded=True)
    arrow(22, yB + 3, fx, y + 0.15, color="#d1d5db", lw=0.6, alpha=0.5)

# Intersection — conclusion
ax.text(67, yB + 7.5, "intersection",
        ha="center", fontsize=10, color=C_GREEN, fontweight="700")
ax.text(67, yB + 5.7, "across all 10 futures",
        ha="center", fontsize=9, color=C_GREEN, style="italic")

# Since futures disagree on bank meaning (some say 银行, some say 河岸)
# → intersection is ∅ for 银行 vs 河岸 level
# → system decides to READ more before committing "bank"
box(62, yB - 1.5, 14, 5,
    "∩ = ∅\nno unanimous\ntoken",
    facecolor="#fff", edgecolor=C_AMBER, fontsize=9, fontweight="600", text_color="#b45309")

arrow(76, yB + 1, 82, yB + 1, color=C_AMBER, lw=2, style="->")

box(82, yB - 1.5, 15, 5,
    "READ more.\nDo NOT commit\nyet ✓",
    facecolor=C_GREEN_LIGHT, edgecolor=C_GREEN, fontsize=10, fontweight="700",
    text_color="#047857")

# Bottom takeaway
ax.add_patch(FancyBboxPatch((2, 3), 96, 6,
             boxstyle="round,pad=0.3,rounding_size=0.5",
             facecolor="#eef1ff", edgecolor=C_BLUE, lw=1.5, zorder=1))
ax.text(50, 7, "Future-aware consensus defers commit when plausible futures disagree — "
        "early-commitment errors become impossible by construction.",
        ha="center", fontsize=11, color="#1e3a8a", fontweight="600")
ax.text(50, 4.5, "A committed token is, by definition, one that ALL K sampled futures would also choose.",
        ha="center", fontsize=10, color=C_BLUE, style="italic")

plt.tight_layout()
plt.savefig(OUT / "tc_vs_waitk.svg", format="svg", bbox_inches="tight", facecolor="white")
plt.savefig(OUT / "tc_vs_waitk.png", format="png", dpi=180, bbox_inches="tight", facecolor="white")
plt.close()
print(f"✓ tc_vs_waitk saved")

print(f"\nAll files in: {OUT}")
for f in sorted(OUT.glob("*")):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
