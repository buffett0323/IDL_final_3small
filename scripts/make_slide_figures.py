#!/usr/bin/env python3
"""Generate BLEU-vs-AL scatter figures for slides (SVG, high-DPI PNG)."""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("/data/user_data/haolingp/IDL_final_3small/outputs/slide_figures")
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "font.family": "sans-serif",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "axes.edgecolor": "#374151",
    "axes.linewidth": 1.2,
})

# ─── Fig 1 · NLLB wait-k on WMT19 (1997): shows plateau ───
fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
ks = [3, 5, 7, 9]
bleus = [11.65, 12.52, 12.88, 13.78]
als = [7.00, 8.46, 9.95, 11.36]

ax.plot(als, bleus, "-o", color="#4a5cff", linewidth=2, markersize=10,
        markerfacecolor="#4a5cff", markeredgecolor="white", markeredgewidth=1.5)
for k, al, b in zip(ks, als, bleus):
    ax.annotate(f"k={k}", xy=(al, b), xytext=(6, 8), textcoords="offset points",
                fontsize=11, color="#1a1a2e", fontweight="600")

ax.set_xlabel("Average Lagging (AL)  →  latency", fontsize=12, color="#374151")
ax.set_ylabel("BLEU  ↑", fontsize=12, color="#374151")
ax.set_title("NLLB-600M wait-k on WMT19-1997 · pure wait-k can't escape plateau",
             fontsize=13, color="#0f1220", pad=14, fontweight="600")

ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(5.5, 12.5)
ax.set_ylim(10.5, 14.8)
ax.set_facecolor("#fafbfc")
fig.patch.set_facecolor("white")

# Annotate delta between k=3 and k=9
ax.annotate(
    "+2.13 BLEU\nfor +4.36 AL\n(quality bought with latency)",
    xy=(11.36, 13.78), xytext=(8.1, 11.4),
    fontsize=10, color="#dc2626", ha="center", fontweight="600",
    arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2)
)

for spine in ax.spines.values():
    spine.set_color("#d1d5db")

plt.tight_layout()
plt.savefig(OUT / "waitk_plateau.svg", format="svg", bbox_inches="tight",
            facecolor="white")
plt.savefig(OUT / "waitk_plateau.png", format="png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"✓ waitk_plateau: BLEU range {min(bleus)}-{max(bleus)}, AL range {min(als)}-{max(als)}")

# ─── Fig 2 · Headline scatter — CoVoST + main methods ───
# Using current CoVoST-100 data (will update when CoVoST-1997 finishes)
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

methods = [
    # (label, AL, BLEU, color, size, is_ours)
    ("NLLB greedy",          5.60, 24.63, "#9ca3af", 80,  False),
    ("NLLB + DD veto",       7.90, 29.39, "#9ca3af", 80,  False),
    ("Qwen direct",          5.60, 30.23, "#4a5cff", 110, False),
    ("+ DD + JS",            6.13, 31.21, "#7b49ff", 110, False),
    ("+ Future LCP",         5.78, 34.17, "#f59e0b", 120, False),
    ("+ Token Consensus ⭐",  5.66, 34.92, "#22c55e", 180, True),
]

for label, al, bleu, color, size, ours in methods:
    edge = "#047857" if ours else "white"
    lw = 2.5 if ours else 1.0
    ax.scatter(al, bleu, s=size, c=color, edgecolors=edge, linewidths=lw,
               zorder=5 if ours else 3)

# Label positions — careful to avoid overlap
for label, al, bleu, color, size, ours in methods:
    if label == "NLLB greedy":
        ax.annotate(label, xy=(al, bleu), xytext=(10, 8), textcoords="offset points",
                    fontsize=10, color="#4b5563")
    elif label == "NLLB + DD veto":
        ax.annotate(label, xy=(al, bleu), xytext=(10, -4), textcoords="offset points",
                    fontsize=10, color="#4b5563")
    elif label == "Qwen direct":
        ax.annotate(label, xy=(al, bleu), xytext=(10, -14), textcoords="offset points",
                    fontsize=11, color="#4a5cff", fontweight="600")
    elif label == "+ DD + JS":
        ax.annotate(label, xy=(al, bleu), xytext=(10, 0), textcoords="offset points",
                    fontsize=11, color="#7b49ff", fontweight="600")
    elif label == "+ Future LCP":
        ax.annotate(label, xy=(al, bleu), xytext=(10, 0), textcoords="offset points",
                    fontsize=11, color="#b45309", fontweight="600")
    elif "Token Consensus" in label:
        ax.annotate(label, xy=(al, bleu), xytext=(15, 5), textcoords="offset points",
                    fontsize=13, color="#047857", fontweight="700")

# Dashed arrow from direct → TC showing the improvement
ax.annotate("",
            xy=(5.66, 34.9), xytext=(5.60, 30.5),
            arrowprops=dict(arrowstyle="->", color="#22c55e", lw=1.8, alpha=0.6,
                            linestyle="--"))
ax.text(4.5, 32.5, "+4.69 BLEU\n+0.043 COMET\n≈ 0 AL cost",
        fontsize=10, color="#047857", ha="center", fontweight="600",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#ecfdf5",
                  edgecolor="#86efac", lw=1))

ax.set_xlabel("Average Lagging (AL)  →  latency", fontsize=13, color="#374151")
ax.set_ylabel("BLEU  ↑   quality", fontsize=13, color="#374151")
ax.set_title("CoVoST-2 EN→ZH cascaded ST · quality–latency frontier",
             fontsize=14, color="#0f1220", pad=14, fontweight="600")
ax.text(0.99, -0.14, "Pilot 100 utt · CoVoST-1997 rerun in progress",
        transform=ax.transAxes, ha="right", fontsize=9, color="#9ca3af",
        style="italic")

ax.grid(True, alpha=0.3, linestyle="--")
ax.set_xlim(3.5, 9.5)
ax.set_ylim(22, 37)
ax.set_facecolor("#fafbfc")
fig.patch.set_facecolor("white")

for spine in ax.spines.values():
    spine.set_color("#d1d5db")

# Annotate "top-left = better" guide
ax.text(0.02, 0.98, "↖ better (higher quality, lower latency)",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        color="#6b7280", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#e5e7eb", lw=0.8))

plt.tight_layout()
plt.savefig(OUT / "covost_headline.svg", format="svg", bbox_inches="tight",
            facecolor="white")
plt.savefig(OUT / "covost_headline.png", format="png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"✓ covost_headline: 6 methods, TC at top-left corner")

# ─── Fig 3 · TC ablations combined (K + top-k) ───
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

# K ablation
Ks = [4, 6, 10, 15]
K_bleus = [33.01, 32.65, 32.86, 33.37]
K_comets = [0.847, 0.851, 0.849, 0.853]

ax1b = ax1.twinx()
ax1.plot(Ks, K_bleus, "-o", color="#4a5cff", linewidth=2.5, markersize=10,
         markerfacecolor="#4a5cff", markeredgecolor="white", markeredgewidth=1.5,
         label="BLEU")
ax1b.plot(Ks, K_comets, "--s", color="#22c55e", linewidth=2, markersize=8,
          markerfacecolor="#22c55e", markeredgecolor="white", markeredgewidth=1.5,
          label="COMET")

ax1.set_xlabel("K (number of futures)", fontsize=12, color="#374151")
ax1.set_ylabel("BLEU", color="#4a5cff", fontsize=12, fontweight="600")
ax1b.set_ylabel("COMET", color="#22c55e", fontsize=12, fontweight="600")
ax1.set_title("TC ablation · K (top-k=10 fixed)\n→ robust to K",
              fontsize=12, color="#0f1220", fontweight="600", pad=8)
ax1.grid(True, alpha=0.3, linestyle="--")
ax1.set_ylim(32, 34)
ax1b.set_ylim(0.840, 0.860)
ax1.set_facecolor("#fafbfc")
ax1.tick_params(axis="y", labelcolor="#4a5cff")
ax1b.tick_params(axis="y", labelcolor="#22c55e")
ax1.set_xticks(Ks)

# top-k ablation
tops = [5, 10, 20]
T_bleus = [33.47, 32.86, 32.81]
T_comets = [0.851, 0.849, 0.848]

ax2b = ax2.twinx()
ax2.plot(tops, T_bleus, "-o", color="#4a5cff", linewidth=2.5, markersize=10,
         markerfacecolor="#4a5cff", markeredgecolor="white", markeredgewidth=1.5,
         label="BLEU")
ax2b.plot(tops, T_comets, "--s", color="#22c55e", linewidth=2, markersize=8,
          markerfacecolor="#22c55e", markeredgecolor="white", markeredgewidth=1.5,
          label="COMET")

ax2.set_xlabel("top-k per-future candidate pool", fontsize=12, color="#374151")
ax2.set_ylabel("BLEU", color="#4a5cff", fontsize=12, fontweight="600")
ax2b.set_ylabel("COMET", color="#22c55e", fontsize=12, fontweight="600")
ax2.set_title("TC ablation · top-k (K=10 fixed)\n→ larger pool pollutes intersection",
              fontsize=12, color="#0f1220", fontweight="600", pad=8)
ax2.grid(True, alpha=0.3, linestyle="--")
ax2.set_ylim(32, 34)
ax2b.set_ylim(0.840, 0.860)
ax2.set_facecolor("#fafbfc")
ax2.tick_params(axis="y", labelcolor="#4a5cff")
ax2b.tick_params(axis="y", labelcolor="#22c55e")
ax2.set_xticks(tops)

# Mark top-5 as sweet spot
ax2.annotate("highest\nBLEU",
             xy=(5, 33.47), xytext=(8, 33.7),
             fontsize=10, color="#dc2626", ha="center", fontweight="600",
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.2))

fig.patch.set_facecolor("white")
for ax in [ax1, ax2, ax1b, ax2b]:
    for spine in ax.spines.values():
        spine.set_color("#d1d5db")

plt.tight_layout()
plt.savefig(OUT / "tc_ablations.svg", format="svg", bbox_inches="tight",
            facecolor="white")
plt.savefig(OUT / "tc_ablations.png", format="png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print(f"✓ tc_ablations: K ∈ {{4,6,10,15}}, top-k ∈ {{5,10,20}}")

print(f"\nOutputs in: {OUT}")
for f in sorted(OUT.glob("*")):
    print(f"  {f.name}  ({f.stat().st_size // 1024} KB)")
