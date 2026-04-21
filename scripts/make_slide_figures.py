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

# ─── Fig 2 · Headline scatter — CoVoST-1997 main methods ───
# Full CoVoST-1997 results from outputs/covost1997/*/scores.
fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=180)

methods = [
    # label, AL, BLEU, COMET, color, marker, size, family
    ("NLLB greedy",       5.64, 22.27, 0.729, "#9ca3af", "o", 90,  "small baseline"),
    ("NLLB + DD veto",    7.95, 26.88, 0.761, "#6b7280", "o", 90,  "small baseline"),
    ("Qwen direct",       5.64, 32.18, 0.795, "#2563eb", "o", 130, "Qwen baseline"),
    ("+ DD/JS gate",      6.21, 33.27, 0.807, "#7c3aed", "o", 120, "future-aware"),
    ("+ Future LCP",      5.88, 35.05, 0.823, "#d97706", "o", 120, "future-aware"),
    ("+ Token Consensus", 5.76, 35.85, 0.832, "#059669", "*", 360, "ours"),
]

# Light guide region: lower AL is better.
ax.axvspan(5.45, 6.05, color="#ecfdf5", alpha=0.75, zorder=0)

for label, al, bleu, comet, color, marker, size, family in methods:
    edge = "#064e3b" if family == "ours" else "white"
    lw = 1.8 if family == "ours" else 1.1
    ax.scatter(al, bleu, s=size, marker=marker, c=color, edgecolors=edge,
               linewidths=lw, zorder=5 if family == "ours" else 3)

# Clear labels. Keep every label near its point, but avoid line crossings.
label_style = dict(fontsize=11, bbox=dict(boxstyle="round,pad=0.18",
                                          fc="white", ec="none", alpha=0.82))
offsets = {
    "NLLB greedy": (10, -6, "#6b7280", "left"),
    "NLLB + DD veto": (10, -4, "#4b5563", "left"),
    "Qwen direct": (10, -16, "#1d4ed8", "left"),
    "+ DD/JS gate": (12, -2, "#6d28d9", "left"),
    "+ Future LCP": (12, -6, "#b45309", "left"),
    "+ Token Consensus": (24, -1, "#047857", "left"),
}
for label, al, bleu, comet, color, marker, size, family in methods:
    dx, dy, txt_color, ha = offsets[label]
    fs = 13 if family == "ours" else 11
    fw = "800" if family == "ours" else "600"
    ax.annotate(label, xy=(al, bleu), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="center", color=txt_color, fontsize=fs,
                fontweight=fw, bbox=label_style["bbox"], zorder=7)

# Main story arrow: compare against the strong Qwen direct baseline, not NLLB.
qwen_al, qwen_bleu = 5.64, 32.18
tc_al, tc_bleu = 5.76, 35.85
ax.annotate("",
            xy=(tc_al, tc_bleu - 0.08), xytext=(qwen_al, qwen_bleu + 0.22),
            arrowprops=dict(arrowstyle="->", color="#059669", lw=2.6,
                            mutation_scale=18, shrinkA=8, shrinkB=8))
ax.text(6.55, 31.0,
        "TC vs Qwen direct:\n+3.67 BLEU\n+0.12 AL",
        fontsize=11.5, color="#065f46", ha="left", va="center", fontweight="700",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="#ecfdf5",
                  edgecolor="#6ee7b7", lw=1.2))

ax.set_xlabel("Average Lagging (AL, source words)  →  lower is better",
              fontsize=13, color="#111827", labelpad=8)
ax.set_ylabel("BLEU  →  higher is better", fontsize=13, color="#111827", labelpad=8)
ax.set_title("CoVoST-2 EN→ZH cascaded speech translation: quality vs latency",
             fontsize=15, color="#111827", pad=14, fontweight="800")
ax.text(0.985, 0.025,
        "Full CoVoST-1997 · Whisper-small ASR → streaming text agent",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9.5, color="#6b7280",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                  edgecolor="none", alpha=0.75))

ax.grid(True, alpha=0.28, linestyle="--", linewidth=0.8)
ax.set_xlim(4.45, 8.55)
ax.set_ylim(21.4, 36.9)
ax.set_xticks([5, 6, 7, 8])
ax.set_yticks([22, 26, 30, 34, 36])
ax.tick_params(axis="both", labelsize=11, colors="#374151")
ax.set_facecolor("#fbfdff")
fig.patch.set_facecolor("white")

for spine in ax.spines.values():
    spine.set_color("#cbd5e1")
    spine.set_linewidth(1.1)

ax.text(0.02, 0.98, "Best region: upper-left",
        transform=ax.transAxes, ha="left", va="top", fontsize=10,
        color="#374151", fontweight="600",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white",
                  edgecolor="#d1d5db", lw=0.9))

plt.tight_layout(pad=1.0)
plt.savefig(OUT / "covost_headline.svg", format="svg", bbox_inches="tight",
            facecolor="white")
plt.savefig(OUT / "covost_headline.png", format="png", dpi=200, bbox_inches="tight",
            facecolor="white")
plt.close()
print("✓ covost_headline: full CoVoST-1997, clear Qwen-direct → TC story")

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
