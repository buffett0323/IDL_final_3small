# 全实验结果汇总（EN→ZH Simultaneous Translation）

一共 **~80+ runs**，分 12 组。所有数字来自 `outputs/**/scores` 和 `comet_score.txt`。

**规范**：BLEU / COMET 越高越好；AL / LAAL / AP / DAL 越低越好。
不同 slice（WMT19-1997 / wmt500 / rand100 / CoVoST-subset100）绝对 BLEU 不可横向比。

---

## 1. 遗留 EN→DE（原始 proposal，已弃用）

OPUS-MT EN-DE + WMT14。用来验证 beam refinement 增益不明显 → 决定转向 EN-ZH。

| Method | BLEU | AL |
|:--|--:|--:|
| wait-k=3, beam=1 | 14.17 | 2.50 |
| wait-k=5, beam=1 | 16.72 | 4.14 |
| wait-k=7, beam=1 | 18.43 | 6.08 |
| wait-k=9, beam=1 | 19.63 | 8.03 |
| wait-k=5, beam=8 | 16.92 | 4.16 |

**结论**：beam=8 只多 +0.2 BLEU，不值 4× 算力 → 转向 read-more + LCP + 大模型 rerank 思路。

---

## 2. 早期 EN→ZH NLLB 探索（`cmp_*`, `rand100_*`, 100 句）

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| NLLB wait-k=3 | 12.72 | 7.34 | 0.545 |
| NLLB wait-k=5 | 13.46 | 8.80 | 0.558 |
| NLLB wait-k=7 | 14.30 | 10.24 | 0.575 |
| NLLB wait-k=9 | 14.69 | 11.60 | 0.600 |
| entropy-only gate (τ=3.0) | 14.92 | 10.35 | 0.579 |
| DD full τ=0.03 | 15.66 | 10.74 | 0.608 |
| DD full τ=0.05 | 15.45 | 10.41 | 0.600 |
| DD veto τ=0.03 | 16.94 | 11.96 | 0.629 |
| DD veto τ=0.05 | 16.73 | 11.65 | 0.623 |
| Continuation baseline k=5 | 8.67 | 8.68 | 0.595 |
| Qwen continuation k=5 | 8.46 | 8.70 | 0.645 |

**发现**：DD veto > DD full > entropy-only；continuation 想法被否。

---

## 3. LM-sampled DD（LM-sDD，对照 oracle futures 的消融）

同 100 句 rand100，看 LM 采样 futures 是否能替代 oracle truncation。

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| baseline k=5 | 13.46 | 8.80 | 0.558 |
| DD oracle τ=0.03 | 15.66 | 10.74 | 0.608 |
| DD oracle τ=0.05 | 15.45 | 10.41 | 0.600 |
| **DD LM τ=0.03** | **16.64** | 13.15 | 0.641 |
| DD LM τ=0.05 | 16.28 | 12.42 | 0.638 |

**结论**：LM futures 比 oracle 还好（+1 BLEU），因为 LM 能探索更多样化的 continuation。

---

## 4. 早期 SemLCP 100 句实验（探索阶段）

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| NLLB baseline k=5 | 13.46 | 8.80 | 0.558 |
| Qwen30B direct k=3 | 21.02 | 7.25 | 0.624 |
| Qwen30B direct k=5 | 22.45 | 8.75 | 0.666 |
| Qwen SemLCP K=4 | 25.75 | 8.94 | 0.713 |
| Qwen SemLCP K=6 | 25.65 | 8.91 | 0.719 |

证实在 100 句上 SemLCP > Qwen direct +3.3 BLEU / +0.05 COMET。

---

## 5. 全量 wmt19（1997 句）— NLLB 支持线（`full/`）

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| NLLB k=3 | 13.26 | 7.03 | 0.556 |
| NLLB k=5 | 14.14 | 8.52 | 0.586 |
| NLLB k=7 | 14.86 | 9.99 | 0.610 |
| NLLB DD full τ=0.05 | 15.56 | 10.08 | 0.626 |
| NLLB DD veto τ=0.03 | 16.67 | 11.53 | 0.651 |
| NLLB LM-DD τ=0.03 | 16.59 | 12.54 | 0.664 |

⚠️ 这些是**非 prefix-anchored** 的旧版，下面 §7 有更严格的 rerun。

---

## 6. wmt500（500 句）— NLLB / Qwen 多方法（`full/`）

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| NLLB k=3 | 12.05 | 6.81 | 0.544 |
| NLLB k=5 greedy | 12.82 | 8.34 | 0.575 |
| **NLLB k=5 beam=8** | **18.31** | 8.39 | **0.689** |
| NLLB k=7 | 13.49 | 9.84 | 0.600 |
| NLLB DD full τ=0.05 | 14.95 | 11.92 | 0.651 |
| NLLB DD veto τ=0.03 | 16.30 | 13.41 | 0.667 |
| NLLB LCP K=1 | 14.53 | 8.37 | 0.659 |
| NLLB LCP K=2 | 14.17 | 8.37 | 0.648 |
| NLLB LCP K=4 | 14.47 | 8.38 | 0.642 |
| NLLB LCP K=8 | 14.43 | 8.39 | 0.640 |
| Qwen direct k=3 | 21.44 | 6.89 | 0.622 |
| Qwen direct k=5 | 21.73 | 8.40 | 0.668 |
| Qwen direct k=7 | 20.77 | 9.89 | 0.695 |
| Qwen SemLCP k=3 K=4 | 26.52 | 7.28 | 0.722 |
| Qwen SemLCP k=5 K=4 | 24.11 | 8.64 | 0.703 |
| Qwen SemLCP k=5 K=6 | 24.42 | 8.58 | 0.707 |
| Qwen SemLCP k=7 K=4 | 22.35 | 10.06 | 0.693 |

**观察**：NLLB beam=8 是个**上界（oracle-like）**；LCP K={1,2,4,8} 几乎不敏感；Qwen k=3 出人意料比 k=5 SemLCP 好（源信息少反而 Qwen 倾向"翻保守")。

---

## 7. 🔬 Prefix-anchored rerun（权威版，`rerun_20260330/`）

### 7a. NLLB 支持线 WMT19 (1997)

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| NLLB k=3 | 11.65 | 7.00 | 0.619 |
| NLLB k=5 | 12.52 | 8.46 | 0.635 |
| NLLB k=7 | 12.88 | 9.95 | 0.646 |
| NLLB k=9 | 13.78 | 11.36 | 0.656 |
| NLLB DD full τ=0.05 | 14.04 | 11.89 | 0.664 |
| **NLLB DD veto τ=0.03** | **15.75** | 13.29 | **0.679** |
| NLLB LM-DD τ=0.03 | 14.40 | 12.51 | 0.669 |

### 7b. Qwen main wmt500 (500)

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| Qwen direct k=5 | 21.73 | 8.40 | 0.668 |
| Qwen SemLCP K=4 | 24.42 | 8.70 | 0.709 |

### 7c. Qwen bridge wmt500（加了 DD gate 对比，bridging 5 和 7b）

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| Qwen direct k=5 | 21.73 | 8.40 | 0.668 |
| Qwen DD+JS k=5 K=4 τ=0.15 | 22.27 | 9.17 | 0.692 |
| Qwen SemLCP k=5 K=4 | 24.07 | 8.68 | 0.708 |

---

## 8. 🏆 主 headline — WMT19 全量 Qwen 4 方法（`wmt19_qwen/`）

| Method | BLEU | AL | LAAL | AP | DAL | COMET |
|:--|--:|--:|--:|--:|--:|--:|
| Qwen30B direct k=5 | 28.15 | 8.55 | 9.19 | 0.940 | 11.25 | 0.7741 |
| + DD + JS gate (K=4, τ=0.15) | 29.14 | 9.33 | 9.86 | 0.939 | 12.05 | 0.7912 |
| + Future literal LCP (K=4) | 31.95 | 8.79 | 9.10 | 0.911 | 12.01 | 0.8281 |
| **+ Token Consensus (K=10, top=10)** | **34.16** | **8.83** | 9.15 | 0.899 | **11.17** | **0.8477** |
| SemLCP (no-instruct prompt, 消融) | 29.62 | 8.79 | 9.02 | 0.912 | 11.99 | 0.826 |

**ΔBLEU vs direct**: DD +1.00 · LCP +3.80 · **TC +6.01**
**ΔCOMET vs direct**: DD +0.017 · LCP +0.054 · **TC +0.074**
"bad_noinst_prompt" 证明去掉 chat template 后 SemLCP 掉 2.33 BLEU → prompt 格式很关键。

---

## 9. 公平 rand100 4 方法（bug-fixed, `fair_*`）

| Method | BLEU | AL | COMET |
|:--|--:|--:|--:|
| Qwen direct k=5 | 27.44 | 8.75 | 0.779 |
| DD + JS gate K=4 | 29.42 | 9.82 | 0.790 |
| Future literal LCP K=4 | 31.51 | 8.80 | 0.827 |
| **Token Consensus K=10** | **33.84** | 9.08 | **0.850** |

和 WMT19 全量 pattern 完全一致（TC +6.40 BLEU over direct）。

---

## 10. 🎤 CoVoST cascaded ST — 语音管线 subset100（`covost_cascaded/`）

Pipeline: 英文语音 → Whisper-small ASR → 文本流 → SimulEval。

| Method | BLEU | AL | LAAL | AP | COMET |
|:--|--:|--:|--:|--:|--:|
| NLLB wait-k=5 (broken early run) | 0.04 | — | — | — | 0.710 |
| NLLB wait-k=5 greedy (fixed) | 24.63 | 5.60 | 5.67 | 0.911 | 0.710 |
| NLLB DD veto τ=0.03 | 29.39 | 7.90 | 7.91 | 0.968 | 0.746 |
| Qwen direct k=5 | 30.23 | 5.60 | 5.81 | 1.044 | 0.769 |
| Qwen DD+JS K=4 τ=0.15 | 31.21 | 6.13 | 6.29 | 1.068 | 0.786 |
| Qwen SemLCP K=4 | 34.17 | 5.78 | 5.95 | 1.046 | 0.801 |
| **Qwen Token Consensus K=10** | **34.92** | **5.66** | 5.84 | 1.060 | **0.812** |

语音管线上 TC 再次最优，**+4.69 BLEU / +0.043 COMET vs Qwen direct, AL 几乎不变**。

---

## 11. Token Consensus ablation（`tc_ablation/`, rand100）

### 11a. K 扫描（top=10 固定）

| K | BLEU | AL | COMET |
|--:|--:|--:|--:|
| 4 | 33.01 | 8.89 | 0.847 |
| 6 | 32.65 | 8.95 | 0.851 |
| 10 | 32.86 | 9.04 | 0.849 |
| 15 | 33.37 | 9.27 | 0.853 |

BLEU 随 K 几乎不变（±0.7）→ TC 对 futures 数量鲁棒。

### 11b. top-k 扫描（K=10 固定）

| top_logprobs | BLEU | AL | COMET |
|--:|--:|--:|--:|
| **5** | **33.47** | 9.35 | 0.851 |
| 10 | 32.86 | 9.04 | 0.849 |
| 20 | 32.81 | 9.10 | 0.848 |

**top=5 BLEU 最高**，候选池越大反而受低概率噪声污染；top=10 是实用 sweet spot（AL 稍低）。

### 11c. TC 早期版本（开发过程）

| Run | N | BLEU | AL | COMET |
|:--|--:|--:|--:|--:|
| token_consensus_test5 (smoke) | 5 | 33.44 | 6.61 | — |
| token_consensus_test5_v2 (smoke) | 5 | 39.94 | 6.32 | — |
| token_consensus_rand100_v2 | 100 | 33.04 | 9.05 | 0.850 |

v2 是修完 bug（prefix-force continuation、force-finish）后的版本，对应了 §9 fair_consensus_k5。

---

## 12. 其他（basline/smoke/重复）

| Run | 用途 |
|:--|:--|
| `demo_semantic_lcp/` | 5 句 smoke |
| `semlcp_baseline_k5/lmsdd_baseline_k5/` | 早期 NLLB k=5 重复跑（与 cmp_baseline_k5 一致） |
| `token_consensus_debug/`, `token_consensus_rand100/` | 开发期 bug 版本（已弃） |
| `verbose_traces/`, `cmp_qwen30b_cont_k{5,7,9}/` | trace 生成 & 无结果旧实验 |

---

## 🧭 三行总结

1. **文本 1997 句 / 文本 100 句 / 语音 100 句** 三个 slice 上 progression 都成立：
   `direct < DD/JS < SemLCP < Token Consensus`（BLEU 和 COMET 同向增长）。

2. **TC 在主 headline 上 +6.01 BLEU / +0.074 COMET vs Qwen direct**，延迟只多 +0.28 AL，是最大的贡献。

3. **Ablation 结论**：TC 对 K 鲁棒（4–15 同级），对 top-k 敏感（5 > 10 > 20），prompt 格式关键（无 chat template 掉 2.3 BLEU）；DD 路线在 NLLB 小模型和 speech cascade 上也都证实了（+3.2 / +4.8 BLEU），但延迟代价大。

---

## 文件索引

- 主 headline：[`outputs/wmt19_qwen/`](outputs/wmt19_qwen/)
- Fair rand100：[`outputs/fair_*`](outputs/fair_consensus_k5/)
- NLLB 支持线：[`outputs/rerun_20260330/nllb_support/`](outputs/rerun_20260330/nllb_support/)、[`outputs/full/nllb_*`](outputs/full/)
- Qwen main wmt500：[`outputs/rerun_20260330/qwen_main/`](outputs/rerun_20260330/qwen_main/)、[`outputs/rerun_20260330/qwen_bridge/`](outputs/rerun_20260330/qwen_bridge/)
- 语音：[`outputs/covost_cascaded/`](outputs/covost_cascaded/)
- TC ablation：[`outputs/tc_ablation/`](outputs/tc_ablation/)
- 早期 100-句探索：[`outputs/cmp_*`](outputs/), [`outputs/lmsdd_*`](outputs/), [`outputs/semlcp_*`](outputs/)
- 全 COMET 索引：[`outputs/comet_scores.md`](outputs/comet_scores.md), [`outputs/comet_scores.csv`](outputs/comet_scores.csv)
