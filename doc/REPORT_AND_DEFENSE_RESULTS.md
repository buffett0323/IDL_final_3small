# 报告与答辩素材：从 DD + JS 到 future literal LCP

本文档可直接粘贴到幻灯片/论文「实验设置与结果」一节，或用作答辩口播提纲。  
数值与 `outputs/full_comparison.md`、`outputs/covost_cascaded/.../scores` 一致（截至生成时的仓库状态）。

**与课程初版 proposal 的关系**（小组 PDF vs. 本仓库实现）已写在 `FINAL_REPORT.md` **§1.1**（表格：动机对应关系、未做项、挑战与目标达成度）。答辩时若被问「为什么和 proposal 语言对/数据集不一致」，直接按 §1.1 回答即可。

---

## 一、中文版（报告正文 / 答辩叙述）

### 1. 主结果：future-aware 方法演化，文本流（SimulEval）

我们把最终 story 讲成一个**方法演化链**。第一步是 **DD + JS**：先判断“现在该不该写”，如果未来分歧很大，就先别写。第二步是 **future literal LCP**：发现单纯靠“等更多 source”会让 latency 明显升高后，进一步改成“现在到底有哪一段前缀还是安全可写的”。整个设定仍是 SimulEval 下 **源逐词到达** 的文本同时翻译，与级联语音翻译里 **ASR 输出后再翻译** 的接口完全同构。

**评测指标**：BLEU（越高越好）、AL / LAAL / AP（延迟相关，越低越好）。ΔBLEU 相对 **NLLB wait-k=5** 主基线。

**主要结论（一句话）**：这份 project 最顺的讲法不是“两个平行方法”，而是“一个方法升级过程”。**DD + JS** 先证明了 future disagreement 确实能帮助系统判断 **when to wait**，但代价是 latency 明显升高；随后 **future literal LCP** 把同样的 future-aware intuition 变成 **what is safe to write** 的规则，在固定 translator 上拿到了更好的 quality–latency tradeoff。整体叙事见 **`FINAL_REPORT.md` §4–5**。

**图表路径（仓库内）**：

- 汇总表：`outputs/full_comparison.md`
- 质量–延迟散点图：`outputs/full_comparison_plot.png`（**左右两栏**：wmt19 vs wmt500）
- BLEU 柱状图：`outputs/full_bleu_bar.png`
- 复现汇总脚本：`python scripts/compare_full.py`

### 2. 补充实验：CoVoST 子集 + Whisper ASR + SimulEval（级联 ST）

为与 **语音翻译（speech translation）** 叙事对齐，我们增加 **级联管线**：**英语语音 → 自动语音识别（ASR）→ 英文文本流 → 现有 Simul 智能体 → 中文**。

| 项目 | 说明 |
|------|------|
| **语音数据** | Hugging Face `AudioLLMs/covost2_en_zh_test`（CoVoST EN→ZH test 的 Parquet 镜像），**前 100 条** test 样本（`test[:100]`）。 |
| **ASR** | `openai/whisper-small`（Hugging Face Transformers），英语转写，逐句生成一行英文。 |
| **翻译** | 与主实验相同的 `sttr_enzh_agent.py`，NLLB-600M，`wait-k=5`，纯 wait-k（不确定性阈值 999，不触发多读）。 |
| **中文参考** | 数据集原始 `answer` 经 **SimulEval 对齐预处理**：按字/词元加空格，与 `data/enzh/wmt19_target.txt` 及 SimulEval 按字写出时的 **空格拼接方式** 一致；否则默认 sacrebleu 分词下 BLEU 会异常偏低。实现见 `scripts/format_zh_target_for_simul.py` / `prepare_covost_asr_texts.py --target-format simul`。 |
| **输出目录** | `outputs/covost_cascaded/nllb_baseline_k5_asr_subset100_fixed/`（若你本地目录名不同，以实际 `scores` 为准）。 |

**该子集上的数值（NLLB wait-k=5，ASR 源）**：

| BLEU | LAAL | AL | AP |
|-----:|-----:|---:|---:|
| 24.629 | 5.668 | 5.6 | 0.911 |

**答辩时可强调的边界**：

- 这是 **pilot（100 句）**，用于证明 **管线打通** 与 **语音条件** 下的可行性，**不与 WMT19 表内数字直接横向比大小**（域、句数、源是否为 ASR 均不同）。
- 主表建议直接讲 **`wmt500` 固定 backbone** 的两张表；CoVoST+ASR 为 **级联 ST 补充**。

### 3. 答辩口播要点（30 秒 / 90 秒）

**30 秒版**  
「这份 project 最好的讲法是一个方法升级过程。第一步是 DD+JS，它先解决『现在该不该写』，证明 future disagreement 真的有用；但它主要靠多等一点 source 来换质量，所以 latency 会变高。第二步是 future literal LCP，它改成直接问『现在哪一段前缀还是安全可写的』，因此在固定 translator 上拿到了更好的 quality–latency tradeoff。除此之外，我们还用 CoVoST 抽了 100 句语音，Whisper 转英文再接同一个 SimulEval agent，说明 speech 路线的 pipeline 也是通的。」

**90 秒版**  
「同时翻译真正难的地方不是只有『要不要等』，还有『现在到底哪一段前缀可以安全写出来』。我们先做了 DD+JS：用 future disagreement 去判断 when to wait，结果发现它的确提质量，但 latency 也上去了。于是我们进一步做 future literal LCP：不只是 veto，而是从多条 future-conditioned hypotheses 里提取一个稳定前缀来写。这个版本在固定 Qwen translator 上效果最好，而在固定 NLLB 上也能看到同样 intuition 的收益。为了对齐 speech translation，我们还做了 CoVoST + Whisper 的级联 pilot；中文参考按 SimulEval 需要做空格对齐，否则 BLEU 会假低。100 句 pilot 上 wait-k=5 的 BLEU 约 24.6，说明整条链路是通的。」

---

## 二、英文版（可直接贴进英文报告）

### Main results: EN→ZH text stream (SimulEval)

We evaluate incremental-source EN→ZH under SimulEval (same interface as **ASR-fed** cascaded ST).
The clearest final narrative is a **method progression**: **DD + JS** is the first future-aware
answer to **when to wait**, and **future literal LCP** is the stronger follow-up answer to
**what is safe to write now**. The most presentation-friendly comparisons are the fixed-backbone
`wmt500` tables: one on **NLLB**, one on **Qwen30B**.

**Evidence-chain narrative (Pareto / compute / ablations)**: **FINAL_REPORT §4–5**.

**Artifacts**: `outputs/full_comparison.md`; `outputs/full_comparison_plot.png`; `outputs/full_bleu_bar.png`; regenerate with `python scripts/compare_full.py`.

### Supplement: CoVoST subset with Whisper ASR (cascaded ST pilot)

To connect to **speech translation**, we instantiate a **cascaded pipeline**: **English speech → ASR (English text) → existing SimulEval agent → Chinese**.

- **Speech data**: Hugging Face dataset `AudioLLMs/covost2_en_zh_test`, **first 100** test utterances (`test[:100]`).
- **ASR**: `openai/whisper-small` (Transformers), English transcription, one line per utterance.
- **ST model / policy**: `agents/sttr_enzh_agent.py`, `facebook/nllb-200-distilled-600M`, `wait-k=5`, `--uncertainty-threshold 999` (pure wait-k).
- **Chinese references**: formatted with **SimulEval-compatible spacing** (token-separated characters, consistent with `data/enzh/wmt19_target.txt` and the agent’s character-wise `WriteAction` concatenation). Scripts: `scripts/format_zh_target_for_simul.py`, `scripts/prepare_covost_asr_texts.py` (`--target-format simul`).

**Pilot numbers** (100 sentences, ASR source, NLLB k=5): **BLEU 24.629**, **LAAL 5.668**, **AL 5.6**, **AP 0.911** (`outputs/covost_cascaded/nllb_baseline_k5_asr_subset100_fixed/scores`).

**Scope note**: this pilot validates **end-to-end feasibility** under **spoken input**; it is **not** directly comparable in absolute BLEU to the WMT19 table due to **domain shift**, **subset size**, and **ASR noise**.

---

## 三、大规模主表（粘贴用：`python scripts/compare_full.py` 生成）

以下为 **`outputs/full_comparison.md`** 的结构摘要（含 **N** 与 **Wall** 列）；完整版请以仓库文件为准。

**表 A — 固定 NLLB translator，`wmt500`，N=500**

| Method | BLEU | ΔBLEU* | AL | LAAL | AP |
|:-------|-----:|-------:|---:|-----:|---:|
| NLLB baseline k=5, beam=1 | 12.821 | +0.000 | 8.335 | 8.365 | 0.574 |
| NLLB baseline k=5, beam=8 | 18.311 | +5.490 | 8.393 | 8.517 | 0.711 |
| NLLB DD + JS | 14.951 | +2.130 | 11.916 | 11.931 | 0.640 |
| NLLB literal LCP K=4 | 14.472 | +1.651 | 8.384 | 8.449 | 0.638 |

\* ΔBLEU 相对表内 greedy `k=5, beam=1`。

**表 B — 固定 Qwen30B translator，`wmt500`，N=500**

| Method | BLEU | ΔBLEU** | AL | LAAL | AP |
|:-------|-----:|--------:|---:|-----:|---:|
| Qwen30B direct k=5 | 21.726 | +0.000 | 8.400 | 8.662 | 0.668 |
| Qwen30B DD + JS | 22.266 | +0.540 | 9.170 | 9.387 | 0.673 |
| Future literal LCP K=4 | 24.072 | **+2.346** | 8.678 | 8.897 | 0.658 |

\*\* ΔBLEU 相对表内 Qwen30B direct（同一 500 句）。

*BLEU 越高越好；AL/LAAL/AP 越低越好。*

---

## 四、与早期「100 句开发集」表格的说明

`FINAL_REPORT.md` 中更早的 Section 3–9 有一些 **`rand100` 小集** 或早期 rerun 的开发阶段表格。**课程/答辩主结果建议直接以上文两张 `wmt500` 固定-backbone 表为准**；若需要补充大规模 supporting evidence，再引用 `FINAL_REPORT.md §4.3` 的 **WMT19 prefix-anchored DD veto** 即可。
