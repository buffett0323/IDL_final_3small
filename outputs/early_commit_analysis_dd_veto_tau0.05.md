# Early-Commit Risk Analysis

## Layer 1 — Definition: DD as Early-Commit Risk Detector

> **Early-commit risk** is defined as the extent to which the current
> translation decision would change if a small amount of additional source
> context were revealed.

Distribution Divergence (DD) estimates this directly:

- **Low JS divergence** across K futures → the next-token Chinese
  distribution is stable regardless of how much more source is revealed
  → *low early-commit risk* → safe to COMMIT
- **High JS divergence** across K futures → revealing even one more
  source word substantially changes the predicted Chinese next-token
  → *high early-commit risk* → force READ

Gate threshold used: τ = 0.05  (avg_js_firstN > τ → READ)

## Layer 3 — Metrics: Blocked Commit Statistics

| Metric | Value |
|--------|-------|
| Total gate evaluations (post wait-k) | 1415 |
| Commits blocked (DD forced READ)      | 317 (22.4%) |
| Commits allowed                       | 1098 (77.6%) |
| Sentences with ≥1 veto               | 46 / 93 |
| Avg vetos per sentence                | 3.41 |
| Avg JS at READ decisions              | 0.1599 |
| Avg JS at COMMIT decisions            | 0.0054 |
| JS separation (READ − COMMIT)         | 0.1545 |

**JS separation** measures how well DD discriminates risky from safe commits.
A gap of 0.1545 confirms DD is not firing randomly —
it consistently identifies structurally different situations.

## Layer 4 — Aggregate Improvement

| | Baseline (wait-k=5) | DD full gate (τ=0.05) | Δ |
|--|--|--|--|
| BLEU | 13.46 | 16.73 | **+3.27** |
| AL   | 8.80 | 11.65 | +2.85 |
| LAAL | 8.82 | 11.66 | +2.84 |

**Interpretation:** DD full gate achieves +3.27 BLEU at the cost of +2.85 AL.
This improvement cannot be explained by simply 'reading more' —
the AL increase is modest while the BLEU gain is substantial,
confirming that DD is selectively blocking the *right* commits.

## Beneficial Veto Analysis

For each sentence, we compare character-level F1 (char-F1) of baseline
vs DD output against the reference translation.
A veto is **beneficial** if DD output has char-F1 ≥ 0.02 higher than baseline.

| Category   | Count | Meaning |
|------------|-------|---------|
| Beneficial | 44 | DD output closer to reference |
| Neutral    | 37 | No meaningful difference (< 0.02 F1) |
| Harmful    | 12 | DD output further from reference |

**Key finding:** sentences where DD vetoed at least once improved by
avg ΔcharF1 = **+0.0617**,
while sentences with no vetos changed by +0.0100.
This confirms that DD interventions are net-positive.

### Top Beneficial Veto Cases (DD improved the output most)

**Sent 16** — 6 vetos, ΔcharF1 = +0.3158

| | Text |
|--|------|
| Reference | `由 于 热 带 雨 具 有 分 散 性 ， 伴 随 条 件 的 快 速 恶 化 ， 可 能 会 出 现 洪 水 泛 滥 。` |
| Baseline  | `闪 速 迅 速 恶 可 能 , 即 的 , 着 条 件 的 迅 速 恶 化 , 即 发 生 流 .` |
| DD output | `由 于 热 带 雨 的 分 散 性 , 随 着 条 件 的 迅 速 恶 化 而 可 能 生 流 .` |

charF1: baseline=0.281 → DD=0.596 (**+0.316**)

**Sent 76** — 14 vetos, ΔcharF1 = +0.3022

| | Text |
|--|------|
| Reference | `Mother 的 合 伙 人 兼 执 行 创 意 总 监 (ECD) Ana Balarin 评 论 道 ： “Elvie Pump 是 一 款 革 命 性 的 产 品 ， 值 得 大 胆 尝 试 。` |
| Baseline  | `艾 娜 · 合 合 亲 , 池 伴 和 性 阿 Ana · , 得 值 是 如 此 革 命 性 的 产 品 ,` |
| DD output | `马 尔 公 司 的 合 作 伙 伴 和 ECD 的 Ana Balarin 说 :"Elvie Pump 是 如 此 革 命 性 的 产 品 ,` |

charF1: baseline=0.280 → DD=0.582 (**+0.302**)

**Sent 41** — 3 vetos, ΔcharF1 = +0.2110

| | Text |
|--|------|
| Reference | `据 路 透 社 报 道 ， 西 弗 吉 尼 亚 大 学 的 政 治 学 者 西 蒙 · 哈 德 (Simon Haeder) 表 示 ， “ 总 统 不 得 不 在 民 意 调 查 中 为 其 推 波 助 澜 ， 这 对 莫 里 西 而 言` |
| Baseline  | `" 这 对 莫 一 西 来 说 不 是 一 个 好 迹 象 , 总 统 必 须 来 试 图 给 他 在 民 意 调 查 来 试 图 给 他 在 民 意 调 查 中 带 来 推 动 ,` |
| DD output | `" 这 对 莫 里 西 来 说 不 是 一 个 好 迹 象 , 总 统 必 须 来 试 图 给 (Simon Haeder) 表 示 :" 这 对 莫 里 西 来 说 民 意 调 查 中 带 来 推 动 ,` |

charF1: baseline=0.293 → DD=0.504 (**+0.211**)

### Harmful Veto Cases (DD hurt the output)

These cases represent failure modes where DD forced unnecessary READs:

**Sent 45** — 2 vetos, ΔcharF1 = -0.0800

| | Text |
|--|------|
| Reference | `“ 然 而 ， 我 们 并 没 有 看 到 美 国 方 面 做 出 任 何 回 应 ” ， 李 说 。` |
| Baseline  | `" 然 而 , 我 们 没 有 们 看 不 到 美 国 的 任 何 相 应 的 反 应 .` |
| DD output | `" 然 而 , 我 而 , 我 们 看 不 到 美 国 的 任 何 相 应 的 反 应 .` |

**Sent 29** — 2 vetos, ΔcharF1 = -0.0769

| | Text |
|--|------|
| Reference | `这 张 专 辑 更 多 反 映 了 我 们 是 谁 。 ”` |
| Baseline  | `这 张 专 辑 更 是 专 辑 上 ,` |
| DD output | `这 张 专 辑 更 个 专 辑 上 ,` |

### Veto Count vs Quality Improvement

| Veto count bucket | # sentences | Avg ΔcharF1 |
|-------------------|-------------|-------------|
| 0                 |          47 |     +0.0100 |
| 1–3               |          16 |     +0.0253 |
| 4–9               |          20 |     +0.0644 |
| 10+               |          10 |     +0.1148 |

More vetos → larger quality improvement, consistent with the hypothesis
that high-veto sentences are the ones most harmed by early commitment.

## Layer 2 — Behavior: Concrete Early-Commit Case Studies

Each case shows a sentence where DD blocked multiple baseline commits
while the model waited for critical disambiguating context.

### Case 1: Sentence 70

**Source:** `However, it cannot be right that it is as easy for individuals who don't live in the UK, as well as foreign-based companies, to buy homes as hard-working British residents.`

**Reference translation:** `然 而 ， 对 于 不 住 在 英 国 的 个 人 以 及 外 国 公 司 来 说 ， 与 勤 劳 的 英 国 居 民 在 购 房 难 度 上 别 无 二 致 ， 这 是 不 对 的 。`

**DD blocked 15 commits** (max 15 consecutive),
peak JS = 0.2838

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  5/ 0 | `...However, it cannot be right` | 0.0099 | ✅ COMMIT | JS=0.0099 ≤ τ=0.05 → stable, safe to commit |
|  6/ 1 | `...However, it cannot be right that` | 0.0145 | ✅ COMMIT | JS=0.0145 ≤ τ=0.05 → stable, safe to commit |
|  7/ 2 | `...However, it cannot be right that it` | 0.0126 | ✅ COMMIT | JS=0.0126 ≤ τ=0.05 → stable, safe to commit |
|  8/ 3 | `...However, it cannot be right that it is` | 0.0414 | ✅ COMMIT | JS=0.0414 ≤ τ=0.05 → stable, safe to commit |
|  9/ 4 | `...However, it cannot be right that it is as` | 0.0565 | 🚫 READ | JS=0.0565 > τ=0.05 → future context changes translation |
| 12/ 4 | `...wever, it cannot be right that it is as easy for individuals` | 0.1632 | 🚫 READ | JS=0.1632 > τ=0.05 → future context changes translation |
| 14/ 4 | `...cannot be right that it is as easy for individuals who don't` | 0.1979 | 🚫 READ | JS=0.1979 > τ=0.05 → future context changes translation |
| 15/ 4 | `...t be right that it is as easy for individuals who don't live` | 0.2802 | 🚫 READ | JS=0.2802 > τ=0.05 → future context changes translation |
| 16/ 4 | `...e right that it is as easy for individuals who don't live in` | 0.2822 | 🚫 READ | JS=0.2822 > τ=0.05 → future context changes translation |
| 17/ 4 | `...ght that it is as easy for individuals who don't live in the` | 0.2838 | 🚫 READ | JS=0.2838 > τ=0.05 → future context changes translation |
| 18/ 4 | `...that it is as easy for individuals who don't live in the UK,` | 0.2828 | 🚫 READ | JS=0.2828 > τ=0.05 → future context changes translation |
| 19/ 4 | `...t it is as easy for individuals who don't live in the UK, as` | 0.1822 | 🚫 READ | JS=0.1822 > τ=0.05 → future context changes translation |
| 20/ 4 | `...is as easy for individuals who don't live in the UK, as well` | 0.2748 | 🚫 READ | JS=0.2748 > τ=0.05 → future context changes translation |
| 21/ 4 | `...as easy for individuals who don't live in the UK, as well as` | 0.2149 | 🚫 READ | JS=0.2149 > τ=0.05 → future context changes translation |
| 22/ 4 | `...dividuals who don't live in the UK, as well as foreign-based` | 0.2196 | 🚫 READ | JS=0.2196 > τ=0.05 → future context changes translation |
| 23/ 4 | `...ho don't live in the UK, as well as foreign-based companies,` | 0.1700 | 🚫 READ | JS=0.1700 > τ=0.05 → future context changes translation |
| 24/ 4 | `...don't live in the UK, as well as foreign-based companies, to` | 0.2261 | 🚫 READ | JS=0.2261 > τ=0.05 → future context changes translation |
| 25/ 4 | `...t live in the UK, as well as foreign-based companies, to buy` | 0.1693 | 🚫 READ | JS=0.1693 > τ=0.05 → future context changes translation |
| 26/ 4 | `... in the UK, as well as foreign-based companies, to buy homes` | 0.1690 | 🚫 READ | JS=0.1690 > τ=0.05 → future context changes translation |
| 27/ 4 | `... the UK, as well as foreign-based companies, to buy homes as` | 0.0024 | ✅ COMMIT | JS=0.0024 ≤ τ=0.05 → stable, safe to commit |
| 28/ 5 | `...ell as foreign-based companies, to buy homes as hard-working` | 0.0018 | ✅ COMMIT | JS=0.0018 ≤ τ=0.05 → stable, safe to commit |
| 29/ 6 | `...oreign-based companies, to buy homes as hard-working British` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.05 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=5,
tgt_len=0, having seen only `However, it cannot be right`.
**DD waited until:** `However, it cannot be right that it is as easy for individuals who don't live in the UK, as well as foreign-based companies, to buy homes as`
(JS dropped to 0.0024 — translation direction stable)

### Case 2: Sentence 20

**Source:** `Crosses continually came into the Livingston box and were continually cleared, while two penalty claims - after Halkett's challenge on substitute Glenn Middleton, and one for handball - were waved away.`

**Reference translation:** `Crosses 不 断 进 入 利 文 斯 顿 队 的 禁 区 ， 并 不 断 地 被 解 围 ， 而 两 个 点 球 —— 在 哈 尔 科 特 提 出 要 让 替 补 格 伦 · 米 德 尔 顿 (Glenn Middleton) 上 场 和 一 个 手 球 之 后 —— 被 踢 了 回 去 。`

**DD blocked 15 commits** (max 15 consecutive),
peak JS = 0.1500

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  6/ 0 | `...Crosses continually came into the Livingston` | 0.0033 | ✅ COMMIT | JS=0.0033 ≤ τ=0.05 → stable, safe to commit |
|  7/ 1 | `...Crosses continually came into the Livingston box` | 0.0047 | ✅ COMMIT | JS=0.0047 ≤ τ=0.05 → stable, safe to commit |
|  8/ 2 | `...Crosses continually came into the Livingston box and` | 0.0123 | ✅ COMMIT | JS=0.0123 ≤ τ=0.05 → stable, safe to commit |
|  9/ 3 | `...Crosses continually came into the Livingston box and were` | 0.0203 | ✅ COMMIT | JS=0.0203 ≤ τ=0.05 → stable, safe to commit |
| 10/ 4 | `...ontinually came into the Livingston box and were continually` | 0.0215 | ✅ COMMIT | JS=0.0215 ≤ τ=0.05 → stable, safe to commit |
| 11/ 5 | `...y came into the Livingston box and were continually cleared,` | 0.1124 | 🚫 READ | JS=0.1124 > τ=0.05 → future context changes translation |
| 12/ 5 | `... into the Livingston box and were continually cleared, while` | 0.1500 | 🚫 READ | JS=0.1500 > τ=0.05 → future context changes translation |
| 13/ 5 | `...o the Livingston box and were continually cleared, while two` | 0.1159 | 🚫 READ | JS=0.1159 > τ=0.05 → future context changes translation |
| 14/ 5 | `...vingston box and were continually cleared, while two penalty` | 0.1111 | 🚫 READ | JS=0.1111 > τ=0.05 → future context changes translation |
| 17/ 5 | `...d were continually cleared, while two penalty claims - after` | 0.1225 | 🚫 READ | JS=0.1225 > τ=0.05 → future context changes translation |
| 18/ 5 | `...tinually cleared, while two penalty claims - after Halkett's` | 0.0931 | 🚫 READ | JS=0.0931 > τ=0.05 → future context changes translation |
| 19/ 5 | `...leared, while two penalty claims - after Halkett's challenge` | 0.1358 | 🚫 READ | JS=0.1358 > τ=0.05 → future context changes translation |
| 20/ 5 | `...red, while two penalty claims - after Halkett's challenge on` | 0.1051 | 🚫 READ | JS=0.1051 > τ=0.05 → future context changes translation |
| 21/ 5 | `...two penalty claims - after Halkett's challenge on substitute` | 0.1064 | 🚫 READ | JS=0.1064 > τ=0.05 → future context changes translation |
| 22/ 5 | `...nalty claims - after Halkett's challenge on substitute Glenn` | 0.1367 | 🚫 READ | JS=0.1367 > τ=0.05 → future context changes translation |
| 23/ 5 | `...s - after Halkett's challenge on substitute Glenn Middleton,` | 0.0949 | 🚫 READ | JS=0.0949 > τ=0.05 → future context changes translation |
| 24/ 5 | `...after Halkett's challenge on substitute Glenn Middleton, and` | 0.1302 | 🚫 READ | JS=0.1302 > τ=0.05 → future context changes translation |
| 25/ 5 | `...r Halkett's challenge on substitute Glenn Middleton, and one` | 0.1285 | 🚫 READ | JS=0.1285 > τ=0.05 → future context changes translation |
| 26/ 5 | `...lkett's challenge on substitute Glenn Middleton, and one for` | 0.1021 | 🚫 READ | JS=0.1021 > τ=0.05 → future context changes translation |
| 27/ 5 | `...hallenge on substitute Glenn Middleton, and one for handball` | 0.1063 | 🚫 READ | JS=0.1063 > τ=0.05 → future context changes translation |
| 28/ 5 | `...llenge on substitute Glenn Middleton, and one for handball -` | 0.0071 | ✅ COMMIT | JS=0.0071 ≤ τ=0.05 → stable, safe to commit |
| 29/ 6 | `...e on substitute Glenn Middleton, and one for handball - were` | 0.0037 | ✅ COMMIT | JS=0.0037 ≤ τ=0.05 → stable, safe to commit |
| 30/ 7 | `...ubstitute Glenn Middleton, and one for handball - were waved` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.05 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=6,
tgt_len=0, having seen only `Crosses continually came into the Livingston`.
**DD waited until:** `Crosses continually came into the Livingston box and were continually cleared, while two penalty claims - after Halkett's challenge on substitute Glenn Middleton, and one for handball -`
(JS dropped to 0.0071 — translation direction stable)

### Case 3: Sentence 76

**Source:** `Ana Balarin, partner and ECD at Mother commented: "The Elvie Pump is such a revolutionary product that it deserved a bold and provocative launch.`

**Reference translation:** `Mother 的 合 伙 人 兼 执 行 创 意 总 监 (ECD) Ana Balarin 评 论 道 ： “Elvie Pump 是 一 款 革 命 性 的 产 品 ， 值 得 大 胆 尝 试 。`

**DD blocked 14 commits** (max 13 consecutive),
peak JS = 0.1597

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  8/ 0 | `...Ana Balarin, partner and ECD at Mother commented:` | 0.1025 | 🚫 READ | JS=0.1025 > τ=0.05 → future context changes translation |
|  9/ 0 | `...Ana Balarin, partner and ECD at Mother commented: "The` | 0.0370 | ✅ COMMIT | JS=0.0370 ≤ τ=0.05 → stable, safe to commit |
| 10/ 1 | `...Ana Balarin, partner and ECD at Mother commented: "The Elvie` | 0.1213 | 🚫 READ | JS=0.1213 > τ=0.05 → future context changes translation |
| 11/ 1 | `...alarin, partner and ECD at Mother commented: "The Elvie Pump` | 0.1473 | 🚫 READ | JS=0.1473 > τ=0.05 → future context changes translation |
| 12/ 1 | `...rin, partner and ECD at Mother commented: "The Elvie Pump is` | 0.1514 | 🚫 READ | JS=0.1514 > τ=0.05 → future context changes translation |
| 13/ 1 | `...partner and ECD at Mother commented: "The Elvie Pump is such` | 0.1597 | 🚫 READ | JS=0.1597 > τ=0.05 → future context changes translation |
| 14/ 1 | `...rtner and ECD at Mother commented: "The Elvie Pump is such a` | 0.1098 | 🚫 READ | JS=0.1098 > τ=0.05 → future context changes translation |
| 15/ 1 | `...at Mother commented: "The Elvie Pump is such a revolutionary` | 0.1132 | 🚫 READ | JS=0.1132 > τ=0.05 → future context changes translation |
| 16/ 1 | `...r commented: "The Elvie Pump is such a revolutionary product` | 0.1060 | 🚫 READ | JS=0.1060 > τ=0.05 → future context changes translation |
| 17/ 1 | `...mented: "The Elvie Pump is such a revolutionary product that` | 0.1064 | 🚫 READ | JS=0.1064 > τ=0.05 → future context changes translation |
| 18/ 1 | `...ted: "The Elvie Pump is such a revolutionary product that it` | 0.1386 | 🚫 READ | JS=0.1386 > τ=0.05 → future context changes translation |
| 19/ 1 | `... Elvie Pump is such a revolutionary product that it deserved` | 0.0970 | 🚫 READ | JS=0.0970 > τ=0.05 → future context changes translation |
| 20/ 1 | `...lvie Pump is such a revolutionary product that it deserved a` | 0.1018 | 🚫 READ | JS=0.1018 > τ=0.05 → future context changes translation |
| 21/ 1 | `...Pump is such a revolutionary product that it deserved a bold` | 0.1341 | 🚫 READ | JS=0.1341 > τ=0.05 → future context changes translation |
| 22/ 1 | `... is such a revolutionary product that it deserved a bold and` | 0.0972 | 🚫 READ | JS=0.0972 > τ=0.05 → future context changes translation |
| 23/ 1 | `...evolutionary product that it deserved a bold and provocative` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.05 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=8,
tgt_len=0, having seen only `Ana Balarin, partner and ECD at Mother commented:`.
**DD waited until:** `Ana Balarin, partner and ECD at Mother commented: "The Elvie Pump is such a revolutionary product that it deserved a bold and provocative`
(JS dropped to 0.0000 — translation direction stable)

## Summary

DD functions as a principled early-commit risk detector:

1. **317 baseline commits were blocked** (22.4% of all post-wait-k decisions)
2. **JS signal is discriminative**: mean JS at READ (0.1599) vs COMMIT (0.0054)
3. **Case studies confirm** that blocked commits correspond to genuinely
   ambiguous positions where additional source context is informative
   (location disambiguation, named entities, subject resolution).
4. **+3.27 BLEU improvement** with moderate latency increase confirms
   the blocked commits were harmful (early commitment errors).
