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

Gate threshold used: τ = 0.03  (avg_js_firstN > τ → READ)

## Layer 3 — Metrics: Blocked Commit Statistics

| Metric | Value |
|--------|-------|
| Total gate evaluations (post wait-k) | 1669 |
| Commits blocked (DD forced READ)      | 426 (25.5%) |
| Commits allowed                       | 1243 (74.5%) |
| Sentences with ≥1 veto               | 56 / 97 |
| Avg vetos per sentence                | 4.39 |
| Avg JS at READ decisions              | 0.1496 |
| Avg JS at COMMIT decisions            | 0.0049 |
| JS separation (READ − COMMIT)         | 0.1447 |

**JS separation** measures how well DD discriminates risky from safe commits.
A gap of 0.1447 confirms DD is not firing randomly —
it consistently identifies structurally different situations.

## Layer 4 — Aggregate Improvement

| | Baseline (wait-k=5) | DD full gate (τ=0.03) | Δ |
|--|--|--|--|
| BLEU | 13.46 | 15.66 | **+2.19** |
| AL   | 8.80 | 10.74 | +1.94 |
| LAAL | 8.82 | 10.75 | +1.94 |

**Interpretation:** DD full gate achieves +2.19 BLEU at the cost of +1.94 AL.
This improvement cannot be explained by simply 'reading more' —
the AL increase is modest while the BLEU gain is substantial,
confirming that DD is selectively blocking the *right* commits.

## Beneficial Veto Analysis

For each sentence, we compare character-level F1 (char-F1) of baseline
vs DD output against the reference translation.
A veto is **beneficial** if DD output has char-F1 ≥ 0.02 higher than baseline.

| Category   | Count | Meaning |
|------------|-------|---------|
| Beneficial | 27 | DD output closer to reference |
| Neutral    | 61 | No meaningful difference (< 0.02 F1) |
| Harmful    | 9 | DD output further from reference |

**Key finding:** sentences where DD vetoed at least once improved by
avg ΔcharF1 = **+0.0373**,
while sentences with no vetos changed by +0.0000.
This confirms that DD interventions are net-positive.

### Top Beneficial Veto Cases (DD improved the output most)

**Sent 76** — 18 vetos, ΔcharF1 = +0.3022

| | Text |
|--|------|
| Reference | `Mother 的 合 伙 人 兼 执 行 创 意 总 监 (ECD) Ana Balarin 评 论 道 ： “Elvie Pump 是 一 款 革 命 性 的 产 品 ， 值 得 大 胆 尝 试 。` |
| Baseline  | `艾 娜 · 合 合 亲 , 池 伴 和 性 阿 Ana · , 得 值 是 如 此 革 命 性 的 产 品 ,` |
| DD output | `艾 德 公 司 的 合 作 伙 伴 和 ECD 的 Ana Balarin 说 :"Elvie Pump 是 如 此 革 命 性 的 产 品 ,` |

charF1: baseline=0.280 → DD=0.582 (**+0.302**)

**Sent 16** — 9 vetos, ΔcharF1 = +0.2456

| | Text |
|--|------|
| Reference | `由 于 热 带 雨 具 有 分 散 性 ， 伴 随 条 件 的 快 速 恶 化 ， 可 能 会 出 现 洪 水 泛 滥 。` |
| Baseline  | `闪 速 迅 速 恶 可 能 , 即 的 , 着 条 件 的 迅 速 恶 化 , 即 发 生 流 .` |
| DD output | `由 于 热 带 雨 的 分 散 性 , 随 着 条 件 的 迅 速 恶 化 , 即 发 生 流 .` |

charF1: baseline=0.281 → DD=0.526 (**+0.246**)

**Sent 6** — 4 vetos, ΔcharF1 = +0.2105

| | Text |
|--|------|
| Reference | `马 洛 总 部 还 有 整 整 一 仓 库 的 货 物 等 着 出 售 呢 。 “` |
| Baseline  | `库 里 个 这 有 个 一 个 装 仓 , 物 的 仓 库 ,` |
| DD output | `货 马 洛 总 部 有 大 量 装 满 货 物 的 仓 库 ,` |

charF1: baseline=0.316 → DD=0.526 (**+0.211**)

### Harmful Veto Cases (DD hurt the output)

These cases represent failure modes where DD forced unnecessary READs:

**Sent 29** — 3 vetos, ΔcharF1 = -0.1538

| | Text |
|--|------|
| Reference | `这 张 专 辑 更 多 反 映 了 我 们 是 谁 。 ”` |
| Baseline  | `这 张 专 辑 更 是 专 辑 上 ,` |
| DD output | `这 张 专 在 这 个 专 辑 上 ,` |

**Sent 72** — 1 vetos, ΔcharF1 = -0.0870

| | Text |
|--|------|
| Reference | `酒 水 饮 料 为 主 的 酒 吧 表 现 最 佳 ， 同 比 之 下 ， 餐 厅 的 交 易 量 却 持 续 下 降 。` |
| Baseline  | `酒 吧 和 酒 吧 , 以 喝 者 最 餐 为 的 ,` |
| DD output | `酒 吧 和 酒 吧 , 以 导 远 比 今 大 的 ,` |

### Veto Count vs Quality Improvement

| Veto count bucket | # sentences | Avg ΔcharF1 |
|-------------------|-------------|-------------|
| 0                 |          41 |     +0.0000 |
| 1–3               |          20 |     -0.0143 |
| 4–9               |          21 |     +0.0443 |
| 10+               |          15 |     +0.0964 |

More vetos → larger quality improvement, consistent with the hypothesis
that high-veto sentences are the ones most harmed by early commitment.

## Layer 2 — Behavior: Concrete Early-Commit Case Studies

Each case shows a sentence where DD blocked multiple baseline commits
while the model waited for critical disambiguating context.

### Case 1: Sentence 76

**Source:** `Ana Balarin, partner and ECD at Mother commented: "The Elvie Pump is such a revolutionary product that it deserved a bold and provocative launch.`

**Reference translation:** `Mother 的 合 伙 人 兼 执 行 创 意 总 监 (ECD) Ana Balarin 评 论 道 ： “Elvie Pump 是 一 款 革 命 性 的 产 品 ， 值 得 大 胆 尝 试 。`

**DD blocked 18 commits** (max 18 consecutive),
peak JS = 0.1838

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  5/ 0 | `...Ana Balarin, partner and ECD` | 0.1838 | 🚫 READ | JS=0.1838 > τ=0.03 → future context changes translation |
|  6/ 0 | `...Ana Balarin, partner and ECD at` | 0.1787 | 🚫 READ | JS=0.1787 > τ=0.03 → future context changes translation |
|  7/ 0 | `...Ana Balarin, partner and ECD at Mother` | 0.1189 | 🚫 READ | JS=0.1189 > τ=0.03 → future context changes translation |
|  8/ 0 | `...Ana Balarin, partner and ECD at Mother commented:` | 0.1025 | 🚫 READ | JS=0.1025 > τ=0.03 → future context changes translation |
|  9/ 0 | `...Ana Balarin, partner and ECD at Mother commented: "The` | 0.0370 | 🚫 READ | JS=0.0370 > τ=0.03 → future context changes translation |
| 10/ 0 | `...Ana Balarin, partner and ECD at Mother commented: "The Elvie` | 0.1213 | 🚫 READ | JS=0.1213 > τ=0.03 → future context changes translation |
| 11/ 0 | `...alarin, partner and ECD at Mother commented: "The Elvie Pump` | 0.1473 | 🚫 READ | JS=0.1473 > τ=0.03 → future context changes translation |
| 12/ 0 | `...rin, partner and ECD at Mother commented: "The Elvie Pump is` | 0.1514 | 🚫 READ | JS=0.1514 > τ=0.03 → future context changes translation |
| 13/ 0 | `...partner and ECD at Mother commented: "The Elvie Pump is such` | 0.1597 | 🚫 READ | JS=0.1597 > τ=0.03 → future context changes translation |
| 14/ 0 | `...rtner and ECD at Mother commented: "The Elvie Pump is such a` | 0.1098 | 🚫 READ | JS=0.1098 > τ=0.03 → future context changes translation |
| 15/ 0 | `...at Mother commented: "The Elvie Pump is such a revolutionary` | 0.1132 | 🚫 READ | JS=0.1132 > τ=0.03 → future context changes translation |
| 16/ 0 | `...r commented: "The Elvie Pump is such a revolutionary product` | 0.1060 | 🚫 READ | JS=0.1060 > τ=0.03 → future context changes translation |
| 17/ 0 | `...mented: "The Elvie Pump is such a revolutionary product that` | 0.1064 | 🚫 READ | JS=0.1064 > τ=0.03 → future context changes translation |
| 18/ 0 | `...ted: "The Elvie Pump is such a revolutionary product that it` | 0.1386 | 🚫 READ | JS=0.1386 > τ=0.03 → future context changes translation |
| 19/ 0 | `... Elvie Pump is such a revolutionary product that it deserved` | 0.0970 | 🚫 READ | JS=0.0970 > τ=0.03 → future context changes translation |
| 20/ 0 | `...lvie Pump is such a revolutionary product that it deserved a` | 0.1018 | 🚫 READ | JS=0.1018 > τ=0.03 → future context changes translation |
| 21/ 0 | `...Pump is such a revolutionary product that it deserved a bold` | 0.1341 | 🚫 READ | JS=0.1341 > τ=0.03 → future context changes translation |
| 22/ 0 | `... is such a revolutionary product that it deserved a bold and` | 0.0972 | 🚫 READ | JS=0.0972 > τ=0.03 → future context changes translation |
| 23/ 0 | `...evolutionary product that it deserved a bold and provocative` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.03 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=5,
tgt_len=0, having seen only `Ana Balarin, partner and ECD`.
**DD waited until:** `Ana Balarin, partner and ECD at Mother commented: "The Elvie Pump is such a revolutionary product that it deserved a bold and provocative`
(JS dropped to 0.0000 — translation direction stable)

### Case 2: Sentence 20

**Source:** `Crosses continually came into the Livingston box and were continually cleared, while two penalty claims - after Halkett's challenge on substitute Glenn Middleton, and one for handball - were waved away.`

**Reference translation:** `Crosses 不 断 进 入 利 文 斯 顿 队 的 禁 区 ， 并 不 断 地 被 解 围 ， 而 两 个 点 球 —— 在 哈 尔 科 特 提 出 要 让 替 补 格 伦 · 米 德 尔 顿 (Glenn Middleton) 上 场 和 一 个 手 球 之 后 —— 被 踢 了 回 去 。`

**DD blocked 17 commits** (max 17 consecutive),
peak JS = 0.1526

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  5/ 0 | `...Crosses continually came into the` | 0.0059 | ✅ COMMIT | JS=0.0059 ≤ τ=0.03 → stable, safe to commit |
|  6/ 1 | `...Crosses continually came into the Livingston` | 0.0033 | ✅ COMMIT | JS=0.0033 ≤ τ=0.03 → stable, safe to commit |
|  7/ 2 | `...Crosses continually came into the Livingston box` | 0.0047 | ✅ COMMIT | JS=0.0047 ≤ τ=0.03 → stable, safe to commit |
|  8/ 3 | `...Crosses continually came into the Livingston box and` | 0.0123 | ✅ COMMIT | JS=0.0123 ≤ τ=0.03 → stable, safe to commit |
|  9/ 4 | `...Crosses continually came into the Livingston box and were` | 0.0203 | ✅ COMMIT | JS=0.0203 ≤ τ=0.03 → stable, safe to commit |
| 10/ 5 | `...ontinually came into the Livingston box and were continually` | 0.0215 | ✅ COMMIT | JS=0.0215 ≤ τ=0.03 → stable, safe to commit |
| 11/ 6 | `...y came into the Livingston box and were continually cleared,` | 0.1124 | 🚫 READ | JS=0.1124 > τ=0.03 → future context changes translation |
| 12/ 6 | `... into the Livingston box and were continually cleared, while` | 0.1500 | 🚫 READ | JS=0.1500 > τ=0.03 → future context changes translation |
| 13/ 6 | `...o the Livingston box and were continually cleared, while two` | 0.1159 | 🚫 READ | JS=0.1159 > τ=0.03 → future context changes translation |
| 14/ 6 | `...vingston box and were continually cleared, while two penalty` | 0.1111 | 🚫 READ | JS=0.1111 > τ=0.03 → future context changes translation |
| 15/ 6 | `...n box and were continually cleared, while two penalty claims` | 0.1526 | 🚫 READ | JS=0.1526 > τ=0.03 → future context changes translation |
| 16/ 6 | `...box and were continually cleared, while two penalty claims -` | 0.1473 | 🚫 READ | JS=0.1473 > τ=0.03 → future context changes translation |
| 17/ 6 | `...d were continually cleared, while two penalty claims - after` | 0.1225 | 🚫 READ | JS=0.1225 > τ=0.03 → future context changes translation |
| 18/ 6 | `...tinually cleared, while two penalty claims - after Halkett's` | 0.0931 | 🚫 READ | JS=0.0931 > τ=0.03 → future context changes translation |
| 19/ 6 | `...leared, while two penalty claims - after Halkett's challenge` | 0.1358 | 🚫 READ | JS=0.1358 > τ=0.03 → future context changes translation |
| 20/ 6 | `...red, while two penalty claims - after Halkett's challenge on` | 0.1051 | 🚫 READ | JS=0.1051 > τ=0.03 → future context changes translation |
| 21/ 6 | `...two penalty claims - after Halkett's challenge on substitute` | 0.1064 | 🚫 READ | JS=0.1064 > τ=0.03 → future context changes translation |
| 22/ 6 | `...nalty claims - after Halkett's challenge on substitute Glenn` | 0.1367 | 🚫 READ | JS=0.1367 > τ=0.03 → future context changes translation |
| 23/ 6 | `...s - after Halkett's challenge on substitute Glenn Middleton,` | 0.0949 | 🚫 READ | JS=0.0949 > τ=0.03 → future context changes translation |
| 24/ 6 | `...after Halkett's challenge on substitute Glenn Middleton, and` | 0.1302 | 🚫 READ | JS=0.1302 > τ=0.03 → future context changes translation |
| 25/ 6 | `...r Halkett's challenge on substitute Glenn Middleton, and one` | 0.1285 | 🚫 READ | JS=0.1285 > τ=0.03 → future context changes translation |
| 26/ 6 | `...lkett's challenge on substitute Glenn Middleton, and one for` | 0.1021 | 🚫 READ | JS=0.1021 > τ=0.03 → future context changes translation |
| 27/ 6 | `...hallenge on substitute Glenn Middleton, and one for handball` | 0.1063 | 🚫 READ | JS=0.1063 > τ=0.03 → future context changes translation |
| 28/ 6 | `...llenge on substitute Glenn Middleton, and one for handball -` | 0.0071 | ✅ COMMIT | JS=0.0071 ≤ τ=0.03 → stable, safe to commit |
| 29/ 7 | `...e on substitute Glenn Middleton, and one for handball - were` | 0.0037 | ✅ COMMIT | JS=0.0037 ≤ τ=0.03 → stable, safe to commit |
| 30/ 8 | `...ubstitute Glenn Middleton, and one for handball - were waved` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.03 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=5,
tgt_len=0, having seen only `Crosses continually came into the`.
**DD waited until:** `Crosses continually came into the Livingston box and were continually cleared, while two penalty claims - after Halkett's challenge on substitute Glenn Middleton, and one for handball -`
(JS dropped to 0.0071 — translation direction stable)

### Case 3: Sentence 1

**Source:** `AMs are apparently suggesting alternative options, but the struggle to reach consensus could be a headache for the Presiding Officer, Elin Jones, who is expected to submit draft legislation on the changes within weeks.`

**Reference translation:** `显 然 ， AM 建 议 采 用 其 他 名 称 ， 但 会 议 主 席 Elin Jones 感 到 头 痛 的 是 ， 迟 迟 达 不 成 一 致 ， 因 为 他 要 在 几 周 内 提 交 名 称 变 更 的 立 法 草 案 。`

**DD blocked 21 commits** (max 16 consecutive),
peak JS = 0.1366

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  5/ 0 | `...AMs are apparently suggesting alternative` | 0.0431 | 🚫 READ | JS=0.0431 > τ=0.03 → future context changes translation |
|  6/ 0 | `...AMs are apparently suggesting alternative options,` | 0.0366 | 🚫 READ | JS=0.0366 > τ=0.03 → future context changes translation |
|  7/ 0 | `...AMs are apparently suggesting alternative options, but` | 0.0426 | 🚫 READ | JS=0.0426 > τ=0.03 → future context changes translation |
|  8/ 0 | `...AMs are apparently suggesting alternative options, but the` | 0.0327 | 🚫 READ | JS=0.0327 > τ=0.03 → future context changes translation |
|  9/ 0 | `... apparently suggesting alternative options, but the struggle` | 0.0348 | 🚫 READ | JS=0.0348 > τ=0.03 → future context changes translation |
| 10/ 0 | `...parently suggesting alternative options, but the struggle to` | 0.0429 | 🚫 READ | JS=0.0429 > τ=0.03 → future context changes translation |
| 11/ 0 | `...ly suggesting alternative options, but the struggle to reach` | 0.0409 | 🚫 READ | JS=0.0409 > τ=0.03 → future context changes translation |
| 12/ 0 | `...ing alternative options, but the struggle to reach consensus` | 0.0427 | 🚫 READ | JS=0.0427 > τ=0.03 → future context changes translation |
| 13/ 0 | `...ternative options, but the struggle to reach consensus could` | 0.0325 | 🚫 READ | JS=0.0325 > τ=0.03 → future context changes translation |
| 14/ 0 | `...native options, but the struggle to reach consensus could be` | 0.0340 | 🚫 READ | JS=0.0340 > τ=0.03 → future context changes translation |
| 15/ 0 | `...tive options, but the struggle to reach consensus could be a` | 0.0400 | 🚫 READ | JS=0.0400 > τ=0.03 → future context changes translation |
| 16/ 0 | `...ons, but the struggle to reach consensus could be a headache` | 0.1192 | 🚫 READ | JS=0.1192 > τ=0.03 → future context changes translation |
| 17/ 0 | `... but the struggle to reach consensus could be a headache for` | 0.1162 | 🚫 READ | JS=0.1162 > τ=0.03 → future context changes translation |
| 18/ 0 | `... the struggle to reach consensus could be a headache for the` | 0.1279 | 🚫 READ | JS=0.1279 > τ=0.03 → future context changes translation |
| 19/ 0 | `...gle to reach consensus could be a headache for the Presiding` | 0.1241 | 🚫 READ | JS=0.1241 > τ=0.03 → future context changes translation |
| 20/ 0 | `...ach consensus could be a headache for the Presiding Officer,` | 0.0363 | 🚫 READ | JS=0.0363 > τ=0.03 → future context changes translation |
| 21/ 0 | `...onsensus could be a headache for the Presiding Officer, Elin` | 0.0082 | ✅ COMMIT | JS=0.0082 ≤ τ=0.03 → stable, safe to commit |
| 22/ 1 | `...s could be a headache for the Presiding Officer, Elin Jones,` | 0.0330 | 🚫 READ | JS=0.0330 > τ=0.03 → future context changes translation |
| 23/ 1 | `...uld be a headache for the Presiding Officer, Elin Jones, who` | 0.0426 | 🚫 READ | JS=0.0426 > τ=0.03 → future context changes translation |
| 24/ 1 | `... be a headache for the Presiding Officer, Elin Jones, who is` | 0.1223 | 🚫 READ | JS=0.1223 > τ=0.03 → future context changes translation |
| 25/ 1 | `...dache for the Presiding Officer, Elin Jones, who is expected` | 0.1366 | 🚫 READ | JS=0.1366 > τ=0.03 → future context changes translation |
| 26/ 1 | `...he for the Presiding Officer, Elin Jones, who is expected to` | 0.1034 | 🚫 READ | JS=0.1034 > τ=0.03 → future context changes translation |
| 27/ 1 | `...the Presiding Officer, Elin Jones, who is expected to submit` | 0.0047 | ✅ COMMIT | JS=0.0047 ≤ τ=0.03 → stable, safe to commit |
| 28/ 2 | `...esiding Officer, Elin Jones, who is expected to submit draft` | 0.0031 | ✅ COMMIT | JS=0.0031 ≤ τ=0.03 → stable, safe to commit |
| 29/ 3 | `...cer, Elin Jones, who is expected to submit draft legislation` | 0.0026 | ✅ COMMIT | JS=0.0026 ≤ τ=0.03 → stable, safe to commit |
| 30/ 4 | `..., Elin Jones, who is expected to submit draft legislation on` | 0.0045 | ✅ COMMIT | JS=0.0045 ≤ τ=0.03 → stable, safe to commit |
| 31/ 5 | `...in Jones, who is expected to submit draft legislation on the` | 0.0037 | ✅ COMMIT | JS=0.0037 ≤ τ=0.03 → stable, safe to commit |
| 32/ 6 | `..., who is expected to submit draft legislation on the changes` | 0.0025 | ✅ COMMIT | JS=0.0025 ≤ τ=0.03 → stable, safe to commit |
| 33/ 7 | `...s expected to submit draft legislation on the changes within` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.03 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=5,
tgt_len=0, having seen only `AMs are apparently suggesting alternative`.
**DD waited until:** `AMs are apparently suggesting alternative options, but the struggle to reach consensus could be a headache for the Presiding Officer, Elin Jones, who is expected to submit`
(JS dropped to 0.0047 — translation direction stable)

## Summary

DD functions as a principled early-commit risk detector:

1. **426 baseline commits were blocked** (25.5% of all post-wait-k decisions)
2. **JS signal is discriminative**: mean JS at READ (0.1496) vs COMMIT (0.0049)
3. **Case studies confirm** that blocked commits correspond to genuinely
   ambiguous positions where additional source context is informative
   (location disambiguation, named entities, subject resolution).
4. **+2.19 BLEU improvement** with moderate latency increase confirms
   the blocked commits were harmful (early commitment errors).
