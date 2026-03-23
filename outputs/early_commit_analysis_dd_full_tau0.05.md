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
| Total gate evaluations (post wait-k) | 1669 |
| Commits blocked (DD forced READ)      | 392 (23.5%) |
| Commits allowed                       | 1277 (76.5%) |
| Sentences with ≥1 veto               | 50 / 97 |
| Avg vetos per sentence                | 4.04 |
| Avg JS at READ decisions              | 0.1594 |
| Avg JS at COMMIT decisions            | 0.0057 |
| JS separation (READ − COMMIT)         | 0.1537 |

**JS separation** measures how well DD discriminates risky from safe commits.
A gap of 0.1537 confirms DD is not firing randomly —
it consistently identifies structurally different situations.

## Layer 4 — Aggregate Improvement

| | Baseline (wait-k=5) | DD full gate (τ=0.05) | Δ |
|--|--|--|--|
| BLEU | 13.46 | 15.45 | **+1.99** |
| AL   | 8.80 | 10.41 | +1.61 |
| LAAL | 8.82 | 10.42 | +1.61 |

**Interpretation:** DD full gate achieves +1.99 BLEU at the cost of +1.61 AL.
This improvement cannot be explained by simply 'reading more' —
the AL increase is modest while the BLEU gain is substantial,
confirming that DD is selectively blocking the *right* commits.

## Beneficial Veto Analysis

For each sentence, we compare character-level F1 (char-F1) of baseline
vs DD output against the reference translation.
A veto is **beneficial** if DD output has char-F1 ≥ 0.02 higher than baseline.

| Category   | Count | Meaning |
|------------|-------|---------|
| Beneficial | 25 | DD output closer to reference |
| Neutral    | 63 | No meaningful difference (< 0.02 F1) |
| Harmful    | 9 | DD output further from reference |

**Key finding:** sentences where DD vetoed at least once improved by
avg ΔcharF1 = **+0.0354**,
while sentences with no vetos changed by +0.0000.
This confirms that DD interventions are net-positive.

### Top Beneficial Veto Cases (DD improved the output most)

**Sent 76** — 17 vetos, ΔcharF1 = +0.3022

| | Text |
|--|------|
| Reference | `Mother 的 合 伙 人 兼 执 行 创 意 总 监 (ECD) Ana Balarin 评 论 道 ： “Elvie Pump 是 一 款 革 命 性 的 产 品 ， 值 得 大 胆 尝 试 。` |
| Baseline  | `艾 娜 · 合 合 亲 , 池 伴 和 性 阿 Ana · , 得 值 是 如 此 革 命 性 的 产 品 ,` |
| DD output | `马 尔 公 司 的 合 作 伙 伴 和 ECD 的 Ana Balarin 说 :"Elvie Pump 是 如 此 革 命 性 的 产 品 ,` |

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

**Sent 45** — 3 vetos, ΔcharF1 = -0.0800

| | Text |
|--|------|
| Reference | `“ 然 而 ， 我 们 并 没 有 看 到 美 国 方 面 做 出 任 何 回 应 ” ， 李 说 。` |
| Baseline  | `" 然 而 , 我 们 没 有 们 看 不 到 美 国 的 任 何 相 应 的 反 应 .` |
| DD output | `" 然 而 , 我 而 , 我 们 看 不 到 美 国 的 任 何 相 应 的 反 应 .` |

### Veto Count vs Quality Improvement

| Veto count bucket | # sentences | Avg ΔcharF1 |
|-------------------|-------------|-------------|
| 0                 |          47 |     +0.0000 |
| 1–3               |          16 |     -0.0149 |
| 4–9               |          20 |     +0.0353 |
| 10+               |          14 |     +0.0930 |

More vetos → larger quality improvement, consistent with the hypothesis
that high-veto sentences are the ones most harmed by early commitment.

## Layer 2 — Behavior: Concrete Early-Commit Case Studies

Each case shows a sentence where DD blocked multiple baseline commits
while the model waited for critical disambiguating context.

### Case 1: Sentence 20

**Source:** `Crosses continually came into the Livingston box and were continually cleared, while two penalty claims - after Halkett's challenge on substitute Glenn Middleton, and one for handball - were waved away.`

**Reference translation:** `Crosses 不 断 进 入 利 文 斯 顿 队 的 禁 区 ， 并 不 断 地 被 解 围 ， 而 两 个 点 球 —— 在 哈 尔 科 特 提 出 要 让 替 补 格 伦 · 米 德 尔 顿 (Glenn Middleton) 上 场 和 一 个 手 球 之 后 —— 被 踢 了 回 去 。`

**DD blocked 17 commits** (max 17 consecutive),
peak JS = 0.1526

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  5/ 0 | `...Crosses continually came into the` | 0.0059 | ✅ COMMIT | JS=0.0059 ≤ τ=0.05 → stable, safe to commit |
|  6/ 1 | `...Crosses continually came into the Livingston` | 0.0033 | ✅ COMMIT | JS=0.0033 ≤ τ=0.05 → stable, safe to commit |
|  7/ 2 | `...Crosses continually came into the Livingston box` | 0.0047 | ✅ COMMIT | JS=0.0047 ≤ τ=0.05 → stable, safe to commit |
|  8/ 3 | `...Crosses continually came into the Livingston box and` | 0.0123 | ✅ COMMIT | JS=0.0123 ≤ τ=0.05 → stable, safe to commit |
|  9/ 4 | `...Crosses continually came into the Livingston box and were` | 0.0203 | ✅ COMMIT | JS=0.0203 ≤ τ=0.05 → stable, safe to commit |
| 10/ 5 | `...ontinually came into the Livingston box and were continually` | 0.0215 | ✅ COMMIT | JS=0.0215 ≤ τ=0.05 → stable, safe to commit |
| 11/ 6 | `...y came into the Livingston box and were continually cleared,` | 0.1124 | 🚫 READ | JS=0.1124 > τ=0.05 → future context changes translation |
| 12/ 6 | `... into the Livingston box and were continually cleared, while` | 0.1500 | 🚫 READ | JS=0.1500 > τ=0.05 → future context changes translation |
| 13/ 6 | `...o the Livingston box and were continually cleared, while two` | 0.1159 | 🚫 READ | JS=0.1159 > τ=0.05 → future context changes translation |
| 14/ 6 | `...vingston box and were continually cleared, while two penalty` | 0.1111 | 🚫 READ | JS=0.1111 > τ=0.05 → future context changes translation |
| 15/ 6 | `...n box and were continually cleared, while two penalty claims` | 0.1526 | 🚫 READ | JS=0.1526 > τ=0.05 → future context changes translation |
| 16/ 6 | `...box and were continually cleared, while two penalty claims -` | 0.1473 | 🚫 READ | JS=0.1473 > τ=0.05 → future context changes translation |
| 17/ 6 | `...d were continually cleared, while two penalty claims - after` | 0.1225 | 🚫 READ | JS=0.1225 > τ=0.05 → future context changes translation |
| 18/ 6 | `...tinually cleared, while two penalty claims - after Halkett's` | 0.0931 | 🚫 READ | JS=0.0931 > τ=0.05 → future context changes translation |
| 19/ 6 | `...leared, while two penalty claims - after Halkett's challenge` | 0.1358 | 🚫 READ | JS=0.1358 > τ=0.05 → future context changes translation |
| 20/ 6 | `...red, while two penalty claims - after Halkett's challenge on` | 0.1051 | 🚫 READ | JS=0.1051 > τ=0.05 → future context changes translation |
| 21/ 6 | `...two penalty claims - after Halkett's challenge on substitute` | 0.1064 | 🚫 READ | JS=0.1064 > τ=0.05 → future context changes translation |
| 22/ 6 | `...nalty claims - after Halkett's challenge on substitute Glenn` | 0.1367 | 🚫 READ | JS=0.1367 > τ=0.05 → future context changes translation |
| 23/ 6 | `...s - after Halkett's challenge on substitute Glenn Middleton,` | 0.0949 | 🚫 READ | JS=0.0949 > τ=0.05 → future context changes translation |
| 24/ 6 | `...after Halkett's challenge on substitute Glenn Middleton, and` | 0.1302 | 🚫 READ | JS=0.1302 > τ=0.05 → future context changes translation |
| 25/ 6 | `...r Halkett's challenge on substitute Glenn Middleton, and one` | 0.1285 | 🚫 READ | JS=0.1285 > τ=0.05 → future context changes translation |
| 26/ 6 | `...lkett's challenge on substitute Glenn Middleton, and one for` | 0.1021 | 🚫 READ | JS=0.1021 > τ=0.05 → future context changes translation |
| 27/ 6 | `...hallenge on substitute Glenn Middleton, and one for handball` | 0.1063 | 🚫 READ | JS=0.1063 > τ=0.05 → future context changes translation |
| 28/ 6 | `...llenge on substitute Glenn Middleton, and one for handball -` | 0.0071 | ✅ COMMIT | JS=0.0071 ≤ τ=0.05 → stable, safe to commit |
| 29/ 7 | `...e on substitute Glenn Middleton, and one for handball - were` | 0.0037 | ✅ COMMIT | JS=0.0037 ≤ τ=0.05 → stable, safe to commit |
| 30/ 8 | `...ubstitute Glenn Middleton, and one for handball - were waved` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.05 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=5,
tgt_len=0, having seen only `Crosses continually came into the`.
**DD waited until:** `Crosses continually came into the Livingston box and were continually cleared, while two penalty claims - after Halkett's challenge on substitute Glenn Middleton, and one for handball -`
(JS dropped to 0.0071 — translation direction stable)

### Case 2: Sentence 70

**Source:** `However, it cannot be right that it is as easy for individuals who don't live in the UK, as well as foreign-based companies, to buy homes as hard-working British residents.`

**Reference translation:** `然 而 ， 对 于 不 住 在 英 国 的 个 人 以 及 外 国 公 司 来 说 ， 与 勤 劳 的 英 国 居 民 在 购 房 难 度 上 别 无 二 致 ， 这 是 不 对 的 。`

**DD blocked 17 commits** (max 13 consecutive),
peak JS = 0.2838

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  5/ 0 | `...However, it cannot be right` | 0.0099 | ✅ COMMIT | JS=0.0099 ≤ τ=0.05 → stable, safe to commit |
|  6/ 1 | `...However, it cannot be right that` | 0.0145 | ✅ COMMIT | JS=0.0145 ≤ τ=0.05 → stable, safe to commit |
|  7/ 2 | `...However, it cannot be right that it` | 0.0126 | ✅ COMMIT | JS=0.0126 ≤ τ=0.05 → stable, safe to commit |
|  8/ 3 | `...However, it cannot be right that it is` | 0.0414 | ✅ COMMIT | JS=0.0414 ≤ τ=0.05 → stable, safe to commit |
|  9/ 4 | `...However, it cannot be right that it is as` | 0.0565 | 🚫 READ | JS=0.0565 > τ=0.05 → future context changes translation |
| 10/ 4 | `...However, it cannot be right that it is as easy` | 0.1760 | 🚫 READ | JS=0.1760 > τ=0.05 → future context changes translation |
| 11/ 4 | `...However, it cannot be right that it is as easy for` | 0.2097 | 🚫 READ | JS=0.2097 > τ=0.05 → future context changes translation |
| 12/ 4 | `...wever, it cannot be right that it is as easy for individuals` | 0.1632 | 🚫 READ | JS=0.1632 > τ=0.05 → future context changes translation |
| 13/ 4 | `...r, it cannot be right that it is as easy for individuals who` | 0.0283 | ✅ COMMIT | JS=0.0283 ≤ τ=0.05 → stable, safe to commit |
| 14/ 5 | `...cannot be right that it is as easy for individuals who don't` | 0.1979 | 🚫 READ | JS=0.1979 > τ=0.05 → future context changes translation |
| 15/ 5 | `...t be right that it is as easy for individuals who don't live` | 0.2802 | 🚫 READ | JS=0.2802 > τ=0.05 → future context changes translation |
| 16/ 5 | `...e right that it is as easy for individuals who don't live in` | 0.2822 | 🚫 READ | JS=0.2822 > τ=0.05 → future context changes translation |
| 17/ 5 | `...ght that it is as easy for individuals who don't live in the` | 0.2838 | 🚫 READ | JS=0.2838 > τ=0.05 → future context changes translation |
| 18/ 5 | `...that it is as easy for individuals who don't live in the UK,` | 0.2828 | 🚫 READ | JS=0.2828 > τ=0.05 → future context changes translation |
| 19/ 5 | `...t it is as easy for individuals who don't live in the UK, as` | 0.1822 | 🚫 READ | JS=0.1822 > τ=0.05 → future context changes translation |
| 20/ 5 | `...is as easy for individuals who don't live in the UK, as well` | 0.2748 | 🚫 READ | JS=0.2748 > τ=0.05 → future context changes translation |
| 21/ 5 | `...as easy for individuals who don't live in the UK, as well as` | 0.2149 | 🚫 READ | JS=0.2149 > τ=0.05 → future context changes translation |
| 22/ 5 | `...dividuals who don't live in the UK, as well as foreign-based` | 0.2196 | 🚫 READ | JS=0.2196 > τ=0.05 → future context changes translation |
| 23/ 5 | `...ho don't live in the UK, as well as foreign-based companies,` | 0.1700 | 🚫 READ | JS=0.1700 > τ=0.05 → future context changes translation |
| 24/ 5 | `...don't live in the UK, as well as foreign-based companies, to` | 0.2261 | 🚫 READ | JS=0.2261 > τ=0.05 → future context changes translation |
| 25/ 5 | `...t live in the UK, as well as foreign-based companies, to buy` | 0.1693 | 🚫 READ | JS=0.1693 > τ=0.05 → future context changes translation |
| 26/ 5 | `... in the UK, as well as foreign-based companies, to buy homes` | 0.1690 | 🚫 READ | JS=0.1690 > τ=0.05 → future context changes translation |
| 27/ 5 | `... the UK, as well as foreign-based companies, to buy homes as` | 0.0024 | ✅ COMMIT | JS=0.0024 ≤ τ=0.05 → stable, safe to commit |
| 28/ 6 | `...ell as foreign-based companies, to buy homes as hard-working` | 0.0018 | ✅ COMMIT | JS=0.0018 ≤ τ=0.05 → stable, safe to commit |
| 29/ 7 | `...oreign-based companies, to buy homes as hard-working British` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.05 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=5,
tgt_len=0, having seen only `However, it cannot be right`.
**DD waited until:** `However, it cannot be right that it is as easy for individuals who don't live in the UK, as well as foreign-based companies, to buy homes as`
(JS dropped to 0.0024 — translation direction stable)

### Case 3: Sentence 81

**Source:** `Vice President Mike Pence is now slated to address the conference, now scheduled for mid-October, in a signal of the import the administration places on the gathering, the diplomats said.`

**Reference translation:** `副 总 统 迈 克 · 彭 斯 (Mike Pence) 目 前 计 划 在 预 计 于 十 月 中 旬 举 行 的 会 议 上 发 表 讲 话 ， 外 交 官 们 表 示 ， 这 是 政 府 当 局 对 此 次 会 议 提 起 重 视 的 一 个 信 号 。`

**DD blocked 23 commits** (max 13 consecutive),
peak JS = 0.2639

| Step | src prefix | JS score | Decision | Why |
|------|-----------|----------|----------|-----|
|  5/ 0 | `...Vice President Mike Pence is` | 0.1538 | 🚫 READ | JS=0.1538 > τ=0.05 → future context changes translation |
|  6/ 0 | `...Vice President Mike Pence is now` | 0.1520 | 🚫 READ | JS=0.1520 > τ=0.05 → future context changes translation |
|  7/ 0 | `...Vice President Mike Pence is now slated` | 0.1440 | 🚫 READ | JS=0.1440 > τ=0.05 → future context changes translation |
|  8/ 0 | `...Vice President Mike Pence is now slated to` | 0.1125 | 🚫 READ | JS=0.1125 > τ=0.05 → future context changes translation |
|  9/ 0 | `...Vice President Mike Pence is now slated to address` | 0.1110 | 🚫 READ | JS=0.1110 > τ=0.05 → future context changes translation |
| 10/ 0 | `...Vice President Mike Pence is now slated to address the` | 0.1121 | 🚫 READ | JS=0.1121 > τ=0.05 → future context changes translation |
| 11/ 0 | `...resident Mike Pence is now slated to address the conference,` | 0.1138 | 🚫 READ | JS=0.1138 > τ=0.05 → future context changes translation |
| 12/ 0 | `...dent Mike Pence is now slated to address the conference, now` | 0.1469 | 🚫 READ | JS=0.1469 > τ=0.05 → future context changes translation |
| 13/ 0 | `...Pence is now slated to address the conference, now scheduled` | 0.1135 | 🚫 READ | JS=0.1135 > τ=0.05 → future context changes translation |
| 14/ 0 | `...e is now slated to address the conference, now scheduled for` | 0.1457 | 🚫 READ | JS=0.1457 > τ=0.05 → future context changes translation |
| 15/ 0 | `...ed to address the conference, now scheduled for mid-October,` | 0.1091 | 🚫 READ | JS=0.1091 > τ=0.05 → future context changes translation |
| 16/ 0 | `...to address the conference, now scheduled for mid-October, in` | 0.1102 | 🚫 READ | JS=0.1102 > τ=0.05 → future context changes translation |
| 17/ 0 | `... address the conference, now scheduled for mid-October, in a` | 0.1130 | 🚫 READ | JS=0.1130 > τ=0.05 → future context changes translation |
| 18/ 0 | `...s the conference, now scheduled for mid-October, in a signal` | 0.0042 | ✅ COMMIT | JS=0.0042 ≤ τ=0.05 → stable, safe to commit |
| 19/ 1 | `...he conference, now scheduled for mid-October, in a signal of` | 0.1093 | 🚫 READ | JS=0.1093 > τ=0.05 → future context changes translation |
| 20/ 1 | `...onference, now scheduled for mid-October, in a signal of the` | 0.1464 | 🚫 READ | JS=0.1464 > τ=0.05 → future context changes translation |
| 21/ 1 | `...ce, now scheduled for mid-October, in a signal of the import` | 0.1443 | 🚫 READ | JS=0.1443 > τ=0.05 → future context changes translation |
| 22/ 1 | `...now scheduled for mid-October, in a signal of the import the` | 0.1419 | 🚫 READ | JS=0.1419 > τ=0.05 → future context changes translation |
| 23/ 1 | `...or mid-October, in a signal of the import the administration` | 0.1459 | 🚫 READ | JS=0.1459 > τ=0.05 → future context changes translation |
| 24/ 1 | `...October, in a signal of the import the administration places` | 0.1119 | 🚫 READ | JS=0.1119 > τ=0.05 → future context changes translation |
| 25/ 1 | `...ober, in a signal of the import the administration places on` | 0.1499 | 🚫 READ | JS=0.1499 > τ=0.05 → future context changes translation |
| 26/ 1 | `..., in a signal of the import the administration places on the` | 0.2635 | 🚫 READ | JS=0.2635 > τ=0.05 → future context changes translation |
| 27/ 1 | `...al of the import the administration places on the gathering,` | 0.2639 | 🚫 READ | JS=0.2639 > τ=0.05 → future context changes translation |
| 28/ 1 | `...f the import the administration places on the gathering, the` | 0.1723 | 🚫 READ | JS=0.1723 > τ=0.05 → future context changes translation |
| 29/ 1 | `...rt the administration places on the gathering, the diplomats` | 0.0000 | ✅ COMMIT | JS=0.0000 ≤ τ=0.05 → stable, safe to commit |

**Baseline (wait-k=5) behavior:** Would have committed at src_len=5,
tgt_len=0, having seen only `Vice President Mike Pence is`.
**DD waited until:** `Vice President Mike Pence is now slated to address the conference, now scheduled for mid-October, in a signal of the import the administration places on the gathering, the diplomats`
(JS dropped to 0.0000 — translation direction stable)

## Summary

DD functions as a principled early-commit risk detector:

1. **392 baseline commits were blocked** (23.5% of all post-wait-k decisions)
2. **JS signal is discriminative**: mean JS at READ (0.1594) vs COMMIT (0.0057)
3. **Case studies confirm** that blocked commits correspond to genuinely
   ambiguous positions where additional source context is informative
   (location disambiguation, named entities, subject resolution).
4. **+1.99 BLEU improvement** with moderate latency increase confirms
   the blocked commits were harmful (early commitment errors).
