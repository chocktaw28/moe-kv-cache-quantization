# Recency Analysis — publication-date clustering

Generated from `data/papers.csv` (365 rows). All counts below are reproducible from the CSV.

## 0. Month derivation and coverage

`year` is the only date column in the CSV, so month-level dates were derived from the
`paper_id` where it is an arXiv ID: arXiv IDs encode YYMM (`2401.12345` -> 2024-01).
Rule applied: `paper_id` matching `^(\d{2})(\d{2})\.\d{4,5}$`, with the month part
validated to 01-12; year = 2000 + first two digits.

| Date resolution | Papers |
|---|---|
| month derivable from arXiv ID | 322 |
| month NOT derivable (DOI-only, 40; Semantic-Scholar hash IDs, 3) | 43 |
| **total** | **365** |

Handling of the 43 month-less rows: they are EXCLUDED from every monthly histogram
below (they cannot be placed in a month without fabricating one) but are retained in
all year-level counts and in the author-overlap analysis. Their year distribution is
2024: 10, 2025: 12, 2026: 18, not stated: 3. Their omission is not uniform across
series -- DOI-only records are disproportionately journal/conference MoE papers, so
the is_moe==true monthly series slightly UNDERCOUNTS relative to its true size.

Month-less rows by series: is_moe==true 8 of 64; reference 35 of 301; baseline-methods 1 of 12.

Month-less is_moe==true paper_ids: `10.1016/j.patcog.2026.114286`, `10.1109/DAC63849.2025.11132853`, `10.1109/ICEIC69189.2026.11386441`, `10.1609/aaai.v40i32.39899`, `10.18653/v1/2025.findings-acl.1386`, `10.18653/v1/2026.acl-long.982`, `ss:b49d11d5be5f05af7767b1a28c6ef0f12e3b2abf`, `10.1109/TCCN.2026.3694872`

## 1. Series definitions

- **(i) is_moe==true** — 64 papers (60 as originally extracted + 4 reclassified in Step 1:
  `2605.27646`, `2507.04610`, `2405.14366`, `2607.08029`). 56 have a derivable month.
- **(ii) reference (is_moe false/unclear)** — 301 papers, 266 with a derivable month.
- **(iii) named-baseline-methods** — 12 papers, 11 with a derivable month.
  **Matching rule:** case-insensitive regex `\bAWQ\b|SpinQuant|QuaRot|LLM-QAT|\bGPTQ\b`
  applied to the `method_name` field OR the `title` field. This captures both papers that
  PROPOSE these methods and papers that USE/compare against them by name, so series (iii)
  measures 'activity in the named-baseline orbit', not 'papers introducing a new baseline'.
  Membership is independent of is_moe status, so series (iii) OVERLAPS series (i) and (ii);
  the three series are not a partition. Baseline papers that are also is_moe==true: `2607.26316`.

## 2. Monthly histogram (actual counts)

| Month | (i) is_moe==true | (ii) reference | (iii) baseline-methods |
|---|---|---|---|
| 2024-01 | 0 | 2 | 0 |
| 2024-02 | 0 | 6 | 0 |
| 2024-03 | 0 | 4 | 0 |
| 2024-04 | 0 | 1 | 1 |
| 2024-05 | 2 | 7 | 2 |
| 2024-06 | 1 | 5 | 0 |
| 2024-07 | 0 | 4 | 0 |
| 2024-08 | 0 | 4 | 0 |
| 2024-09 | 0 | 2 | 1 |
| 2024-10 | 1 | 8 | 0 |
| 2024-11 | 1 | 5 | 0 |
| 2024-12 | 0 | 8 | 0 |
| 2025-01 | 0 | 7 | 0 |
| 2025-02 | 0 | 11 | 0 |
| 2025-03 | 1 | 7 | 0 |
| 2025-04 | 1 | 5 | 0 |
| 2025-05 | 3 | 11 | 0 |
| 2025-06 | 2 | 8 | 0 |
| 2025-07 | 2 | 7 | 2 |
| 2025-08 | 2 | 11 | 2 |
| 2025-09 | 1 | 6 | 1 |
| 2025-10 | 1 | 12 | 0 |
| 2025-11 | 4 | 9 | 0 |
| 2025-12 | 0 | 9 | 0 |
| 2026-01 | 3 | 7 | 0 |
| 2026-02 | 4 | 8 | 0 |
| 2026-03 | 3 | 9 | 0 |
| 2026-04 | 2 | 15 | 0 |
| 2026-05 | 3 | 28 | 1 |
| 2026-06 | 7 | 12 | 0 |
| 2026-07 | 9 | 16 | 1 |
| 2026-08 | 3 | 12 | 0 |
| **total (month-assigned)** | **56** | **266** | **11** |

### Half-year rollup (same data, less noise)

| Bucket | (i) is_moe==true | (ii) reference | (iii) baseline-methods |
|---|---|---|---|
| 2024H1 | 3 | 25 | 3 |
| 2024H2 | 2 | 31 | 1 |
| 2025H1 | 7 | 49 | 0 |
| 2025H2 | 10 | 54 | 5 |
| 2026H1 | 22 | 79 | 1 |
| 2026H2 | 12 | 28 | 1 |

Note 2026H2 is a PARTIAL bucket (corpus cutoff 2026-08-14, i.e. ~1.5 of 6 months
elapsed) and must not be compared like-for-like against full buckets.

## 3. THE CAP ARTIFACT — read before comparing any two series

`logs/run_summary.md` §3 documents a collection artifact that systematically distorts
series (ii) and (iii): papers whose cluster hits were confined to {B, C} were **capped at
15 per 6-month bucket, selected by citation count descending**. Recent papers have had
less time to accrue citations, so they were disproportionately excluded. The damage is
severe and grows with recency:

| Bucket | B/C-only candidates | Kept | Excluded | Kept fraction |
|---|---|---|---|---|
| 2024H1 | 30 | 15 | 15 | 50% |
| 2024H2 | 21 | 15 | 6 | 71% |
| 2025H1 | 55 | 15 | 40 | 27% |
| 2025H2 | 49 | 15 | 34 | 31% |
| 2026H1 | 117 | 15 | 102 | 13% |
| 2026H2 (partial) | 19 | 15 | 4 | 79% |

2026H1 is the extreme case: 117 candidates truncated to 15 (87% discarded). Series (i)
is largely uncapped because is_moe==true papers overwhelmingly carry a Cluster-A or
Cluster-D hit, and A/D papers were kept unconditionally. Concretely, of the 64
is_moe==true papers, 60 have an A or D cluster hit (uncapped) and 4 are B/C-only
(subject to the cap).

**Consequence:** series (i) is the only series whose shape can be read as roughly
reflecting the real literature. Series (ii) and (iii) have an artificial ceiling of
~15 B/C-only papers per half-year imposed on their recent buckets. Any statement of the
form 'MoE is accelerating relative to the baselines' is UNSUPPORTABLE from this data,
because the cap alone predicts exactly that pattern regardless of the underlying truth.

## 4. Is the is_moe==true subset accelerating?

Raw monthly counts for series (i), in order: 2024-05=2, 2024-06=1, 2024-10=1, 2024-11=1, 2025-03=1, 2025-04=1, 2025-05=3, 2025-06=2, 2025-07=2, 2025-08=2, 2025-09=1, 2025-10=1, 2025-11=4, 2026-01=3, 2026-02=4, 2026-03=3, 2026-04=2, 2026-05=3, 2026-06=7, 2026-07=9, 2026-08=3

- Papers in the last 12 months of the window (2025-09 through 2026-08): **40**
- Papers before 2025-09 (2024-01 through 2025-08, 20 months): **16**
- Rate: 3.33/month recent vs 0.80/month earlier.

**Answer, stated with the caution the sample size demands:** the direction is upward and
the monthly counts do rise, but **the monthly counts are too small to call a trend
reliably**. The per-month values in series (i) are single digits throughout -- most months
are 0, 1, 2 or 3 papers. At that magnitude, a single month's variation of +/-2 papers is
indistinguishable from Poisson noise, and no month-to-month comparison in this series is
statistically meaningful. What CAN be said at the half-year level, where counts are larger:
series (i) grows from a low base in 2024 to its largest full-bucket value in 2026H1.
Even that is a coarse observation from a ~60-paper series, and it is partly a general
arXiv-volume effect (the whole quantization literature grew over this window), not
necessarily MoE-specific. Two further deflators: (a) 2026H2 is partial and will look
artificially small; (b) the 8 month-less is_moe==true rows are absent from these counts.

A cleaner way to phrase the finding: **MoE-specific quantization work exists in every
half-year bucket from 2024H1 onward and is more common recently than at the start of the
window, but this corpus cannot distinguish 'accelerating field' from 'growing literature
overall' or from month-scale noise.**

## 5. Author overlap WITHIN the is_moe==true subset

Computed only over series (i) (64 papers), per the brief -- author overlap in the
dense/reference subset says nothing about MoE-quant competitiveness. Author strings were
split on `;` and `,`, Unicode-normalised, and lowercased; no cross-record identity
resolution beyond exact normalised-string match was attempted, so distinct people sharing
a name would collide and the same person spelled differently would not match.

### 5a. Authors appearing on 2+ is_moe==true papers

| Author | # is_moe==true papers | paper_ids |
|---|---|---|
| Tianlong Chen | 3 | `2406.08155`, `2602.19938`, `2605.23078` |
| Xing Hu | 3 | `2505.03804`, `2511.02302`, `2602.11184` |
| Amnon Geifman | 2 | `2602.11937`, `2607.04371` |
| Dawei Yang | 2 | `2505.03804`, `2602.11184` |
| Dingwen Tao | 2 | `10.1109/DAC63849.2025.11132853`, `10.1609/aaai.v40i32.39899` |
| Donglei Wu | 2 | `10.1109/DAC63849.2025.11132853`, `10.1609/aaai.v40i32.39899` |
| Elad Segal | 2 | `2602.11937`, `2607.04371` |
| Fei Chao | 2 | `10.18653/v1/2025.findings-acl.1386`, `10.18653/v1/2026.acl-long.982` |
| Fengjuan Wang | 2 | `2511.02302`, `2603.02731` |
| Haoru Tan | 2 | `2410.06270`, `2510.10962` |
| Ido Galil | 2 | `2602.11937`, `2607.04371` |
| Ido Shahaf | 2 | `2602.11937`, `2607.04371` |
| Itay Levy | 2 | `2602.11937`, `2607.04371` |
| Jian Cheng | 2 | `2508.01625`, `2606.17118` |
| Jianhui Liu | 2 | `2410.06270`, `2510.10962` |
| Jinda Jia | 2 | `10.1109/DAC63849.2025.11132853`, `10.1609/aaai.v40i32.39899` |
| Jing Liu | 2 | `2405.14366`, `2606.17118` |
| Mohammad Dabbah | 2 | `2602.11937`, `2607.04371` |
| Mou Sun | 2 | `2511.02302`, `2603.02731` |
| Najeeb Nabwani | 2 | `2602.11937`, `2607.04371` |
| Nave Assaf | 2 | `2602.11937`, `2607.04371` |
| Nir Ailon | 2 | `2602.11937`, `2607.04371` |
| Omri Puny | 2 | `2602.11937`, `2607.04371` |
| Oren Tropp | 2 | `2602.11937`, `2607.04371` |
| Pavlo Molchanov | 2 | `2602.11937`, `2607.04371` |
| Peisong Wang | 2 | `2508.01625`, `2606.17118` |
| Ran El-Yaniv | 2 | `2602.11937`, `2607.04371` |
| Ran Zilberstein | 2 | `2602.11937`, `2607.04371` |
| Roi Koren | 2 | `2602.11937`, `2607.04371` |
| Shiming Zhang | 2 | `2410.06270`, `2510.10962` |
| Si Liu | 2 | `2410.06270`, `2510.10962` |
| Size Zheng | 2 | `2503.21135`, `2505.05799` |
| Tao Liu | 2 | `10.1109/TCCN.2026.3694872`, `2606.17118` |
| Tomer Asida | 2 | `2602.11937`, `2607.04371` |
| Tomer Ronen | 2 | `2602.11937`, `2607.04371` |
| Vladimir Anisimov | 2 | `2602.11937`, `2607.04371` |
| Wei Huang | 2 | `2410.06270`, `2510.10962` |
| Wen Xia | 2 | `10.1109/DAC63849.2025.11132853`, `10.1609/aaai.v40i32.39899` |
| Xiaojuan Qi | 2 | `2410.06270`, `2510.10962` |
| Xiawu Zheng | 2 | `10.18653/v1/2025.findings-acl.1386`, `2606.04980` |
| Yonatan Geifman | 2 | `2602.11937`, `2607.04371` |
| Yuanteng Chen | 2 | `2508.01625`, `2606.17118` |
| Yuantian Shao | 2 | `2508.01625`, `2606.17118` |
| Yuchen Yang | 2 | `2601.21198`, `2607.16184` |
| Yue Liao | 2 | `2410.06270`, `2510.10962` |
| Yuexiao Ma | 2 | `10.18653/v1/2025.findings-acl.1386`, `2606.04980` |
| Zach Moshe | 2 | `2602.11937`, `2607.04371` |
| Zhihang Yuan | 2 | `2505.03804`, `2505.05799` |
| Zhixuan Chen | 2 | `2505.03804`, `2602.11184` |
| Zijie Liu | 2 | `2602.19938`, `2605.23078` |
| Zukang Xu | 2 | `2505.03804`, `2602.11184` |

### 5b. Recurring groups (connected components over shared authorship)

Papers are linked when they share at least one normalised author name, then grouped into
connected components. A component is a proxy for 'a lab or collaboration publishing
repeatedly in MoE quantization'. Components of size 1 are omitted.

**8 groups of 2+ papers, covering 24 of the 64 is_moe==true papers (38%).**

**Group 1 — 6 papers.** Recurring authors: Xing Hu, Size Zheng, Zhixuan Chen, Dawei Yang, Zhihang Yuan, Zukang Xu, Mou Sun, Fengjuan Wang
  - `2503.21135` (2025) DynaMo: Runtime Switchable Quantization for MoE with Cross-Dataset Adaptation
  - `2505.03804` (2025) MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanc
  - `2505.05799` (2025) MxMoE: Mixed-precision Quantization for MoE with Accuracy and Performance Co-Design
  - `2511.02302` (2025) FP8-Flow-MoE: A Casting-Free FP8 Recipe without Double Quantization Error
  - `2602.11184` (2026) KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models
  - `2603.02731` (2026) Practical FP4 Training for Large-Scale MoE Models on Hopper GPUs

**Group 2 — 4 papers.** Recurring authors: Peisong Wang, Jian Cheng, Yuantian Shao, Yuanteng Chen, Tao Liu, Jing Liu
  - `10.1109/TCCN.2026.3694872` (not stated) Efficient Edge Deployment of Mixture-of-Experts Models With Hybrid Expert Quantization
  - `2405.14366` (2024) MiniCache: KV Cache Compression in Depth Dimension for Large Language Models
  - `2508.01625` (2025) EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models
  - `2606.17118` (2026) MODE: Modality-Decomposed Expert-Level Mixed-Precision Quantization for MoE Multimodal LLMs

**Group 3 — 3 papers.** Recurring authors: Yuexiao Ma, Xiawu Zheng, Fei Chao
  - `10.18653/v1/2025.findings-acl.1386` (2025) Automated Fine-Grained Mixture-of-Experts Quantization
  - `10.18653/v1/2026.acl-long.982` (2026) Profiling-Free Mixed-Precision Quantization for MoE LLMs via Fuzzy Rule Interpolation
  - `2606.04980` (2026) AlphaQ: Calibration-Free Bit Allocation for Mixture-of-Experts Quantization

**Group 4 — 3 papers.** Recurring authors: Tianlong Chen, Zijie Liu
  - `2406.08155` (2024) QuantMoE-Bench: Examining Post-Training Quantization for Mixture-of-Experts
  - `2602.19938` (2026) A Replicate-and-Quantize Strategy for Plug-and-Play Load Balancing of Sparse Mixture-of-Experts
  - `2605.23078` (2026) GEMQ: Global Expert-Level Mixed-Precision Quantization for MoE LLMs

**Group 5 — 2 papers.** Recurring authors: Donglei Wu, Jinda Jia, Dingwen Tao, Wen Xia
  - `10.1109/DAC63849.2025.11132853` (2025) BirdMoE: Reducing Communication Costs for Mixture-of-Experts Training Using Load-Aware Bi-rando
  - `10.1609/aaai.v40i32.39899` (not stated) RCMoE: A Communication-Efficient Random Compression Framework for Resource-Constrained Mixture-

**Group 6 — 2 papers.** Recurring authors: Shiming Zhang, Si Liu, Xiaojuan Qi, Wei Huang, Haoru Tan, Jianhui Liu, Yue Liao
  - `2410.06270` (2024) MC-MoE: Mixture Compressor for Mixture-of-Experts LLMs Gains More
  - `2510.10962` (2025) MC#: Mixture Compressor for Mixture-of-Experts Large Models

**Group 7 — 2 papers.** Recurring authors: Yuchen Yang
  - `2601.21198` (2026) ZipMoE: Efficient On-Device MoE Serving via Lossless Compression and Cache-Affinity Scheduling
  - `2607.16184` (2026) PagedWeight: Efficient MoE LLM Serving with Dynamic Quality-Aware Weight Quantization

**Group 8 — 2 papers.** Recurring authors: Zach Moshe, Ido Galil, Oren Tropp, Roi Koren, Ran Zilberstein, Amnon Geifman, Elad Segal, Vladimir Anisimov…
  - `2602.11937` (2026) Extending Puzzle for Mixture-of-Experts Reasoning Models with Application to GPT-OSS Accelerati
  - `2607.04371` (2026) Nemotron-Labs-3-Puzzle-75B-A9B: Compressing Hybrid MoE LLMs

**Interpretation.** This is the strongest competitiveness signal in Step 3, and it points
the opposite way from a naive reading of the small corpus size. 51 distinct author names
recur across 2+ is_moe==true papers, and 8 separate groups account for 24 of the 64
papers — **more than a third of the MoE-quantization evidence base is produced by
repeat-publishing collaborations, not by one-off entrants.** The largest group has 6
papers spanning 2025-03 to 2026-03 and covers mixed-precision PTQ, FP8 training, FP4
training and vector quantization for MoE — i.e. one collaboration is working across
most of the bit-width range simultaneously. A second group of 4 spans expert-selection-
aware compression, multimodal MoE mixed precision and edge MoE deployment.

**What this does and does not license.** It supports the claim that the MoE-quantization
space is *actively contested by established groups* rather than empty. It does NOT
support any claim about which sub-problems are contested — group membership says nothing
about topic coverage, which is what the Step 1 matrix is for. Caveats: exact normalised-
name matching means common Chinese-transliterated names (e.g. 'Jing Liu', 'Tao Liu', 'Si
Liu', 'Wei Huang') may produce FALSE links between unrelated researchers, and Group 2 in
particular rests partly on such names — treat group boundaries as approximate. Conversely,
DOI-only and abstract-only rows sometimes carry truncated author lists, which would
produce false NEGATIVES. The large NVIDIA/Nemotron group (Group 8) is an industry lab
with a naturally large shared author roster, which inflates linkage within it.

## 6. Comparison against the baseline series — mature/crowded vs still opening up

| | is_moe==true | reference | baseline-methods |
|---|---|---|---|
| papers (month-assigned) | 56 | 266 | 11 |
| first month present | 2024-05 | 2024-01 | 2024-04 |
| peak single month | 9 | 28 | 2 |

**This comparison is heavily compromised and should carry little weight.** The reference
and baseline series are the two capped series; their recent months are truncated to an
artificial ceiling while series (i) is not. The measured ratio between series (i) and
series (iii) in recent months is therefore an artifact of the sampling design at least as
much as a property of the literature. The one comparison that survives the artifact is
directional and weak: the named-baseline methods (AWQ/GPTQ/QuaRot/SpinQuant/LLM-QAT)
appear across the whole window as an established reference vocabulary that other papers
cite and compare against, whereas MoE-specific quantization methods are still mostly
introducing new named methods rather than converging on a shared baseline set -- which is
consistent with 'still opening up', but is an observation about naming conventions in the
`method_name` field, NOT a measured publication-rate result.

