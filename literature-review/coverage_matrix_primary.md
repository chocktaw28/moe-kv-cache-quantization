# Coverage Matrix — PRIMARY (is_moe == true only)

## Row count feeding this matrix

| Population | Papers |
|---|---|
| `is_moe == "true"` as originally extracted (BASELINE) | **60** |
| + reclassified `unclear` -> `true` in the Step 1 pass | **+4** |
| **Total feeding the matrix below (ADJUSTED)** | **64** |

The 4 reclassified papers are `2605.27646`, `2507.04610`, `2405.14366`, `2607.08029`.
They are marked with a dagger (†) everywhere they appear in a cell listing, so the
original 60-paper baseline stays auditable. Section 6 gives the full
baseline-vs-adjusted delta; no cell's gap classification changes between the two.

Everything below is drawn ONLY from these 64 rows. The other 301 rows are in
`coverage_matrix_reference.md` and are NOT MoE-quantization coverage.

**Cells sum to more than 64.** A paper testing 2-bit, 3-bit and 4-bit on both an A100 and
an RTX 3090 is counted in every applicable (bit x arch x hardware) cell. Multi-label
totals: the 64 papers generate 215 (bit x arch x hw) cell placements.

## 1. Bucketing rules (check my assignments against these)

### 1.1 Bit-width buckets
```
Bit-width buckets are assigned by case-insensitive pattern matching on the raw
`bit_widths_tested` string. A paper is placed in EVERY bucket it matches
(multi-label), so bucket counts sum to MORE than the paper count.

  <=2-bit        : any of 1-bit,1.5,1.57,1.58,1.59,1.6,1.75,1.8,2-bit,2.0-2.5,W2,A2,INT2,
                   binary, ternary, trit, sub-2-bit, 2.06/2.08/2.25/2.54/2.57/2.75-bit
  3-bit          : 3-bit, W3, A3, INT3, 3.03, 3.25, 3.79, IQ/Q3, "3 bits"
  4-bit          : 4-bit, W4, A4, INT4, NVFP4, MXFP4, FP4, Q4, 4.0625, 4.5, any4
  8-bit          : 8-bit, W8, A8, INT8, FP8, E4M3, E5M2, Q8, 8/8/4
  16-bit+ / FP baseline : FP16, BF16, float16, half-precision, FP32, 16-bit
  mixed/adaptive : "mixed precision", "mixed-precision", "per-expert"/"bit allocation",
                   "runtime-switchable", "adaptive", or >=3 distinct integer widths in one row
  not stated     : the field begins with / is exactly "not stated", or is "not applicable",
                   or no numeric width pattern matched
```

### 1.2 Hardware buckets
```
Hardware buckets are assigned by case-insensitive pattern matching on the raw
`hardware_target` string. Multi-label, same summing caveat.

  datacenter GPU     : A100, H100, H200, GH200, B200, V100, A6000, A5000, A40, A800,
                       Ampere/Hopper/Blackwell (as datacenter class), "cluster", NVSwitch,
                       InfiniBand, Quadro RTX 8000, RTX 6000 Ada (workstation-class)
  consumer GPU       : RTX 3090, RTX 4090, RTX 5090, T4, "consumer-grade", "consumer hardware",
                       "commodity hardware"
  edge/mobile/NPU/FPGA: Jetson, Orin, Raspberry Pi, ESP32, NPU, PIM, FPGA, ZCU102, Alveo,
                       "edge", "mobile", "on-device", "embedded", Strix Halo, Ryzen AI,
                       Apple Silicon / M3 / M5 (as a laptop-edge target), chiplet, ReRAM, 22nm
  CPU-only           : "CPU" without an accompanying GPU token, SIMD, aarch64, x86-64, qnnpack, llama.cpp
  simulation-only    : "simulated", "fake-quant", "simulation"
  not stated         : field begins with / equals "not stated"

NOTE on Apple-Silicon laptops: rows naming MacBook / M3 / M5 / Metal are bucketed as
edge/mobile because the papers themselves frame them as local/edge deployment targets;
they are ALSO bucketed CPU-only where the row names CPU/SIMD kernels explicitly.
```

### 1.3 Architecture categories
```
PRIMARY-matrix architecture categories (assigned from `title` +
`model_architecture_tested` + `extraction_notes`, in-CSV text only, single-label,
first rule that fires wins):

  1. "MoE, non-LLM"            : the CSV says the tested MoE is a vision/audio/image model
                                 rather than a language model.
  2. "MoE (fine-grained, many small experts)" : the CSV names a model with >=64 experts per
                                 layer, or names a fine-grained/many-small-expert design
                                 (DeepSeek-MoE 64-expert, DeepSeek-V2/V3, Qwen3-MoE 128/512
                                 experts, OLMoE 64, Switch Transformer 64/128, Snowflake
                                 Arctic, ButterflyMoE 64/256, LightningLM 460 experts),
                                 or the title says "fine-grained".
  3. "MoE LLM"                 : any other MoE language model (default for MoE LLM papers,
                                 e.g. Mixtral-8x7B-only rows: 8 coarse experts).
  4. "MoE, arch not stated"    : is_moe==true but the CSV never names the tested model.

Rule 2 vs 3 is a judgment call made ONLY from expert counts stated in the CSV. Rows
matching both a coarse (Mixtral 8x7B) and a fine-grained (DeepSeek-MoE-16B, Qwen3-30B-A3B)
model are categorised fine-grained, because the fine-grained regime IS exercised.

REFERENCE-matrix architecture categories: "dense LLM" where the CSV names a concrete
dense model family (Llama/Qwen/Mistral/OPT/Phi/Gemma/GPT/Nemotron/etc.) or the notes say
dense; otherwise "unclear/not stated".
```

## 2. Marginals (papers, not cell placements)

**By bit-width bucket** (multi-label; a paper appears in each bucket it tests)

| Bucket | Papers |
|---|---|
| <=2-bit | 28 |
| 3-bit | 20 |
| 4-bit | 39 |
| 8-bit | 24 |
| 16-bit+/FP baseline | 22 |
| mixed/adaptive | 24 |
| not stated | 12 |

**By hardware bucket** (multi-label)

| Bucket | Papers |
|---|---|
| datacenter GPU | 30 |
| consumer GPU | 13 |
| edge/mobile/NPU/FPGA | 14 |
| CPU-only | 4 |
| simulation-only | 1 |
| not stated | 17 |

**By architecture category** (single-label; sums to 64)

| Category | Papers |
|---|---|
| MoE LLM | 11 |
| MoE (fine-grained, many small experts) | 42 |
| MoE, non-LLM | 3 |
| MoE, arch not stated | 8 |
| **total** | **64** |

## 3. Two-way slices (easier to read than the full 3-way)

### 3.1 bit-width x architecture

| bit-width \ arch | MoE LLM | MoE (fine-grained, many small experts) | MoE, non-LLM | MoE, arch not stated | row total |
|---|---|---|---|---|---|
| <=2-bit | 4 | 21 | 1 | 2 | 28 |
| 3-bit | 2 | 16 | 0 | 2 | 20 |
| 4-bit | 8 | 28 | 1 | 2 | 39 |
| 8-bit | 4 | 17 | 1 | 2 | 24 |
| 16-bit+/FP baseline | 4 | 17 | 1 | 0 | 22 |
| mixed/adaptive | 3 | 20 | 0 | 1 | 24 |
| not stated | 2 | 4 | 1 | 5 | 12 |

### 3.2 bit-width x hardware

| bit-width \ hardware | datacenter GPU | consumer GPU | edge/mobile/NPU/FPGA | CPU-only | simulation-only | not stated | row total |
|---|---|---|---|---|---|---|---|
| <=2-bit | 16 | 10 | 8 | 3 | 1 | 2 | 40 |
| 3-bit | 11 | 7 | 3 | 1 | 0 | 3 | 25 |
| 4-bit | 21 | 10 | 8 | 3 | 1 | 8 | 51 |
| 8-bit | 15 | 3 | 7 | 1 | 1 | 2 | 29 |
| 16-bit+/FP baseline | 14 | 5 | 3 | 2 | 0 | 4 | 28 |
| mixed/adaptive | 17 | 6 | 4 | 1 | 1 | 1 | 30 |
| not stated | 1 | 1 | 3 | 0 | 0 | 7 | 12 |

### 3.3 architecture x hardware

| arch \ hardware | datacenter GPU | consumer GPU | edge/mobile/NPU/FPGA | CPU-only | simulation-only | not stated | row total |
|---|---|---|---|---|---|---|---|
| MoE LLM | 6 | 3 | 1 | 0 | 0 | 3 | 13 |
| MoE (fine-grained, many small experts) | 23 | 10 | 7 | 3 | 1 | 10 | 54 |
| MoE, non-LLM | 1 | 0 | 2 | 1 | 0 | 0 | 4 |
| MoE, arch not stated | 0 | 0 | 4 | 0 | 0 | 4 | 8 |

## 4. Full 3-way matrix (non-empty cells only)

Empty cells are omitted here for readability; the complete grid including every zero cell
is in `coverage_matrix_primary.csv` (168 rows = 7 bit x 4 arch x 6 hw).

| bit-width | architecture | hardware | n | paper_ids |
|---|---|---|---|---|
| <=2-bit | MoE LLM | datacenter GPU | 3 | `2410.06270`, `2510.10962`, `2607.06601` |
| <=2-bit | MoE LLM | consumer GPU | 2 | `2410.06270`, `2510.10962` |
| <=2-bit | MoE LLM | not stated | 1 | `2507.04610`† |
| <=2-bit | MoE (fine-grained, many small experts) | datacenter GPU | 13 | `2406.08155`, `2503.21135`, `2506.13329`, `2508.02322`, `2509.25689`, `2511.15015`, `2602.11184`, `2604.06515`, `2604.10496`, `2605.09281`, `2605.23078`, `2606.00079`, `2606.04980` |
| <=2-bit | MoE (fine-grained, many small experts) | consumer GPU | 8 | `2505.05799`, `2508.01625`, `2511.15015`, `2601.13563`, `2603.19172`, `2605.09281`, `2606.17118`, `2608.08081` |
| <=2-bit | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 5 | `2509.25689`, `2601.13563`, `2603.19172`, `2608.08081`, `2608.08910` |
| <=2-bit | MoE (fine-grained, many small experts) | CPU-only | 2 | `2604.10496`, `2608.08910` |
| <=2-bit | MoE (fine-grained, many small experts) | simulation-only | 1 | `2603.19172` |
| <=2-bit | MoE (fine-grained, many small experts) | not stated | 1 | `2507.07145` |
| <=2-bit | MoE, non-LLM | edge/mobile/NPU/FPGA | 1 | `2511.11743` |
| <=2-bit | MoE, non-LLM | CPU-only | 1 | `2511.11743` |
| <=2-bit | MoE, arch not stated | edge/mobile/NPU/FPGA | 2 | `10.1109/ICEIC69189.2026.11386441`, `2607.20981` |
| 3-bit | MoE LLM | datacenter GPU | 1 | `2410.06270` |
| 3-bit | MoE LLM | consumer GPU | 1 | `2410.06270` |
| 3-bit | MoE LLM | not stated | 1 | `2507.04610`† |
| 3-bit | MoE (fine-grained, many small experts) | datacenter GPU | 10 | `2503.21135`, `2504.02658`, `2505.03804`, `2506.13329`, `2602.11184`, `2604.06515`, `2604.10496`, `2605.09281`, `2606.00079`, `2606.04980` |
| 3-bit | MoE (fine-grained, many small experts) | consumer GPU | 6 | `2505.03804`, `2505.05799`, `2508.01625`, `2605.09281`, `2606.17118`, `2608.08081` |
| 3-bit | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 1 | `2608.08081` |
| 3-bit | MoE (fine-grained, many small experts) | CPU-only | 1 | `2604.10496` |
| 3-bit | MoE (fine-grained, many small experts) | not stated | 2 | `2602.01037`, `2605.27646`† |
| 3-bit | MoE, arch not stated | edge/mobile/NPU/FPGA | 2 | `10.1109/ICEIC69189.2026.11386441`, `2607.20981` |
| 4-bit | MoE LLM | datacenter GPU | 6 | `2410.06270`, `2505.13840`, `2510.10962`, `2603.02731`, `2607.06601`, `2607.15810` |
| 4-bit | MoE LLM | consumer GPU | 2 | `2410.06270`, `2510.10962` |
| 4-bit | MoE LLM | not stated | 2 | `2405.14366`†, `2507.04610`† |
| 4-bit | MoE (fine-grained, many small experts) | datacenter GPU | 15 | `2406.08155`, `2503.21135`, `2504.02658`, `2505.03804`, `2506.13329`, `2509.25689`, `2511.14102`, `2511.15015`, `2602.11184`, `2602.11937`, `2604.10496`, `2605.23078`, `2606.04980`, `2606.25092`, `2607.04371` |
| 4-bit | MoE (fine-grained, many small experts) | consumer GPU | 8 | `2505.03804`, `2505.05799`, `2508.01625`, `2511.14102`, `2511.15015`, `2603.19172`, `2606.17118`, `2608.08081` |
| 4-bit | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 5 | `2509.25689`, `2603.19172`, `2607.09686`, `2608.08081`, `2608.08910` |
| 4-bit | MoE (fine-grained, many small experts) | CPU-only | 3 | `2511.14102`, `2604.10496`, `2608.08910` |
| 4-bit | MoE (fine-grained, many small experts) | simulation-only | 1 | `2603.19172` |
| 4-bit | MoE (fine-grained, many small experts) | not stated | 6 | `2601.22101`, `2602.01037`, `2605.27646`†, `2606.05688`, `2606.18304`, `2608.11212` |
| 4-bit | MoE, non-LLM | edge/mobile/NPU/FPGA | 1 | `2506.08496` |
| 4-bit | MoE, arch not stated | edge/mobile/NPU/FPGA | 2 | `2607.08029`†, `2607.20981` |
| 8-bit | MoE LLM | datacenter GPU | 3 | `2603.02731`, `2607.06601`, `2607.15810` |
| 8-bit | MoE LLM | not stated | 1 | `2405.14428` |
| 8-bit | MoE (fine-grained, many small experts) | datacenter GPU | 12 | `2406.08155`, `2503.21135`, `2504.02658`, `2506.13329`, `2509.25689`, `2511.02302`, `2602.11937`, `2602.19938`, `2604.10496`, `2606.00079`, `2606.07404`, `2607.04371` |
| 8-bit | MoE (fine-grained, many small experts) | consumer GPU | 3 | `2505.05799`, `2603.19172`, `2608.08081` |
| 8-bit | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 4 | `2509.25689`, `2603.19172`, `2607.09686`, `2608.08081` |
| 8-bit | MoE (fine-grained, many small experts) | CPU-only | 1 | `2604.10496` |
| 8-bit | MoE (fine-grained, many small experts) | simulation-only | 1 | `2603.19172` |
| 8-bit | MoE (fine-grained, many small experts) | not stated | 1 | `2601.22101` |
| 8-bit | MoE, non-LLM | edge/mobile/NPU/FPGA | 1 | `2506.08496` |
| 8-bit | MoE, arch not stated | edge/mobile/NPU/FPGA | 2 | `2607.08029`†, `2607.20981` |
| 16-bit+/FP baseline | MoE LLM | datacenter GPU | 4 | `2505.13840`, `2603.02731`, `2607.06601`, `2607.15810` |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | datacenter GPU | 10 | `2406.08155`, `2504.02658`, `2509.25689`, `2511.02302`, `2511.14102`, `2511.15015`, `2602.11937`, `2602.19938`, `2606.25092`, `2607.04371` |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | consumer GPU | 5 | `2505.05799`, `2511.14102`, `2511.15015`, `2601.13563`, `2606.17118` |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 2 | `2509.25689`, `2601.13563` |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | CPU-only | 1 | `2511.14102` |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | not stated | 4 | `2601.22101`, `2602.01037`, `2606.05688`, `2608.11212` |
| 16-bit+/FP baseline | MoE, non-LLM | edge/mobile/NPU/FPGA | 1 | `2511.11743` |
| 16-bit+/FP baseline | MoE, non-LLM | CPU-only | 1 | `2511.11743` |
| mixed/adaptive | MoE LLM | datacenter GPU | 2 | `2410.06270`, `2607.06601` |
| mixed/adaptive | MoE LLM | consumer GPU | 1 | `2410.06270` |
| mixed/adaptive | MoE LLM | not stated | 1 | `2507.04610`† |
| mixed/adaptive | MoE (fine-grained, many small experts) | datacenter GPU | 15 | `2406.08155`, `2503.21135`, `2504.02658`, `2506.13329`, `2508.02322`, `2509.25689`, `2511.02302`, `2602.11184`, `2602.19938`, `2604.06515`, `2604.10496`, `2605.23078`, `2606.00079`, `2606.04980`, `2607.16184` |
| mixed/adaptive | MoE (fine-grained, many small experts) | consumer GPU | 5 | `2505.05799`, `2508.01625`, `2603.19172`, `2606.17118`, `2608.08081` |
| mixed/adaptive | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 3 | `2509.25689`, `2603.19172`, `2608.08081` |
| mixed/adaptive | MoE (fine-grained, many small experts) | CPU-only | 1 | `2604.10496` |
| mixed/adaptive | MoE (fine-grained, many small experts) | simulation-only | 1 | `2603.19172` |
| mixed/adaptive | MoE, arch not stated | edge/mobile/NPU/FPGA | 1 | `2607.20981` |
| not stated | MoE LLM | consumer GPU | 1 | `2607.26316` |
| not stated | MoE LLM | edge/mobile/NPU/FPGA | 1 | `2603.15589` |
| not stated | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 1 | `2601.21198` |
| not stated | MoE (fine-grained, many small experts) | not stated | 3 | `10.18653/v1/2025.findings-acl.1386`, `10.18653/v1/2026.acl-long.982`, `ss:b49d11d5be5f05af7767b1a28c6ef0f12e3b2abf` |
| not stated | MoE, non-LLM | datacenter GPU | 1 | `2607.14334` |
| not stated | MoE, arch not stated | edge/mobile/NPU/FPGA | 1 | `10.1109/TCCN.2026.3694872` |
| not stated | MoE, arch not stated | not stated | 4 | `10.1016/j.patcog.2026.114286`, `10.1109/DAC63849.2025.11132853`, `10.1609/aaai.v40i32.39899`, `2411.19402` |

† = reclassified from `unclear` in Step 1; not part of the original 60-paper baseline.

## 5. CANDIDATE GAPS — empty and near-empty cells (0-2 papers)

**Read the denominator first.** 64 papers are spread over 168 cells and generate only 215 cell
placements, so the *average* cell holds under 0.2 papers and **102 of 168 cells are empty.**
**Emptiness is the default state of this matrix, not a finding.** Enumerating every empty
cell would produce a page of noise and would badly mislead: it would make a small field look
like a field full of gaps. This section therefore reports only cells where emptiness is
*interpretable* — where the surrounding row and column are populated enough that the hole
means something, or where a reader would specifically expect the combination to exist.

A negative result up front, because it constrains everything downstream: **the well-covered
regions are broader than the small corpus size would suggest.** Sub-4-bit MoE quantization
is populated at every hardware bucket, not concentrated in datacenters as one might expect
(<=2-bit: 16 datacenter, 10 consumer GPU, 8 edge/mobile, 3 CPU-only). Mixed-precision
per-expert bit allocation is the single densest theme. Anyone proposing 'low-bit MoE on
constrained hardware' as an open angle should read §4 first — it is contested.

Each is tagged with exactly one of:

- **(i) GENUINE gap** — the combination is plausible, would have been surfaced by the run's
  query terms if it existed, and the adjacent cells are populated enough that its emptiness
  is informative.
- **(ii) SEARCH-CORPUS ARTIFACT** — the run's query terms (`logs/query_log.jsonl`) would not
  reliably have surfaced this combination, so absence is evidence about the search, not the
  literature.
- **(iii) STATISTICAL SPARSITY** — the combination is fine and may well be covered in the
  real literature; there are simply not enough papers here to populate the cell. **This is a
  much weaker claim than (i) and is not rounded up to it.**

The tally below is deliberately lopsided toward (iii). That is the honest result for a
64-paper matrix.

| Cell / cross-cut | n | classification |
|---|---|---|
| MoE, non-LLM (whole row, all bit-widths, all hardware) | 3 | (ii) SEARCH-CORPUS ARTIFACT |
| 3-bit x MoE LLM (coarse) x edge/mobile — and the coarse-MoE row generally | 0 | (iii) STATISTICAL SPARSITY — and partly an artifact of my own categorisation |
| simulation-only column (all cells) | 1 | (ii) SEARCH-CORPUS ARTIFACT — definitional/extraction-side |
| not-stated bit-width row (all architectures, all hardware) | 12 | (ii) SEARCH-CORPUS ARTIFACT — extraction-side |
| KV-cache quantization x MoE (any bit-width, any hardware) | 6 | (i) plausible GENUINE gap — the strongest gap-like signal in this matrix |
| aggressive (<=3-bit) quantization OF THE ROUTER/GATE itself | 0 | (i) plausible GENUINE gap |

Tally: **2 classified (i)**, **3 classified (ii)**, **1 classified (iii)**. Note that the
two (i) findings are *cross-cuts* (KV-cache; router precision) rather than bit x arch x hw
cells. That is itself a result worth stating plainly: **the three-way matrix the brief asked
for does not, on inspection, contain a convincing empty cell.** Sub-4-bit MoE is populated
at every hardware bucket (<=2-bit: 16 datacenter, 10 consumer, 8 edge, 3 CPU), 4-bit is the
densest region, and the marginals are broad. The interesting holes are in *what part of the
model* is quantized, which the requested axes do not resolve. Sections 5.1-5.6 report the
axes as asked; the two cross-cuts are reported alongside because suppressing them would
mean reporting only the uninformative version of the analysis.

### MoE, non-LLM (whole row, all bit-widths, all hardware)

**n = 3** — `2506.08496`, `2511.11743`, `2607.14334`

**Classification: (ii) SEARCH-CORPUS ARTIFACT**

The non-LLM MoE row holds 3 papers total, and they are mutually unrelated (a ViT-MoE FPGA accelerator, a small audio-classification model, a learned image codec). This is squarely an artifact of the search design, not a statement about the literature. Every query term in `logs/query_log.jsonl` pairs a quantization term with an LLM or MoE-LLM term — `mixture of experts quantization`, `MoE quantization`, `expert quantization`, `sparse MoE low-bit` (broadened to `"sparse mixture of experts" AND quantization`) — and the Semantic Scholar filter additionally required the topical term to appear in the TITLE. Vision, speech and recommender MoE quantization work that does not put an LLM word in its title is excluded by construction. The three present arrived incidentally on keyword overlap, and `run_summary.md` §8 flags two of them as probable false positives. **Nothing about non-LLM MoE quantization can be inferred from this corpus in either direction.**

### 3-bit x MoE LLM (coarse) x edge/mobile — and the coarse-MoE row generally

**n = 0** — empty

**Classification: (iii) STATISTICAL SPARSITY — and partly an artifact of my own categorisation**

Empty. But 'MoE LLM (coarse)' is a residual category of only 11 papers produced by my architecture rule: any paper that also tests a fine-grained model (DeepSeek-MoE, Qwen3-MoE, OLMoE, Switch) is categorised fine-grained, so the coarse bucket collects only Mixtral-only or unnamed-MoE papers. Its holes measure my bucketing granularity, not the literature. **No research signal — do not build an angle on this cell.**

### simulation-only column (all cells)

**n = 1** — `2603.19172`

**Classification: (ii) SEARCH-CORPUS ARTIFACT — definitional/extraction-side**

1 paper (`2603.19172`, which states 'simulated memory constraints'). This is near-empty for a definitional reason: papers seldom describe themselves as simulation-only in the sentence an extractor reads for `hardware_target`, and `2608.11212` uses fake-quant throughout yet reports no hardware at all. The bucket under-fills mechanically. Do not read it as 'nobody simulates'.

### not-stated bit-width row (all architectures, all hardware)

**n = 12** — `10.1016/j.patcog.2026.114286`, `10.1109/DAC63849.2025.11132853`, `10.1109/TCCN.2026.3694872`, `10.1609/aaai.v40i32.39899`, `10.18653/v1/2025.findings-acl.1386`, `10.18653/v1/2026.acl-long.982`, `2411.19402`, `2601.21198`, `2603.15589`, `2607.14334`, `2607.26316`, `ss:b49d11d5be5f05af7767b1a28c6ef0f12e3b2abf`

**Classification: (ii) SEARCH-CORPUS ARTIFACT — extraction-side**

12 of 64 primary papers (19%) report NO bit-width. This is an extraction gap, not a research gap, and it is the single largest threat to reading anything off this matrix. Most are DOI-only rows (Elsevier, IEEE, AAAI, ACL Anthology) that `run_summary.md` §7 records as unreachable for full text — `10.1016/j.patcog.2026.114286` returned a literally empty abstract. Their bit-widths exist in the papers but not in this CSV. **Every sparse cell below must be discounted by the possibility that some of these 12 belong in it**, and three of them (`10.1016/j.patcog.2026.114286`, `10.18653/v1/2025.findings-acl.1386`, `10.18653/v1/2026.acl-long.982`) are core MoE-quantization papers by title.

### KV-cache quantization x MoE (any bit-width, any hardware)

**n = 6** — `2405.14366`, `2602.11937`, `2605.27646`, `2607.06601`, `2608.08081`, `2608.11212`

**Classification: (i) plausible GENUINE gap — the strongest gap-like signal in this matrix**

This is a cross-cut rather than a single matrix cell, identified by regex on title + `method_name` + `bit_widths_tested`. Only **6 of 64** primary papers touch KV-cache quantization at all, and of those, only `2608.11212` and `2607.06601` treat the interaction between routing and KV precision as the object of study; `2608.08081` and `2602.11937` quantize the KV cache incidentally alongside expert weights, and `2405.14366`/`2605.27646` are architecture-agnostic dense methods that happen to include an MoE model. Against this: **117 reference-matrix papers have 'KV cache' in the title.** A 117-to-6 ratio, where the 6 are mostly incidental, is the clearest asymmetry in the whole dataset. Why (i) and not (ii): Cluster D ran `KV cache quantization` (40 arXiv hits, 333 SS hits) and was **uncapped**, so MoE KV-cache papers had an unobstructed route into this corpus and 117 dense ones took it. The search cannot explain the asymmetry. `run_summary.md` §8 reaches the same conclusion independently, calling it 'a systemic pattern'. Why not stronger than 'plausible': the reclassification pass (§7) leaves `2605.17757` (GLM-4.7 358B, INT2 KV) and `2510.01290` (GPT-OSS) unresolved, and both are probably MoE KV-cache papers — resolving them would move the count to ~8 and soften, though not erase, the asymmetry.

### aggressive (<=3-bit) quantization OF THE ROUTER/GATE itself

**n = 0** — empty

**Classification: (i) plausible GENUINE gap**

**Empty, and informatively so.** 10 primary papers discuss the router or gating network in a quantization context, and *every one of them protects it*: `2410.06270` keeps gating at 4-bit while experts go to 1.57 bits; `2506.13329` pins the router at W8A8 while experts run W2A4; `2606.04980` holds attention and router at 4-bit against 1-bit experts; `2511.14102` keeps gating networks in FP16; `2608.11212` reads a *protected BF16 gate*. The convergence is unanimous across 5 independent groups and 2024-2026. No paper in the primary matrix reports quantizing the router below 4 bits, and none reports the accuracy cost of trying. Why (i) rather than (iii): this is not a thinly-populated cell, it is a *populated neighbourhood with a consistent hole* — the papers all had the router in hand, made an explicit precision decision about it, and all made the same one. Why not a stronger claim: the router is a small fraction of MoE parameters, so the obvious explanation is that quantizing it is simply not worth the memory saved. That would make this a *well-founded engineering consensus* rather than an unexplored gap — and the corpus contains no paper that tests the consensus, which is precisely what makes it interesting and also what prevents ruling out the boring explanation.

### Cells that are NOT gaps — the dense regions

For contrast, these are the best-covered combinations, and any proposed angle overlapping
them is contested territory rather than open ground:

| bit-width | architecture | hardware | n |
|---|---|---|---|
| mixed/adaptive | MoE (fine-grained, many small experts) | datacenter GPU | 15 |
| 4-bit | MoE (fine-grained, many small experts) | datacenter GPU | 15 |
| <=2-bit | MoE (fine-grained, many small experts) | datacenter GPU | 13 |
| 8-bit | MoE (fine-grained, many small experts) | datacenter GPU | 12 |
| 3-bit | MoE (fine-grained, many small experts) | datacenter GPU | 10 |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | datacenter GPU | 10 |
| <=2-bit | MoE (fine-grained, many small experts) | consumer GPU | 8 |
| 4-bit | MoE (fine-grained, many small experts) | consumer GPU | 8 |
| 4-bit | MoE LLM | datacenter GPU | 6 |
| 4-bit | MoE (fine-grained, many small experts) | not stated | 6 |
| 3-bit | MoE (fine-grained, many small experts) | consumer GPU | 6 |
| mixed/adaptive | MoE (fine-grained, many small experts) | consumer GPU | 5 |
| <=2-bit | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 5 |
| 4-bit | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 5 |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | consumer GPU | 5 |
| not stated | MoE, arch not stated | not stated | 4 |
| 8-bit | MoE (fine-grained, many small experts) | edge/mobile/NPU/FPGA | 4 |
| 16-bit+/FP baseline | MoE LLM | datacenter GPU | 4 |
| 16-bit+/FP baseline | MoE (fine-grained, many small experts) | not stated | 4 |

## 6. Baseline (60) vs adjusted (64) — the reclassification delta

Per the brief, the matrix is auditable both ways. The 4 papers added:

| paper_id | title | why reclassified | cells it lands in |
|---|---|---|---|
| `2605.27646` | Hurwitz Quaternion Multiplicative Quantization for KV Cache  | extraction_notes state the abstract explicitly evaluates gpt-oss-20b and describes it as 'sparse MoE'; an MoE architecture is therefore confirmed-test | 3-bit / MoE (fine-grained, many small experts) / not stated; 4-bit / MoE (fine-grained, many small experts) / not stated |
| `2507.04610` | any4: Learned 4-bit Numeric Representation for LLMs | model_architecture_tested lists Mixtral; extraction_notes call Mixtral 'an MoE model' explicitly. MoE architecture confirmed-tested in-CSV. | 3-bit / MoE LLM / not stated; 4-bit / MoE LLM / not stated; <=2-bit / MoE LLM / not stated; mixed/adaptive / MoE LLM / not stated |
| `2405.14366` | MiniCache: KV Cache Compression in Depth Dimension for Large | model_architecture_tested lists Mixtral; extraction_notes call Mixtral 'an MoE model' explicitly. MoE architecture confirmed-tested in-CSV. | 4-bit / MoE LLM / not stated |
| `2607.08029` | Rethinking Small VLM Quantization: From Component-Wise Analy | extraction_notes state the paper 'explicitly compares MoE vs dense backbone quantization sensitivity in small VLMs on edge hardware (Jetson)'. An MoE  | 4-bit / MoE, arch not stated / edge/mobile/NPU/FPGA; 8-bit / MoE, arch not stated / edge/mobile/NPU/FPGA |

**Does adding them change any conclusion? No.** All four are papers where an MoE model is
one of several tested architectures and the method is not MoE-specific — they add breadth
at 3-bit/4-bit on unstated or edge hardware, and they do not fill any of the candidate-gap
cells in Section 5. Marginal deltas (baseline -> adjusted): <=2-bit 27->28, 3-bit 18->20, 4-bit 35->39, 8-bit 23->24, 16-bit+/FP baseline 22->22, mixed/adaptive 23->24, not stated 12->12.

Full baseline-only grid: re-run `scratch/analyze.py`, which emits both `cells_base` and
`cells_adj`; the dagger markers in Section 4 let you subtract the 4 by hand.

## 7. Reclassification pass — full accounting of the 63 `unclear` rows

Method: for each of the 63, I read `model_architecture_tested` and `extraction_notes` and
asked whether the CSV **itself** already settles the question. No papers were re-fetched and
no outside knowledge was used to fill gaps — that constraint is why the reclassified count
is small.

| Outcome | Count |
|---|---|
| `unclear` -> **true** | 4 |
| `unclear` -> **false** | 4 |
| left `unclear` | 55 |
| **total** | **63** |

Resulting distribution: is_moe true 60->**64**, false 242->**246**, unclear 63->**55**.

**-> true (4).** Each names an MoE model that the CSV's own text identifies as MoE:

- `2605.27646` — extraction_notes state the abstract explicitly evaluates gpt-oss-20b and describes it as 'sparse MoE'; an MoE architecture is therefore confirmed-tested by in-CSV text.
- `2507.04610` — model_architecture_tested lists Mixtral; extraction_notes call Mixtral 'an MoE model' explicitly. MoE architecture confirmed-tested in-CSV.
- `2405.14366` — model_architecture_tested lists Mixtral; extraction_notes call Mixtral 'an MoE model' explicitly. MoE architecture confirmed-tested in-CSV.
- `2607.08029` — extraction_notes state the paper 'explicitly compares MoE vs dense backbone quantization sensitivity in small VLMs on edge hardware (Jetson)'. An MoE backbone is confirmed-tested in-CSV, even though the paper is comparative rather than MoE-exclusive.

**-> false (4).** Each is stated in-CSV to test something that is not an MoE model:

- `2602.02958` — extraction_notes state this is an autoregressive VIDEO-DIFFUSION KV-cache paper, not a text-LLM/MoE paper; named models (LongCat Video, HY WorldPlay, Self Forcing) are video generators. No MoE architecture tested.
- `2603.27469` — extraction_notes state this is a video-generation (Wan2.1 Self-Forcing) KV-cache benchmarking study, explicitly 'Not MoE-specific'; the named stack is not an MoE LLM.
- `2602.22592` — extraction_notes state pQuant uses an MoE-LIKE sparse-expert mechanism INTERNAL to its own quantizer and is 'not evaluated on existing MoE base models'. No MoE model is tested, so false under the field's 'tests an MoE architecture' semantics.
- `2604.19528` — extraction_notes state this is a reproducibility note comparing two vector-quantization methods (RaBitQ, TurboQuant) for nearest-neighbour search, 'not an LLM/MoE quantization method paper itself'. No model tested at all.

**Deliberately left `unclear` — the near-misses.** These are the rows most likely to be
MoE and are logged so the 55 remaining are not silently dropped. Every one would have
required outside knowledge about a model's architecture, which the brief forbids:

- `2510.01290` — extraction_notes call GPT-OSS 'a known MoE architecture' but flag that this is outside-knowledge inference, not an in-abstract statement. Applying it would require outside knowledge -> left unclear.
- `2605.17757` — extraction_notes say GLM-4.7 (358B) is 'very likely an MoE model in practice' but the source text does not state it. Outside knowledge required -> left unclear.
- `2604.18556` — Kimi-K2.5 is described in-CSV as a trillion-scale MoE the method 'scales to', but the CSV states primary evaluation is dense Llama; ambiguous whether the MoE was actually evaluated -> left unclear.
- `2605.19645` — 'eight LLMs' unnamed -> cannot be resolved from the CSV.
- `2606.21257` — openPangu 1B/7B architecture type not stated in-CSV -> cannot resolve without outside knowledge.

**The residual risk this leaves.** `2510.01290` (GPT-OSS), `2605.17757` (GLM-4.7 358B) and
`2604.18556` (Kimi-K2.5) almost certainly DO test MoE models; a human with domain knowledge
would likely move all three to `true`. That would make the primary denominator ~67 rather
than 64 and would add sub-4-bit KV-cache MoE evidence specifically (`2605.17757` is INT2
KV cache, `2604.18556` is 2-3 bit). **This materially weakens any sub-4-bit MoE gap claim**
and is carried forward into the Step 4 counter-evidence.

The remaining ~50 `unclear` rows are, on inspection, dominated by dense-model KV-cache
papers whose abstracts never named a model ('three models', 'diverse LLMs', 'eight LLMs').
They are unresolvable from this CSV and are more likely dense than MoE, but that is an
expectation, not a finding.

Decisions logged to `logs/is_moe_reclassification.jsonl` (13 entries: 8 reclassified, 5 explicit leave-as-unclear).

