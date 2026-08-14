# Coverage Matrix — REFERENCE (is_moe false / unclear)

## What this is, and what it is not

**Row count: 301 papers** (365 total - 64 in the primary matrix).

Composition: `is_moe == "false"` 246 (242 as extracted + 4 reclassified from `unclear`), `is_moe == "unclear"` 55 (63 - 4 moved to true - 4 moved to false).

**This matrix is CONTEXT ONLY.** It measures how large the adjacent dense-model
quantization field is. It is **not** MoE-quantization coverage, it must **not** be merged
into the primary matrix, and an empty primary cell must **not** be treated as filled
because the corresponding reference cell is populated. A well-populated reference cell next
to an empty primary cell means the *technique* exists but has not been shown on MoE — which
is the interesting case, not a reason to dismiss the primary hole.

**Severe sampling caveat.** Most of this population passed through the B/C citation cap
(`run_summary.md` §3): 201 of 291 B/C-only candidates were DISCARDED, and in 2026H1 alone
117 candidates were cut to 15. Recent papers were excluded preferentially because they had
fewer citations. Counts here are a floor, badly and non-uniformly undercounted toward the
present. Do not compute rates or trends from this matrix.

**Cells sum to more than 301** for the same multi-label reason as the primary matrix (478 total cell placements).

Bucketing rules for bit-width and hardware are identical to the primary matrix — see
`coverage_matrix_primary.md` §1.1 and §1.2. Architecture is bucketed differently:

```
dense LLM         : model_architecture_tested names a concrete dense family
                    (Llama/Qwen/Mistral/OPT/Phi/Gemma/Falcon/Pythia/Mamba/RWKV/
                     Nemotron/GLM/InternLM/openPangu/ViT/SigLIP/BERT/T5/...)
unclear/not stated: the field is 'not stated' or names no identifiable architecture
```

## Marginals

| Bit-width bucket | Papers |
|---|---|
| <=2-bit | 101 |
| 3-bit | 55 |
| 4-bit | 101 |
| 8-bit | 42 |
| 16-bit+/FP baseline | 19 |
| mixed/adaptive | 33 |
| not stated | 101 |

| Hardware bucket | Papers |
|---|---|
| datacenter GPU | 19 |
| consumer GPU | 15 |
| edge/mobile/NPU/FPGA | 42 |
| CPU-only | 9 |
| simulation-only | 0 |
| not stated | 231 |

| Architecture | Papers |
|---|---|
| dense LLM | 129 |
| unclear/not stated | 172 |
| **total** | **301** |

## Two-way slices

### bit-width x architecture

| bit-width \ arch | dense LLM | unclear/not stated | row total |
|---|---|---|---|
| <=2-bit | 50 | 51 | 101 |
| 3-bit | 29 | 26 | 55 |
| 4-bit | 56 | 45 | 101 |
| 8-bit | 23 | 19 | 42 |
| 16-bit+/FP baseline | 12 | 7 | 19 |
| mixed/adaptive | 17 | 16 | 33 |
| not stated | 28 | 73 | 101 |

### bit-width x hardware

| bit-width \ hardware | datacenter GPU | consumer GPU | edge/mobile/NPU/FPGA | CPU-only | simulation-only | not stated | row total |
|---|---|---|---|---|---|---|---|
| <=2-bit | 6 | 7 | 12 | 2 | 0 | 79 | 106 |
| 3-bit | 3 | 5 | 6 | 0 | 0 | 44 | 58 |
| 4-bit | 9 | 7 | 17 | 0 | 0 | 72 | 105 |
| 8-bit | 4 | 3 | 8 | 1 | 0 | 29 | 45 |
| 16-bit+/FP baseline | 6 | 4 | 5 | 0 | 0 | 8 | 23 |
| mixed/adaptive | 1 | 3 | 7 | 0 | 0 | 24 | 35 |
| not stated | 5 | 1 | 17 | 6 | 0 | 77 | 106 |

### architecture x hardware

| arch \ hardware | datacenter GPU | consumer GPU | edge/mobile/NPU/FPGA | CPU-only | simulation-only | not stated | row total |
|---|---|---|---|---|---|---|---|
| dense LLM | 11 | 8 | 18 | 3 | 0 | 97 | 137 |
| unclear/not stated | 8 | 7 | 24 | 6 | 0 | 134 | 179 |

## Full 3-way matrix (non-empty cells; counts only)

Paper IDs are omitted here because several cells hold 20+ papers and this matrix is
context, not evidence. The complete grid WITH paper_ids, including zero cells, is in
`coverage_matrix_reference.csv`.

| bit-width | architecture | hardware | n |
|---|---|---|---|
| <=2-bit | dense LLM | datacenter GPU | 5 |
| <=2-bit | dense LLM | consumer GPU | 4 |
| <=2-bit | dense LLM | edge/mobile/NPU/FPGA | 6 |
| <=2-bit | dense LLM | CPU-only | 2 |
| <=2-bit | dense LLM | not stated | 36 |
| <=2-bit | unclear/not stated | datacenter GPU | 1 |
| <=2-bit | unclear/not stated | consumer GPU | 3 |
| <=2-bit | unclear/not stated | edge/mobile/NPU/FPGA | 6 |
| <=2-bit | unclear/not stated | not stated | 43 |
| 3-bit | dense LLM | datacenter GPU | 2 |
| 3-bit | dense LLM | consumer GPU | 3 |
| 3-bit | dense LLM | edge/mobile/NPU/FPGA | 5 |
| 3-bit | dense LLM | not stated | 21 |
| 3-bit | unclear/not stated | datacenter GPU | 1 |
| 3-bit | unclear/not stated | consumer GPU | 2 |
| 3-bit | unclear/not stated | edge/mobile/NPU/FPGA | 1 |
| 3-bit | unclear/not stated | not stated | 23 |
| 4-bit | dense LLM | datacenter GPU | 6 |
| 4-bit | dense LLM | consumer GPU | 4 |
| 4-bit | dense LLM | edge/mobile/NPU/FPGA | 10 |
| 4-bit | dense LLM | not stated | 39 |
| 4-bit | unclear/not stated | datacenter GPU | 3 |
| 4-bit | unclear/not stated | consumer GPU | 3 |
| 4-bit | unclear/not stated | edge/mobile/NPU/FPGA | 7 |
| 4-bit | unclear/not stated | not stated | 33 |
| 8-bit | dense LLM | datacenter GPU | 4 |
| 8-bit | dense LLM | consumer GPU | 3 |
| 8-bit | dense LLM | edge/mobile/NPU/FPGA | 3 |
| 8-bit | dense LLM | not stated | 16 |
| 8-bit | unclear/not stated | edge/mobile/NPU/FPGA | 5 |
| 8-bit | unclear/not stated | CPU-only | 1 |
| 8-bit | unclear/not stated | not stated | 13 |
| 16-bit+/FP baseline | dense LLM | datacenter GPU | 4 |
| 16-bit+/FP baseline | dense LLM | consumer GPU | 4 |
| 16-bit+/FP baseline | dense LLM | edge/mobile/NPU/FPGA | 3 |
| 16-bit+/FP baseline | dense LLM | not stated | 5 |
| 16-bit+/FP baseline | unclear/not stated | datacenter GPU | 2 |
| 16-bit+/FP baseline | unclear/not stated | edge/mobile/NPU/FPGA | 2 |
| 16-bit+/FP baseline | unclear/not stated | not stated | 3 |
| mixed/adaptive | dense LLM | datacenter GPU | 1 |
| mixed/adaptive | dense LLM | consumer GPU | 3 |
| mixed/adaptive | dense LLM | edge/mobile/NPU/FPGA | 4 |
| mixed/adaptive | dense LLM | not stated | 11 |
| mixed/adaptive | unclear/not stated | edge/mobile/NPU/FPGA | 3 |
| mixed/adaptive | unclear/not stated | not stated | 13 |
| not stated | dense LLM | datacenter GPU | 1 |
| not stated | dense LLM | edge/mobile/NPU/FPGA | 3 |
| not stated | dense LLM | CPU-only | 1 |
| not stated | dense LLM | not stated | 24 |
| not stated | unclear/not stated | datacenter GPU | 4 |
| not stated | unclear/not stated | consumer GPU | 1 |
| not stated | unclear/not stated | edge/mobile/NPU/FPGA | 14 |
| not stated | unclear/not stated | CPU-only | 5 |
| not stated | unclear/not stated | not stated | 53 |

## The one comparison worth making

| Bucket | PRIMARY (MoE, n=64) | REFERENCE (n=301) | ratio |
|---|---|---|---|
| <=2-bit | 28 | 101 | 1 : 3.6 |
| 3-bit | 20 | 55 | 1 : 2.8 |
| 4-bit | 39 | 101 | 1 : 2.6 |
| 8-bit | 24 | 42 | 1 : 1.8 |
| 16-bit+/FP baseline | 22 | 19 | 1 : 0.9 |
| mixed/adaptive | 24 | 33 | 1 : 1.4 |
| not stated | 12 | 101 | 1 : 8.4 |

The dense/reference field is roughly an order of magnitude larger at every bit-width, and
that ratio is measured AFTER the cap threw away 201 reference papers — the true ratio is
larger still. The practical reading: for essentially every quantization technique in this
corpus, the dense-model version exists and the MoE version is either absent or represented
by a handful of papers. That is the structural fact underlying every candidate angle in
`gap_analysis.md`, and it is also why 'no MoE paper does X' is such weak evidence here —
the MoE column is thin everywhere, uniformly.

