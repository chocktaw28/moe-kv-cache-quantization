# MoE KV-Cache Quantization

An independent research project investigating KV-cache quantization for
Mixture-of-Experts (MoE) language models.

Nearly all KV-cache quantization research targets dense architectures. This project
asks whether that body of work transfers cleanly to MoE models, or whether MoE
architecture creates conditions that architecture-agnostic methods don't account
for.

This repository is the **public methodology record**: how the literature corpus was
built, how baselines were reproduced, and what was verified rather than assumed.
Strategic analysis and the specific research hypothesis are held in a private
working repository and are not published here while the work is in progress.

---

## Status

| Phase | Description | State |
|---|---|---|
| 0 | Literature review — 365-paper corpus, coverage analysis | ✅ Complete |
| 1 | Gap analysis and hypothesis formalisation | ✅ Complete (held privately) |
| 2 | Baseline reproduction — OSCAR on Qwen3-8B, then OLMoE | 🔄 In progress |
| 3+ | Diagnostic pilot, method design, evaluation, write-up | ⏳ Not started |

**No experimental results are published yet.** Phase 2 has validated its
environment and calibration artifacts; it has not yet produced accuracy numbers.

---

## What's here

### [`literature-review/`](literature-review/)

A structured review of KV-cache quantization research across dense and MoE
architectures. 365 papers (2024–2026), built via arXiv and Semantic Scholar API
queries with structured field extraction under a strict no-fabrication policy —
every field not explicitly stated in a source paper is marked "not stated" rather
than inferred.

Headline finding: only 6 of 67 confirmed-MoE papers in the corpus address KV-cache
quantization at all, versus 117 dense-model papers on the same topic. The asymmetry
was tested against two alternative explanations — search-design artifact and
corpus-labelling error — and survives both. Ambiguous architecture classifications
were resolved by hand and logged.

### [`reproduction/`](reproduction/)

Reproduction of [OSCAR](https://github.com/FutureMLS-Lab/OSCAR)
([arXiv:2605.17757](https://arxiv.org/abs/2605.17757)), currently the strongest
open deployable 2-bit KV-cache quantization method, on a single consumer GPU
rather than the H100-class hardware it targets.

Contains environment notes (including a silent FlashAttention-3 failure on sm89
hardware and its fix), a portability patch, and a validation procedure for
rotation calibration checkpoints that runs on CPU and requires no GPU.

---

## Method notes

A few commitments that shape how this repository reads:

**Claims are checked, not asserted.** A pipeline exiting without errors is not
evidence that it produced correct output. Where a claim can be verified cheaply, it
is — and the verification is published alongside the claim.

**Predictions are recorded before results, and wrong ones stay visible.** Several
documents here contain hypotheses that turned out to be incorrect, kept alongside
what was actually found. A methodology record that only shows correct guesses isn't
much of a record.

**Negative results are reportable.** Falsification conditions for the core
hypothesis were written before any data was collected, with pre-registered
thresholds. If the hypothesis fails, that outcome will be reported.

---

## Related work

This project builds directly on OSCAR, by Zhongzhu Zhou, Donglin Zhuang, Jisen Li,
Ziyan Chen, Shuaiwen Leon Song, Ben Athiwaratkun, and Xiaoxia Wu (MIT License). The
reproduction work here is a portability exercise; all method credit is theirs.

```bibtex
@misc{zhou2026oscar,
  title={OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization},
  author={Zhongzhu Zhou and Donglin Zhuang and Jisen Li and Ziyan Chen and Shuaiwen Leon Song and Ben Athiwaratkun and Xiaoxia Wu},
  year={2026},
  eprint={2605.17757},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.17757}
}
```

---

## Contact

Jayin Panesar — independent research. Issues and corrections welcome.
