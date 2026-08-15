# Baseline Reproduction — OSCAR on Consumer Hardware

This directory records an attempt to reproduce
[OSCAR](https://github.com/FutureMLS-Lab/OSCAR)
([arXiv:2605.17757](https://arxiv.org/abs/2605.17757)) — currently the strongest
open, deployable 2-bit KV-cache quantization method — on a single RTX 4090 rather
than the H100-class hardware it targets.

The purpose is environment validation: establishing that the toolchain, kernels,
and evaluation pipeline behave correctly before any new measurement taken in this
environment is trusted.

## Status

| Stage | State |
|---|---|
| Environment setup (driver, venv, backends) | Complete — see `ENVIRONMENT.md` |
| OSCAR Phase 1: QKV activation dump | Complete — 198/198 prompts, 36/36 layers, 0 errors |
| OSCAR Phase 2: rotation calibration | Complete — checkpoints generated, hashes in `rotations_sha256.txt` |
| Checkpoint validation vs. published RotationZoo | In progress |
| OSCAR Phase 3: BF16 vs INT2 accuracy evaluation | Not started |

No accuracy numbers are reported yet. **Nothing in this directory should be read as
a reproduction result** — only as environment and artifact provenance.

## Contents

- **`ENVIRONMENT.md`** — hardware, driver floor, torch/venv setup, the sm89
  FlashAttention-3 incompatibility, memory-fraction tuning, calibration run
  statistics, and RunPod operational notes.
- **`eval_oscar_gpqa.patch`** — 46-line patch against OSCAR commit
  `41ebcdba` enabling the eval script to run on sm89 and outside conda.
- **`rotations_sha256.txt`** — SHA-256 hashes of the generated rotation
  checkpoints, for provenance. The `.pt` files themselves are not redistributed.
- **`inspect_rotations.py`** — schema-agnostic inventory and self-consistency
  checker for rotation checkpoints (see below).

## Validating rotation checkpoints

A calibration run exiting 0 is not evidence that it produced correct output. Two
independent checks are available and neither requires a GPU:

**Reference-free.** Every rotation matrix must be orthogonal. `inspect_rotations.py`
walks a checkpoint of unknown schema and reports, per layer, the orthogonality
residual `‖RRᵀ − I‖_F / √n`, singular-value range, condition number, and
NaN/Inf/all-zero counts. A non-orthogonal "rotation" is not a rotation, and an
all-zero tensor is the signature of a stage that completed while writing nothing.

**Against published reference.** OSCAR publishes precomputed rotations for
Qwen3-4B/8B/32B in
[RotationZoo](https://huggingface.co/Zhongzhu/OSCAR-RotationZoo). For any model in
that set, locally-generated rotations can be compared directly against the
reference — far more informative than a small benchmark run, and free.

Comparison must be **sign- and permutation-invariant**: OSCAR's rotation is
`R = U · H_Hadamard · P_bitreverse`, where only `U` (the eigenbasis of the
attention-aware covariance) is data-dependent. Eigenvectors are defined only up to
sign, and up to arbitrary rotation within degenerate subspaces, so elementwise diffs
are meaningless. Compare principal angles between top-k subspaces instead, and
compare per-layer clipping thresholds directly.

Also worth running: a **hash comparison** between local and published checkpoints.
A match means calibration didn't actually run and precomputed rotations were
downloaded somewhere in the pipeline.

### Pre-registered pass criteria

Written before running, to avoid post-hoc rationalisation:

1. Layer count matches the model (36 for Qwen3-8B).
2. Every rotation satisfies `‖RRᵀ − I‖_F / √n < 1e-4`, all singular values ≈ 1.
3. Zero NaN, zero Inf, no all-zero or constant tensors.
4. No hash collision with published RotationZoo checkpoints.
5. Schema (leaf keys, shapes, dtypes) consistent with the published reference.

## Reported vs. actual: a note on file sizes

Predicting the on-disk layout from file size is a useful sanity check, but only if
the prediction is recorded honestly.

Predicted `[36, 128, 128]` float32 = 36 × 128 × 128 × 4 = **2,359,296 bytes**.
Actual: **2,399,261 bytes** — 39,965 bytes larger, roughly 1.7% overhead. That is
more than bare pickle framing for two tensors would explain, suggesting additional
payload (plausibly per-layer clipping thresholds stored alongside the rotations).

The prediction was directionally correct and numerically wrong. Recorded as such
rather than rounded into a confirmation.

## Reproducing

See `ENVIRONMENT.md` for full setup. Minimum viable path:

1. Provision a GPU host with a CUDA ≥ 12.9 driver.
2. Create an isolated venv on persistent storage; install OSCAR's pinned stack.
3. Apply `eval_oscar_gpqa.patch`.
4. On sm89 hardware, confirm both prefill and decode backends are `triton`.
5. Run OSCAR's calibration pipeline, then validate the output with
   `inspect_rotations.py` before running any benchmark.

## Acknowledgements

OSCAR is by Zhongzhu Zhou, Donglin Zhuang, Jisen Li, Ziyan Chen, Shuaiwen Leon
Song, Ben Athiwaratkun, and Xiaoxia Wu, released under the MIT License. The patch
here is a small portability change; all method credit is theirs.
