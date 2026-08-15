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
| Checkpoint validation vs. published RotationZoo — Stage A (inventory, self-consistency) | Complete — PASS, see `stage_a_report.md` |
| Checkpoint validation vs. published RotationZoo — Stage B (eigenbasis subspaces) | Complete — PASS, see `stage_b_report.md` |
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
- **`stage_a_report.md`** — Stage A: inventory, layout, orthogonality, contamination
  hash check, schema comparison. Raw output in `stage_a_local.txt` / `stage_a_zoo.txt`.
- **`make_scrambled_rotations.py`** — negative control. Produces checkpoints with an
  identical schema but random orthogonal rotations: valid rotations that are
  deliberately the wrong ones, for use as a null baseline.
- **`stage_b_subspace_compare.py`** / **`stage_b_report.md`** — Stage B: principal-angle
  comparison of the recovered eigenbasis against RotationZoo, with the scrambled set as
  the null. Raw output in `stage_b_raw.txt`.
- **`run_config.env`** — resolved runtime configuration (memory fraction, window
  settings) carried over from the working calibration run.
- **`scripts/`** — pod launch scripts for the Phase 3 evaluation (BF16 baseline,
  INT2 with real rotations, INT2 with scrambled rotations).

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
are meaningless. Compare principal angles between top-k subspaces instead.

(An earlier version of this note also suggested comparing per-layer clipping
thresholds directly. Stage A established that the checkpoints contain no clipping
thresholds — see the file-size section below — so that comparison is not available.
The per-layer `eigenvalues` tensor can be compared instead.)

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

**Original prediction (recorded as written, before inspection):**

> Predicted `[36, 128, 128]` float32 = 36 × 128 × 128 × 4 = **2,359,296 bytes**.
> Actual: **2,399,261 bytes** — 39,965 bytes larger, roughly 1.7% overhead. That is
> more than bare pickle framing for two tensors would explain, suggesting additional
> payload (plausibly per-layer clipping thresholds stored alongside the rotations).

**What Stage A actually found — the clipping-threshold hypothesis was wrong.**

There are **no clipping thresholds anywhere**, in the locally-generated checkpoints
or in RotationZoo's. The layout is also not a flat `[36, 128, 128]` tensor: it is a
metadata dict wrapping per-layer entries.

```
{ 'format_version': int(1),
  'objective':      str,          # 'qqt_r_h_pbr' | 'sst_r_h_pbr'
  'source_grouping':str('layer'),
  'layers': { 0..35 : { 'layer_id':    int,
                        'rotation':    Tensor [128,128] float32,
                        'eigenvalues': Tensor [128]     float32 } } }
```

The rotation payload matched the prediction exactly; the surplus is an extra
per-layer `[128]` float32 **eigenvalue spectrum** plus zip/pickle framing:

```
rotations   36 x 128 x 128 x 4 B = 2,359,296   (predicted size, exactly right)
eigenvalues 36 x 128 x 4 B       =    18,432
zip/pickle container framing     =    21,533   (78 zip members; data.pkl = 5,467 B)
                                   ---------
total                              2,399,261   == actual file size, exact

surplus over prediction            18,432 + 21,533 = 39,965   == exact
```

So the byte count was right and the structural explanation was wrong. The surplus
splits **18,432 eigenvalues + 21,533 framing**, with nothing resembling a threshold.
All four checkpoints (local k/v, RotationZoo k/v) land on the identical split, which
is why they are all exactly 2,399,261 bytes despite different calibration data.

Recorded as a miss rather than retrofitted into a confirmation. Full evidence in
[`stage_a_report.md`](stage_a_report.md).

## Stage B: eigenbasis comparison against RotationZoo

Stage A established the checkpoints are *well-formed* (correct schema, orthogonal to
3.7e-08, no hash collision with the reference). That is not the same as *correct* —
the negative control in `make_scrambled_rotations.py` passes every Stage A check while
containing deliberately wrong rotations. Stage B tests structure.

**Method.** Strip the fixed factors from `R = U · H_Hadamard · P_bitreverse` to recover
the data-dependent eigenbasis `U`, then compare against RotationZoo via principal
angles between top-k subspaces — invariant to eigenvector sign and to rotation within
degenerate subspaces, as required.

**Pre-registered test.** Real rotations must be measurably closer to RotationZoo than
the scrambled control at every k. No absolute angle threshold was set: the two
calibrations used different data (83 prompts/seq20000 vs 117/seq30000), so no
principled expected value exists. The null *is* the test.

**Result: PASS at every informative k.** Mean principal angle, real vs null:

| k | K real / null | V real / null |
|---|---|---|
| 8 | 3.22° / 77.63° | 11.72° / 78.05° |
| 16 | 3.54° / 71.87° | 9.40° / 72.15° |
| 32 | 3.03° / 63.33° | 6.64° / 63.12° |
| 64 | 1.75° / 45.07° | 3.53° / 45.05° |
| 128 | 0.0056° / 0.0056° | 0.0056° / 0.0056° — degenerate, excluded |

The distributions do not overlap at all: the worst real layer beats the best null layer
at every informative k, by factors of 6.7×–25.8×.

**Why k=128 is excluded.** At k = n = 128 the top-k subspace is all of ℝ¹²⁸ for *any*
orthogonal matrix, so every principal angle is zero by construction. Verified against
independently drawn random orthogonal pairs, which also measure ~0.000000°. Both real
and null sit at the identical float32 round-off floor (0.0056°), and the sign of the
difference between two equal-to-round-off quantities is noise. The cell is excluded on
grounds that hold regardless of outcome — it would be equally uninformative had the
sign fallen the other way. On the literal criterion ("closer at every k") this reads
9/10; on the four k that can measure anything it is 10/10.

Full method, per-layer tables, the re-composition gate, and the structural validation
of the stripping: [`stage_b_report.md`](stage_b_report.md).

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
