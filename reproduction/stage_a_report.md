# Stage A — Rotation checkpoint inventory & self-consistency

**Date:** 2026-08-15
**Model:** Qwen3-8B
**Env:** conda `rotcheck` — torch 2.13.0, numpy 2.4.6, huggingface_hub 1.27.0, safetensors 0.8.0 (CPU only)

**Verdict: PASS on all seven pre-registered criteria (a–g).**
One prediction was wrong — the extra bytes are **not** clipping thresholds. See (b) and (g).

## Artifacts

| Path | What |
|---|---|
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_a_local.txt` | Raw inspection, my rotations |
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_a_zoo.txt` | Raw inspection, RotationZoo |
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_a_local.json` | Same, machine-readable |
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_a_zoo.json` | Same, machine-readable |
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_a_orthogonality_table.txt` | All 144 matrices, per-layer |
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/inspect_rotations.py` | Extended (batched-square support) |

**Inputs (read-only, unmodified):**
`/Users/jayinpanesar/recovery/rotations/{k_rotation_qqt_r_h_pbr.pt, v_rotation_sst_r_h_pbr.pt}`
**Reference:** `/Users/jayinpanesar/rotation_zoo/Qwen3-8B/seq20000_prompt83_group128/` (downloaded 2026-08-15)

### RotationZoo repo layout (listed before download)

111 files. Only the 3 matching `Qwen3-8B/**` + `README.md` were pulled; the GLM-5.2-FP8 (79 files) and 32B/358B-class subtrees were **not** downloaded.

```
GLM-4.7-FP8  2      MiniMax-M2.7  3    Qwen3-32B               2
GLM-5.2-FP8 79      MiniMax-M3    3    Qwen3-4B-Thinking-2507  4
Gemma4-12B   2      Qwen3-30B-A3B 5    Qwen3-8B                2   <-- pulled
                                        Qwen3.5-35B-A3B 3, Qwen3.5-4B 3
```

> **Config skew, worth noting:** RotationZoo's Qwen3-8B is `seq20000_prompt83_group128`; mine is `seq30000_prompt117_group128`. Same model, **different calibration set** (83 vs 117 prompts). The reference is a same-model/different-data comparison, not a bit-exact target. This is the correct expectation for Stage B.

## (a) Layer count — PASS

36 layers in every one of the four files, keys `int` 0..35 contiguous, matching Qwen3-8B and the Phase 1 log's "Layers captured: 36/36".

```
MINE/k format_version=1 objective='qqt_r_h_pbr' source_grouping='layer' n_layers=36
MINE/v format_version=1 objective='sst_r_h_pbr' source_grouping='layer' n_layers=36
ZOO/k  format_version=1 objective='qqt_r_h_pbr' source_grouping='layer' n_layers=36
ZOO/v  format_version=1 objective='sst_r_h_pbr' source_grouping='layer' n_layers=36
```

## (b) Actual layout and the extra 39,965 bytes — PASS, but your prediction was wrong

**Predicted:** flat `[36,128,128]` float32 tensor + ~39,965 bytes of per-layer clipping thresholds.

**Actual:** not a flat tensor at all. It is a metadata dict wrapping a per-layer dict:

```
{ 'format_version': int(1),
  'objective':      str,          # 'qqt_r_h_pbr' | 'sst_r_h_pbr'
  'source_grouping':str('layer'),
  'layers': { 0..35 : { 'layer_id':    int,
                        'rotation':    Tensor [128,128] float32,
                        'eigenvalues': Tensor [128]     float32 } } }
```

The rotation payload is exactly the predicted size; the surplus is **eigenvalues + zip/pickle framing**, with **no clipping thresholds anywhere**:

```
file          total  rot bytes  eig bytes    payload  overhead
MINE/k    2,399,261  2,359,296     18,432  2,377,728    21,533
MINE/v    2,399,261  2,359,296     18,432  2,377,728    21,533
ZOO/k     2,399,261  2,359,296     18,432  2,377,728    21,533
ZOO/v     2,399,261  2,359,296     18,432  2,377,728    21,533
```

Arithmetic, reconciling against your 39,965:

```
rotations   36 x 128 x 128 x 4 B = 2,359,296   (your prediction, exactly right)
eigenvalues 36 x 128 x 4 B       =    18,432
zip/pickle container overhead    =    21,533   (78 zip members; data.pkl = 5,467 B)
                                   ---------
total                              2,399,261  == actual file size, exact
surplus over prediction            18,432 + 21,533 = 39,965  == your figure, exact
```

So the 39,965 splits **18,432 eigenvalues + 21,533 container framing**. Your byte count was right and your structural hypothesis was wrong — nothing in either file is a clipping threshold. All four files land on the identical byte split, which is why they are all exactly 2,399,261 despite different calibration data.

## (c) Orthogonality — PASS

All **144** matrices (4 files × 36 layers), checked in float64 on both `RR^T` and `R^TR`, worst of the two reported.

```
matrices checked: 144
residual  min=3.5329e-08  max=3.6889e-08  mean=3.6155e-08
threshold 1e-04: failures = 0 / 144
margin: worst residual is 2711x inside threshold
sv_min min=0.999999962419   sv_max max=1.000000038862
max |sv-1| over all singular values: 3.886e-08
```

Worst 5:

```
3.6889e-08  MINE/v layer 23  sv=[0.9999999653,1.0000000346] cond=1.00000007
3.6824e-08  MINE/v layer 26  sv=[0.9999999650,1.0000000361] cond=1.00000007
3.6810e-08  ZOO/v  layer 35  sv=[0.9999999634,1.0000000339] cond=1.00000007
3.6769e-08  ZOO/k  layer 31  sv=[0.9999999641,1.0000000358] cond=1.00000007
3.6752e-08  MINE/v layer 24  sv=[0.9999999644,1.0000000360] cond=1.00000007
```

Residual is flat across all layers and both sides — it is float32 storage round-off, not a calibration defect. My files are indistinguishable from the reference on this axis. Full per-layer table: `stage_a_orthogonality_table.txt`.

**Determinant sign is mixed, and that is fine.** MINE/k 24×(−1)/12×(+1), MINE/v 17/19; ZOO/k 19/17, ZOO/v 16/20. Both sides mix in similar proportion. `det = −1` means a reflection, still exactly orthogonal; eigenvector sign is arbitrary and OSCAR composes with a bit-reversal permutation whose sign depends on `n`. Not a criterion failure and not a discrepancy vs the reference.

## (d) NaN / Inf / degeneracy — PASS

```
tensors scanned=288  NaN=0  Inf=0  all-zero=0  constant=0
```

288 = 4 files × 36 layers × 2 tensors. Every eigenvalue vector is strictly positive and ascending-sorted. (The `** CONSTANT TENSOR **` lines in the raw logs fire on the scalar `layer_id`/`format_version` metadata leaves — an artifact of my flag treating scalars as arrays, not a degenerate rotation.)

## (e) Contamination — PASS, clean

All four hashes distinct; **no local file matches any RotationZoo file.**

```
MINE k  ca8a0198f630dabbb85a186388355c844a1721782e1c08e4355558ca3424205a
MINE v  802088639ac53162193b591ee67a6aff289a867ac533d89a043142172e15af49
ZOO  k  8eacac74882ae887c5e832f38aa4a8c272b37c9899a020637c882bfddb96e1fa
ZOO  v  d8146a3ff3d53aad89af9f58b7044d244598f3f7a24554b56e109f7edbe6b81b
```

My two hashes reproduce `/Users/jayinpanesar/recovery/rotations_sha256.txt` exactly, so the files are the ones Phase 2 wrote. Corroborating at tensor level: **0/36 rotations bitwise-identical** to the reference in either file, max elementwise |diff| 0.665 (k) / 0.669 (v). You really did calibrate.

## (f) Schema comparison — PASS, exact match

Verified programmatically, both k and v:

- layer keys identical: **True** (36, 0..35)
- per-layer field names identical: **True** — `['eigenvalues','layer_id','rotation']`
- all (layer, field, shape, dtype) tuples identical: **True**
- `format_version=1` and `source_grouping='layer'` on both sides; `objective` matches per file

No format-version skew. My pipeline emits byte-for-byte the same container the published reference uses.

## (g) Clipping thresholds — not present; cK/cV comparison not possible

**No clipping thresholds exist in either my files or RotationZoo's.** The only non-tensor fields are `format_version`, `objective`, `source_grouping` (top level) and `layer_id` (per layer). There is no per-layer scalar that could carry cK≈0.96 / cV≈0.92.

This is a **clean PASS** — the criterion was conditional ("if clipping thresholds ARE present"), and their absence is symmetric with the reference, so it is a property of OSCAR's rotation format, not a defect in my output. If cK/cV exist in OSCAR they live elsewhere (quantizer config at inference time, not the rotation checkpoint). **I did not compare against 0.96/0.92 — there was nothing to compare.**

What the extra tensor actually is: `eigenvalues`, `[128]` float32, ascending, strictly positive — the spectrum of the calibration covariance the rotation diagonalizes. As a same-model sanity check, my spectra track RotationZoo's closely despite the different prompt set:

```
k, relative |sum| difference vs zoo:  layer 0: 0.16%  1: 0.08%  34: 0.16%  35: 0.11%
v, relative |sum| difference vs zoo:  layer 0: 0.72%  1: 0.83%  34: 0.57%  35: 0.53%
```

Sub-1% agreement on spectral mass, with rotations that are nowhere near elementwise equal, is the signature of a genuine independent calibration of the same model on different data. That is the strongest positive evidence in this report.

## Summary

| # | Criterion | Result |
|---|---|---|
| a | 36 layers | **PASS** — 36/36, all four files |
| b | Layout + account for 39,965 B | **PASS** — accounted exactly; hypothesis wrong (eigenvalues + zip framing, not clip thresholds) |
| c | Every rotation resid < 1e-4, sv≈1 | **PASS** — 0/144 failures, worst 3.69e-08 (2711× margin) |
| d | No NaN/Inf/zero/constant | **PASS** — 288 tensors, all clean |
| e | No hash match vs RotationZoo | **PASS** — all distinct; 0/36 bitwise-identical |
| f | Schema match | **PASS** — exact, incl. format_version |
| g | Clip thresholds vs cK/cV | **PASS (vacuous)** — absent on both sides; comparison not possible |

**No action taken on anything found.** Nothing was modified; the inputs remain read-only.

### Caveats

- Reference is a **different calibration config** (83 vs 117 prompts). Only same-model/same-format claims are supported; bit-exactness was never expected.
- Orthogonality is reference-free and conclusive. Everything else here is structural — **this validates that the checkpoints are well-formed, not that they are accuracy-optimal.** Only a real eval can show that.
- The one open question this raises: the eigenvalue spectra agree to <1%, but that is a coarse scalar summary. Whether the *subspaces* agree is exactly what Stage B measures.

**Stopped here per instruction. Stage B not started.**
