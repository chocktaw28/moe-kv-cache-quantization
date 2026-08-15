# Stage B — Eigenbasis subspace comparison vs RotationZoo

**Date:** 2026-08-15
**Model:** Qwen3-8B
**Env:** conda `rotcheck` — torch 2.13.0, numpy 2.4.6 (CPU only)

**Result: PASS on the pre-registered test at every informative k (8/16/32/64, both k and v).**
k=128 is mathematically degenerate and cannot discriminate — see below. It is reported as
degenerate, not as a pass or a failure.

## Artifacts

| Path | What |
|---|---|
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_b_subspace_compare.py` | Analysis script |
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_b_raw.txt` | Full raw output |
| `/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_b_report.md` | This report |

**Inputs (all read-only, unmodified, hashes verified at run start and re-verified after):**

```
OK  /Users/jayinpanesar/recovery/rotations/k_rotation_qqt_r_h_pbr.pt
    ca8a0198f630dabbb85a186388355c844a1721782e1c08e4355558ca3424205a
OK  /Users/jayinpanesar/recovery/rotations/v_rotation_sst_r_h_pbr.pt
    802088639ac53162193b591ee67a6aff289a867ac533d89a043142172e15af49
OK  /Users/jayinpanesar/recovery/rotations_scrambled/k_rotation_qqt_r_h_pbr.pt
    cc5313264cc7edc527b55d982bbe4d8c36938e752ddb3f2f09588c5ccc056aa1
OK  /Users/jayinpanesar/recovery/rotations_scrambled/v_rotation_sst_r_h_pbr.pt
    18815a709923f79ae1806b3c86875c6667e4e31cbb2bc4840dfefc299b4c391f
OK  /Users/jayinpanesar/rotation_zoo/Qwen3-8B/seq20000_prompt83_group128/k_rotation_qqt_r_h_pbr.pt
    8eacac74882ae887c5e832f38aa4a8c272b37c9899a020637c882bfddb96e1fa
OK  /Users/jayinpanesar/rotation_zoo/Qwen3-8B/seq20000_prompt83_group128/v_rotation_sst_r_h_pbr.pt
    d8146a3ff3d53aad89af9f58b7044d244598f3f7a24554b56e109f7edbe6b81b
```

## 1. Re-composition residual (the required gate)

Fixed factors built independently: `H` = Sylvester-Hadamard (n=128) normalized by
`1/sqrt(n)`; `P` = bit-reversal permutation. Both verified orthogonal, `H` symmetric,
bit-reversal confirmed involutive, `det(H)=+1`, `det(P)=+1`.

Strip `U = R·(H·P)ᵀ`, recompose `U·H·P`, compare to `R`:

```
REF   k: worst max|U.H.P - R| over 36 layers = 1.943e-16
REF   v: worst max|U.H.P - R| over 36 layers = 2.255e-16
REAL  k: worst max|U.H.P - R| over 36 layers = 1.943e-16
REAL  v: worst max|U.H.P - R| over 36 layers = 1.943e-16
SCRAM k: worst max|U.H.P - R| over 36 layers = 2.082e-16
SCRAM v: worst max|U.H.P - R| over 36 layers = 1.683e-16

worst overall = 2.255e-16   float32 eps = 1.192e-07
```

**Round-trips ~9 orders of magnitude inside float32 precision. Gate passed.**

### Honest caveat on what this proves

I want to be explicit, because the instruction was to confirm the stripping before
trusting any result, and the round-trip alone does **not** do that. `R·(HP)ᵀ·(HP) = R`
is an algebraic identity for *any* orthogonal `HP`. A wrong-but-orthogonal factor would
round-trip to 1e-16 just as cleanly. The residual confirms the factors are orthogonal
and the code path is self-consistent — nothing more.

The actual evidence that these are OSCAR's real `H` and `P` is structural. If the
factor is correct, stripping it should *recover* a sparse eigenbasis; a wrong factor
should leave the matrix looking generic. Measured by mean inverse participation ratio
over columns (higher = sparser), on the reference checkpoint:

```
layer   raw R IPR  stripped IPR   wrong-factor IPR    max|U|
    0     0.02010       0.21340            0.02286   0.99994
    1     0.02071       0.18015            0.02298   0.98802
   17     0.02169       0.09490            0.02299   0.94542
   34     0.02134       0.11909            0.02274   0.99703
   35     0.02006       0.16532            0.02322   0.99151
```

Correct stripping raises IPR **~5–10×** and produces near-unit entries (max |U| ≈ 0.99994,
vs raw R which never exceeds 0.34). A random orthogonal factor in the same position leaves
IPR flat at ~0.023. My own checkpoint shows the same signature (layer 0: 0.02012 → 0.21683,
max 0.99994), confirming both sides came from the same pipeline. **This is what justifies
trusting the comparison**, not the round-trip.

## 2. Pre-registered test — PASS at every informative k

Method: principal angles between top-k column subspaces of `U`, via SVD of `AᵀB`;
angles = `arccos` of singular values. Invariant to eigenvector sign and to arbitrary
rotation within degenerate eigenvalue subspaces, as required. No elementwise differences
are used anywhere in this comparison.

### K rotations

| k | real mean | null mean | gap | real median | null median | verdict |
|---|---|---|---|---|---|---|
| 8 | 3.2153° | 77.6298° | 74.4145° | 3.3738° | 77.7973° | **real closer** |
| 16 | 3.5369° | 71.8704° | 68.3335° | 3.3982° | 71.8106° | **real closer** |
| 32 | 3.0252° | 63.3340° | 60.3088° | 3.0306° | 63.3146° | **real closer** |
| 64 | 1.7459° | 45.0693° | 43.3234° | 1.7987° | 45.0071° | **real closer** |
| 128 | 0.0056° | 0.0056° | 0.0000° | 0.0056° | 0.0056° | degenerate — see §3 |

### V rotations

| k | real mean | null mean | gap | real median | null median | verdict |
|---|---|---|---|---|---|---|
| 8 | 11.7227° | 78.0523° | 66.3296° | 11.6861° | 77.9423° | **real closer** |
| 16 | 9.3991° | 72.1482° | 62.7491° | 9.1764° | 72.1563° | **real closer** |
| 32 | 6.6418° | 63.1159° | 56.4741° | 6.4672° | 63.1497° | **real closer** |
| 64 | 3.5290° | 45.0477° | 41.5187° | 3.4339° | 45.0229° | **real closer** |
| 128 | 0.0056° | 0.0056° | −0.0000° | 0.0056° | 0.0055° | degenerate — see §3 |

### Distributions do not overlap at all

Not just the means — the **worst** real layer beats the **best** null layer at every
informative k, with no overlap anywhere:

```
k k=8    real worst=  7.133  null best= 74.633  overlap=NO  ratio= 24.1x
k k=16   real worst=  7.983  null best= 70.484  overlap=NO  ratio= 20.3x
k k=32   real worst=  4.469  null best= 61.645  overlap=NO  ratio= 20.9x
k k=64   real worst=  2.785  null best= 44.269  overlap=NO  ratio= 25.8x
v k=8    real worst= 19.091  null best= 75.711  overlap=NO  ratio=  6.7x
v k=16   real worst= 13.034  null best= 70.079  overlap=NO  ratio=  7.7x
v k=32   real worst=  9.518  null best= 61.193  overlap=NO  ratio=  9.5x
v k=64   real worst=  4.863  null best= 44.218  overlap=NO  ratio= 12.8x
```

There is no threshold-tuning here and no averaging away of bad cells: every one of the
72 real (layer, k) measurements per side sits below every one of the 72 null measurements.

## 3. Why k=128 is degenerate, not a failure

The script's automated verdict line reads `MIXED — 9/10 cells`, flagging `v k=128`
(real 0.0056° vs null 0.0056°, difference −0.0000°). **That flag is an artifact of the
metric, not a finding.** I am reporting it rather than suppressing it, but it should not
be read as a partial failure.

At k = n = 128, the top-k subspace is all of ℝ¹²⁸ for *any* orthogonal matrix. Every
principal angle is therefore zero by construction, and the comparison carries no
information about either set. Verified directly with independently drawn random
orthogonal matrices:

```
trial 0: k=128 mean angle between two UNRELATED orthogonal bases = 0.000000 deg
trial 1: k=128 mean angle between two UNRELATED orthogonal bases = 0.000001 deg
trial 2: k=128 mean angle between two UNRELATED orthogonal bases = 0.000000 deg
```

Both real and null measure 0.0056° there — the identical float32 round-off floor. The
sign of the difference between two equal-to-round-off quantities is noise. A metric that
returns the same value for genuinely related and genuinely unrelated inputs cannot
discriminate, so `k=128` is excluded as uninformative **on grounds that hold independently
of the result** (it would be equally degenerate had the sign fallen the other way).

**On the pre-registered criterion as written** — "real closer at every k" — the literal
reading is 9/10. On the four k values that can actually measure anything, it is 10/10 with
zero distribution overlap. I judge this a PASS and am flagging the discrepancy so you can
overrule that call.

## 4. Per-layer results (descriptive)

Full 36-layer tables for all k are in
`/Users/jayinpanesar/moe-kv-cache-quantization/reproduction/stage_b_raw.txt`.
Excerpt, K rotations, `real / null` in degrees:

```
layer         k=8              k=16              k=32              k=64
    0   2.795 /  75.052   2.222 /  72.307   2.058 /  63.322   1.023 /  45.694
    1   2.035 /  75.485   2.186 /  72.976   2.684 /  62.517   0.880 /  45.726
    7   7.133 /  78.230   3.412 /  71.720   3.038 /  64.218   1.769 /  45.531
   17   4.040 /  75.134   2.756 /  70.502   3.923 /  63.307   2.785 /  44.557
   29   0.041 /  78.617   1.349 /  71.533   2.178 /  63.298   1.943 /  45.004
   30   4.443 /  78.935   7.983 /  72.964   3.074 /  63.930   1.795 /  45.549
   35   1.756 /  77.418   2.010 /  71.663   2.099 /  64.545   0.974 /  45.490
```

Real-vs-reference angles are small and layer-to-layer noisy (K k=8 ranges 0.04°–7.13°);
null angles are tightly clustered near the random-subspace expectation (74.6°–80.1° at
k=8) with no layer structure, exactly as a null should behave.

### Depth trend

```
k k=8   : slope=-0.0306°/layer  r=-0.179  first6=3.172°  last6=3.999°
v k=8   : slope=+0.1030°/layer  r=+0.369  first6=9.124°  last6=14.361°
```

**K shows no meaningful depth trend** (r = −0.18, slope ≈ −0.03°/layer — negligible against
0.04–8° layer-to-layer scatter). **V drifts mildly upward with depth**: r = +0.37,
+0.10°/layer, with the last six layers averaging ~57% higher than the first six
(14.36° vs 9.12°). So deeper V subspaces agree somewhat less well between the two
calibrations. This is a weak-to-moderate correlation on 36 points, and I would not lean
on it hard; it is consistent with deeper V statistics being more data-dependent, but this
comparison cannot establish that.

Throughout, V agreement is worse than K (11.7° vs 3.2° at k=8) while both remain far from
null. Note V's eigenvalue spectra span ~5 orders of magnitude across layers
(0.094 → 4553), so V subspaces are plausibly more sensitive to calibration data.

## 5. Eigenvalue spectrum comparison (descriptive)

```
--- k ---
layer     real sum      ref sum    scram sum  rel|real-ref|  corr(real,ref)
    0       288.99       288.53       288.99        0.00162      0.99997662
    1       268.97       269.18       268.97        0.00077      0.99999365
   34       327.08       326.55       327.08        0.00161      0.99999891
   35       288.37       288.05       288.37        0.00109      0.99999891
  over all 36: mean rel diff=0.00279  max=0.00841  mean corr=0.99999469

--- v ---
    0     0.094373       0.0937     0.094373        0.00718      0.99992261
    1      0.27859      0.27629      0.27859        0.00832      0.99947352
   34       3922.3         3900       3922.3        0.00572      0.99996392
   35       1776.5       1767.2       1776.5        0.00525      0.99993949
  over all 36: mean rel diff=0.00366  max=0.01090  mean corr=0.99992111
```

Spectra agree with the reference to a mean relative difference of **0.28% (k)** and
**0.37% (v)**, worst case 1.09%, with per-layer Pearson correlation ≥ 0.9999. Two
independent calibrations of the same model on different prompt sets recover nearly the
same covariance spectrum.

**The scrambled set carries real eigenvalues bitwise-identical to the real set** (verified:
`True` both sides) — by construction, since `make_scrambled_rotations.py` replaces only
rotations. This is exactly why the spectrum comparison is *descriptive only* and cannot
serve as the test: the null is indistinguishable from real on this axis by design. Only
the subspace comparison in §2 discriminates.

## 6. Determinant sign distribution (descriptive)

```
set      side   det(R) +1/-1   det(U) +1/-1
real     k          12/24          12/24
real     v          19/17          19/17
scram    k          18/18          18/18
scram    v          21/15          21/15
ref      k          17/19          17/19
ref      v          20/16          20/16
```

`det(U)` matches `det(R)` in every case, as expected since `det(H)=+1` and `det(P)=+1`
(so stripping cannot change the sign — a consistency check on the stripping code, which
it passes).

All three sets mix both signs in broadly similar proportion; real K is the most skewed
(12/24) and scrambled K the most balanced (18/18, as expected for Haar-random). A
`det = −1` basis is a reflection, still exactly orthogonal, and eigenvector sign is
arbitrary — so this is descriptive only and no conclusion is drawn from it.

## 7. Summary

| Item | Result |
|---|---|
| Input hash verification | **PASS** — all 6 files match |
| Re-composition residual | **PASS** — worst 2.255e-16 (float32 eps 1.19e-07) |
| Stripping validated structurally | **PASS** — IPR ~5–10× vs wrong-factor control |
| Pre-registered: real closer than null, k=8 | **PASS** both sides |
| Pre-registered: real closer than null, k=16 | **PASS** both sides |
| Pre-registered: real closer than null, k=32 | **PASS** both sides |
| Pre-registered: real closer than null, k=64 | **PASS** both sides |
| Pre-registered: k=128 | **Degenerate** — metric carries no information at k=n |
| Distribution overlap | **None** — worst real beats best null at every informative k |

**Overall: PASS.** My rotations share genuine eigenbasis structure with the published
RotationZoo reference — 3.2° (K) and 11.7° (V) mean principal angle at k=8, against a
null of ~78°. Combined with Stage A (correct schema, orthogonal to 3.7e-08, no hash
collision with the reference), the calibration produced real, correctly-structured,
independently-derived rotations.

### What this does and does not establish

- **Does:** the data-dependent eigenbasis `U` is close to the reference's, in a
  sign- and rotation-invariant sense, far outside what unrelated orthogonal matrices produce.
- **Does not:** establish downstream accuracy. This is a structural comparison against
  one reference calibrated on *different data* (83 prompts/seq20000 vs my 117/seq30000).
  No absolute angle threshold was set, deliberately, and none is implied by these numbers.
  Only a real eval can show quantization quality.
- **Unexplained residual:** the 3–12° gap is presumably the different calibration sets,
  but this comparison cannot separate "different data" from "pipeline difference." A
  same-data run would be needed to decompose it.
- The V depth trend (r = +0.37) is suggestive, not established, on 36 points.

**No input file was modified. Nothing was fixed.**
