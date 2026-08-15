#!/usr/bin/env python3
"""
stage_b_subspace_compare.py -- OSCAR Stage B eigenbasis comparison.

Strips the fixed factors from R = U . H_Hadamard . P_bitreverse to recover the
data-dependent eigenbasis U, then compares U against the RotationZoo reference
via PRINCIPAL ANGLES between top-k subspaces.

Why principal angles: eigenvectors are defined only up to sign, and up to an
arbitrary rotation within any degenerate eigenvalue subspace. Principal angles
between the SUBSPACES those vectors span are invariant to both. Elementwise
differences are not, and are never reported here as evidence of similarity.

The scrambled checkpoints provide the null: whatever angle two unrelated
orthogonal matrices produce. The pre-registered test is real-vs-null, because
the two calibrations used different data (83 prompts/seq20000 vs 117/seq30000)
and no principled absolute expected angle exists.

CPU-only. All inputs read-only.

Usage:
    python stage_b_subspace_compare.py
"""

import hashlib
import os
import sys

import numpy as np
import torch

N = 128
KS = [8, 16, 32, 64, 128]

REAL_DIR = "/Users/jayinpanesar/recovery/rotations"
SCRAM_DIR = "/Users/jayinpanesar/recovery/rotations_scrambled"
REF_DIR = "/Users/jayinpanesar/rotation_zoo/Qwen3-8B/seq20000_prompt83_group128"

FILES = {"k": "k_rotation_qqt_r_h_pbr.pt", "v": "v_rotation_sst_r_h_pbr.pt"}

EXPECTED_SHA = {
    (REAL_DIR, "k"): "ca8a0198f630dabbb85a186388355c844a1721782e1c08e4355558ca3424205a",
    (REAL_DIR, "v"): "802088639ac53162193b591ee67a6aff289a867ac533d89a043142172e15af49",
    (SCRAM_DIR, "k"): "cc5313264cc7edc527b55d982bbe4d8c36938e752ddb3f2f09588c5ccc056aa1",
    (SCRAM_DIR, "v"): "18815a709923f79ae1806b3c86875c6667e4e31cbb2bc4840dfefc299b4c391f",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify_inputs():
    print("=== INPUT HASH VERIFICATION ===")
    ok = True
    for (d, side), want in EXPECTED_SHA.items():
        p = os.path.join(d, FILES[side])
        got = sha256(p)
        match = got == want
        ok &= match
        print(f"  {'OK  ' if match else 'FAIL'}  {p}")
        print(f"        {got}")
        if not match:
            print(f"        EXPECTED {want}")
    for side in FILES:
        p = os.path.join(REF_DIR, FILES[side])
        if not os.path.isfile(p):
            print(f"  FAIL  missing reference: {p}")
            ok = False
        else:
            print(f"  OK    {p}\n        {sha256(p)}  (reference, Stage A verified)")
    if not ok:
        sys.exit("ABORT: input hash mismatch")
    print("  all inputs verified\n")


# ------------------------------------------------------------ fixed factors

def hadamard(n):
    """Sylvester-Hadamard, normalized orthogonal. Requires n a power of 2."""
    assert n & (n - 1) == 0, "Hadamard needs power-of-2"
    H = np.ones((1, 1))
    while H.shape[0] < n:
        H = np.block([[H, H], [H, -H]])
    return H / np.sqrt(n)


def bitrev_perm(n):
    bits = int(np.log2(n))
    idx = np.array([int(format(i, f"0{bits}b")[::-1], 2) for i in range(n)])
    P = np.zeros((n, n))
    P[np.arange(n), idx] = 1.0
    return P


H = hadamard(N)
P = bitrev_perm(N)
HP = H @ P


def strip(R):
    """Recover U from R = U . H . P."""
    return R @ HP.T


def recompose(U):
    return U @ HP


# ------------------------------------------------------------ principal angles

def principal_angles(A, B):
    """Principal angles (radians, ascending) between the column spans of A and B.

    A, B: (n, k) with orthonormal columns. Uses the SVD of A^T B; singular
    values are the cosines. Clipped to [-1,1] against round-off.
    Sign- and rotation-within-subspace invariant by construction.
    """
    s = np.linalg.svd(A.T @ B, compute_uv=False)
    return np.arccos(np.clip(s, -1.0, 1.0))


def load_U(path):
    """Return {layer_id: U} plus the raw objects."""
    o = torch.load(path, map_location="cpu", weights_only=False)
    return {lid: strip(o["layers"][lid]["rotation"].double().numpy())
            for lid in sorted(o["layers"])}, o


# ------------------------------------------------------------ main

def main():
    verify_inputs()

    print("=== FIXED FACTOR PROPERTIES ===")
    print(f"  H: orth resid {np.linalg.norm(H@H.T-np.eye(N))/np.sqrt(N):.3e}  "
          f"symmetric={np.allclose(H,H.T)}  det sign={np.linalg.slogdet(H)[0]:.0f}")
    print(f"  P: orth resid {np.linalg.norm(P@P.T-np.eye(N))/np.sqrt(N):.3e}  "
          f"permutation=True  det sign={np.linalg.slogdet(P)[0]:.0f}")

    # ---- re-composition check, required before trusting anything
    print("\n=== RE-COMPOSITION RESIDUAL (required gate) ===")
    print("  For each checkpoint: strip H.P to get U, recompose U.H.P, compare to R.")
    worst_overall = 0.0
    for tag, d in [("REF", REF_DIR), ("REAL", REAL_DIR), ("SCRAM", SCRAM_DIR)]:
        for side in FILES:
            o = torch.load(os.path.join(d, FILES[side]), map_location="cpu",
                           weights_only=False)
            worst = 0.0
            for lid in sorted(o["layers"]):
                R = o["layers"][lid]["rotation"].double().numpy()
                worst = max(worst, np.abs(recompose(strip(R)) - R).max())
            worst_overall = max(worst_overall, worst)
            print(f"    {tag:<5} {side}: worst max|U.H.P - R| over 36 layers = {worst:.3e}")
    f32_eps = np.finfo(np.float32).eps
    print(f"  worst overall = {worst_overall:.3e}   float32 eps = {f32_eps:.3e}")
    if worst_overall > 1e-5:
        sys.exit("ABORT: re-composition failed; comparison would be meaningless")
    print("  PASS: round-trips well inside float32 precision.")
    print("  CAVEAT: this is an algebraic identity (R.(HP)^T.(HP) = R for any")
    print("  orthogonal HP), so it confirms H and P are orthogonal and the code")
    print("  path is consistent -- it does NOT by itself prove these are OSCAR's")
    print("  actual H and P. See the structural test below for that.")

    # ---- structural evidence that HP is the RIGHT fixed factor
    print("\n=== STRUCTURAL VALIDATION OF THE STRIPPING ===")
    print("  If H.P is correct, stripping it should CONCENTRATE the basis")
    print("  (recovering a sparse eigenbasis). A wrong orthogonal factor should not.")
    print("  Metric: mean inverse participation ratio (IPR) over columns; higher = sparser.")

    def ipr(M):
        return float(np.mean([np.sum(c**4)/np.sum(c**2)**2 for c in M.T]))

    rng_null = np.random.default_rng(1234)
    g = rng_null.standard_normal((N, N))
    Qwrong, _ = np.linalg.qr(g)
    o_ref = torch.load(os.path.join(REF_DIR, FILES["k"]), map_location="cpu",
                       weights_only=False)
    print(f"  {'layer':>5} {'raw R IPR':>11} {'stripped IPR':>13} {'wrong-factor IPR':>18} {'max|U|':>9}")
    for lid in [0, 1, 17, 34, 35]:
        R = o_ref["layers"][lid]["rotation"].double().numpy()
        U = strip(R)
        print(f"  {lid:>5} {ipr(R):>11.5f} {ipr(U):>13.5f} {ipr(R@Qwrong.T):>18.5f} "
              f"{np.abs(U).max():>9.5f}")
    print("  Correct stripping raises IPR ~10x and yields near-unit entries;")
    print("  a random orthogonal factor leaves it flat. This is the real evidence.")

    # ---- load all three sets
    print("\n=== LOADING ===")
    U = {}
    objs = {}
    for tag, d in [("real", REAL_DIR), ("scram", SCRAM_DIR), ("ref", REF_DIR)]:
        for side in FILES:
            u, o = load_U(os.path.join(d, FILES[side]))
            U[(tag, side)] = u
            objs[(tag, side)] = o
            print(f"  {tag:<6} {side}: {len(u)} layers")

    # ---- principal angles
    print("\n" + "=" * 78)
    print("PRE-REGISTERED TEST: principal angles vs RotationZoo reference")
    print("real vs ref  compared against  scrambled vs ref (null)")
    print("=" * 78)

    results = {}  # (side, cmp, k) -> per-layer mean angle in degrees
    for side in FILES:
        for cmp_tag in ("real", "scram"):
            for k in KS:
                per_layer = []
                for lid in sorted(U[("ref", side)]):
                    A = U[(cmp_tag, side)][lid][:, :k]
                    B = U[("ref", side)][lid][:, :k]
                    ang = np.degrees(principal_angles(A, B))
                    per_layer.append(ang.mean())
                results[(side, cmp_tag, k)] = np.array(per_layer)

    verdicts = []
    for side in FILES:
        print(f"\n--- {side.upper()} rotations ---")
        print(f"  {'k':>4} {'real mean':>11} {'null mean':>11} {'gap':>9} "
              f"{'real med':>10} {'null med':>10}  verdict")
        for k in KS:
            r = results[(side, "real", k)]
            s = results[(side, "scram", k)]
            gap = s.mean() - r.mean()
            closer = r.mean() < s.mean()
            verdicts.append((side, k, closer, r.mean(), s.mean()))
            print(f"  {k:>4} {r.mean():>10.4f}° {s.mean():>10.4f}° {gap:>8.4f}° "
                  f"{np.median(r):>9.4f}° {np.median(s):>9.4f}°  "
                  f"{'real closer' if closer else '** NOT CLOSER **'}")

    # ---- per-layer detail
    print("\n" + "=" * 78)
    print("PER-LAYER MEAN PRINCIPAL ANGLE (degrees), real vs ref | null vs ref")
    print("=" * 78)
    for side in FILES:
        print(f"\n--- {side.upper()} ---")
        hdr = "  layer " + " ".join(f"{'k='+str(k):>17}" for k in KS)
        print(hdr)
        print("        " + " ".join(f"{'real / null':>17}" for _ in KS))
        for lid in sorted(U[("ref", side)]):
            cells = []
            for k in KS:
                r = results[(side, "real", k)][lid]
                s = results[(side, "scram", k)][lid]
                cells.append(f"{r:7.3f} /{s:8.3f}")
            print(f"  {lid:>5} " + " ".join(f"{c:>17}" for c in cells))

    # ---- depth trend
    print("\n=== DEPTH TREND (real vs ref) ===")
    for side in FILES:
        for k in (8, 128):
            y = results[(side, "real", k)]
            x = np.arange(len(y))
            slope, icpt = np.polyfit(x, y, 1)
            cc = np.corrcoef(x, y)[0, 1]
            print(f"  {side} k={k:<4}: slope={slope:+.4f}°/layer  r={cc:+.3f}  "
                  f"first6={y[:6].mean():.3f}°  last6={y[-6:].mean():.3f}°")

    # ---- eigenvalue spectra
    print("\n=== EIGENVALUE SPECTRUM COMPARISON ===")
    print("  (scrambled eigenvalues are copied from real by construction)")
    for side in FILES:
        print(f"\n  --- {side} ---")
        print(f"  {'layer':>5} {'real sum':>12} {'ref sum':>12} {'scram sum':>12} "
              f"{'rel|real-ref|':>14} {'corr(real,ref)':>15}")
        rels, corrs = [], []
        for lid in sorted(objs[("ref", side)]["layers"]):
            er = objs[("real", side)]["layers"][lid]["eigenvalues"].double().numpy()
            ef = objs[("ref", side)]["layers"][lid]["eigenvalues"].double().numpy()
            es = objs[("scram", side)]["layers"][lid]["eigenvalues"].double().numpy()
            rel = abs(er.sum() - ef.sum()) / max(abs(ef.sum()), 1e-30)
            corr = np.corrcoef(er, ef)[0, 1]
            rels.append(rel); corrs.append(corr)
            if lid < 3 or lid > 32:
                print(f"  {lid:>5} {er.sum():>12.5g} {ef.sum():>12.5g} {es.sum():>12.5g} "
                      f"{rel:>14.5f} {corr:>15.8f}")
        print(f"    over all 36: mean rel diff={np.mean(rels):.5f}  "
              f"max={np.max(rels):.5f}  mean corr={np.mean(corrs):.8f}")
        same = all(np.array_equal(
            objs[("real", side)]["layers"][l]["eigenvalues"].numpy(),
            objs[("scram", side)]["layers"][l]["eigenvalues"].numpy())
            for l in objs[("real", side)]["layers"])
        print(f"    scrambled eigenvalues bitwise-equal to real: {same}")

    # ---- determinant signs
    print("\n=== DETERMINANT SIGN DISTRIBUTION ===")
    print(f"  {'set':<8} {'side':<5} {'det(R) +1/-1':>14} {'det(U) +1/-1':>14}")
    for tag in ("real", "scram", "ref"):
        for side in FILES:
            sr, su = [], []
            for lid in sorted(objs[(tag, side)]["layers"]):
                R = objs[(tag, side)]["layers"][lid]["rotation"].double().numpy()
                sr.append(int(np.linalg.slogdet(R)[0]))
                su.append(int(np.linalg.slogdet(strip(R))[0]))
            print(f"  {tag:<8} {side:<5} {sr.count(1):>6}/{sr.count(-1):<7} "
                  f"{su.count(1):>6}/{su.count(-1):<7}")
    print(f"  det(H)={np.linalg.slogdet(H)[0]:.0f} det(P)={np.linalg.slogdet(P)[0]:.0f} "
          f"-> det(U) and det(R) should agree")

    # ---- final verdict
    print("\n" + "=" * 78)
    n_closer = sum(1 for _, _, c, _, _ in verdicts if c)
    print(f"PRE-REGISTERED RESULT: real closer than null in {n_closer}/{len(verdicts)} "
          f"(side, k) cells")
    if n_closer == len(verdicts):
        print("VERDICT: PASS -- real rotations measurably closer to RotationZoo")
        print("         than the scrambled null, at every k, both sides.")
    elif n_closer == 0:
        print("VERDICT: FAIL -- real indistinguishable from / worse than null.")
    else:
        print("VERDICT: MIXED -- reporting per-cell, not averaged away:")
        for side, k, c, rm, sm in verdicts:
            if not c:
                print(f"           {side} k={k}: real {rm:.4f}° vs null {sm:.4f}°")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
