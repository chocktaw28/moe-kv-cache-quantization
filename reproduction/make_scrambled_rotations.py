#!/usr/bin/env python3
"""
make_scrambled_rotations.py -- build a NEGATIVE CONTROL for the OSCAR pipeline.

Reads the calibrated rotation checkpoints and writes copies with an IDENTICAL
schema -- same keys, shapes, dtypes, metadata, format_version -- but with every
per-layer rotation replaced by a different random orthogonal matrix.

The point: the scrambled matrices are perfectly valid rotations. They pass every
reference-free check in inspect_rotations.py (orthogonality, no NaN/Inf, correct
layout). They are simply the WRONG rotations -- they diagonalize nothing.

So if a downstream eval scores the same on these as on the real checkpoints, the
eval is not sensitive to rotation quality and its "pass" on the real ones was
never evidence of anything. That is the only thing this script is for.

Eigenvalue tensors are copied through untouched: they describe the calibration
covariance spectrum, not the rotation, and leaving them intact makes the
scrambled files harder to distinguish by a superficial check -- which is
precisely what makes this a useful control.

Determinism: numpy default_rng(0), drawn once and used for every layer in a
fixed (file, layer) order, so the output is byte-reproducible across runs.

CPU-only. Reads the source files read-only; never writes to the source dir.

Usage:
    python make_scrambled_rotations.py
    python make_scrambled_rotations.py --seed 0 --out ~/recovery/rotations_scrambled
"""

import argparse
import os
import sys

import numpy as np
import torch

SRC_DEFAULT = os.path.expanduser("~/recovery/rotations")
OUT_DEFAULT = os.path.expanduser("~/recovery/rotations_scrambled")
FILENAMES = ["k_rotation_qqt_r_h_pbr.pt", "v_rotation_sst_r_h_pbr.pt"]


def random_orthogonal(n, rng):
    """A Haar-distributed random orthogonal matrix via QR of a Gaussian.

    numpy's QR does not fix the sign of R's diagonal, so Q alone is not Haar
    uniform. Multiplying by sign(diag(R)) corrects that -- the standard
    Mezzadri (2007) fix. Without it the ensemble is subtly biased, which would
    weaken the control.
    """
    a = rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    q *= np.sign(np.diag(r))
    return q


def scramble_file(src_path, out_path, rng, verbose=True):
    obj = torch.load(src_path, map_location="cpu", weights_only=False)
    if "layers" not in obj:
        sys.exit(f"unexpected schema, no 'layers' key: {src_path}")

    n_rot = 0
    for layer_id in sorted(obj["layers"].keys()):
        entry = obj["layers"][layer_id]
        rot = entry["rotation"]
        n = rot.shape[-1]
        if rot.ndim != 2 or rot.shape[0] != rot.shape[1]:
            sys.exit(f"rotation at layer {layer_id} is not square: {tuple(rot.shape)}")

        new = random_orthogonal(n, rng)
        # Preserve dtype exactly; the schema must be indistinguishable.
        entry["rotation"] = torch.from_numpy(new).to(rot.dtype)
        n_rot += 1
        # 'eigenvalues' and 'layer_id' deliberately untouched.

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(obj, out_path)
    if verbose:
        print(f"  {os.path.basename(src_path)}: scrambled {n_rot} rotations "
              f"-> {out_path} ({os.path.getsize(out_path):,} B)")
    return n_rot


def verify_pair(src_path, out_path):
    """Confirm schema identity and that every rotation actually changed."""
    a = torch.load(src_path, map_location="cpu", weights_only=False)
    b = torch.load(out_path, map_location="cpu", weights_only=False)

    meta_ok = all(a[k] == b[k] for k in a if k != "layers")
    keys_ok = sorted(a["layers"]) == sorted(b["layers"])

    shapes_ok = eig_ok = True
    changed = 0
    worst_resid = 0.0
    for lid in sorted(a["layers"]):
        ea, eb = a["layers"][lid], b["layers"][lid]
        if set(ea) != set(eb):
            shapes_ok = False
            continue
        for f in ea:
            if torch.is_tensor(ea[f]):
                if ea[f].shape != eb[f].shape or ea[f].dtype != eb[f].dtype:
                    shapes_ok = False
        if not torch.equal(ea["eigenvalues"], eb["eigenvalues"]):
            eig_ok = False
        if not torch.equal(ea["rotation"], eb["rotation"]):
            changed += 1
        r = eb["rotation"].double().numpy()
        n = r.shape[0]
        resid = np.linalg.norm(r @ r.T - np.eye(n)) / np.sqrt(n)
        worst_resid = max(worst_resid, resid)

    return {
        "metadata_identical": meta_ok,
        "layer_keys_identical": keys_ok,
        "shapes_dtypes_identical": shapes_ok,
        "eigenvalues_untouched": eig_ok,
        "rotations_changed": changed,
        "n_layers": len(a["layers"]),
        "worst_resid": worst_resid,
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = os.path.abspath(os.path.expanduser(args.src))
    out = os.path.abspath(os.path.expanduser(args.out))
    if os.path.realpath(src) == os.path.realpath(out):
        sys.exit("refusing to write scrambled output over the source directory")

    print(f"source: {src}")
    print(f"output: {out}")
    print(f"seed:   {args.seed}\n")

    rng = np.random.default_rng(args.seed)

    print("scrambling:")
    pairs = []
    for fn in FILENAMES:
        sp = os.path.join(src, fn)
        if not os.path.isfile(sp):
            sys.exit(f"missing source file: {sp}")
        op = os.path.join(out, fn)
        scramble_file(sp, op, rng)
        pairs.append((sp, op))

    print("\nverification (schema identity + rotations actually differ):")
    all_ok = True
    for sp, op in pairs:
        v = verify_pair(sp, op)
        ok = (v["metadata_identical"] and v["layer_keys_identical"]
              and v["shapes_dtypes_identical"] and v["eigenvalues_untouched"]
              and v["rotations_changed"] == v["n_layers"])
        all_ok &= ok
        print(f"  {os.path.basename(op)}")
        print(f"    metadata identical:        {v['metadata_identical']}")
        print(f"    layer keys identical:      {v['layer_keys_identical']}")
        print(f"    shapes/dtypes identical:   {v['shapes_dtypes_identical']}")
        print(f"    eigenvalues untouched:     {v['eigenvalues_untouched']}")
        print(f"    rotations changed:         {v['rotations_changed']}/{v['n_layers']}")
        print(f"    worst ||RR^T-I||_F/sqrt(n): {v['worst_resid']:.4e}")

    print(f"\nresult: {'OK' if all_ok else 'PROBLEM -- see above'}")
    print("\nNOTE: these are valid rotations that are deliberately WRONG. "
          "They exist to be failed by a sensitive eval.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
