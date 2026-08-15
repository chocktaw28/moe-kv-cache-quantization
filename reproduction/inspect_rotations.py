#!/usr/bin/env python3
"""
inspect_rotations.py -- Stage 1 of the OSCAR RotationZoo diff.

Format-agnostic inventory of a directory of rotation checkpoints, plus
reference-free self-consistency checks on any square 2D tensor found.

Deliberately makes NO assumptions about OSCAR's on-disk schema. It walks
whatever is there, reports it, and checks orthogonality where applicable.

Usage:
    python inspect_rotations.py <dir> --label local
    python inspect_rotations.py <dir> --label zoo
    python inspect_rotations.py <dir> --label local --json out.json

CPU-only. Read-only. Does not import sglang or touch the GPU.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

# ---------------------------------------------------------------- loaders

def _load_torch(path):
    import torch
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj


def _load_safetensors(path):
    from safetensors.numpy import load_file
    return load_file(path)


def _load_npz(path):
    return dict(np.load(path, allow_pickle=True))


LOADERS = {
    ".pt": _load_torch,
    ".pth": _load_torch,
    ".bin": _load_torch,
    ".ckpt": _load_torch,
    ".safetensors": _load_safetensors,
    ".npz": _load_npz,
    ".npy": lambda p: {"<array>": np.load(p, allow_pickle=True)},
}


def load_any(path):
    ext = os.path.splitext(path)[1].lower()
    loader = LOADERS.get(ext)
    if loader is None:
        return None, f"no loader for extension {ext!r}"
    try:
        return loader(path), None
    except Exception as e:  # noqa: BLE001
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------- walking

def to_numpy(x):
    """Best-effort conversion of a leaf to a float64 numpy array."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().to(torch.float64).numpy()
    except ImportError:
        pass
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            return None
        return x.astype(np.float64)
    if isinstance(x, (int, float)):
        return np.asarray([float(x)])
    return None


def dtype_of(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return str(x.dtype)
    except ImportError:
        pass
    if isinstance(x, np.ndarray):
        return str(x.dtype)
    return type(x).__name__


def walk(obj, prefix=""):
    """Yield (dotted_key, leaf) for nested dicts/lists/tuples."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from walk(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from walk(v, f"{prefix}[{i}]")
    else:
        yield prefix or "<root>", obj


# ---------------------------------------------------------------- checks

def batched_orthogonality_report(a):
    """Orthogonality checks over a tensor whose trailing 2 axes are square.

    Handles any number of LEADING batch axes: a [36,128,128] tensor is checked
    as 36 separate 128x128 matrices, a [4,8,64,64] as 32, and a bare [128,128]
    as one. Returns a list of (index_tuple, report) or None if not applicable.

    This exists because OSCAR may store rotations batched per layer/head rather
    than as bare 2D matrices; checking only ndim==2 would silently skip them.
    """
    if a is None or a.ndim < 2 or a.shape[-1] != a.shape[-2]:
        return None
    n = a.shape[-1]
    lead = a.shape[:-2]
    flat = a.reshape(-1, n, n)
    out = []
    for i in range(flat.shape[0]):
        idx = np.unravel_index(i, lead) if lead else ()
        out.append((tuple(int(x) for x in idx), orthogonality_report(flat[i])))
    return out


def orthogonality_report(a):
    """Reference-free checks for a square 2D matrix. Returns dict or None."""
    if a is None or a.ndim != 2 or a.shape[0] != a.shape[1]:
        return None
    n = a.shape[0]
    if n > 8192:
        return {"n": n, "skipped": "matrix too large for dense check"}
    ident = np.eye(n)
    resid_rt = np.linalg.norm(a @ a.T - ident) / np.sqrt(n)
    resid_tr = np.linalg.norm(a.T @ a - ident) / np.sqrt(n)
    try:
        sign, logdet = np.linalg.slogdet(a)
        det_note = f"sign={int(sign)} log|det|={logdet:.3e}"
    except np.linalg.LinAlgError:
        det_note = "slogdet failed"
    try:
        sv = np.linalg.svd(a, compute_uv=False)
        sv_note = {"min": float(sv.min()), "max": float(sv.max()),
                   "cond": float(sv.max() / max(sv.min(), 1e-300))}
    except np.linalg.LinAlgError:
        sv_note = None
    return {
        "n": n,
        "orth_resid_RRt": float(resid_rt),
        "orth_resid_RtR": float(resid_tr),
        "det": det_note,
        "singular_values": sv_note,
        "is_orthogonal": bool(max(resid_rt, resid_tr) < 1e-4),
    }


def stats(a):
    if a is None or a.size == 0:
        return None
    finite = np.isfinite(a)
    return {
        "min": float(np.min(a[finite])) if finite.any() else None,
        "max": float(np.max(a[finite])) if finite.any() else None,
        "mean": float(np.mean(a[finite])) if finite.any() else None,
        "absmean": float(np.mean(np.abs(a[finite]))) if finite.any() else None,
        "n_nan": int(np.isnan(a).sum()),
        "n_inf": int(np.isinf(a).sum()),
        "n_zero": int((a == 0).sum()),
    }


def file_digest(path, nbytes=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(nbytes):
            h.update(chunk)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------- main

def inspect_dir(root, label, max_files=None, do_hash=True):
    records = []
    paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            paths.append(os.path.join(dirpath, fn))
    paths.sort()
    if max_files:
        paths = paths[:max_files]

    print(f"\n{'=' * 78}")
    print(f"LABEL: {label}   ROOT: {root}   FILES: {len(paths)}")
    print(f"{'=' * 78}")

    schema_counter = Counter()
    shape_by_key = defaultdict(set)
    orth_summary = []

    for path in paths:
        rel = os.path.relpath(path, root)
        size = os.path.getsize(path)
        obj, err = load_any(path)
        if err:
            print(f"\n-- {rel}  ({size:,} B)  [UNREADABLE: {err}]")
            records.append({"file": rel, "bytes": size, "error": err})
            continue

        digest = file_digest(path) if do_hash else None
        print(f"\n-- {rel}  ({size:,} B)"
              + (f"  sha256:{digest}" if digest else ""))
        print(f"   container: {type(obj).__name__}")

        leaves = list(walk(obj))
        schema_counter[tuple(sorted(k for k, _ in leaves))] += 1

        file_rec = {"file": rel, "bytes": size, "sha256_16": digest,
                    "container": type(obj).__name__, "leaves": []}

        for key, leaf in leaves:
            arr = to_numpy(leaf)
            shape = tuple(getattr(leaf, "shape", ())) if arr is None else arr.shape
            shape_by_key[key].add(shape)
            st = stats(arr)
            batch = batched_orthogonality_report(arr)
            line = f"   {key:<44} shape={str(shape):<22} dtype={dtype_of(leaf)}"
            print(line)
            if st:
                print(f"       stats: min={st['min']:.6g} max={st['max']:.6g} "
                      f"mean={st['mean']:.6g} absmean={st['absmean']:.6g} "
                      f"nan={st['n_nan']} inf={st['n_inf']} zero={st['n_zero']}")
            if st and st["n_nan"] == 0 and st["n_inf"] == 0 and arr is not None:
                if arr.size and float(np.ptp(arr)) == 0.0:
                    print(f"       ** CONSTANT TENSOR ** every element == {arr.flat[0]:.6g}")
            leaf_orths = []
            for idx, orth in (batch or []):
                if orth is None:
                    continue
                tag = f"[{','.join(map(str, idx))}]" if idx else ""
                if "skipped" in orth:
                    print(f"       orth{tag}: {orth['skipped']}")
                    continue
                flag = "OK" if orth["is_orthogonal"] else "** NOT ORTHOGONAL **"
                print(f"       orth{tag}: ||RR^T-I||_F/sqrt(n)="
                      f"{orth['orth_resid_RRt']:.3e} {flag}  {orth['det']}")
                if orth["singular_values"]:
                    sv = orth["singular_values"]
                    print(f"       svd{tag}:  min={sv['min']:.6g} max={sv['max']:.6g} "
                          f"cond={sv['cond']:.6g}")
                orth_summary.append((rel, f"{key}{tag}", orth["is_orthogonal"],
                                     orth["orth_resid_RRt"]))
                leaf_orths.append({"index": list(idx), **orth})

            file_rec["leaves"].append({
                "key": key, "shape": list(shape), "dtype": dtype_of(leaf),
                "stats": st, "orth": leaf_orths,
            })

        records.append(file_rec)

    # ---- aggregate
    print(f"\n{'-' * 78}")
    print(f"SCHEMA SUMMARY for {label}")
    print(f"{'-' * 78}")
    for schema, count in schema_counter.most_common():
        print(f"  {count:>4} file(s) with {len(schema)} leaf key(s):")
        for k in schema[:24]:
            shapes = sorted(shape_by_key[k], key=str)
            shown = ", ".join(str(s) for s in shapes[:4])
            more = "" if len(shapes) <= 4 else f" (+{len(shapes) - 4} more)"
            print(f"        {k:<44} shapes: {shown}{more}")
        if len(schema) > 24:
            print(f"        ... and {len(schema) - 24} more keys")

    if orth_summary:
        bad = [r for r in orth_summary if not r[2]]
        print(f"\n  Square matrices checked: {len(orth_summary)}   "
              f"Non-orthogonal: {len(bad)}")
        worst = sorted(orth_summary, key=lambda r: -r[3])[:5]
        print("  Worst orthogonality residuals:")
        for rel, key, ok, resid in worst:
            print(f"    {resid:.3e}  {'OK ' if ok else 'BAD'}  {rel}::{key}")
    else:
        print("\n  No square 2D matrices found -- rotations may be stored "
              "batched (e.g. [n_heads, d, d]) or flattened. Check shapes above.")

    return {"label": label, "root": root, "files": records}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("directory")
    ap.add_argument("--label", default="unlabeled")
    ap.add_argument("--max-files", type=int, default=None,
                    help="inspect only the first N files (sorted)")
    ap.add_argument("--no-hash", action="store_true")
    ap.add_argument("--json", default=None, help="write full report to JSON")
    args = ap.parse_args()

    if not os.path.isdir(args.directory):
        sys.exit(f"not a directory: {args.directory}")

    report = inspect_dir(args.directory, args.label,
                         max_files=args.max_files, do_hash=not args.no_hash)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
