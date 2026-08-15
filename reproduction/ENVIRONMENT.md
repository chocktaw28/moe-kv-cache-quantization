# Reproduction Environment Notes

Notes from reproducing [OSCAR](https://github.com/FutureMLS-Lab/OSCAR) (Offline
Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization,
[arXiv:2605.17757](https://arxiv.org/abs/2605.17757)) on a **single consumer GPU**
rather than the H100-class hardware the paper and repo assume.

Recorded because several of these cost hours to diagnose and none of them are
documented upstream. Nothing here is a criticism of OSCAR — it is a
production-oriented system built for datacenter GPUs, and these are the friction
points you hit when you take it somewhere it wasn't targeted.

**Scope:** this documents environment setup and baseline reproduction only.

---

## Hardware and base environment

| Item | Value |
|---|---|
| GPU | NVIDIA RTX 4090, 24 GB (24564 MiB reported) |
| Compute capability | **sm89** (Ada Lovelace) |
| Provider | RunPod (community cloud) |
| Base image | `runpod/pytorch:2.8.0-py3.11-cuda12.4.1-devel-ubuntu2204` |
| OSCAR commit | `41ebcdba3db5f0ce1339c3727caea80df575d437` (2026-06-28) |
| Torch (OSCAR venv) | `2.9.1+cu129` |
| Attention backends used | prefill: `triton`, decode: `triton` |

---

## 1. CUDA driver floor: 12.9

OSCAR's pinned `torch==2.9.1+cu129` requires a host driver supporting CUDA 12.9.
The first pod provisioned had a 12.8 host driver and failed at import time.

**Check before doing anything else on a new pod:**

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
nvidia-smi | grep "CUDA Version"
```

If the host shows < 12.9, destroy the pod and provision a different one. This is a
host-level property and cannot be fixed from inside the container.

## 2. Torch version conflict — use an isolated venv

The base image ships `torch 2.8.0+cu128`. Installing OSCAR's requirements on top of
it produces a broken mixed environment (mismatched `flashinfer` / `sgl_kernel`
against the wrong torch ABI).

Create an isolated venv and install OSCAR's pinned stack into it cleanly. Do **not**
`pip install` OSCAR's requirements into the image's default environment.

> **TODO:** capture `pip freeze` from a working venv and commit it as
> `requirements.lock.txt`. This was not captured before the pod was released, so
> the exact resolved dependency set is currently unrecorded. Do this first thing on
> the next working pod.

## 3. **fa3 prefill crashes silently on sm89**

The most expensive finding here, and the one most likely to affect others.

OSCAR's vendored FlashAttention-3 build supports **sm90 / sm100 / sm120**
(Hopper / Blackwell). On **sm89** (Ada Lovelace — RTX 4090, RTX 4000 Ada, L40S),
the fa3 *prefill* kernel crashes **silently** when combined with OSCAR's custom
INT2 KV-cache path. No traceback, no CUDA error, no non-zero exit — the process
simply dies.

The silence is what makes this costly: it looks like an OOM, a hang, or a RunPod
infrastructure event rather than an architecture mismatch.

**Fix:** use the Triton prefill backend, matching the decode backend that OSCAR
already defaults to.

```diff
 SERVER_ARGS=(
     --model-path "${MODEL}"
     --tensor-parallel-size "${TP_SIZE}"
-    --prefill-attention-backend fa3
+    --prefill-attention-backend triton
     --decode-attention-backend triton
     --kv-cache-dtype int2
```

See `eval_oscar_gpqa.patch` in this directory for the applied change.

Note also that `rotation/eval_oscar_gpqa.sh` documents a `PRE_ROPE_FA3` environment
variable in its header comment which is **never read anywhere in the codebase** —
the backends are hardcoded in `SERVER_ARGS`. Setting it has no effect.

**Cost of Triton prefill:** first server start takes ~320 s for kernel compilation
for a given config. Subsequent starts with an identical config are fast (cached).
Budget for this on every config change, and note that Triton cache is per-rank —
OSCAR already prepends a per-rank cache redirector so TP workers don't race.

## 4. conda assumption in the eval script

`rotation/eval_oscar_gpqa.sh` unconditionally runs:

```bash
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"
```

Many cloud images (including RunPod's PyTorch images) have no conda. With
`set -euo pipefail` this aborts the run immediately. Patched to fall back to an
already-active `$VIRTUAL_ENV`, and to fail loudly if neither is available. See the
patch file.

## 5. MEM_FRAC is tuned for multi-GPU production

The script default `MEM_FRAC=0.8` assumes a 4-GPU `TP_SIZE=4` production
deployment. On a single 24 GB card this OOMs.

`MEM_FRAC=0.72` with reduced batch and KV-window settings worked at **smoke-test
scale only**. This has **not** been validated at the full 32K-context production
scale, and should not be assumed to hold there.

> **~~TODO~~ RESOLVED.** The original note read: *"the working overrides were passed
> as environment variables at invocation rather than committed to a file, and were
> lost with the pod. Record them in a committed `run_config.env` next time rather
> than as shell-history-only command-line overrides."*
>
> Done — the working values are now committed in
> [`run_config.env`](run_config.env), which the launch scripts in `scripts/` source
> directly.
>
> No patch is or was required for this. `MEM_FRAC` is a plain environment variable
> with a shell default at `rotation/eval_oscar_gpqa.sh:38`:
>
> ```sh
> MEM_FRAC="${MEM_FRAC:-0.8}"
> ```
>
> so exporting `MEM_FRAC=0.72` before invocation is sufficient to override it. The
> value was only ever "lost" in the sense of being unrecorded, not in the sense of
> needing to be re-derived through a code change.

**Lesson worth generalising:** any tuning value discovered by trial and error
belongs in a committed file, not a command line.

## 6. Calibration reference run

For anyone comparing against a known-good run. Qwen3-8B, GPQA-Diamond calibration
set, `METHOD=qqt_sst`, composition `r_h_pbr`, `chunk=all`, `head_dim=128`:

| Stage | Result |
|---|---|
| Server ready | 320 s (first-time Triton kernel compilation) |
| Prompts sent | 198 GPQA-Diamond, ok=198, err=0 |
| Dump elapsed | 129.8 s |
| Prompts captured | **117** (dump token budget — see below; *not* dedup) |
| Layers captured | 36/36 |
| Peak VRAM | 20152 MiB of 24564 MiB |
| Rotation outputs | `k_rotation_qqt_r_h_pbr.pt`, `v_rotation_sst_r_h_pbr.pt` — 2,399,261 bytes each |

**Note the 198 → 117 reduction.** All 198 requests succeeded, but only 117 prompts
appear in the captured calibration set. These are two different numbers and should not
be conflated when reporting. The output directory name encodes the surviving count
(`seq30000_prompt117_group128`), which is a useful built-in provenance check.

**Update — the full chain is now established from source.** Read against OSCAR commit
`41ebcdba`. The original note (*"whether 117 is the expected post-dedup count or
represents a silent drop has not been confirmed against upstream"*) is resolved: 117 is
expected behaviour, nothing was silently dropped, and the mechanism is a token budget,
not dedup.

**1. A token budget is set.** `save_qkv_8b.sh:21`:

```sh
export DUMP_KVCACHE_TOKENS="${DUMP_KVCACHE_TOKENS:-30000}"
```

Documented in the OSCAR repo's own `README.md:368`, in the calibration-knobs table:
`DUMP_KVCACHE_TOKENS` — default 30000 — "Total token budget for calibration".

**2. The dump hook accumulates per layer and truncates the final chunk.**
`sglang-dump-qkv/python/sglang/srt/layers/attention/triton_backend.py:850-857`:

```python
dump_tokens   = get_int_env_var("DUMP_KVCACHE_TOKENS", 100)
saved_so_far  = self._dump_saved_tokens.get(layer_id, 0)
remaining     = dump_tokens - saved_so_far
if remaining > 0:
    num_tokens = q.shape[0]
    tokens_to_save = min(num_tokens, remaining)      # truncates the final batch
```

**3. Once the budget is filled, the layer is marked done.** Lines 918-923:

```python
self._dump_saved_tokens[layer_id] = saved_so_far + tokens_to_save
self._dump_chunk_idx[layer_id] = chunk_idx + 1
if saved_so_far + tokens_to_save >= dump_tokens:
    self._dump_kv_done_layers.add(layer_id)      # stop dumping this layer
```

**4. The per-sequence lengths recorded in that final chunk are themselves clipped to
what fit**, so the last prompt counted is likely partial, not complete. Lines 873-882:

```python
chunk_seq_lens = []
if forward_batch.extend_seq_lens is not None:
    remain = tokens_to_save
    for slen in forward_batch.extend_seq_lens.tolist():
        if remain <= 0:
            break
        take = min(slen, remain)          # <-- clipped to remaining budget
        chunk_seq_lens.append(take)
        remain -= take
```

**5. The prompt count is computed after the fact**, by summing those recorded lengths —
it is an output of the dump, not an input to it. `save_qkv_8b.sh:149-160`:

```sh
N_PROMPTS=$("${PY}" - "${DUMP_KVCACHE_DIR}/layer_0/seq_lens" <<'PYEOF'
import os, sys, torch
seq_dir = sys.argv[1]
total = 0
for f in sorted(os.listdir(seq_dir), key=lambda x: int(x.split('.')[0])):
    s = torch.load(os.path.join(seq_dir, f), weights_only=True, map_location='cpu')
    total += len(s.tolist())
print(total)
PYEOF
)
```

**6. That count is written into the output directory name.** `save_qkv_8b.sh:162`:

```sh
FINAL_TAG="seq${DUMP_KVCACHE_TOKENS}_prompt${N_PROMPTS}_group${GROUP_SIZE}"
```
— which is exactly `seq30000_prompt117_group128`, the directory this run produced.

**In plain terms:** all 198 requests genuinely succeeded — nothing failed, nothing was
dropped by the server or the client. The dump hook simply stopped *recording* once it
had banked the 30,000 tokens it was told to collect. "117" means "prompts that
contributed at least one recorded token," not "complete prompts captured" — per point 4,
the last one is likely truncated mid-sequence.

**Independent corroboration from a second run.** RotationZoo's own Qwen3-8B checkpoint
(compared against directly in Stage A/B) carries the directory name
`seq20000_prompt83_group128` — a different budget (20,000 vs my 30,000), same model,
same GPQA-Diamond-style calibration set:

```
mine:        30000 / 117 = 256.4 tokens/prompt
RotationZoo: 20000 /  83 = 241.0 tokens/prompt
```

Two independent runs, two different token budgets, land within ~6% of each other at
~250 tokens/prompt. The budget mechanism doesn't just explain my run in isolation — it
quantitatively predicts the RotationZoo count too, from an unrelated calibration run.
This is real corroboration, not just an unfalsifiable story fitted to one data point.

**A hypothesis that turned out to be wrong, recorded as a miss.** The 198 → 117 gap was
initially attributed to `load_tensor()` in `rotation/compute_kv_rotation.py:49-72`,
whose `"all"` mode does deliberately discard chunk 0:

> ```
> "all" mode skips chunk 0 (a 6-token warmup batch produced by the prefill
> schedule that would dominate hessian estimation with degenerate samples).
> ```
>
> ```python
> chunk_paths = [p for p in chunk_paths if int(p.stem) != 0]
> ```

That skip is real, and this calibration ran `chunk=all`, so it did apply. But it does
**not** explain the prompt count: the skip happens in the *calibration* step, while 117
is counted in the *dump* step, before `compute_kv_rotation.py` ever runs. The two are
sequential, not the same reduction. The discarded chunk 0 is 6 tokens of warmup, which
is also far too small to account for a 198 → 117 gap.

**What remains unverified, even now.** The budget explains the mechanism, predicts the
right order of magnitude, and is corroborated by a second independent run landing at a
consistent tokens/prompt figure. It does **not** amount to having independently confirmed
that 117 is the *exact* count 30,000 tokens should yield for this specific set of
GPQA-Diamond prompts and this scheduler's batching — that would require summing the
dumped `seq_lens` tensors directly for this run, which has not been done. As before:
OSCAR's paper describes calibration as lightweight and low-sensitivity to calibration
domain, so 117 prompts is plausibly sufficient — that part remains an assumption, not a
verified claim.

Artifact hashes for the run above are in `rotations_sha256.txt`.

## 7. RunPod-specific operational notes

**File transfer.** The `ssh.runpod.io` proxy is **PTY-only**. It silently rejects:

- `scp` (both SFTP and legacy `-O` modes) — fails with
  `subsystem request failed on channel 0`, or exits 0 having transferred nothing
- non-interactive `ssh host "cat file" > local` — returns
  `Error: Your SSH client doesn't support PTY` written *into* the output file,
  producing a 43-byte "archive"

Use `runpodctl` instead. Both ends must be live simultaneously:

```bash
# on the pod
runpodctl send /workspace/recovery.tar.gz     # prints a one-time code

# on the local machine (writes to CWD, not to a specified path)
runpodctl receive <code>
```

On macOS install via the **cask** (prebuilt binary) — the formula builds from
source and will fail on outdated Command Line Tools:

```bash
brew install --cask runpod/runpodctl/runpodctl
```

Alternatively, provision pods with a direct TCP port exposed, which gives a real
`sshd` that `scp` works against normally. Worth doing by default.

**Container disk vs volume.** Only `/workspace` (the network volume) survives pod
migration and hardware reclaim events. Anything on the container disk — including
venvs, installed CLI tools, and logs written outside `/workspace` — is lost. Place
venvs on the volume.

**Hardware reclaim is a real event.** This project survived one mid-run reclaim.
Assume the pod is disposable and treat a pulled artifact bundle as the only durable
state.

## 8. Verifying artifacts — a process note

Agent-driven pipelines (Claude Code and similar) report output paths **relative to
whatever working directory they happened to be in**. A summary line reading
`Output: rotation/qwen3-8B/.../k_rotation.pt` is ambiguous by construction and cost
a substantial detour here, because the true location was under the nested `OSCAR/`
repo directory rather than the project root.

Two habits that remove this class of problem entirely:

1. **Require absolute paths in run summaries.** Cheap to specify, eliminates the
   ambiguity.
2. **Human-verify every claimed artifact.** End each phase with a manually run
   `ls -la` and `sha256sum` on the claimed outputs, then pull them off the pod
   immediately. An automated summary asserting a file exists is not evidence that
   it does — but note that in this case the summary was in fact accurate, and the
   apparent discrepancy was purely path resolution. Verify rather than assume in
   either direction.

---

## Applied patch

`eval_oscar_gpqa.patch` — 46 lines against
`41ebcdba3db5f0ce1339c3727caea80df575d437`, covering items 3 and 4 above. Item 5
(MEM_FRAC) was applied as a runtime override and is not in the patch.

## License

OSCAR is MIT-licensed. This directory contains reproduction notes and a small patch
against it; the rotation checkpoint files themselves are not redistributed here.
