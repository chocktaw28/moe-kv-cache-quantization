# Phase 2 launch scripts (baseline reproduction)

> ## ⚠ Phase numbering: two different schemes
>
> **"Phase 2" here means THIS PROJECT's phase 2 — baseline reproduction.**
>
> OSCAR's own documentation numbers *its internal pipeline* — QKV dump,
> rotation calibration, evaluation — as phases 1, 2 and 3. Those are **not**
> this project's phases. In particular, OSCAR's "Phase 3" (its eval step) is
> what these scripts run, but it belongs to **this project's Phase 2**. This
> project's Phase 3 is a separate routing-instability pilot with no connection
> to these scripts.
>
> This collision has already caused confusion once. When reading logs or
> commit messages, check which scheme is in use.

Three arms for the BF16 vs INT2 evaluation, plus the shared driver they source.

| Script | KV cache | Rotations |
|---|---|---|
| `run_bf16_baseline.sh` | `bfloat16` | none (Oscar path disabled) |
| `run_int2_real.sh` | `int2`, group 128 | `$ROT_REAL` — locally calibrated |
| `run_int2_scrambled.sh` | `int2`, group 128 | `$ROT_SCRAMBLED` — negative control |
| `_common.sh` | — | shared logic, not executable on its own |
| `smoke_prompt.txt` | — | fixed smoke input (see `smoke_prompt_SOURCE.md`) |

The two INT2 arms differ in exactly three substantive lines (rotation dir,
expected hashes, arm name). Everything else lives in `_common.sh` so they
cannot drift apart.

## Usage

```sh
# on the pod, with the venv active

# smoke: free-form continuation, forced INT2 path (prefix=4, recent=8)
SMOKE=1 ./run_bf16_baseline.sh
SMOKE=1 ./run_int2_real.sh
SMOKE=1 ./run_int2_scrambled.sh

# real benchmark: GPQA, paper-knee windows (64/256), full 198 questions
./run_bf16_baseline.sh
./run_int2_real.sh
./run_int2_scrambled.sh
```

Run all three arms before drawing any conclusion. A single arm in isolation is
not interpretable.

## The smoke test is not GPQA

The smoke test checks **kernel liveness, not accuracy**, so it deliberately does
not use GPQA. GPQA's output is a single letter (A/B/C/D): a completely broken
INT2 path still emits a plausible-looking answer, so the test would pass while
testing nothing. GPQA prompts may also be short enough to sit entirely inside
the sink+recent windows, meaning zero tokens ever get quantized.

Instead, `SMOKE=1` runs a **free-form continuation**:

- Fixed prompt from `smoke_prompt.txt`, byte-identical across all three arms
  (public-domain Melville + deterministic labelled filler; see
  `smoke_prompt_SOURCE.md`).
- The script tokenizes it with **the model's own tokenizer** and **aborts** if
  the length does not exceed sink+recent — otherwise the INT2 path would not be
  exercised.
- ~300 tokens of greedy continuation, fixed seed, via `/v1/completions`
  (`http_server.py:1468`) — raw text, no chat template, no answer parsing, no
  scoring harness, no dataset download.
- Raw output written to `smoke_output.txt` per arm.

**Pass criterion is human-readable, by design.** Read the three outputs side by
side:

- BF16 and INT2-real → both coherent prose.
- INT2-scrambled → visibly degraded or broken.

If **real and scrambled read identically, the rotations are not being loaded**,
and every downstream number is meaningless.

## The two automated controls

**Positive control — KV pool size.** Each run writes `kv_pool_size.txt` and
`token_capacity.txt` from the server's own startup logs
(`memory_pool.py:808,814`, `scheduler.py:725`).

> **Gate: INT2 token capacity must be at least 3× BF16.**

The check screens for one specific failure — *INT2 path not engaged at all* —
which presents as a ratio near 1×. Bit-width arithmetic suggests ~8×, but the
BF16 sink/recent windows and allocator overhead pull the real figure down by an
amount **not stated in any source**. A tight bar would invite post-hoc
adjustment if the real answer came back at, say, 5.5×. The measured ratio is
reported prominently; 3× is only the automated gate.

**Negative control — scrambled rotations.** See the smoke test section above.
The interpretation is written into `run_int2_scrambled.sh` in advance so it
cannot be rationalised after the fact.

## Known risk: BF16 at 32K context may not fit

Before launching, each arm prints a KV-memory estimate and **warns loudly if it
looks marginal** — it never aborts, and never silently reduces context. An
asymmetric comparison between arms would be worse than no comparison, so
changing context length or hardware is an operator decision, applied to all
three arms together.

The estimate for Qwen3-8B (36 layers × 8 KV heads × 128 head_dim) at
`MEM_FRAC=0.72` on a 24 GB card:

| dtype | KV/token | @32K context | headroom after ~16 GB weights |
|---|---|---|---|
| bfloat16 | 144.0 KB | 4.50 GB | **−3.22 GB — does not fit** |
| int2 | 18.0 KB | 0.56 GB | +0.72 GB — marginal |

Override the estimator's assumptions with `EST_CONTEXT_LEN`, `EST_GPU_GB`,
`EST_WEIGHTS_GB`. These figures are estimates, not measurements; the server log
is authoritative.

## Overridable paths

| Var | Default |
|---|---|
| `OSCAR_SRC` | `~/oscar-src` |
| `ROT_REAL` | `~/recovery/rotations` |
| `ROT_SCRAMBLED` | `~/recovery/rotations_scrambled` |
| `RUN_ROOT` | `~/phase2_runs` |
| `MODEL` | `Qwen/Qwen3-8B` |

Output goes to `$RUN_ROOT/<UTC timestamp>_<arm>_<full|smoke>/`, containing
`resolved_config.txt`, `server.log`, `kv_pool_size.txt`, `token_capacity.txt`,
`SUMMARY.txt`, plus either `smoke_output.txt` or `eval.log` (absolute paths
throughout).

## Unvalidated starting values

`MAX_RUNNING=16`, `CUDA_GRAPH_MAX_BS=8`, `NUM_WORKERS=8` are conservative
reductions from upstream's 4-GPU defaults (64/32/32), chosen for a single 24 GB
card. They are **not derived from source and not validated on this hardware** —
expect to adjust them on first contact with the pod. They affect throughput and
memory headroom, not the correctness of the comparison.

## Deviations from upstream `eval_oscar_gpqa.sh`

Each is deliberate and traceable to source at commit `41ebcdba`:

- **Both attention backends `triton`.** Upstream hardcodes
  `--prefill-attention-backend fa3` (line 86); fa3 prefill crashes silently on
  sm89 with the INT2 KV path. See `ENVIRONMENT.md` §3.
- **`--kv-cache-dtype` is per-arm.** Upstream hardcodes `int2` (line 88), so a
  BF16 baseline is not expressible through it. `bfloat16` is a documented choice
  in `server_args.py:4170-4183`.
- **`--kv-cache-quant-group-size` omitted for BF16.** `server_args.py:6491-6498`
  raises `ValueError` if it is set with any dtype other than `int2`.
- **Greedy decoding, fixed seed.** Upstream defaults to `temperature=1.0`,
  `top_p=0.95`, `top_k=40` (sampled). These scripts use `0.0 / 1.0 / -1` with
  `--random-seed 0` so the arms differ only in the KV path. `top_k=-1` is
  SGLang's "disable" value and `temperature=0` selects greedy
  (`sampling_params.py:99-103`).
- **`TP_SIZE=1`, one GPU.** Upstream assumes `TP_SIZE=4` across 4 GPUs.
- **No conda.** Upstream `source`s a conda profile unconditionally (lines 49-50);
  these scripts use whatever Python environment is already active.
- **Smoke path bypasses the eval harness entirely.** Upstream has no smoke mode.
