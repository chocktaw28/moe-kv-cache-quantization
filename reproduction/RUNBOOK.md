# Phase 2 Pod Session Runbook

Anhydrous / Angle B, Phase 2 (baseline reproduction). This is an execution
document: every decision that needed thought should already be made below. If
you find yourself thinking during the session, that's a sign this runbook was
incomplete, not a sign to improvise — note the gap and come back to it after.

**Hardware for this session: RunPod A100 SXM 80GB, $1.59/hr, 117GB RAM, 16
vCPU.**

> ⚠ **This is sm80, not sm89.** `ENVIRONMENT.md` was written against an RTX
> 4090 (sm89). The A100 SXM is Ampere — compute capability sm80. **Both attention backends stay `triton` anyway, deliberately.** This is
> not an oversight to "fix" mid-session: sm80 is one of the architectures fa3
> explicitly claims to support in `flash_attention_v3.py:76` ("fa3 is supported
> for sm80/sm87 and sm86/sm89"), so fa3 might well work here — but switching
> backends now would introduce a new variable into a comparison that was
> validated end-to-end on triton/triton. The point of this session is running
> the validated configuration at scale, not re-opening the backend question.
> Leave it alone.

**Budget: ~$15 credit ≈ 9.4 hours at $1.59/hr.** Cost checkpoints are called
out at each major step. **Hard checkpoint: if 6 hours elapse without reaching
Step 6 (the benchmark), stop and reassess rather than burning the remainder.**
Running out of budget mid-benchmark is worse than stopping deliberately with
time to spare.

Every command below is traceable to a committed file — `ENVIRONMENT.md`,
`run_config.env`, `scripts/*.sh`, `scripts/README.md`, or the Stage A/B
reports. Nothing here is invented. Where a value is genuinely pod-specific
(pod ID, IP), that's called out explicitly with where to get it.

---

## Condensed checklist (session reference)

Tick these off in order. Full detail for each is below.

```
[ ] 0.  Pre-flight: local files + hashes in hand, patch in hand, budget noted
[ ] 1a. Provision A100 SXM 80GB pod
[ ] 1b. Check host CUDA driver >= 12.9        -- HARD GATE, destroy+reprovision if not
[ ] 1c. Confirm compute_cap == 8.0 (sm80)
[ ] 2a. Create venv on /workspace
[ ] 2b. Install OSCAR's pinned stack into it
[ ] 2c. pip freeze > /workspace/pip_freeze.txt
[ ] 2d. Pull pip_freeze.txt off the pod IMMEDIATELY (before anything else)
[ ] 3a. runpodctl send both rotation dirs from local machine
[ ] 3b. runpodctl receive on pod
[ ] 3c. Verify all 4 sha256 hashes ON THE POD
[ ] 3d. Apply eval_oscar_gpqa.patch
[ ] 4a. Edit run_config.env: comment 4090 block, uncomment 80GB block
[ ] 4b. cat run_config.env -- confirm exactly ONE live MEM_FRAC line
[ ] 5a. SMOKE=1 run_bf16_baseline.sh
[ ] 5b. SMOKE=1 run_int2_real.sh
[ ] 5c. SMOKE=1 run_int2_scrambled.sh
[ ] 5d. Check token_capacity ratio >= 3x
[ ] 5e. Read all 3 smoke outputs side by side, judge coherence
[ ] 5f. GATE: real vs scrambled must differ. If not, STOP.
[ ] 6a. run_bf16_baseline.sh (full GPQA, no SMOKE)                -- COST CHECKPOINT
[ ] 6b. run_int2_real.sh (full GPQA, no SMOKE)
[ ] 6c. Compare scores: single-digit gap = pass, collapse = investigate
[ ] 7.  OPTIONAL, only if >1hr budget left: OLMoE launch check (~15 min, tight scope)
[ ] 8a. Pull all artifacts off the pod via runpodctl
[ ] 8b. Verify pulled artifacts by hash where hashes exist
[ ] 8c. Stop the pod
[ ] 8d. Record GPU hours and cost actually spent
```

---

## Step 0 — Pre-flight (before provisioning anything)

Have these in hand on the local machine before spending a cent of pod time.

**Rotation directories and their hashes** — confirm they're present locally:

```bash
ls -la ~/recovery/rotations ~/recovery/rotations_scrambled
shasum -a 256 ~/recovery/rotations/*.pt ~/recovery/rotations_scrambled/*.pt
```

Expect exactly these four lines (order may vary):

```
ca8a0198f630dabbb85a186388355c844a1721782e1c08e4355558ca3424205a  k_rotation_qqt_r_h_pbr.pt   (real)
802088639ac53162193b591ee67a6aff289a867ac533d89a043142172e15af49  v_rotation_sst_r_h_pbr.pt   (real)
cc5313264cc7edc527b55d982bbe4d8c36938e752ddb3f2f09588c5ccc056aa1  k_rotation_qqt_r_h_pbr.pt   (scrambled)
18815a709923f79ae1806b3c86875c6667e4e31cbb2bc4840dfefc299b4c391f  v_rotation_sst_r_h_pbr.pt   (scrambled)
```

If any hash differs from these, **stop** — do not provision. Something in the
local state has changed since Stage A/B validated it, and that needs
resolving before spending pod time.

**Patch present:**

```bash
ls -la ~/moe-kv-cache-quantization/reproduction/eval_oscar_gpqa.patch
```

**Budget awareness:** $15 credit, A100 SXM 80GB at $1.59/hr ≈ 9.4 hours total.
6-hour hard checkpoint before the benchmark step. Know the time now, before
the clock starts.

**runpodctl installed locally** (needed for Step 3 and Step 8):

```bash
brew install --cask runpod/runpodctl/runpodctl
```

(cask, not formula — the formula builds from source and fails on outdated
Command Line Tools; `ENVIRONMENT.md` §7.)

---

## Step 1 — Provision and verify

**1a. Provision.** In the RunPod console: select **A100 SXM 80GB**, community
or secure cloud, confirm the advertised rate is ~$1.59/hr before deploying.
Use the base image already validated in `ENVIRONMENT.md`:
`runpod/pytorch:2.8.0-py3.11-cuda12.4.1-devel-ubuntu2204`, unless a newer
equivalent is the only one offered — if so, note the substitution explicitly
in the run summary at Step 8.

> Pod ID / SSH details are assigned at provision time and can't be given
> literally here. Get them from the RunPod console's "Connect" panel for this
> pod once it's running.

**1b. CUDA driver floor — HARD GATE.** Per `ENVIRONMENT.md` §1, OSCAR's pinned
`torch==2.9.1+cu129` requires a host driver supporting CUDA ≥ 12.9. This is a
host-level property; it cannot be patched from inside the container.

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv
nvidia-smi | grep "CUDA Version"
```

**Expected:** GPU name shows `A100-SXM4-80GB` (or equivalent A100 SXM
variant), CUDA Version ≥ 12.9.

**If CUDA Version < 12.9:** destroy this pod and provision a different one.
Do not attempt to work around it in-container — the first 4090 session hit
exactly this and it cannot be fixed short of a different host.

**1c. Confirm compute capability is sm80.**

```bash
python3 -c "import torch; print(torch.cuda.get_device_capability())"
```

**Expected:** `(8, 0)`. This is the whole point of the sm80-vs-sm89 note at
the top of this document — confirm it now so there's no ambiguity later about
which architecture produced which results.

---

## Step 2 — Environment (capture pip freeze IMMEDIATELY)

Per `ENVIRONMENT.md` §2: the base image ships `torch 2.8.0+cu128`; installing
OSCAR's requirements on top of it produces a broken mixed environment. Use an
isolated venv, and place it on `/workspace` — per §7, only the network volume
survives migration/reclaim; the container disk does not.

**2a. Create the venv on the volume:**

```bash
python3 -m venv /workspace/oscar-venv
source /workspace/oscar-venv/bin/activate
python -m pip install --upgrade pip
```

**2b. Install OSCAR's pinned stack.** Use whatever OSCAR's own install
instructions specify (its `README.md` / `requirements*.txt` — not
reproduced here since the pinned versions may have moved; check the OSCAR
repo at the pinned commit `41ebcdba3db5f0ce1339c3727caea80df575d437` for the
current install command). Confirm afterward:

```bash
python -c "import torch; print(torch.__version__)"
```

**Expected:** `2.9.1+cu129` (matching `ENVIRONMENT.md`'s hardware table). If a
different version resolves, note it — it changes what this run can be
compared against.

**2c/2d. Capture and pull `pip_freeze.txt` before anything else.** This is
called out because it was lost in the previous session and its absence is a
standing, previously-flagged gap (`ENVIRONMENT.md` §2 TODO). Do this now,
before Step 3, so it can't be lost to a reclaim event later:

```bash
pip freeze > /workspace/pip_freeze.txt
wc -l /workspace/pip_freeze.txt   # sanity check it's non-trivial, expect 100+ lines
```

Then, **from the local machine** (not the pod):

```bash
# on the pod first:
runpodctl send /workspace/pip_freeze.txt
# prints a one-time code, e.g. "8146-galileo-bronze-tokyo" -- copy it

# on the local machine, immediately (both ends must be live simultaneously):
runpodctl receive <code-from-above>
```

Confirm the file landed locally (`ls -la pip_freeze.txt`) before proceeding.
This closes the standing gap — do not skip it even if the session feels
time-pressured.

**Cost so far:** ~15-30 min → ~$0.40-0.80.

---

## Step 3 — Restore state

Per `ENVIRONMENT.md` §7: **`ssh.runpod.io` is PTY-only.** `scp` (both modes)
and non-interactive `ssh host "cat file" > local` both fail — the latter
silently, writing a 43-byte error string into what looks like a transferred
file. Use `runpodctl` for everything, or a direct TCP port if the pod offers
one.

**3a/3b. Transfer both rotation directories.** From the local machine:

```bash
cd ~/recovery
tar czf rotations.tar.gz rotations rotations_scrambled
runpodctl send rotations.tar.gz
# prints a one-time code
```

On the pod, immediately:

```bash
mkdir -p /workspace/recovery
cd /workspace/recovery
runpodctl receive <code-from-above>
tar xzf rotations.tar.gz
ls -la rotations rotations_scrambled
```

**Expected:** both directories present, each containing
`k_rotation_qqt_r_h_pbr.pt` and `v_rotation_sst_r_h_pbr.pt`, 2,399,261 bytes
each (per Stage A).

**3c. Verify all four hashes ON THE POD** — never trust a transfer without
re-verifying on the receiving end:

```bash
sha256sum /workspace/recovery/rotations/k_rotation_qqt_r_h_pbr.pt
sha256sum /workspace/recovery/rotations/v_rotation_sst_r_h_pbr.pt
sha256sum /workspace/recovery/rotations_scrambled/k_rotation_qqt_r_h_pbr.pt
sha256sum /workspace/recovery/rotations_scrambled/v_rotation_sst_r_h_pbr.pt
```

**Expected, exactly:**

```
ca8a0198f630dabbb85a186388355c844a1721782e1c08e4355558ca3424205a  .../rotations/k_rotation_qqt_r_h_pbr.pt
802088639ac53162193b591ee67a6aff289a867ac533d89a043142172e15af49  .../rotations/v_rotation_sst_r_h_pbr.pt
cc5313264cc7edc527b55d982bbe4d8c36938e752ddb3f2f09588c5ccc056aa1  .../rotations_scrambled/k_rotation_qqt_r_h_pbr.pt
18815a709923f79ae1806b3c86875c6667e4e31cbb2bc4840dfefc299b4c391f  .../rotations_scrambled/v_rotation_sst_r_h_pbr.pt
```

> ⚠ **Both k-files start with `c`** (`ca8a…` vs `cc53…`). Compare the **full**
> hash, never a prefix. This is exactly the mix-up
> `scripts/_common.sh::verify_rotations()` exists to catch automatically at
> launch time — but verify here too, by hand, before relying on the automated
> check.

**If any hash mismatches:** stop. Do not proceed to Step 4. Re-transfer that
file; do not assume a "close enough" partial transfer is usable.

**3d. Apply the patch:**

```bash
cd ~/oscar-src   # or wherever OSCAR was cloned on this pod
git apply --check /path/to/moe-kv-cache-quantization/reproduction/eval_oscar_gpqa.patch
git apply /path/to/moe-kv-cache-quantization/reproduction/eval_oscar_gpqa.patch
git diff --stat   # expect rotation/eval_oscar_gpqa.sh, ~46 lines changed
```

**Expected:** `git apply --check` exits 0 with no output. If it reports
conflicts, the pod's OSCAR checkout isn't at `41ebcdba3db5f0ce1339c3727caea80df575d437`
— fix the checkout before continuing rather than force-applying.

Note: the three launch scripts (`scripts/run_*.sh`) build their own
`SERVER_ARGS` directly and don't invoke `eval_oscar_gpqa.sh` at all — see
`scripts/README.md`'s "Deviations from upstream" section. The patch is
recorded for provenance and is relevant if `eval_oscar_gpqa.sh` is ever run
directly, but the runbook's Steps 5-6 use the committed scripts, which don't
need it to function.

**Cost so far:** ~20-40 min → ~$0.70-1.30.

---

## Step 4 — Config for 80GB  ⚠ EASY TO FORGET, MAKE THIS PROMINENT

`run_config.env` currently has the **24GB/4090 block ACTIVE** (uncommented)
and the **larger-card block COMMENTED OUT**. If this step is skipped, the
scripts will run — nothing will error — but at `MEM_FRAC=0.72` and
`MAX_RUNNING=16`/`CUDA_GRAPH_MAX_BS=8` tuned for a 24GB card, most of an 80GB
card's memory and throughput will sit unused. The run will "work" and waste
most of the session's value. **Nothing will tell you this happened — check it
explicitly.**

**4a. Edit `run_config.env`.** Comment out the active 4090 block, uncomment
the reference 80GB block:

```bash
cd ~/moe-kv-cache-quantization/reproduction
```

Open `run_config.env` and change it from:

```sh
# --- ACTIVE: RTX 4090, 24GB, sm89 -------------------------------------------
MEM_FRAC=0.72
# Paper-knee KV windows (defaults, stated explicitly for the record):
SGLANG_MIXED_KV_PREFIX_TOKENS=64
SGLANG_MIXED_KV_RECENT_TOKENS=256

# --- REFERENCE (not active): larger card for the benchmark run -------------
# ...
# MEM_FRAC=0.8
# MAX_RUNNING=64
# CUDA_GRAPH_MAX_BS=32
```

to:

```sh
# --- REFERENCE (not active): RTX 4090, 24GB, sm89 ---------------------------
# MEM_FRAC=0.72
# Paper-knee KV windows (defaults, stated explicitly for the record):
SGLANG_MIXED_KV_PREFIX_TOKENS=64
SGLANG_MIXED_KV_RECENT_TOKENS=256

# --- ACTIVE: larger card for the benchmark run ------------------------------
# ...
MEM_FRAC=0.8
MAX_RUNNING=64
CUDA_GRAPH_MAX_BS=32
```

The `SGLANG_MIXED_KV_PREFIX_TOKENS`/`RECENT_TOKENS` lines are model/method
properties, not hardware — leave them uncommented and unchanged either way,
per the comment already in the file.

**4b. Confirm exactly one `MEM_FRAC` line is live:**

```bash
grep -n "^MEM_FRAC\|^#.*MEM_FRAC\|^# MEM_FRAC" run_config.env
```

**Expected:** exactly one line **without** a leading `#`, reading
`MEM_FRAC=0.8`. If you see `MEM_FRAC=0.72` uncommented, or both lines
uncommented, or neither, fix it before proceeding — `_common.sh:75` will
otherwise either use the wrong value or (if genuinely both are somehow active)
take whichever the shell resolves last, silently.

Also confirm `MAX_RUNNING` and `CUDA_GRAPH_MAX_BS` are live:

```bash
grep -n "^MAX_RUNNING\|^CUDA_GRAPH_MAX_BS" run_config.env
```

**Expected:** `MAX_RUNNING=64` and `CUDA_GRAPH_MAX_BS=32`, both uncommented.
Per the file's own comments, these are upstream's own 4-GPU defaults reused
as **starting points for one larger GPU — not derived, not validated**. If
Step 5 or 6 shows signs of memory pressure or throughput problems, these two
are the first things to revisit, but do not preemptively tune them now.

**Cost so far:** ~25-45 min → ~$0.80-1.50.

---

## Step 5 — Smoke test (3 arms, short, SMOKE=1)

Per `scripts/README.md`: the smoke test checks **kernel liveness, not
accuracy** — it deliberately does not use GPQA. It forces the INT2 mixed-KV
window path to engage at short context (`prefix=4, recent=8` instead of the
paper-knee `64/256`), and produces free-form prose that a human reads for
coherence.

```bash
cd ~/moe-kv-cache-quantization/reproduction/scripts
export OSCAR_SRC=~/oscar-src              # adjust if OSCAR is checked out elsewhere
export ROT_REAL=/workspace/recovery/rotations
export ROT_SCRAMBLED=/workspace/recovery/rotations_scrambled
export RUN_ROOT=/workspace/phase2_runs

SMOKE=1 ./run_bf16_baseline.sh
SMOKE=1 ./run_int2_real.sh
SMOKE=1 ./run_int2_scrambled.sh
```

Each run creates its own timestamped directory under `$RUN_ROOT` and prints
its path at the end (`log "done: ${RUN_DIR}"`). Note all three `RUN_DIR`
paths as they print — they're needed for the checks below and for Step 8.

**Expected during each run, in order:**
1. Resolved config echoed to terminal (model, TP, mem_frac, KV args, etc.)
2. `server ready` within ~20 min (first run of a new config compiles Triton
   kernels, ~320s per `ENVIRONMENT.md` §3 — budget for this once per distinct
   config, not once per arm, since triton cache persists across arms in the
   same session as long as `$RUN_ROOT`'s triton cache isn't cleared)
3. KV pool size block printed (the positive control, see below)
4. `smoke prompt: N tokens` + `OK: N tokens exceeds window by ...`
5. `wrote .../smoke_output.txt (N chars)`
6. `done: <RUN_DIR>`

**If a server dies during startup** (`[FATAL] server died during startup` /
`[FATAL] server not ready after 20 min`): this is exactly the silent-death
failure mode documented in `ENVIRONMENT.md` §3, except that on sm80 fa3
*should* be nominally supported per source, and these scripts already force
both backends to `triton` regardless — so this class of failure being
architecture-related is less likely here than it was on the 4090. Check
`tail -100 <RUN_DIR>/server.log` for whatever the real cause is (most likely
here: an actual OOM, since `MAX_RUNNING=64`/`CUDA_GRAPH_MAX_BS=32` are
unvalidated starting values per Step 4). If it's OOM, that's informative on
its own — record it, don't just retry blindly.

### Pre-registered smoke criteria

| # | Check | Pass | Fail action |
|---|---|---|---|
| 1 | KV token capacity ratio | INT2 `token_capacity.txt` ≥ 3× BF16's | Near 1× means the INT2 path did not engage — **STOP**, do not proceed to Step 6 |
| 2 | Smoke prompt length | Script's own assertion passes (it aborts automatically if not — `_common.sh::assert_smoke_prompt_length`) | Script already refuses to run if this fails; nothing to check manually beyond confirming it printed `OK:` not `[FATAL]` |
| 3 | BF16 output | Coherent prose | Investigate before proceeding |
| 4 | INT2-real output | Coherent prose | Investigate before proceeding |
| 5 | INT2-scrambled output | **Visibly degraded** vs INT2-real | If it reads the same as real, the rotations are **not being loaded** — **STOP**, do not proceed to Step 6 |
| 6 | Any silent process death | None | **STOP** and investigate; do not retry-and-ignore |

**5d. Check the capacity ratio:**

```bash
BF16_DIR=$(ls -dt /workspace/phase2_runs/*_bf16_baseline_smoke | head -1)
REAL_DIR=$(ls -dt /workspace/phase2_runs/*_int2_real_smoke | head -1)
SCRAM_DIR=$(ls -dt /workspace/phase2_runs/*_int2_scrambled_smoke | head -1)

echo "BF16:      $(cat "${BF16_DIR}/token_capacity.txt")"
echo "INT2 real: $(cat "${REAL_DIR}/token_capacity.txt")"
echo "INT2 scr:  $(cat "${SCRAM_DIR}/token_capacity.txt")"
python3 -c "
bf16 = int(open('${BF16_DIR}/token_capacity.txt').read())
real = int(open('${REAL_DIR}/token_capacity.txt').read())
print(f'ratio real/bf16 = {real/bf16:.2f}x  ({\"PASS\" if real/bf16 >= 3 else \"FAIL — STOP\"})')"
```

**5e. Read all three smoke outputs side by side:**

```bash
echo "=== BF16 ==="; cat "${BF16_DIR}/smoke_output.txt"; echo
echo "=== INT2 REAL ==="; cat "${REAL_DIR}/smoke_output.txt"; echo
echo "=== INT2 SCRAMBLED ==="; cat "${SCRAM_DIR}/smoke_output.txt"; echo
```

Judge by eye per the table above. This is deliberately not automated — per
`scripts/README.md`, "pass criterion is human-readable, by design."

**5f. Gate.** If real and scrambled read the same, or capacity ratio is near
1×, **stop the session here**. Every downstream number would be
uninterpretable. Do not proceed to Step 6 hoping it resolves itself at scale.

**Cost so far:** ~45-75 min → ~$1.20-2.00 (mostly the one-time ~320s Triton
compile plus three short generations).

---

## Step 6 — Benchmark (only if Step 5 passes)

**⚠ COST CHECKPOINT: check elapsed session time now. If more than 6 hours
have elapsed since Step 1 began, stop and reassess rather than starting this
step.** Full GPQA (198 questions × up to 32768 tokens, two arms) is the
expensive part of this session; starting it without headroom risks a run that
gets killed mid-way, wasting the spend already made.

Full GPQA, one seed, greedy decoding, **paper-knee windows (64/256)** — i.e.
**do not set `SMOKE`**. BF16 and INT2-real only; the scrambled arm's job was
done in Step 5 and does not need a full benchmark run.

```bash
cd ~/moe-kv-cache-quantization/reproduction/scripts
./run_bf16_baseline.sh
./run_int2_real.sh
```

**Expected:** each takes substantially longer than the smoke runs (198
real questions vs. one 300-token continuation). Watch `server.log` /
`runner.log` for the running eval progress if curious, but the scripts are
designed to run unattended — `tee` handles the logging.

**On completion, each prints a `SUMMARY.txt` with a `--- score ---` block**
(grep for `gpqa/score`/`gpqa/chars` in `eval.log`). Read both:

```bash
BF16_FULL=$(ls -dt /workspace/phase2_runs/*_bf16_baseline_full | head -1)
REAL_FULL=$(ls -dt /workspace/phase2_runs/*_int2_real_full | head -1)
echo "=== BF16 ==="; cat "${BF16_FULL}/SUMMARY.txt"
echo "=== INT2 REAL ==="; cat "${REAL_FULL}/SUMMARY.txt"
```

### How to read the result — stated plainly, in advance

**Compare MY INT2 arm against MY BF16 arm from this same session, on this
same hardware, with this same prefill backend and this same locally-calibrated
rotation set.** Do **not** compare against OSCAR's published BF16 numbers
from their paper or repo. Different hardware, different prefill backend
(triton here vs. whatever they benchmarked with), different calibration set
(this run's own rotations vs. RotationZoo's) — a cross-comparison against
their published figure would be uninterpretable, not just imprecise.

**Phase 2 passes if the INT2-vs-BF16 gap here is single-digit points, not a
collapse.** It does **not** need to reproduce OSCAR's reported 1.42-point gap
exactly — claiming a quantitative match to their figure from a different
hardware/calibration setup would be overclaiming, given everything Stage A/B
already established about this being a genuinely independent calibration
(different prompt count, different token budget) rather than a bit-identical
reproduction.

If the gap looks like a collapse (INT2 dramatically worse, not single digits):
the smoke test already confirmed the rotations load and are visibly different
from garbage, and the capacity ratio confirmed the path engages — so a
collapse at this stage is a real, informative negative result about accuracy,
not a plumbing bug to chase. Record it as such rather than assuming
something's broken.

**Cost so far:** highly variable depending on GPQA throughput on this
hardware; budget the remainder of the 6-hour checkpoint window for this step
and stop if it runs long, per the hard checkpoint above.

---

## Step 7 — Optional: OLMoE launch check (only with >1hr budget left)

**Scope this tightly. ~15 minutes total. Launch check only.** Do not expand
this into a calibration run, a benchmark, or a debugging session. This step
exists to answer exactly two yes/no questions and stop.

**Precondition:** confirm `sglang-research/python/sglang/srt/models/olmoe.py`
exists in this OSCAR checkout (it does, as of commit `41ebcdba`) — meaning
base architecture support is already present. The only open question is
whether the INT2 KV path works with it; this step does not investigate why
if it doesn't.

**Question 1: does the server start with OLMoE and `--kv-cache-dtype int2`?**

```bash
cd ~/oscar-src
source /workspace/oscar-venv/bin/activate
export PYTHONPATH="${OSCAR_SRC}/rotation/_triton_per_rank:${OSCAR_SRC}/sglang-research/python:${PYTHONPATH:-}"
timeout 300 python -m sglang.launch_server \
    --model-path allenai/OLMoE-1B-7B-0924 \
    --tensor-parallel-size 1 \
    --prefill-attention-backend triton \
    --decode-attention-backend triton \
    --kv-cache-dtype int2 \
    --kv-cache-quant-group-size 128 \
    --mem-fraction-static 0.8 \
    --host 127.0.0.1 --port 31099 \
    --trust-remote-code \
    2>&1 | tee /workspace/olmoe_launch_check.log &
sleep 120
curl -s http://127.0.0.1:31099/health && echo " -- SERVER UP" || echo " -- SERVER DID NOT COME UP"
kill %1 2>/dev/null || true
```

**Record whichever happens — pass or fail — and move on either way.**

**Question 2 (only if Question 1 passed): does OSCAR's QKV dump script run for
a handful of prompts?** There's no OLMoE-specific dump script in this repo;
`rotation/qwen3-8B/save_qkv_8b.sh` reads `MODEL` from the environment with a
default, so reuse it with `MODEL` overridden and a tiny prompt count:

```bash
cd ~/oscar-src
MODEL=allenai/OLMoE-1B-7B-0924 \
DUMP_KVCACHE_TOKENS=2000 \
timeout 300 bash rotation/qwen3-8B/save_qkv_8b.sh \
    2>&1 | tee /workspace/olmoe_dump_check.log
```

(Note: this script's own prompt count is hardcoded to 198 GPQA prompts
internally, per `ENVIRONMENT.md` §6 point 1 — the small `DUMP_KVCACHE_TOKENS`
budget here is what keeps this a quick check rather than a full dump; it will
still send all 198 requests but stop recording almost immediately.)

**If either check fails: record the exact error text from the log file and
stop. Do not debug it in this session.** This step's only job is to tell
future work whether OLMoE is worth pursuing further, not to make it work now.

**Cost:** ~15 min → ~$0.40.

---

## Step 8 — Close out

**8a. Pull everything off the pod.** From the pod:

```bash
cd /workspace
tar czf phase2_session_artifacts.tar.gz \
    phase2_runs/ \
    pip_freeze.txt \
    olmoe_launch_check.log olmoe_dump_check.log 2>/dev/null || true
runpodctl send phase2_session_artifacts.tar.gz
```

From the local machine, immediately:

```bash
cd ~/moe-kv-cache-quantization/reproduction   # or wherever you want the pull to land
runpodctl receive <code>
tar xzf phase2_session_artifacts.tar.gz
```

**What this bundle must contain** (check before trusting it's complete):
- All `phase2_runs/*/` run directories — each has `resolved_config.txt`,
  `server.log`, `kv_pool_size.txt`, `token_capacity.txt`, `SUMMARY.txt`, plus
  `smoke_output.txt`+`smoke_response.json` (smoke arms) or `eval.log`+
  `runner.log` (full arms)
- `pip_freeze.txt` (already pulled once in Step 2, but include again here for
  a single complete bundle)
- The two OLMoE check logs, if Step 7 ran

**8b. Verify by hash where hashes exist.** The rotation files themselves
aren't part of this bundle (they're inputs, already verified in Step 3, and
shouldn't need re-pulling) — but if in doubt whether the pod-side copies
degraded during the session, re-hash them before destroying the pod:

```bash
sha256sum /workspace/recovery/rotations/*.pt /workspace/recovery/rotations_scrambled/*.pt
```

Compare against the Step 3 values one more time. For everything else in the
bundle (logs, summaries), there's no separate hash to check against — the
verification is that the files exist, are non-empty, and their content reads
as expected (`SUMMARY.txt` files especially).

**8c. Stop the pod.** Via the RunPod console, or:

```bash
runpodctl stop pod <pod-id>
```

(Pod ID from the console — see the note under Step 1.) Confirm it shows
stopped in the console before closing the session; a pod left running
un-billed-for is the single most avoidable cost overrun here.

**8d. Record actual GPU hours and cost.** Note the pod's actual running time
(RunPod console shows this) and multiply by $1.59/hr — compare against the
9.4-hour budget and note any variance for next time.

---

## Known failure modes (from `ENVIRONMENT.md`)

- **Silent process death.** No traceback, no CUDA error, no non-zero exit —
  the process just stops. Documented at length in `ENVIRONMENT.md` §3 for the
  fa3+INT2+sm89 case specifically; the mechanism there remains only
  partially understood (see `UPSTREAM_ISSUE.md`) and there's no guarantee an
  unrelated silent death couldn't occur on different hardware for a different
  reason. If a server dies with no error output at any step in this runbook,
  don't assume it's the known sm89 issue (this is sm80) — read `server.log`
  in full and treat it as a fresh failure to diagnose.
- **Relative-path ambiguity in any written summary.** `ENVIRONMENT.md` §8:
  agent-written or hand-written summaries that give paths relative to
  "whatever directory I happened to be in" cost real time once already, when
  a claimed output path was ambiguous between the project root and a nested
  `OSCAR/` checkout. **Require absolute paths in every run summary and
  artifact list produced during this session** — including ones written by
  hand while executing this runbook. The scripts already do this
  (`resolved_config.txt` uses `$RUN_DIR` built from `$RUN_ROOT`, itself meant
  to be set to an absolute path per Step 5's `export`).
- **Container disk loss on migration/reclaim.** Only `/workspace` (the
  network volume) survives a pod migration or hardware reclaim event —
  everything else, including a venv or logs placed outside `/workspace`, is
  gone if that happens. `ENVIRONMENT.md` §7 records that this project
  survived one mid-run reclaim already. Every path used for anything durable
  in this runbook is under `/workspace` for exactly this reason — if
  improvising a path during the session, keep it under `/workspace` too.
