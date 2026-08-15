#!/usr/bin/env bash
# _common.sh -- shared logic for the three Phase 2 launch scripts.
#
# Sourced by run_bf16_baseline.sh / run_int2_real.sh / run_int2_scrambled.sh.
# Those three differ ONLY in the arm they select; everything else lives here so
# the arms cannot silently drift apart.
#
# NOTE ON PHASE NUMBERING: "Phase 2" here means THIS PROJECT's phase 2
# (baseline reproduction). OSCAR's own docs number their internal pipeline
# dump/calibrate/eval as phases 1/2/3 -- those are NOT this project's phases.
# See scripts/README.md.
#
# Not executable on its own.
#
# Every SGLang flag and env var set here is traceable to OSCAR commit
# 41ebcdba3db5f0ce1339c3727caea80df575d437:
#   rotation/eval_oscar_gpqa.sh                                  (driver interface)
#   sglang-research/python/sglang/srt/environ.py:215-239          (SGLANG_* vars)
#   sglang-research/python/sglang/srt/mem_cache/memory_pool.py:90-180
#   sglang-research/python/sglang/srt/server_args.py:4170-4183    (--kv-cache-dtype)
#   sglang-research/python/sglang/srt/entrypoints/http_server.py:1468 (/v1/completions)

set -euo pipefail

# ---------------------------------------------------------------- paths

REPRO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPTS_DIR="${REPRO_DIR}/scripts"
OSCAR_SRC="${OSCAR_SRC:-${HOME}/oscar-src}"
RUN_ROOT="${RUN_ROOT:-${HOME}/phase2_runs}"

# Expected calibration checkpoint hashes. Both k-files begin with "c", so a
# mix-up between the real and scrambled arms would otherwise be invisible --
# this is the guard against silently invalidating a run.
SHA_REAL_K="ca8a0198f630dabbb85a186388355c844a1721782e1c08e4355558ca3424205a"
SHA_REAL_V="802088639ac53162193b591ee67a6aff289a867ac533d89a043142172e15af49"
SHA_SCRAM_K="cc5313264cc7edc527b55d982bbe4d8c36938e752ddb3f2f09588c5ccc056aa1"
SHA_SCRAM_V="18815a709923f79ae1806b3c86875c6667e4e31cbb2bc4840dfefc299b4c391f"

ROT_REAL="${ROT_REAL:-${HOME}/recovery/rotations}"
ROT_SCRAMBLED="${ROT_SCRAMBLED:-${HOME}/recovery/rotations_scrambled}"

K_ROT_FILENAME="k_rotation_qqt_r_h_pbr.pt"
V_ROT_FILENAME="v_rotation_sst_r_h_pbr.pt"

SMOKE_PROMPT="${SCRIPTS_DIR}/smoke_prompt.txt"

die() { echo "[FATAL] $*" >&2; exit 1; }
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
warn_banner() {
    echo "" >&2
    echo "################################################################" >&2
    echo "# $*" >&2
    echo "################################################################" >&2
    echo "" >&2
}

# ---------------------------------------------------------------- config

# run_config.env carries the values that were tuned by trial and error on this
# hardware (MEM_FRAC=0.72) plus the paper-knee KV windows. Committed rather
# than passed on a command line -- see ENVIRONMENT.md section 5.
CONFIG_ENV="${REPRO_DIR}/run_config.env"
[[ -f "${CONFIG_ENV}" ]] || die "missing config: ${CONFIG_ENV}"
# shellcheck disable=SC1090
set -a; source "${CONFIG_ENV}"; set +a

MODEL="${MODEL:-Qwen/Qwen3-8B}"

# Single 4090: TP=1, one GPU. Upstream defaults assume TP=4 on 4 GPUs.
TP_SIZE="${TP_SIZE:-1}"
GPUS="${GPUS:-0}"
PORT="${PORT:-31057}"
DIST_PORT="${DIST_PORT:-41057}"
MEM_FRAC="${MEM_FRAC:?MEM_FRAC must come from run_config.env}"

# UNVALIDATED STARTING VALUES. These are conservative reductions from upstream's
# 4-GPU defaults (MAX_RUNNING=64, CUDA_GRAPH_MAX_BS=32, NUM_WORKERS=32), chosen
# for a single 24GB card. They are NOT derived from source and NOT validated on
# this hardware -- expect to adjust them on first contact with the pod. They
# affect throughput and memory headroom, not correctness of the comparison.
MAX_RUNNING="${MAX_RUNNING:-16}"
CUDA_GRAPH_MAX_BS="${CUDA_GRAPH_MAX_BS:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"

GROUP_SIZE="${GROUP_SIZE:-128}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32768}"
N_REPEATS="${N_REPEATS:-1}"

# Clip ratios: env vars read by load_oscar_rotation_config(), NOT stored in the
# checkpoint (Stage A confirmed no thresholds on disk). Defaults from
# eval_oscar_gpqa.sh:114-115, which match the paper's reported cK/cV.
K_CLIP="${K_CLIP:-0.96}"
V_CLIP="${V_CLIP:-0.92}"

# SMOKE=1 selects the free-form continuation test instead of GPQA.
#
# Why not GPQA for smoke: GPQA emits a single letter (A/B/C/D), so a completely
# broken INT2 path still produces a plausible-looking answer -- the test would
# pass while testing nothing. Its prompts may also be short enough to sit
# entirely inside the sink+recent windows, quantizing zero tokens. The smoke
# test therefore uses a long fixed prompt and free-form generation, where
# breakage is visible as incoherent prose.
if [[ "${SMOKE:-0}" == "1" ]]; then
    MIXED_PREFIX_TOKENS=4
    MIXED_RECENT_TOKENS=8
    SMOKE_TAG="smoke"
    SMOKE_GEN_TOKENS="${SMOKE_GEN_TOKENS:-300}"
else
    # Paper knee, also the defaults in eval_oscar_gpqa.sh:108-109.
    MIXED_PREFIX_TOKENS="${SGLANG_MIXED_KV_PREFIX_TOKENS:-64}"
    MIXED_RECENT_TOKENS="${SGLANG_MIXED_KV_RECENT_TOKENS:-256}"
    SMOKE_TAG="full"
fi

# ---------------------------------------------------------------- checks

# Read one sha256 from a file, failing closed.
#
# Deliberately NOT written as `sha256sum f | awk '{print $1}'`: awk succeeds even
# when sha256sum fails, so a pipeline masks the real exit status and yields an
# empty string. An empty string then compares equal to an empty expected value,
# and the guard passes on garbage. Verified experimentally. This version checks
# the tool's exit status, then validates the shape of what came back.
sha256_of() {
    local f="$1" out hash
    if command -v sha256sum >/dev/null 2>&1; then
        out="$(sha256sum -- "${f}")" || return 1
    elif command -v shasum >/dev/null 2>&1; then
        out="$(shasum -a 256 -- "${f}")" || return 1
    else
        return 1
    fi
    hash="${out%% *}"
    # Must be exactly 64 lowercase hex chars, or we do not trust it.
    [[ "${hash}" =~ ^[0-9a-f]{64}$ ]] || return 1
    printf '%s' "${hash}"
}

verify_rotations() {
    # verify_rotations <dir> <expected_k_sha> <expected_v_sha>
    # Fails closed: aborts on missing dir, missing file, unreadable file,
    # unavailable/malfunctioning hash tool, or hash mismatch.
    local dir="$1" want_k="$2" want_v="$3"

    [[ -n "${want_k}" && -n "${want_v}" ]] \
        || die "verify_rotations called with an empty expected hash -- refusing to launch"

    [[ -d "${dir}" ]] || die "rotation dir not found: ${dir}"
    local kf="${dir}/${K_ROT_FILENAME}" vf="${dir}/${V_ROT_FILENAME}"
    [[ -f "${kf}" ]] || die "missing rotation file: ${kf}"
    [[ -f "${vf}" ]] || die "missing rotation file: ${vf}"
    [[ -r "${kf}" ]] || die "rotation file not readable: ${kf}"
    [[ -r "${vf}" ]] || die "rotation file not readable: ${vf}"
    [[ -s "${kf}" ]] || die "rotation file is empty: ${kf}"
    [[ -s "${vf}" ]] || die "rotation file is empty: ${vf}"

    local got_k got_v
    got_k="$(sha256_of "${kf}")" \
        || die "could not compute sha256 of ${kf} (no working sha256sum/shasum, or bad output)"
    got_v="$(sha256_of "${vf}")" \
        || die "could not compute sha256 of ${vf} (no working sha256sum/shasum, or bad output)"

    log "verifying rotation checkpoints in ${dir}"
    log "  k: ${got_k}"
    log "  v: ${got_v}"

    [[ "${got_k}" == "${want_k}" ]] || die \
        "K rotation sha256 MISMATCH
    file:     ${kf}
    expected: ${want_k}
    actual:   ${got_k}
  Refusing to launch. Both k-files start with 'c'; this is exactly the
  mix-up this check exists to catch."
    [[ "${got_v}" == "${want_v}" ]] || die \
        "V rotation sha256 MISMATCH
    file:     ${vf}
    expected: ${want_v}
    actual:   ${got_v}
  Refusing to launch."
    log "  both hashes OK"
}

preflight() {
    # Hash tooling must exist before anything trusts verify_rotations().
    if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
        die "neither sha256sum nor shasum found -- cannot verify rotation checkpoints"
    fi
    command -v curl   >/dev/null 2>&1 || die "curl not found"
    command -v python >/dev/null 2>&1 || die "python not found"
    [[ -d "${OSCAR_SRC}" ]] || die "OSCAR source not found: ${OSCAR_SRC} (set OSCAR_SRC)"
    [[ -d "${SGLANG_RESEARCH_DIR}" ]] || die "sglang-research not found: ${SGLANG_RESEARCH_DIR}"
    command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found -- no GPU?"
    if [[ "${SMOKE:-0}" == "1" ]]; then
        [[ -f "${SMOKE_PROMPT}" ]] || die "missing smoke prompt: ${SMOKE_PROMPT}"
        [[ -s "${SMOKE_PROMPT}" ]] || die "smoke prompt is empty: ${SMOKE_PROMPT}"
    fi
}

# ---- BF16 context-length memory estimate (DECISION 6) ---------------------
# Warns loudly, never aborts, and never silently reduces context. An asymmetric
# comparison between arms would be worse than no comparison, so the decision to
# change hardware or context belongs to the operator, not to this script.
estimate_kv_memory() {
    # estimate_kv_memory <bytes_per_element_label>
    local arm_dtype="$1"
    python - "$arm_dtype" "$MEM_FRAC" "$MAX_NEW_TOKENS" <<'PYEOF'
import sys, json, os, urllib.request
dtype_label = sys.argv[1]
mem_frac = float(sys.argv[2])

# Qwen3-8B architecture. Values are from the published model config; if the
# real config differs at runtime the server log is authoritative, not this.
N_LAYERS, N_KV_HEADS, HEAD_DIM = 36, 8, 128
CTX = int(os.environ.get("EST_CONTEXT_LEN", "32768"))
GPU_GB = float(os.environ.get("EST_GPU_GB", "24"))
WEIGHTS_GB = float(os.environ.get("EST_WEIGHTS_GB", "16"))

bytes_per_elem = {"bfloat16": 2.0, "int2": 0.25}[dtype_label]
# K and V, per token, per layer.
per_token = N_LAYERS * N_KV_HEADS * HEAD_DIM * 2 * bytes_per_elem
kv_gb = per_token * CTX / (1024 ** 3)
budget_gb = GPU_GB * mem_frac
headroom = budget_gb - WEIGHTS_GB - kv_gb

print(f"  dtype:              {dtype_label} ({bytes_per_elem} B/element)")
print(f"  layers x kv_heads x head_dim: {N_LAYERS} x {N_KV_HEADS} x {HEAD_DIM}")
print(f"  KV bytes/token:     {per_token:,.0f} B  ({per_token/1024:.1f} KB)")
print(f"  context length:     {CTX:,} tokens")
print(f"  KV cache total:     {kv_gb:.2f} GB")
print(f"  mem_fraction:       {mem_frac}  -> budget {budget_gb:.2f} GB of {GPU_GB:.0f} GB")
print(f"  weights (est):      {WEIGHTS_GB:.2f} GB")
print(f"  headroom:           {headroom:+.2f} GB")
print(f"MARGINAL={'1' if headroom < 2.0 else '0'}")
PYEOF
}

check_context_memory() {
    local arm_dtype="$1"
    log "estimating KV memory for context length ${EST_CONTEXT_LEN:-32768}..."
    local out
    out="$(estimate_kv_memory "${arm_dtype}")" || {
        log "WARNING: memory estimate failed to run; continuing without it"
        return 0
    }
    echo "${out}" | grep -v '^MARGINAL='
    if echo "${out}" | grep -q '^MARGINAL=1'; then
        warn_banner "MEMORY WARNING: ${arm_dtype} KV cache at this context length looks MARGINAL.
#
# The arithmetic above is a rough estimate, not a measurement. The run may OOM,
# or SGLang may silently cap max_total_num_tokens below the requested context.
#
# This script will NOT reduce context to make it fit: an asymmetric comparison
# between arms is worse than no comparison. If this OOMs, change the hardware or
# change the context length for ALL THREE ARMS together, deliberately.
#
# Proceeding anyway in 10s. Ctrl-C to abort."
        sleep 10
    fi
}

# ---- smoke prompt length assertion ---------------------------------------
assert_smoke_prompt_length() {
    # Uses the MODEL's own tokenizer -- no estimate, no heuristic. Aborts if the
    # prompt does not exceed sink+recent, because in that case zero tokens would
    # be quantized and the smoke test would prove nothing.
    local window=$(( MIXED_PREFIX_TOKENS + MIXED_RECENT_TOKENS ))
    log "tokenizing smoke prompt with ${MODEL}'s tokenizer..."
    local n
    n="$(python - "$MODEL" "$SMOKE_PROMPT" <<'PYEOF'
import sys
try:
    from transformers import AutoTokenizer
except ImportError:
    sys.stderr.write("transformers not installed -- cannot tokenize\n")
    sys.exit(2)
model, path = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
text = open(path, encoding="utf-8").read()
print(len(tok(text, add_special_tokens=False)["input_ids"]))
PYEOF
    )" || die "failed to tokenize smoke prompt with ${MODEL} (is transformers installed?)"

    [[ "${n}" =~ ^[0-9]+$ ]] || die "tokenizer returned non-numeric length: ${n}"
    SMOKE_PROMPT_TOKENS="${n}"

    log "  smoke prompt: ${n} tokens (measured, ${MODEL} tokenizer)"
    log "  sink+recent window: ${MIXED_PREFIX_TOKENS} + ${MIXED_RECENT_TOKENS} = ${window}"
    if (( n <= window )); then
        die "smoke prompt is ${n} tokens but sink+recent window is ${window}.
  Zero tokens would be quantized -- the INT2 path would not be exercised and
  this test would pass while testing nothing. Refusing to run."
    fi
    log "  OK: ${n} tokens exceeds window by $(( n - window )) tokens ($(( n / window ))x)"
}

# ---------------------------------------------------------------- launch

run_arm() {
    # run_arm <arm_name> ; expects ARM_KV_ARGS / ARM_OSCAR_ENV to be set
    local arm="$1"
    local stamp; stamp="$(date -u +%Y%m%dT%H%M%SZ)"
    RUN_DIR="${RUN_ROOT}/${stamp}_${arm}_${SMOKE_TAG}"
    mkdir -p "${RUN_DIR}"

    local server_log="${RUN_DIR}/server.log"
    local runner_log="${RUN_DIR}/runner.log"
    local kvpool_log="${RUN_DIR}/kv_pool_size.txt"
    local config_log="${RUN_DIR}/resolved_config.txt"
    : > "${server_log}"

    if [[ "${SMOKE:-0}" == "1" ]]; then
        assert_smoke_prompt_length
    fi
    check_context_memory "${ARM_DTYPE_LABEL}"

    {
        echo "=== Phase 2 (baseline reproduction) run: ${arm} (${SMOKE_TAG}) ==="
        echo "NOTE: 'Phase 2' is THIS PROJECT's numbering, not OSCAR's internal"
        echo "      pipeline numbering (they call dump/calibrate/eval 1/2/3)."
        echo "utc_start:        ${stamp}"
        echo "run_dir:          ${RUN_DIR}"
        echo "oscar_src:        ${OSCAR_SRC}"
        echo "oscar_commit:     $(cd "${OSCAR_SRC}" && git rev-parse HEAD 2>/dev/null || echo UNKNOWN)"
        echo "sglang_research:  ${SGLANG_RESEARCH_DIR}"
        echo "config_env:       ${CONFIG_ENV}"
        echo "--- model / server ---"
        echo "model:            ${MODEL}"
        echo "tp_size:          ${TP_SIZE}"
        echo "gpus:             ${GPUS}"
        echo "port:             ${PORT}"
        echo "mem_fraction:     ${MEM_FRAC}"
        echo "max_running:      ${MAX_RUNNING}      (unvalidated starting value)"
        echo "cuda_graph_max_bs:${CUDA_GRAPH_MAX_BS}      (unvalidated starting value)"
        echo "num_workers:      ${NUM_WORKERS}      (unvalidated starting value)"
        echo "prefill_backend:  triton"
        echo "decode_backend:   triton"
        echo "kv_args:          ${ARM_KV_ARGS[*]}"
        echo "--- kv windows ---"
        echo "smoke:            ${SMOKE:-0}"
        echo "prefix_tokens:    ${MIXED_PREFIX_TOKENS}"
        echo "recent_tokens:    ${MIXED_RECENT_TOKENS}"
        if [[ "${SMOKE:-0}" == "1" ]]; then
        echo "smoke_prompt:     ${SMOKE_PROMPT}"
        echo "smoke_prompt_sha: $(sha256_of "${SMOKE_PROMPT}" || echo UNKNOWN)"
        echo "smoke_prompt_tok: ${SMOKE_PROMPT_TOKENS:-UNMEASURED} tokens (model tokenizer)"
        echo "smoke_gen_tokens: ${SMOKE_GEN_TOKENS}"
        fi
        echo "--- rotations ---"
        if [[ -n "${ARM_ROT_DIR:-}" ]]; then
            echo "rot_dir:          ${ARM_ROT_DIR}"
            echo "k_rotation:       ${ARM_ROT_DIR}/${K_ROT_FILENAME}"
            echo "v_rotation:       ${ARM_ROT_DIR}/${V_ROT_FILENAME}"
            echo "k_sha256:         $(sha256_of "${ARM_ROT_DIR}/${K_ROT_FILENAME}" || echo UNKNOWN)"
            echo "v_sha256:         $(sha256_of "${ARM_ROT_DIR}/${V_ROT_FILENAME}" || echo UNKNOWN)"
            echo "k_clip_ratio:     ${K_CLIP}"
            echo "v_clip_ratio:     ${V_CLIP}"
        else
            echo "rot_dir:          (none -- BF16 baseline, Oscar path disabled)"
        fi
        echo "--- decoding ---"
        echo "temperature:      ${TEMPERATURE}"
        echo "top_p:            ${TOP_P}"
        echo "top_k:            ${TOP_K}"
        echo "seed:             ${RUN_SEED}"
        echo "n_repeats:        ${N_REPEATS}"
        echo "--- gpu ---"
        nvidia-smi --query-gpu=name,driver_version,memory.total \
                   --format=csv,noheader 2>/dev/null || echo "nvidia-smi query failed"
    } | tee "${config_log}"

    export PYTHONPATH="${OSCAR_SRC}/rotation/_triton_per_rank:${SGLANG_RESEARCH_DIR}/python:${PYTHONPATH:-}"
    export PYTHONUNBUFFERED=1
    export OSCAR_TRITON_PER_RANK_BASE="${RUN_DIR}/triton_cache"
    export TRITON_CACHE_DIR="${OSCAR_TRITON_PER_RANK_BASE}/main"
    mkdir -p "${OSCAR_TRITON_PER_RANK_BASE}" "${TRITON_CACHE_DIR}"

    local server_args=(
        --model-path "${MODEL}"
        --tensor-parallel-size "${TP_SIZE}"
        # Both backends triton: fa3 prefill crashes silently on sm89 (Ada
        # Lovelace) with the int2 KV path. See ENVIRONMENT.md section 3.
        --prefill-attention-backend triton
        --decode-attention-backend triton
        "${ARM_KV_ARGS[@]}"
        --mem-fraction-static "${MEM_FRAC}"
        --max-running-requests "${MAX_RUNNING}"
        --enable-cache-report
        --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS}"
        --host 127.0.0.1
        --port "${PORT}"
        --dist-init-addr "127.0.0.1:${DIST_PORT}"
        --trust-remote-code
        --random-seed "${RUN_SEED}"
    )

    log "launching server -> ${server_log}"
    env "${ARM_OSCAR_ENV[@]}" \
        SGLANG_ENABLE_MIXED_KV_WINDOWS="${MIXED_WINDOWS_ENABLED}" \
        SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
        SGLANG_MIXED_KV_HP_MAX_SPLITS=8 \
        SGLANG_MIXED_KV_PREFIX_TOKENS="${MIXED_PREFIX_TOKENS}" \
        SGLANG_MIXED_KV_RECENT_TOKENS="${MIXED_RECENT_TOKENS}" \
        SGLANG_MIXED_KV_HP_DTYPE=bfloat16 \
        SGLANG_MIXED_KV_SCALE_DTYPE=float32 \
        CUDA_VISIBLE_DEVICES="${GPUS}" \
        python -m sglang.launch_server "${server_args[@]}" >> "${server_log}" 2>&1 &
    SERVER_PID=$!

    local ready=0
    for _ in $(seq 1 240); do
        if curl -s "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
            ready=1; break
        fi
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "[FATAL] server died during startup" >&2
            tail -100 "${server_log}" >&2 || true
            exit 1
        fi
        sleep 5
    done
    [[ "${ready}" == "1" ]] || {
        tail -100 "${server_log}" >&2 || true
        die "server not ready after 20 min"
    }
    log "server ready"

    # ---- POSITIVE CONTROL: KV cache pool size.
    # memory_pool.py:808,814 logs "KV Cache is allocated. #tokens: N ..."
    # scheduler.py:725 logs "max_total_num_tokens=N ..."
    #
    # The gate is INT2 >= 3x BF16 token capacity. This screens for the failure
    # mode that matters -- "INT2 path not engaged at all", which presents as a
    # ratio near 1x. A precise ratio was never needed: bit-width arithmetic
    # suggests ~8x, but the BF16 sink/recent windows and allocator overhead pull
    # the real figure down by an amount not stated in any source. A tight bar
    # would invite post-hoc adjustment if the real answer came back at, say,
    # 5.5x. The measured ratio is reported prominently; 3x is only the gate.
    {
        echo "=== KV cache pool size (positive control) ==="
        echo "arm:     ${arm}"
        echo "run_dir: ${RUN_DIR}"
        echo "--- 'KV Cache is allocated' lines (memory_pool.py:808,814) ---"
        grep -E "KV Cache is allocated" "${server_log}" || echo "(none found)"
        echo "--- 'max_total_num_tokens' lines (scheduler.py:725) ---"
        grep -Eo "max_total_num_tokens=[0-9]+" "${server_log}" || echo "(none found)"
        echo "--- Oscar rotation load lines (memory_pool.py:175) ---"
        grep -E "Loaded Oscar rotation" "${server_log}" || echo "(none -- expected for BF16 arm)"
    } | tee "${kvpool_log}"

    local token_cap
    token_cap="$(grep -Eo 'max_total_num_tokens=[0-9]+' "${server_log}" \
                 | tail -1 | cut -d= -f2 || true)"
    if [[ -z "${token_cap}" ]]; then
        log "WARNING: could not parse max_total_num_tokens from server log."
        log "         The positive control is INCONCLUSIVE for this run."
    else
        log "token capacity: ${token_cap}"
        echo "${token_cap}" > "${RUN_DIR}/token_capacity.txt"
    fi

    # ---- workload
    if [[ "${SMOKE:-0}" == "1" ]]; then
        run_smoke_generation "${arm}"
    else
        log "launching GPQA eval -> ${runner_log}"
        python "${OSCAR_SRC}/rotation/_eval_runner/run_simple_eval.py" \
            --task gpqa \
            --model "${MODEL}" \
            --base-url "http://127.0.0.1:${PORT}/v1" \
            --max-tokens "${MAX_NEW_TOKENS}" \
            --temperature "${TEMPERATURE}" \
            --top-p "${TOP_P}" \
            --top-k "${TOP_K}" \
            --n-repeats "${N_REPEATS}" \
            --num-threads "${NUM_WORKERS}" \
            ${NUM_EXAMPLES:+--num-examples ${NUM_EXAMPLES}} \
            --output-dir "${RUN_DIR}" \
            2>&1 | tee "${runner_log}"
    fi

    {
        echo "=== SUMMARY (${arm}, ${SMOKE_TAG}) ==="
        echo "run_dir:         ${RUN_DIR}"
        echo "resolved_config: ${config_log}"
        echo "server_log:      ${server_log}"
        echo "kv_pool_size:    ${kvpool_log}"
        echo "token_capacity:  ${token_cap:-UNPARSED}"
        if [[ "${SMOKE:-0}" == "1" ]]; then
            echo "smoke_output:    ${RUN_DIR}/smoke_output.txt"
            echo "smoke_prompt_tok:${SMOKE_PROMPT_TOKENS:-UNMEASURED}"
            echo ""
            echo "--- first 400 chars of continuation (read it) ---"
            head -c 400 "${RUN_DIR}/smoke_output.txt" 2>/dev/null || echo "(no output)"
            echo ""
        else
            echo "runner_log:      ${runner_log}"
            echo "eval_log:        ${RUN_DIR}/eval.log"
            echo "--- score ---"
            grep -iE "gpqa/score|gpqa/chars" "${RUN_DIR}/eval.log" 2>/dev/null | tail -10 \
                || echo "(no score lines found in ${RUN_DIR}/eval.log)"
        fi
    } | tee "${RUN_DIR}/SUMMARY.txt"

    log "done: ${RUN_DIR}"
}

# ---- free-form continuation smoke test -----------------------------------
# Uses /v1/completions (http_server.py:1468) for RAW continuation -- no chat
# template, no answer parsing, no scoring harness. The output is meant to be
# read by a human side by side across the three arms.
run_smoke_generation() {
    local arm="$1"
    local out="${RUN_DIR}/smoke_output.txt"
    local raw="${RUN_DIR}/smoke_response.json"

    log "running free-form continuation smoke test (${SMOKE_GEN_TOKENS} tokens)"
    python - "$PORT" "$MODEL" "$SMOKE_PROMPT" "$SMOKE_GEN_TOKENS" \
             "$TEMPERATURE" "$TOP_P" "$TOP_K" "$RUN_SEED" "$out" "$raw" <<'PYEOF'
import json, sys, urllib.request

port, model, prompt_path, ntok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
temp, top_p, top_k, seed = float(sys.argv[5]), float(sys.argv[6]), int(sys.argv[7]), int(sys.argv[8])
out_path, raw_path = sys.argv[9], sys.argv[10]

prompt = open(prompt_path, encoding="utf-8").read()
body = {
    "model": model,
    "prompt": prompt,
    "max_tokens": ntok,
    "temperature": temp,
    "top_p": top_p,
    "seed": seed,
    "stream": False,
}
# top_k is a SGLang extension on the OpenAI-compatible endpoint; -1 disables it
# (sampling_params.py:99-103).
if top_k != -1:
    body["top_k"] = top_k

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps(body).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=1800) as r:
    resp = json.load(r)

with open(raw_path, "w") as f:
    json.dump(resp, f, indent=2)

text = resp["choices"][0]["text"]
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)

usage = resp.get("usage", {})
print(f"  prompt_tokens={usage.get('prompt_tokens')} "
      f"completion_tokens={usage.get('completion_tokens')}")
print(f"  wrote {out_path} ({len(text)} chars)")
PYEOF
    log "  smoke output: ${out}"
}

cleanup() {
    if [[ -n "${SERVER_PID:-}" ]]; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
        pkill -TERM -P "${SERVER_PID}" 2>/dev/null || true
        sleep 2
        kill  -KILL "${SERVER_PID}" 2>/dev/null || true
        pkill -KILL -P "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# ---------------------------------------------------------------- shared defaults

SGLANG_RESEARCH_DIR="${SGLANG_RESEARCH_DIR:-${OSCAR_SRC}/sglang-research}"

# Greedy decoding + fixed seed. NOTE: upstream eval_oscar_gpqa.sh defaults to
# temperature 1.0 / top_p 0.95 / top_k 40 (sampled, not greedy). These are
# deliberately overridden here so the three arms differ only in the KV path.
TEMPERATURE="${TEMPERATURE:-0.0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
RUN_SEED="${RUN_SEED:-0}"
export HF_HOME="${HF_HOME:-/shared/huggingface}"
