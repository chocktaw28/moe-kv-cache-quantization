#!/usr/bin/env bash
# run_int2_real.sh -- Phase 2 (baseline reproduction) -- arm 2 of 3:
# INT2 KV cache with the REAL rotations.
#
# "Phase 2" is THIS PROJECT's numbering. OSCAR's own docs number their internal
# pipeline (dump / calibrate / eval) as phases 1/2/3 -- unrelated. See README.md.
#
# The actual reproduction arm. Uses the locally-calibrated rotations validated
# in Stage A (schema, orthogonality) and Stage B (eigenbasis vs RotationZoo).
#
# Identical to run_int2_scrambled.sh except for ROT dir, expected hashes, and
# the arm label. If this arm and the scrambled arm behave the same, the eval is
# not sensitive to rotation quality and neither number means anything.
#
# Usage:
#   ./run_int2_real.sh          # full GPQA, paper-knee windows (64/256)
#   SMOKE=1 ./run_int2_real.sh  # free-form continuation smoke test (4/8)

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

preflight

ARM_ROT_DIR="${ROT_REAL}"
verify_rotations "${ARM_ROT_DIR}" "${SHA_REAL_K}" "${SHA_REAL_V}"

# --- arm-specific: INT2 KV cache + real rotations --------------------------
ARM_KV_ARGS=(
    --kv-cache-dtype int2
    --kv-cache-quant-group-size "${GROUP_SIZE}"
)
ARM_DTYPE_LABEL="int2"

MIXED_WINDOWS_ENABLED=1

# Read by load_oscar_rotation_config() (memory_pool.py:117-130) via
# environ.py:224-227. Clip ratios are env vars, not checkpoint fields.
# SGLANG_OSCAR_ABSORB_V_ROTATION=1 matches eval_oscar_gpqa.sh:106.
ARM_OSCAR_ENV=(
    "SGLANG_OSCAR_K_ROTATION_PATH=${ARM_ROT_DIR}/${K_ROT_FILENAME}"
    "SGLANG_OSCAR_V_ROTATION_PATH=${ARM_ROT_DIR}/${V_ROT_FILENAME}"
    "SGLANG_OSCAR_K_CLIP_RATIO=${K_CLIP}"
    "SGLANG_OSCAR_V_CLIP_RATIO=${V_CLIP}"
    "SGLANG_OSCAR_ABSORB_V_ROTATION=1"
)

run_arm "int2_real"
