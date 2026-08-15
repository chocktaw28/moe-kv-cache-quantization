#!/usr/bin/env bash
# run_bf16_baseline.sh -- Phase 2 (baseline reproduction) -- arm 1 of 3:
# BF16 KV cache, no rotations.
#
# "Phase 2" is THIS PROJECT's numbering. OSCAR's own docs number their internal
# pipeline (dump / calibrate / eval) as phases 1/2/3 -- unrelated. See README.md.
#
# The accuracy ceiling. Whatever INT2 scores, it is measured against this.
#
# The Oscar rotation path is fully disabled here: SGLANG_OSCAR_*_ROTATION_PATH
# are left empty, which OscarRotationConfig (memory_pool.py:96-99) treats as
# "Oscar path off". No rotation files are read, so no hash check is needed --
# there is nothing to mix up in this arm.
#
# Usage:
#   ./run_bf16_baseline.sh          # full GPQA, paper-knee windows (64/256)
#   SMOKE=1 ./run_bf16_baseline.sh  # free-form continuation smoke test

set -euo pipefail
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/_common.sh"

preflight

# --- arm-specific: BF16 KV cache ------------------------------------------
# "bfloat16" is a valid --kv-cache-dtype choice (server_args.py:4170-4183).
# --kv-cache-quant-group-size is deliberately OMITTED: server_args.py:6491-6498
# raises ValueError if it is set with any dtype other than int2.
ARM_KV_ARGS=(
    --kv-cache-dtype bfloat16
)
ARM_DTYPE_LABEL="bfloat16"

# Mixed KV windows are an INT2-path feature (they keep a high-precision prefix
# and recent window around the quantized bulk). With a BF16 cache the whole
# cache is already high precision, so the feature is disabled for this arm.
MIXED_WINDOWS_ENABLED=0

# No rotation env vars -> Oscar path disabled.
ARM_OSCAR_ENV=()
ARM_ROT_DIR=""

run_arm "bf16_baseline"
