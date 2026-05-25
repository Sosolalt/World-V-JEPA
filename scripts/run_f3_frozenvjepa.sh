#!/usr/bin/env bash
# F3 second-leg: pixel baseline with frozen V1 encoder.
#
# Single-knob change from the original pixel baseline
# (runs/baseline_pixel/20260519_225206/): adds --init-from-vjepa and
# --freeze-encoder. All other hyperparameters are inherited from
# configs/default.yaml (including baseline_overrides: lr_peak=2e-5,
# grad_value_clip=0.5).
#
# Output dir: runs/baseline_pixel_frozenvjepa/<YYYYMMDD_HHMMSS>/
#
# Usage:
#   bash scripts/run_f3_frozenvjepa.sh smoke   # 30-epoch stability check
#   bash scripts/run_f3_frozenvjepa.sh full    # 150-epoch production run

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
V1_CKPT="$ROOT/runs/20260519_195827_iter3_full_150/ckpt_best.pt"

if [ ! -f "$V1_CKPT" ]; then
    echo "ERROR: V1 checkpoint not found: $V1_CKPT" >&2
    exit 1
fi

mkdir -p "$ROOT/logs"

COMMON=(
    --config  "$ROOT/configs/default.yaml"
    --data    "$ROOT/data/default.npz"
    --baseline pixel
    --init-from-vjepa "$V1_CKPT"
    --freeze-encoder
)

smoke() {
    echo "=== F3 smoke test (30 epochs) ==="
    "$PY" scripts/train.py "${COMMON[@]}" --epochs 30 \
        2>&1 | tee "$ROOT/logs/f3_frozenvjepa_smoke.log"
    echo "=== smoke done — check logs/f3_frozenvjepa_smoke.log for NaN/abort ==="
}

full() {
    echo "=== F3 full run (150 epochs) ==="
    echo "Output: runs/baseline_pixel_frozenvjepa/<timestamp>/"
    "$PY" scripts/train.py "${COMMON[@]}" \
        2>&1 | tee "$ROOT/logs/f3_frozenvjepa_full.log"
    echo "=== full run done ==="
}

case "${1:-smoke}" in
    smoke) smoke ;;
    full)  full  ;;
    *)     echo "usage: $0 [smoke|full]" >&2; exit 1 ;;
esac
