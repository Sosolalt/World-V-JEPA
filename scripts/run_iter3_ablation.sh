#!/usr/bin/env bash
# Iteration-3 collapse ablation runner.
#
# Runs each single-knob experiment on default.npz with the original
# 150-epoch LR/EMA schedule but caps the wall time at 30 epochs.
# All output goes to logs/iter3/.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PY=".venv/bin/python"
COMMON=(--config configs/default.yaml --data data/default.npz --epochs 30 --lr-schedule-epochs 150)

mkdir -p logs/iter3

run_one () {
    local tag="$1"; shift
    local log="logs/iter3/${tag}.log"
    echo "=== running ${tag} ==="
    "$PY" scripts/train.py "${COMMON[@]}" --tag "$tag" "$@" 2>&1 | tee "$log"
    echo "=== ${tag} done ==="
}

# Pick which experiment(s) to run based on positional args; default = all.
# Usage: bash scripts/run_iter3_ablation.sh E1 E2 ...
if [ "$#" -eq 0 ]; then
    set -- E1 E2 E3 E4 E5 E6 E7
fi

for exp in "$@"; do
    case "$exp" in
        E1)
            run_one "E1_def_cov25" \
                --override training.cov_loss_weight=25.0
            ;;
        E2)
            run_one "E2_def_regctx" \
                --override training.reg_target=context
            ;;
        E3)
            run_one "E3_def_ema099" \
                --override training.ema_tau_start=0.99
            ;;
        E4)
            run_one "E4_def_mask075" \
                --override training.mask_ratio=0.75
            ;;
        E5)
            run_one "E5_def_smooth_l1" \
                --override training.loss_fn=smooth_l1
            ;;
        E6)
            run_one "E6_def_kitchen_sink" \
                --override training.reg_target=both \
                --override training.cov_loss_weight=25.0 \
                --override training.ema_tau_start=0.99
            ;;
        E7)
            run_one "E7_def_kitchen_plus_mask" \
                --override training.reg_target=both \
                --override training.cov_loss_weight=25.0 \
                --override training.ema_tau_start=0.99 \
                --override training.mask_ratio=0.75
            ;;
        *)
            echo "unknown experiment: $exp" >&2; exit 2;;
    esac
done
