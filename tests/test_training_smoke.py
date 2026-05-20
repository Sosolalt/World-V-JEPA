"""Smoke test: 2-epoch training on the sample NPZ."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_NPZ = REPO_ROOT / "data" / "sample" / "sample.npz"


@pytest.mark.skipif(not SAMPLE_NPZ.exists(), reason="sample dataset missing")
def test_training_debug_runs():
    runs_dir = REPO_ROOT / "runs"
    before = set(p.name for p in runs_dir.glob("*")) if runs_dir.exists() else set()

    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "train.py"),
        "--config", str(REPO_ROOT / "configs" / "simple.yaml"),
        "--debug",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, f"train failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"

    after = set(p.name for p in runs_dir.glob("*"))
    new_runs = sorted(after - before)
    assert new_runs, "no new run directory was created"
    run_dir = runs_dir / new_runs[-1]

    metrics_path = run_dir / "metrics.csv"
    assert metrics_path.exists(), "metrics.csv was not created"

    with open(metrics_path, "r") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2, f"expected 2 rows in metrics.csv, got {len(rows)}"

    expected_cols = {
        "epoch", "lr", "tau", "loss", "mse", "var_reg", "cov_reg",
        "avg_std", "effective_rank", "avg_cosine_sim", "time_s",
    }
    assert expected_cols.issubset(rows[0].keys()), \
        f"metrics.csv missing columns: {expected_cols - set(rows[0].keys())}"

    for row in rows:
        eff = float(row["effective_rank"])
        assert eff > 1.0, f"effective_rank should be > 1, got {eff}"
        for key in ("loss", "mse", "avg_std", "avg_cosine_sim", "cov_reg"):
            v = float(row[key])
            assert v == v, f"{key} is NaN"

    # ckpt_best.pt is only written after warmup_epochs and only when not
    # collapsed; --debug runs 2 epochs which is below warmup, so it may be
    # absent (this is the new, intended behavior - the previous default
    # was selecting the untrained random init as "best").
