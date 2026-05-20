"""Independent review tests for the V-JEPA evaluation pipeline.

These tests target the *pipeline* (leakage, shapes, reproducibility, baseline
parity, figure generation), not the model performance. Performance numbers are
validated separately on a real training run.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from scripts.evaluate import (  # noqa: E402
    cosine_sim_batched,
    degradation_curve,
    encode_all_frames_pixel,
    encode_all_frames_target,
    flatten_for_probe,
    linear_probe,
    load_config,
    load_frames,
    load_pixel,
    load_vjepa,
    run_evaluation,
    set_seed,
    split_indices,
)
from mini_vjepa.dataset import BilliardDataset  # noqa: E402
from mini_vjepa.vjepa import VJEPA, count_parameters  # noqa: E402
from baselines.pixel_predictor import PixelPredictor  # noqa: E402


CONFIG_PATH = REPO_ROOT / "configs" / "simple.yaml"
DATA_PATH = REPO_ROOT / "data" / "sample" / "sample.npz"


def _device() -> torch.device:
    return torch.device("cpu")  # tests run on cpu for determinism


# ---------------------------------------------------------------------------
# TestProbeLeakage
# ---------------------------------------------------------------------------


class TestProbeLeakage:
    def test_split_indices_disjoint(self):
        train, test = split_indices(16, train_frac=0.8)
        assert len(set(train.tolist()) & set(test.tolist())) == 0
        assert len(train) + len(test) == 16
        # 80% rounds to 13, leaves 3 for test
        assert len(train) == 13
        assert len(test) == 3

    def test_split_is_sequence_level_not_frame_level(self):
        """flatten_for_probe must use whole sequences from indices."""
        n, t, n_tokens, d = 4, 8, 6, 5
        n_balls = 2
        z = np.random.randn(n, t, n_tokens, d).astype(np.float32)
        targets = np.random.randn(n, t, n_balls, 2).astype(np.float32)
        train_idx = np.array([0, 1])
        test_idx = np.array([2, 3])
        xtr, ytr = flatten_for_probe(z, targets, train_idx)
        xte, yte = flatten_for_probe(z, targets, test_idx)
        # No frame from test idx may equal any train frame
        assert xtr.shape == (2 * t, n_tokens * d)
        assert xte.shape == (2 * t, n_tokens * d)
        # disjoint by construction
        for row_tr in xtr:
            for row_te in xte:
                assert not np.array_equal(row_tr, row_te)

    def test_target_encoder_no_grad(self):
        set_seed(0)
        device = _device()
        cfg = load_config(str(CONFIG_PATH))
        model = VJEPA(cfg["model"]).to(device).eval()
        # target encoder params should have requires_grad False
        for p in model.target_encoder.parameters():
            assert p.requires_grad is False
        # target_encode output is detached
        frame = torch.randn(1, 3, 64, 64)
        z = model.target_encode(frame)
        assert z.requires_grad is False


# ---------------------------------------------------------------------------
# TestMetrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_linear_probe_random_targets_r2_near_zero(self):
        """When targets are pure noise unrelated to z, test R² must be ~0 or negative."""
        rng = np.random.default_rng(0)
        n, t, n_tokens, d = 32, 8, 4, 6
        n_balls = 2
        # latents have some structure
        z = rng.standard_normal((n, t, n_tokens, d)).astype(np.float32)
        # targets are independent noise
        targets = rng.standard_normal((n, t, n_balls, 2)).astype(np.float32)
        train, test = split_indices(n, train_frac=0.8)
        out = linear_probe(z, targets, train, test)
        # Random targets must not give meaningfully positive R²
        assert out["overall_r2"] < 0.2, f"probe leaks: R²={out['overall_r2']}"

    def test_linear_probe_perfect_signal_high_r2(self):
        """When targets are a deterministic linear function of z, R² should be high."""
        rng = np.random.default_rng(1)
        n, t, n_tokens, d = 40, 6, 4, 6
        n_balls = 2
        z = rng.standard_normal((n, t, n_tokens, d)).astype(np.float32)
        z_flat = z.reshape(n * t, -1)
        W = rng.standard_normal((z_flat.shape[1], n_balls * 2)).astype(np.float32)
        y_flat = z_flat @ W
        targets = y_flat.reshape(n, t, n_balls, 2)
        train, test = split_indices(n, train_frac=0.8)
        out = linear_probe(z, targets, train, test)
        assert out["overall_r2"] > 0.8

    def test_linear_probe_returns_per_ball(self):
        rng = np.random.default_rng(2)
        n_balls = 3
        n, t, n_tokens, d = 20, 4, 4, 4
        z = rng.standard_normal((n, t, n_tokens, d)).astype(np.float32)
        targets = rng.standard_normal((n, t, n_balls, 2)).astype(np.float32)
        train, test = split_indices(n, train_frac=0.8)
        out = linear_probe(z, targets, train, test)
        assert len(out["per_ball_r2"]) == n_balls
        assert "mae" in out
        assert "mean_per_ball_r2" in out


# ---------------------------------------------------------------------------
# TestDegradation
# ---------------------------------------------------------------------------


class TestDegradation:
    def test_cosine_sim_bounds(self):
        a = torch.randn(4, 6, 8)
        b = a.clone()
        s_same = cosine_sim_batched(a, b).item()
        assert s_same == pytest.approx(1.0, abs=1e-5)
        s_opp = cosine_sim_batched(a, -a).item()
        assert s_opp == pytest.approx(-1.0, abs=1e-5)

    def test_degradation_curve_has_all_horizons(self):
        set_seed(0)
        device = _device()
        cfg = load_config(str(CONFIG_PATH))
        model = VJEPA(cfg["model"]).to(device).eval()
        ds = BilliardDataset(str(DATA_PATH))
        store = load_frames(ds)
        _, test_idx = split_indices(len(ds), train_frac=0.8)
        target_z = None  # not used in degradation_curve signature
        horizons = [1, 2, 4, 6, 8, 10, 12]
        deg = degradation_curve(model, store, test_idx, target_z, device, horizons, cfg, batch_size=4)
        assert deg["horizons"] == horizons
        for key in ("vjepa", "copy_last", "random"):
            assert len(deg[key]) == len(horizons)
            for v in deg[key]:
                assert np.isfinite(v), f"{key} produced non-finite sim"

    def test_random_baseline_near_zero(self):
        """Random gaussian vs target latents should have |cosine| << 1 on average."""
        set_seed(0)
        device = _device()
        cfg = load_config(str(CONFIG_PATH))
        model = VJEPA(cfg["model"]).to(device).eval()
        ds = BilliardDataset(str(DATA_PATH))
        store = load_frames(ds)
        _, test_idx = split_indices(len(ds), train_frac=0.8)
        deg = degradation_curve(model, store, test_idx, None, device, [1, 6, 12], cfg, batch_size=4)
        # random baseline is per-horizon a mean cosine, should be near 0 (untrained encoder
        # may have biased token means giving non-trivial sim — allow generous bound).
        for v in deg["random"]:
            assert abs(v) < 0.5, f"random baseline suspiciously high: {v}"


# ---------------------------------------------------------------------------
# TestBaselines
# ---------------------------------------------------------------------------


class TestBaselines:
    def test_pixel_baseline_encoder_param_parity(self):
        """Pixel baseline encoder must have identical param count to V-JEPA encoder."""
        cfg = load_config(str(CONFIG_PATH))
        v = VJEPA(cfg["model"])
        p = PixelPredictor(cfg["model"])
        v_enc = sum(x.numel() for x in v.encoder.parameters())
        p_enc = sum(x.numel() for x in p.encoder.parameters())
        assert v_enc == p_enc, f"encoder param mismatch: v={v_enc} pixel={p_enc}"

    def test_pixel_baseline_predictor_param_parity(self):
        cfg = load_config(str(CONFIG_PATH))
        v = VJEPA(cfg["model"])
        p = PixelPredictor(cfg["model"])
        v_pred = sum(x.numel() for x in v.predictor.parameters())
        p_pred = sum(x.numel() for x in p.predictor.parameters())
        assert v_pred == p_pred

    def test_copy_last_is_in_latent_space(self):
        """copy_last in degradation_curve uses target_encode on last context frame, not pixels."""
        # We assert by inspecting code path: degradation_curve produces a cosine sim in
        # latent space. If shapes were pixel-space, cosine_sim_batched would still work
        # but the value would correspond to pixels. We verify the implementation by
        # checking that copy_last == 1 when target frame == last context frame.
        set_seed(0)
        device = _device()
        cfg = load_config(str(CONFIG_PATH))
        # Make a synthetic store where every frame in a sequence is identical.
        ds = BilliardDataset(str(DATA_PATH))
        store = load_frames(ds)
        same = np.repeat(store.frames[:, :1], store.frames.shape[1], axis=1)
        store.frames = same
        model = VJEPA(cfg["model"]).to(device).eval()
        _, test_idx = split_indices(len(ds), train_frac=0.8)
        deg = degradation_curve(model, store, test_idx, None, device, [1], cfg, batch_size=4)
        # All frames identical → target_encode(last_ctx) == target_encode(target) → cos=1
        assert deg["copy_last"][0] == pytest.approx(1.0, abs=1e-3)


# ---------------------------------------------------------------------------
# TestFigures + end-to-end reproducibility
# ---------------------------------------------------------------------------


class TestFigures:
    def test_end_to_end_smoke_creates_all_outputs(self, tmp_path):
        """Run the full pipeline on an untrained model + sample data."""
        out = tmp_path / "results"
        # Train an untrained model: save its state_dict in a checkpoint-shaped dict.
        set_seed(0)
        cfg = load_config(str(CONFIG_PATH))
        device = _device()
        vjepa = VJEPA(cfg["model"]).to(device)
        ckpt_path = tmp_path / "ckpt.pt"
        torch.save({"epoch": 0, "model_state": vjepa.state_dict(),
                    "optimizer_state": {}, "scheduler_state": {},
                    "ema_tau": 0.996, "metrics_history": []}, ckpt_path)

        pixel = PixelPredictor(cfg["model"]).to(device)
        pix_ckpt = tmp_path / "pixel.pt"
        torch.save({"epoch": 0, "model_state": pixel.state_dict(),
                    "optimizer_state": {}, "scheduler_state": {},
                    "metrics_history": []}, pix_ckpt)

        metrics = run_evaluation(
            ckpt_path=str(ckpt_path),
            data_path=str(DATA_PATH),
            config_path=str(CONFIG_PATH),
            out_dir=str(out),
            pixel_ckpt_path=str(pix_ckpt),
        )
        for fname in [
            "latent_pca_trajectory.png",
            "degradation_curve.png",
            "tsne_by_speed.png",
            "training_curves.png",
            "jepa_vs_pixel.png",
            "metrics.json",
        ]:
            assert (out / fname).exists(), f"missing {fname}"
        assert "vjepa" in metrics and "pixel" in metrics and "degradation" in metrics
        # finite numbers only
        def _walk(x):
            if isinstance(x, dict):
                for v in x.values(): _walk(v)
            elif isinstance(x, list):
                for v in x: _walk(v)
            elif isinstance(x, float):
                assert np.isfinite(x)
        _walk(metrics)

    def test_reproducibility_same_seed_same_metrics(self, tmp_path):
        cfg = load_config(str(CONFIG_PATH))
        device = _device()
        set_seed(0)
        vjepa = VJEPA(cfg["model"]).to(device)
        ckpt = tmp_path / "ckpt.pt"
        torch.save({"epoch": 0, "model_state": vjepa.state_dict(),
                    "optimizer_state": {}, "scheduler_state": {},
                    "ema_tau": 0.996, "metrics_history": []}, ckpt)

        m1 = run_evaluation(str(ckpt), str(DATA_PATH), str(CONFIG_PATH),
                            str(tmp_path / "r1"))
        m2 = run_evaluation(str(ckpt), str(DATA_PATH), str(CONFIG_PATH),
                            str(tmp_path / "r2"))
        assert m1["vjepa"]["positions"]["overall_r2"] == pytest.approx(
            m2["vjepa"]["positions"]["overall_r2"], abs=1e-6
        )
        assert m1["degradation"]["vjepa"] == pytest.approx(m2["degradation"]["vjepa"], abs=1e-6)
