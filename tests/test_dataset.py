"""Dataset / NPZ pipeline contract tests (Mini V-JEPA).

Verifies the contract documented in PLAN_Mini_V-JEPA.md section 3.4 and the
sampling assumptions in section 4.2: frame dtype/shape, channel-first layout
returned by the Dataset, off-by-one bounds for t_start + 3 + delta_t, RGB
ordering preserved end-to-end, and roundtrip determinism for a fixed seed.
"""
from __future__ import annotations

import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_vjepa.dataset import BilliardDataset
from simulation.generator import generate_dataset, load_config
from simulation.renderer import render_frame


SAMPLE_NPZ = ROOT / "data" / "sample" / "sample.npz"
SIMPLE_CFG = ROOT / "configs" / "simple.yaml"
DEFAULT_CFG = ROOT / "configs" / "default.yaml"
STRESS_CFG = ROOT / "configs" / "stress.yaml"


@pytest.fixture(scope="module")
def tiny_npz(tmp_path_factory) -> str:
    cfg, _ = load_config(str(SIMPLE_CFG))
    cfg.n_sequences = 4
    out = tmp_path_factory.mktemp("ds") / "tiny.npz"
    cfg.output_path = str(out)
    generate_dataset(cfg, show_progress=False)
    return str(out)


class TestNPZContract:
    def test_sample_keys_and_dtypes(self):
        d = np.load(SAMPLE_NPZ, allow_pickle=False)
        assert "frames" in d.files
        assert "positions" in d.files
        assert "velocities" in d.files
        assert d["frames"].dtype == np.uint8
        assert d["positions"].dtype == np.float32
        assert d["velocities"].dtype == np.float32

    def test_sample_shapes(self):
        d = np.load(SAMPLE_NPZ, allow_pickle=False)
        N, T, H, W, C = d["frames"].shape
        assert T == 32 and H == 64 and W == 64 and C == 3
        assert d["positions"].shape == (N, T, d["positions"].shape[2], 2)
        assert d["velocities"].shape == d["positions"].shape

    def test_sample_value_ranges(self):
        d = np.load(SAMPLE_NPZ, allow_pickle=False)
        f = d["frames"]
        assert f.min() >= 0 and f.max() <= 255
        assert (f.astype(np.float32).var(axis=(1, 2, 3, 4)) > 0).all()
        pos = d["positions"]
        assert np.isfinite(pos).all()
        assert pos[..., 0].min() >= 0.0 and pos[..., 0].max() <= 1.0
        assert pos[..., 1].min() >= 0.0 and pos[..., 1].max() <= 0.5
        vel = d["velocities"]
        assert np.isfinite(vel).all()
        assert np.abs(vel).max() < 10.0

    def test_tiny_npz_is_compressed(self, tiny_npz):
        # Compressed NPZ stores entries deflated; raw .npy size > file size.
        d = np.load(tiny_npz, allow_pickle=False)
        raw = sum(arr.nbytes for arr in (d["frames"], d["positions"], d["velocities"]))
        assert os.path.getsize(tiny_npz) < raw

    def test_sample_no_initial_overlap(self):
        d = np.load(SAMPLE_NPZ, allow_pickle=False)
        P = d["positions"]
        N, T, n, _ = P.shape
        # ball_radius from config_json
        import json

        cfg = json.loads(str(d["config_json"]))
        r = float(cfg["ball_radius"])
        for s in range(N):
            P0 = P[s, 0]
            for a in range(n):
                for b in range(a + 1, n):
                    dist = math.hypot(P0[a, 0] - P0[b, 0], P0[a, 1] - P0[b, 1])
                    overlap = max(0.0, 2 * r - dist)
                    assert overlap < 1e-6, f"initial overlap {overlap:.3e} (seq {s}, balls {a},{b})"


class TestDatasetIndexing:
    def test_len_matches_npz(self):
        ds = BilliardDataset(str(SAMPLE_NPZ))
        d = np.load(SAMPLE_NPZ, allow_pickle=False)
        assert len(ds) == d["frames"].shape[0]

    def test_getitem_channel_first_uint8(self):
        ds = BilliardDataset(str(SAMPLE_NPZ))
        b = ds[0]
        f = b["frames"]
        assert tuple(f.shape) == (32, 3, 64, 64)
        assert str(f.dtype) == "torch.uint8"

    def test_getitem_pos_vel_alignment(self):
        ds = BilliardDataset(str(SAMPLE_NPZ))
        b = ds[3]
        assert b["positions"].shape[0] == b["frames"].shape[0]
        assert b["velocities"].shape == b["positions"].shape
        assert str(b["positions"].dtype) == "torch.float32"

    def test_max_window_index_in_bounds(self):
        # plan §4.2: t_start in [0,16], delta_t in [1,12], target = t_start+3+delta_t
        ds = BilliardDataset(str(SAMPLE_NPZ))
        T = ds[0]["frames"].shape[0]
        assert 16 + 3 + 12 < T, f"target index out of bounds: T={T}"
        assert 0 + 0 + 1 < T


class TestConfigs:
    @pytest.mark.parametrize("name", ["simple", "default", "stress"])
    def test_config_schema(self, name):
        with open(ROOT / "configs" / f"{name}.yaml") as f:
            raw = yaml.safe_load(f)
        d = raw["data"]
        assert d["sequence_length"] == 32
        assert d["frame_size"] == 64
        assert d["table_ratio"] == [2, 1]
        weights = d["init_strategies"]
        assert set(weights.keys()) == {
            "break",
            "midgame_cluster",
            "midgame_spread",
            "two_ball",
            "random_velocities",
        }
        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_n_balls_per_config(self):
        for name, expected in [("simple", 3), ("default", 9), ("stress", 15)]:
            with open(ROOT / "configs" / f"{name}.yaml") as f:
                raw = yaml.safe_load(f)
            assert raw["data"]["n_balls"] == expected


class TestRoundtrip:
    def test_seed_determinism(self, tmp_path):
        cfg, _ = load_config(str(SIMPLE_CFG))
        cfg.n_sequences = 4
        cfg.output_path = str(tmp_path / "a.npz")
        generate_dataset(cfg, show_progress=False)
        cfg.output_path = str(tmp_path / "b.npz")
        generate_dataset(cfg, show_progress=False)
        a = np.load(tmp_path / "a.npz")
        b = np.load(tmp_path / "b.npz")
        assert np.array_equal(a["frames"], b["frames"])
        assert np.array_equal(a["positions"], b["positions"])
        assert np.array_equal(a["velocities"], b["velocities"])

    def test_rgb_channel_preserved(self):
        # Felt is RGB green (34,139,34). Verify green > red and green > blue in
        # a felt-only pixel of a stored frame.
        d = np.load(SAMPLE_NPZ, allow_pickle=False)
        # pixel (y=2,x=32) sits in the top border area; pixel (20,32) lies in
        # the felt band for the 64x64 letterboxed render of a 2:1 table.
        px = d["frames"][0, 0, 20, 32]
        assert px[1] > px[0] and px[1] > px[2], f"channel order looks wrong: {px}"

    def test_frame_matches_re_render(self, tiny_npz):
        d = np.load(tiny_npz, allow_pickle=False)
        # Re-render frame[2] of sequence 0 from its stored positions.
        ref = render_frame(
            d["positions"][0, 2],
            frame_size=64,
            table_width=1.0,
            table_height=0.5,
            ball_radius=0.03,
        )
        assert np.array_equal(d["frames"][0, 2], ref)


class TestMemory:
    def test_sample_fits_in_ram(self):
        # Sample is small; ensure load doesn't crash and arrays stay uint8/float32.
        ds = BilliardDataset(str(SAMPLE_NPZ))
        assert ds.frames.dtype == np.uint8
        assert ds.positions.dtype == np.float32
        nbytes = ds.frames.nbytes + ds.positions.nbytes + ds.velocities.nbytes
        assert nbytes < 50 * 1024 * 1024  # well under 50 MB for 16 sequences
