"""V-JEPA evaluation pipeline.

Produces linear-probe R^2 (positions, velocities), a degradation curve vs
horizon (V-JEPA / copy-last-frame / random), latent PCA trajectories, a
t-SNE colored by ball speed, training curves, and an optional side-by-side
comparison against a pixel-baseline encoder.

CLI:
    python scripts/evaluate.py \
        --ckpt PATH --data PATH --config configs/simple.yaml \
        --out assets/results/ [--pixel-ckpt PATH]

The model is reconstructed from --config (the training checkpoint does not
embed the config). Train/test split is 80/20 on the sequence axis.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import resource
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def cap_process_memory(gb: float | None) -> None:
    """Hard-cap this process's virtual address space (macOS / Linux).

    `resource.setrlimit(RLIMIT_AS, ...)` causes the next allocation past the
    limit to raise MemoryError rather than letting the OS swap or kill the
    whole machine. On macOS this honors the soft limit but PyTorch MPS uses
    unified memory, so the cap covers both Python heap and MPS allocations.

    Caveat: if your eval genuinely needs more than `gb`, you'll get a
    MemoryError mid-run. Treat this as a defense-in-depth ceiling, not a
    workaround for an actually-too-big workload — pair it with
    `--max-eval-sequences` to shrink the working set first.
    """
    if gb is None or gb <= 0:
        return
    soft_bytes = int(gb * (1024 ** 3))
    try:
        cur_soft, cur_hard = resource.getrlimit(resource.RLIMIT_AS)
        new_hard = cur_hard if cur_hard != resource.RLIM_INFINITY else soft_bytes
        resource.setrlimit(resource.RLIMIT_AS, (soft_bytes, new_hard))
        print(f"[eval] RAM cap (RLIMIT_AS) set to {gb:.1f} GB", flush=True)
    except (ValueError, OSError) as exc:
        print(f"[eval] WARNING: could not set RLIMIT_AS to {gb} GB: {exc}",
              file=sys.stderr, flush=True)
# Headless matplotlib (no display required).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge, RidgeCV  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.metrics import r2_score  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mini_vjepa.dataset import BilliardDataset  # noqa: E402
from mini_vjepa.vjepa import VJEPA  # noqa: E402

SEED = 42
ENCODE_BATCH = 64


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def set_seed(seed: int = SEED) -> None:
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_config(path: str) -> Mapping[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_vjepa(ckpt_path: str, cfg: Mapping[str, Any], device: torch.device) -> VJEPA:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = VJEPA(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def load_pixel(ckpt_path: str, cfg: Mapping[str, Any], device: torch.device):
    from baselines.pixel_predictor import PixelPredictor
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = PixelPredictor(cfg["model"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Frame encoding
# ---------------------------------------------------------------------------


@dataclass
class FrameStore:
    """Per-sequence frames, positions, velocities; all CPU numpy."""
    frames: np.ndarray       # (N, T, 3, H, W) uint8
    positions: np.ndarray    # (N, T, n_balls, 2) float32
    velocities: np.ndarray   # (N, T, n_balls, 2) float32


def load_frames(dataset: BilliardDataset) -> FrameStore:
    frames_u8 = np.ascontiguousarray(
        np.transpose(dataset.frames, (0, 1, 4, 2, 3))
    )  # (N, T, 3, H, W)
    return FrameStore(
        frames=frames_u8,
        positions=dataset.positions,
        velocities=dataset.velocities,
    )


def _encode_frames_with(
    encode_fn,
    frames_u8_flat: np.ndarray,
    device: torch.device,
    batch_size: int = ENCODE_BATCH,
) -> np.ndarray:
    """Encode (M, 3, H, W) uint8 frames via encode_fn; return (M, n_tokens, D) numpy.

    Pre-allocates the output array and writes batches directly into it. The
    previous "accumulate-in-list + torch.cat" pattern doubled peak memory
    during the cat (e.g. the context-window cache went from 7.6 GB final to
    15.6 GB transient at 2000 sequences) and was a contributor to the eval
    spiking to ~25 GB despite the documented "12 GB" budget.
    """
    n = frames_u8_flat.shape[0]
    # Run one tiny pass to learn the encoder's output shape without
    # hard-coding (n_tokens, d) — keeps this helper generic across the
    # per-frame and pixel-baseline encoders.
    with torch.no_grad():
        sample = encode_fn(
            torch.from_numpy(frames_u8_flat[:1]).to(device).float() / 255.0
        )
    n_tokens, d = sample.shape[1], sample.shape[2]
    out = np.empty((n, n_tokens, d), dtype=np.float32)
    del sample
    for start in range(0, n, batch_size):
        chunk = frames_u8_flat[start : start + batch_size]
        x = torch.from_numpy(chunk).to(device).float() / 255.0
        with torch.no_grad():
            z = encode_fn(x)
        out[start : start + chunk.shape[0]] = z.detach().to("cpu").numpy()
    return out


def encode_all_frames_target(model: VJEPA, store: FrameStore, device: torch.device) -> np.ndarray:
    """Use the EMA target encoder; returns (N, T, n_tokens, D) float32 numpy."""
    n, t = store.frames.shape[:2]
    flat = store.frames.reshape(n * t, *store.frames.shape[2:])
    z = _encode_frames_with(lambda x: model.target_encode(x), flat, device)
    n_tokens, d = z.shape[1], z.shape[2]
    return z.reshape(n, t, n_tokens, d)


def encode_all_frames_online(model: VJEPA, store: FrameStore, device: torch.device) -> np.ndarray:
    """V-JEPA online encoder (no EMA, no temporal pos); (N, T, n_tokens, D) float32.

    The headline V1 probe used `target_encode` (EMA copy lagged by τ→1.0 cosine
    over 150 epochs). The pixel baseline's probe uses its online encoder. This
    helper lets us probe both encoders on equal footing and disentangle "is the
    EMA stale?" from "is the representation worse?".
    """
    n, t = store.frames.shape[:2]
    flat = store.frames.reshape(n * t, *store.frames.shape[2:])
    z = _encode_frames_with(lambda x: model.encoder(x), flat, device)
    n_tokens, d = z.shape[1], z.shape[2]
    return z.reshape(n, t, n_tokens, d)


def _encode_windows_with(
    encode_fn,
    store_frames: np.ndarray,
    device: torch.device,
    context_frames: int,
    batch_size: int = ENCODE_BATCH,
) -> np.ndarray:
    """Encode 4-frame sliding windows into a pre-allocated array.

    Shared between V-JEPA and pixel-baseline context-window encoders. Same
    pre-allocation strategy as `_encode_frames_with` — never accumulates a
    list of batch tensors (which the old `torch.cat`-based version did, at
    a 2× peak-memory cost; that was the headline bug behind the eval
    spiking to ~24 GB despite the budget claim of ~12 GB).
    """
    n, t = store_frames.shape[:2]
    t_eff = t - context_frames + 1
    if t_eff <= 0:
        raise ValueError(f"sequence_length={t} too short for context_frames={context_frames}")
    # Discover output shape from a single window.
    sample_window = store_frames[:1, :context_frames]  # (1, ctx, 3, H, W)
    with torch.no_grad():
        sample = encode_fn(
            torch.from_numpy(sample_window).to(device).float() / 255.0
        )
    n_ctx, d = sample.shape[1], sample.shape[2]
    out = np.empty((n, t_eff, n_ctx, d), dtype=np.float32)
    del sample

    # Iterate batches of sequences; for each batch, encode all t_eff windows.
    # `windows` is reused across iterations of the outer loop (max ~91 MB at
    # batch=64, ctx=4, 64x64).
    h, w = store_frames.shape[3], store_frames.shape[4]
    windows = np.empty(
        (batch_size, t_eff, context_frames, store_frames.shape[2], h, w),
        dtype=np.uint8,
    )
    for start in range(0, n, batch_size):
        chunk = store_frames[start : start + batch_size]
        b = chunk.shape[0]
        windows_view = windows[:b]
        for w_idx in range(t_eff):
            windows_view[:, w_idx] = chunk[:, w_idx : w_idx + context_frames]
        flat = windows_view.reshape(b * t_eff, context_frames, store_frames.shape[2], h, w)
        x = torch.from_numpy(flat).to(device).float() / 255.0
        with torch.no_grad():
            tokens = encode_fn(x)
        out[start : start + b] = (
            tokens.detach().to("cpu").numpy().reshape(b, t_eff, n_ctx, d)
        )
    return out


def encode_all_context_windows(
    model: VJEPA, store: FrameStore, device: torch.device, context_frames: int = 4
) -> np.ndarray:
    """V-JEPA's actual training representation: 4-frame context with temporal pos.

    Returns (N, T_eff, n_context_tokens, D) where T_eff = T - context_frames + 1
    and the t-th slot holds the representation for window [t, t+context_frames).
    The last-frame ground truth at index `t + context_frames - 1` is the natural
    probe target — that frame's velocity is decodable from the 4-frame window
    even though it isn't from a single frame.
    """
    return _encode_windows_with(
        lambda x: model.encode_context(x), store.frames, device, context_frames
    )


def encode_all_frames_pixel(pixel_model, store: FrameStore, device: torch.device) -> np.ndarray:
    """Use the pixel baseline's online encoder; (N, T, n_tokens, D) float32."""
    n, t = store.frames.shape[:2]
    flat = store.frames.reshape(n * t, *store.frames.shape[2:])
    z = _encode_frames_with(lambda x: pixel_model.encoder(x), flat, device)
    n_tokens, d = z.shape[1], z.shape[2]
    return z.reshape(n, t, n_tokens, d)


def encode_all_context_windows_pixel(
    pixel_model, store: FrameStore, device: torch.device, context_frames: int = 4
) -> np.ndarray:
    """Pixel baseline's online encoder applied window-by-window (same protocol).

    PixelPredictor exposes `encode_context(frames_4)` with the same signature
    as VJEPA.encode_context (verbatim reuse — see baselines/pixel_predictor.py).
    Provides an apples-to-apples context-window probe against V-JEPA.
    """
    return _encode_windows_with(
        lambda x: pixel_model.encode_context(x), store.frames, device, context_frames
    )


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def split_indices(n_sequences: int, train_frac: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    n_train = max(1, int(round(n_sequences * train_frac)))
    n_train = min(n_train, n_sequences - 1) if n_sequences > 1 else n_sequences
    return np.arange(n_train), np.arange(n_train, n_sequences)


def shuffled_split_indices(
    n_sequences: int, train_frac: float = 0.8, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    """Sequence-level random split. V1 used an unshuffled `np.arange` split
    that was de-facto non-iid across init strategies if the generator emitted
    sequences strategy-by-strategy. Shuffling fixes the immediate bias; for
    a real OOD test see `regime_split_indices`."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    n_train = max(1, int(round(n_sequences * train_frac)))
    n_train = min(n_train, n_sequences - 1) if n_sequences > 1 else n_sequences
    return perm[:n_train], perm[n_train:]


def regime_split_indices(
    strategies: np.ndarray | None,
    held_out: tuple[str, ...] = ("random_velocities",),
) -> tuple[np.ndarray, np.ndarray] | None:
    """Held-out-by-strategy split. Train on every strategy NOT in `held_out`;
    test on those that are. Returns None if `strategies` is unavailable (older
    NPZs) or if either side of the split would be empty.

    The default holds out `random_velocities`, the strategy least visually
    similar to the training-heavy `break` and `midgame_*` patterns. The
    DESIGN.md "Evaluation honesty notes" item 1 explicitly flagged the missing
    OOD probe as a known weakness.
    """
    if strategies is None:
        return None
    mask_test = np.isin(strategies, list(held_out))
    test_idx = np.where(mask_test)[0]
    train_idx = np.where(~mask_test)[0]
    if len(train_idx) == 0 or len(test_idx) == 0:
        return None
    return train_idx, test_idx


def flatten_for_probe(
    z_seq: np.ndarray, target_seq: np.ndarray, indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """z_seq: (N, T, n_tokens, D). target_seq: (N, T, n_balls, 2). Returns flat (M, F), (M, K)."""
    z_sel = z_seq[indices]
    t_sel = target_seq[indices]
    m = z_sel.shape[0] * z_sel.shape[1]
    x = z_sel.reshape(m, -1)
    y = t_sel.reshape(m, -1)
    return x.astype(np.float32, copy=False), y.astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


RIDGE_CV_ALPHAS = np.logspace(-3, 3, 13)


def linear_probe(
    z_seq: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    alpha: float | None = None,
    max_train_sequences: int = 2000,
    solver: str = "auto",
) -> dict:
    """Fit Ridge from latents to targets; return overall + per-ball R^2 and MAE.

    On the 10000-sequence default dataset the flatten step would allocate
    ~8 GB per side; combined with target_z + pixel_z already in memory, this
    OOMs the 32GB-Mac eval. Cap train_idx at max_train_sequences so the probe
    runs in ~2 GB. Test set is always used in full (only ~2k sequences).

    `alpha=None` (default) uses RidgeCV over `RIDGE_CV_ALPHAS`; pass an explicit
    float to bypass CV (used by per-horizon rollout probes where re-fitting CV
    for every horizon would be wasteful).
    """
    if len(train_idx) > max_train_sequences:
        # deterministic stride subsample preserves sequence diversity
        stride = max(1, len(train_idx) // max_train_sequences)
        train_idx = train_idx[::stride][:max_train_sequences]
    x_tr, y_tr = flatten_for_probe(z_seq, targets, train_idx)
    x_te, y_te = flatten_for_probe(z_seq, targets, test_idx)
    if alpha is None:
        model = RidgeCV(alphas=RIDGE_CV_ALPHAS)
        model.fit(x_tr, y_tr)
        chosen_alpha = float(model.alpha_)
    else:
        model = Ridge(alpha=alpha, solver=solver)
        model.fit(x_tr, y_tr)
        chosen_alpha = float(alpha)
    y_pred = model.predict(x_te)
    # free big intermediates before next probe — sklearn can hold internal
    # copies of x_tr that take 8 GB on the default dataset.
    del x_tr, y_tr
    import gc; gc.collect()
    overall_r2 = float(r2_score(y_te, y_pred))
    mae = float(np.mean(np.abs(y_te - y_pred)))
    n_balls = targets.shape[2]
    per_ball_r2: list[float] = []
    y_te_b = y_te.reshape(-1, n_balls, 2)
    y_pred_b = y_pred.reshape(-1, n_balls, 2)
    for b in range(n_balls):
        per_ball_r2.append(float(r2_score(y_te_b[:, b, :], y_pred_b[:, b, :])))
    return {
        "overall_r2": overall_r2,
        "per_ball_r2": per_ball_r2,
        "mean_per_ball_r2": float(np.mean(per_ball_r2)),
        "mae": mae,
        "alpha": chosen_alpha,
        "n_train_sequences": int(len(train_idx)),
        "n_test_sequences": int(len(test_idx)),
    }


def context_window_probe(
    z_ctx: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    context_frames: int = 4,
    max_train_sequences: int = 2000,
) -> dict:
    """Probe from 4-frame context latents to the target at the last context frame.

    A single 64×64 frame contains positions but no velocity information; a per-
    frame Ridge probe of velocities is therefore physically ill-posed (V1's
    0.022 vs -0.042 result is noise on both sides). This probe gives the
    regression access to 4 consecutive frames — the same temporal window
    V-JEPA's encoder is trained on — which is the minimum needed for velocity
    to be linearly decodable.

    z_ctx: (N, T_eff, n_ctx_tokens, D) from `encode_all_context_windows`.
    targets: (N, T, n_balls, 2). The window at slot t covers original frames
    [t, t+context_frames); the natural ground-truth slot is t+context_frames-1.

    Uses `alpha=1.0` with sklearn's iterative `lsqr` solver: the per-frame
    probe has 64*128 = 8192 features and RidgeCV's SVD path is fine, but the
    context-window probe has 4*64*128 = 32768 features and the SVD would
    allocate ~7 GB (a 29000×29000 U matrix at our sample count). lsqr never
    materializes the Gram matrix; the trade is no alpha tuning for that probe.
    """
    last_t_idx = context_frames - 1
    t_eff = z_ctx.shape[1]
    targets_aligned = targets[:, last_t_idx : last_t_idx + t_eff]
    return linear_probe(
        z_ctx, targets_aligned, train_idx, test_idx,
        alpha=1.0, solver="lsqr",
        max_train_sequences=max_train_sequences,
    )


# ---------------------------------------------------------------------------
# Degradation curve
# ---------------------------------------------------------------------------


def cosine_sim_batched(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a, b: (B, n_tokens, D). Returns scalar mean cosine sim per sample, averaged."""
    a_flat = a.reshape(a.shape[0], -1)
    b_flat = b.reshape(b.shape[0], -1)
    cs = torch.nn.functional.cosine_similarity(a_flat, b_flat, dim=1)
    return cs.mean()


def degradation_curve(
    model: VJEPA,
    store: FrameStore,
    test_idx: np.ndarray,
    target_z: np.ndarray,
    device: torch.device,
    horizons: list[int],
    cfg: Mapping[str, Any],
    batch_size: int = 16,
    t_start_stride: int = 4,
) -> dict:
    """For each horizon, average cosine sim of predicted vs target latent on test sequences.

    Three predictors:
      - V-JEPA: model.predict(context_tokens, dt)
      - copy-last: encoded last context frame (target encoder), tiled
      - random: gaussian noise matching the target shape

    V1 fixed `t_start = max_t_start // 2`; that hides the variance across the
    sequence. We now average over every `t_start_stride`-th valid start and
    also report the standard deviation across starts, giving the curve a real
    error bar.
    """
    context_frames = int(cfg["model"]["context_frames"])
    seq_len = int(cfg["data"]["sequence_length"])
    n_tokens = int(cfg["model"]["spatial_tokens"])
    latent_dim = int(cfg["model"]["latent_dim"])
    # Match training-time masking so the predictor sees the distribution it
    # was trained on. Without this the degradation curve measures predictor
    # output on an unseen unmasked-context distribution and reports ~0.07
    # cosine similarity, which is a train/eval-mismatch artifact, not a
    # real prediction failure.
    mask_ratio = float(cfg.get("training", {}).get("mask_ratio", 0.0))

    test_frames = store.frames[test_idx]  # (Nte, T, 3, H, W)
    n_te = test_frames.shape[0]

    results = {
        "horizons": list(horizons),
        "vjepa": [], "copy_last": [], "random": [],
        "vjepa_std": [], "copy_last_std": [],
        "n_t_starts_per_horizon": [],
    }

    rng = np.random.default_rng(SEED)

    for h in horizons:
        max_t_start = seq_len - context_frames - h
        if max_t_start < 0:
            for k in ("vjepa", "copy_last", "random", "vjepa_std", "copy_last_std"):
                results[k].append(float("nan"))
            results["n_t_starts_per_horizon"].append(0)
            continue

        t_starts = list(range(0, max_t_start + 1, t_start_stride))
        if not t_starts:
            t_starts = [0]

        vjepa_per_start: list[float] = []
        copy_per_start: list[float] = []
        rand_per_start: list[float] = []

        for t_start in t_starts:
            vjepa_sims: list[float] = []
            copy_sims: list[float] = []
            rand_sims: list[float] = []

            for start in range(0, n_te, batch_size):
                chunk = test_frames[start : start + batch_size]
                b = chunk.shape[0]
                x = torch.from_numpy(chunk).to(device).float() / 255.0

                context = x[:, t_start : t_start + context_frames].contiguous()
                target_idx = t_start + context_frames - 1 + h
                target_frame = x[:, target_idx].contiguous()
                dt_tensor = torch.full((b,), h, device=device, dtype=torch.long)

                with torch.no_grad():
                    ctx_tokens = model.encode_context(context)
                    predictor_input = model.apply_context_mask(ctx_tokens, mask_ratio)
                    z_pred = model.predict(predictor_input, dt_tensor)
                    z_target = model.target_encode(target_frame)
                    last_frame = x[:, t_start + context_frames - 1].contiguous()
                    z_copy = model.target_encode(last_frame)

                vjepa_sims.append(cosine_sim_batched(z_pred, z_target).item() * b)
                copy_sims.append(cosine_sim_batched(z_copy, z_target).item() * b)
                z_rand = torch.from_numpy(
                    rng.standard_normal((b, n_tokens, latent_dim)).astype(np.float32)
                ).to(device)
                rand_sims.append(cosine_sim_batched(z_rand, z_target).item() * b)

            denom = float(n_te)
            vjepa_per_start.append(sum(vjepa_sims) / denom)
            copy_per_start.append(sum(copy_sims) / denom)
            rand_per_start.append(sum(rand_sims) / denom)

        results["vjepa"].append(float(np.mean(vjepa_per_start)))
        results["copy_last"].append(float(np.mean(copy_per_start)))
        results["random"].append(float(np.mean(rand_per_start)))
        results["vjepa_std"].append(float(np.std(vjepa_per_start)))
        results["copy_last_std"].append(float(np.std(copy_per_start)))
        results["n_t_starts_per_horizon"].append(len(t_starts))

    return results


def rollout_position_error(
    model: VJEPA,
    store: FrameStore,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    target_z: np.ndarray,
    device: torch.device,
    horizons: list[int],
    cfg: Mapping[str, Any],
    batch_size: int = 16,
    t_start_stride: int = 4,
) -> dict:
    """Predict z_pred at horizon h, decode to ball positions via a frozen Ridge
    probe trained on the EMA target encoder, and report per-horizon position MAE.

    This replaces the cos-sim degradation curve as the headline rollout metric.
    The V1 curve compared cos(z_pred, z_target) to cos(copy_last, z_target) and
    reported 0.075 vs 0.95 — which mostly measures that with mask_ratio=0.75 the
    predictor outputs the target centroid (high MSE-of-mean accuracy, near-zero
    per-sample cosine sim). A probe-decoded MAE in *position space* is the
    physically meaningful quantity: "if we trust the predictor, where would it
    place the balls, and how far off would it be?".

    Two reference baselines:
      - copy-last-frame: encode the last context frame, decode to positions
      - identity-positions: assume balls don't move (MAE of position(t) vs
        position(t+context_frames-1+h)) — the *true* trivial-physics baseline
    """
    context_frames = int(cfg["model"]["context_frames"])
    seq_len = int(cfg["data"]["sequence_length"])
    mask_ratio = float(cfg.get("training", {}).get("mask_ratio", 0.0))

    # Fit the decode-probe once on the train split using EMA-target latents at
    # every frame the probe will be asked to decode. Use a fixed alpha to keep
    # the multi-horizon cost bounded.
    probe_train = linear_probe(target_z, store.positions, train_idx, test_idx, alpha=1.0)
    # Refit on the full (capped) train set so we can call .predict directly.
    max_train_sequences = 2000
    if len(train_idx) > max_train_sequences:
        stride = max(1, len(train_idx) // max_train_sequences)
        train_idx_cap = train_idx[::stride][:max_train_sequences]
    else:
        train_idx_cap = train_idx
    x_tr, y_tr = flatten_for_probe(target_z, store.positions, train_idx_cap)
    decode = Ridge(alpha=probe_train["alpha"] if "alpha" in probe_train else 1.0)
    decode.fit(x_tr, y_tr)
    del x_tr, y_tr
    import gc; gc.collect()

    test_frames = store.frames[test_idx]
    test_positions = store.positions[test_idx]
    n_te = test_frames.shape[0]
    n_balls = test_positions.shape[2]

    results = {
        "horizons": list(horizons),
        "vjepa_mae": [], "copy_last_mae": [], "identity_mae": [],
        "n_t_starts_per_horizon": [],
        "probe_alpha": float(probe_train.get("alpha", 1.0)),
        "probe_train_r2": float(probe_train["overall_r2"]),
    }

    for h in horizons:
        max_t_start = seq_len - context_frames - h
        if max_t_start < 0:
            for k in ("vjepa_mae", "copy_last_mae", "identity_mae"):
                results[k].append(float("nan"))
            results["n_t_starts_per_horizon"].append(0)
            continue

        t_starts = list(range(0, max_t_start + 1, t_start_stride))
        if not t_starts:
            t_starts = [0]

        vj_maes: list[float] = []
        cp_maes: list[float] = []
        id_maes: list[float] = []

        for t_start in t_starts:
            target_t = t_start + context_frames - 1 + h
            last_ctx_t = t_start + context_frames - 1
            true_pos = test_positions[:, target_t].reshape(n_te, n_balls * 2)
            last_pos = test_positions[:, last_ctx_t].reshape(n_te, n_balls * 2)
            id_maes.append(float(np.mean(np.abs(true_pos - last_pos))))

            z_pred_all: list[np.ndarray] = []
            z_copy_all: list[np.ndarray] = []
            for start in range(0, n_te, batch_size):
                chunk = test_frames[start : start + batch_size]
                b = chunk.shape[0]
                x = torch.from_numpy(chunk).to(device).float() / 255.0
                context = x[:, t_start : t_start + context_frames].contiguous()
                dt_tensor = torch.full((b,), h, device=device, dtype=torch.long)
                with torch.no_grad():
                    ctx_tokens = model.encode_context(context)
                    pred_input = model.apply_context_mask(ctx_tokens, mask_ratio)
                    z_pred = model.predict(pred_input, dt_tensor)
                    last_frame = x[:, last_ctx_t].contiguous()
                    z_copy = model.target_encode(last_frame)
                z_pred_all.append(z_pred.detach().to("cpu").numpy())
                z_copy_all.append(z_copy.detach().to("cpu").numpy())

            z_pred_flat = np.concatenate(z_pred_all, axis=0).reshape(n_te, -1)
            z_copy_flat = np.concatenate(z_copy_all, axis=0).reshape(n_te, -1)
            pos_pred = decode.predict(z_pred_flat)
            pos_copy = decode.predict(z_copy_flat)

            vj_maes.append(float(np.mean(np.abs(true_pos - pos_pred))))
            cp_maes.append(float(np.mean(np.abs(true_pos - pos_copy))))

        results["vjepa_mae"].append(float(np.mean(vj_maes)))
        results["copy_last_mae"].append(float(np.mean(cp_maes)))
        results["identity_mae"].append(float(np.mean(id_maes)))
        results["n_t_starts_per_horizon"].append(len(t_starts))

    return results


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def plot_latent_pca_trajectory(
    target_z: np.ndarray, store: FrameStore, out_path: Path, n_sequences: int = 4
) -> None:
    n, t = target_z.shape[:2]
    # mean-pool over spatial tokens to get a per-frame vector
    pooled = target_z.mean(axis=2)  # (N, T, D)
    all_vecs = pooled.reshape(n * t, -1)
    pca = PCA(n_components=2, random_state=SEED)
    pca.fit(all_vecs)
    pick = np.linspace(0, n - 1, num=min(n_sequences, n), dtype=int)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i in pick:
        traj = pca.transform(pooled[i])
        sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(t), cmap="viridis", s=18)
        ax.plot(traj[:, 0], traj[:, 1], linewidth=0.8, alpha=0.5)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("frame index (t)")
    ax.set_title("Latent PCA trajectory (per-frame, target encoder)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_degradation_curve(curve: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    h = curve["horizons"]
    vjepa_mean = np.array(curve["vjepa"], dtype=float)
    copy_mean = np.array(curve["copy_last"], dtype=float)
    rand_mean = np.array(curve["random"], dtype=float)
    vjepa_std = np.array(curve.get("vjepa_std", [0.0] * len(h)), dtype=float)
    copy_std = np.array(curve.get("copy_last_std", [0.0] * len(h)), dtype=float)
    ax.errorbar(h, vjepa_mean, yerr=vjepa_std, fmt="o-", label="V-JEPA", capsize=3)
    ax.errorbar(h, copy_mean, yerr=copy_std, fmt="s--", label="copy-last-frame", capsize=3)
    ax.plot(h, rand_mean, "x:", label="random")
    ax.set_xlabel("prediction horizon Δt")
    ax.set_ylabel("cosine similarity (z_pred, z_target)")
    ax.set_title("Latent prediction degradation vs horizon\n(error bars: std across t_start)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_rollout_mae(roll: dict, out_path: Path) -> None:
    """Position-space rollout MAE; the physically meaningful counterpart to
    the cos-sim degradation curve. Lower is better."""
    fig, ax = plt.subplots(figsize=(7, 5))
    h = roll["horizons"]
    ax.plot(h, roll["vjepa_mae"], "o-", label="V-JEPA (predictor + probe)")
    ax.plot(h, roll["copy_last_mae"], "s--", label="copy-last-frame + probe")
    ax.plot(h, roll["identity_mae"], "^:", label="identity (assume no motion)")
    ax.set_xlabel("prediction horizon Δt")
    ax.set_ylabel("position MAE (normalized table coords)")
    title = (
        f"Rollout position error vs horizon\n"
        f"probe α={roll.get('probe_alpha', 1.0):.2g}, "
        f"probe train R²={roll.get('probe_train_r2', float('nan')):.3f}"
    )
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_probe_grid(metrics: dict, out_path: Path) -> None:
    """Side-by-side bar chart of every probe variant we computed: per-frame
    target/online, 4-frame context window, regime-held-out where available,
    each for V-JEPA and pixel baseline. The headline 0.567-vs-0.616 number
    is one of ~10 numbers here; the point of the figure is to surface the
    full picture rather than pick a favorable one."""
    labels: list[str] = []
    vjepa_vals: list[float] = []
    pixel_vals: list[float] = []

    def add(label, vj, px):
        labels.append(label)
        vjepa_vals.append(float("nan") if vj is None else float(vj))
        pixel_vals.append(float("nan") if px is None else float(px))

    vj = metrics["vjepa"]
    px = metrics.get("pixel", {})
    add("pos / per-frame / target (V1)", vj["positions"]["overall_r2"],
        px.get("positions", {}).get("overall_r2"))
    add("pos / per-frame / online", vj.get("positions_online", {}).get("overall_r2"),
        px.get("positions", {}).get("overall_r2"))
    add("pos / 4-frame ctx-window", vj.get("positions_ctx_window", {}).get("overall_r2"),
        px.get("positions_ctx_window", {}).get("overall_r2"))
    add("vel / per-frame / target (V1)", vj["velocities"]["overall_r2"],
        px.get("velocities", {}).get("overall_r2"))
    add("vel / per-frame / online", vj.get("velocities_online", {}).get("overall_r2"),
        px.get("velocities", {}).get("overall_r2"))
    add("vel / 4-frame ctx-window", vj.get("velocities_ctx_window", {}).get("overall_r2"),
        px.get("velocities_ctx_window", {}).get("overall_r2"))
    if "regime_held_out" in metrics:
        ro = metrics["regime_held_out"]
        px_ro = px.get("regime_held_out", {})
        add("pos / ctx-window / held-out",
            ro["ctx_window"]["positions"]["overall_r2"],
            px_ro.get("ctx_window", {}).get("positions", {}).get("overall_r2"))
        add("vel / ctx-window / held-out",
            ro["ctx_window"]["velocities"]["overall_r2"],
            px_ro.get("ctx_window", {}).get("velocities", {}).get("overall_r2"))

    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, vjepa_vals, width, label="V-JEPA")
    ax.bar(x + width / 2, pixel_vals, width, label="pixel baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("R² (test split)")
    ax.set_title("Probe-by-probe R² across every variant\n"
                 "(higher = encoder linearly contains the target)")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_tsne_by_speed(
    target_z: np.ndarray, store: FrameStore, out_path: Path, n_samples: int = 500
) -> None:
    n, t = target_z.shape[:2]
    pooled = target_z.mean(axis=2)  # (N, T, D)
    speed = np.linalg.norm(store.velocities, axis=-1).mean(axis=-1)  # (N, T)
    flat_vec = pooled.reshape(n * t, -1)
    flat_speed = speed.reshape(n * t)
    n_total = flat_vec.shape[0]
    take = min(n_samples, n_total)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n_total, size=take, replace=False)
    sub_vec = flat_vec[idx]
    sub_speed = flat_speed[idx]
    perplexity = max(5, min(30, (take - 1) // 3))
    tsne = TSNE(n_components=2, random_state=SEED, perplexity=perplexity, init="pca")
    emb = tsne.fit_transform(sub_vec)
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=sub_speed, cmap="plasma", s=12)
    fig.colorbar(sc, ax=ax, label="avg ball speed")
    ax.set_title(f"t-SNE of latents colored by mean speed (N={take})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_training_curves(metrics_csv: Path | None, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if metrics_csv is None or not metrics_csv.exists():
        for ax, name in zip(axes, ["loss", "effective_rank", "avg_std"]):
            ax.set_title(f"{name} (no metrics.csv found)")
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(out_path, dpi=140)
        plt.close(fig)
        return

    epochs: list[int] = []
    cols: dict[str, list[float]] = {"loss": [], "effective_rank": [], "avg_std": []}
    with open(metrics_csv, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                epochs.append(int(row["epoch"]))
                for k in cols:
                    cols[k].append(float(row[k]))
            except (KeyError, ValueError):
                continue
    for ax, (name, vals) in zip(axes, cols.items()):
        ax.plot(epochs, vals, "o-")
        ax.set_title(name)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_jepa_vs_pixel(
    target_z: np.ndarray,
    pixel_z: np.ndarray,
    store: FrameStore,
    probe_results: dict,
    out_path: Path,
    n_sequences: int = 4,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, z_arr, title in (
        (axes[0, 0], target_z, "V-JEPA target encoder — latent PCA"),
        (axes[0, 1], pixel_z, "Pixel baseline encoder — latent PCA"),
    ):
        n, t = z_arr.shape[:2]
        pooled = z_arr.mean(axis=2)
        flat = pooled.reshape(n * t, -1)
        pca = PCA(n_components=2, random_state=SEED)
        pca.fit(flat)
        pick = np.linspace(0, n - 1, num=min(n_sequences, n), dtype=int)
        for i in pick:
            traj = pca.transform(pooled[i])
            sc = ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(t), cmap="viridis", s=14)
            ax.plot(traj[:, 0], traj[:, 1], linewidth=0.7, alpha=0.5)
        ax.set_title(title)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=axes[0, :].ravel().tolist(), label="frame index", shrink=0.7)

    # Bar chart of probe R^2
    ax = axes[1, 0]
    labels = ["positions", "velocities"]
    vjepa_r2 = [probe_results["vjepa"]["positions"]["overall_r2"],
                probe_results["vjepa"]["velocities"]["overall_r2"]]
    pixel_r2 = [probe_results["pixel"]["positions"]["overall_r2"],
                probe_results["pixel"]["velocities"]["overall_r2"]]
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width / 2, vjepa_r2, width, label="V-JEPA")
    ax.bar(x + width / 2, pixel_r2, width, label="pixel baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("R²")
    ax.set_title("Linear probe R² (test split)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    axes[1, 1].axis("off")
    text = (
        "V-JEPA vs Pixel baseline — downstream probing\n"
        f"positions R²: V-JEPA={vjepa_r2[0]:.3f}  pixel={pixel_r2[0]:.3f}\n"
        f"velocities R²: V-JEPA={vjepa_r2[1]:.3f}  pixel={pixel_r2[1]:.3f}\n"
    )
    axes[1, 1].text(0.05, 0.5, text, fontsize=11, va="center", family="monospace")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def find_train_metrics_csv(ckpt_path: Path) -> Path | None:
    candidate = ckpt_path.parent / "metrics.csv"
    if candidate.exists():
        return candidate
    return None


def _subsample_store(store: FrameStore, indices: np.ndarray, strategies: np.ndarray | None
                     ) -> tuple[FrameStore, np.ndarray | None]:
    """Return a memory-aligned subset of `store` containing only `indices`.
    The full 10K-sequence dataset (10.5 GB per encoder latent at 128-dim,
    38 GB per context-window cache) does not fit in 20 GB of RAM. The probes
    are statistically saturated well below 10K sequences anyway."""
    sub = FrameStore(
        frames=np.ascontiguousarray(store.frames[indices]),
        positions=np.ascontiguousarray(store.positions[indices]),
        velocities=np.ascontiguousarray(store.velocities[indices]),
    )
    sub_strategies = None if strategies is None else strategies[indices]
    return sub, sub_strategies


def run_evaluation(
    ckpt_path: str,
    data_path: str,
    config_path: str,
    out_dir: str,
    pixel_ckpt_path: str | None = None,
    horizons: list[int] | None = None,
    max_eval_sequences: int = 2000,
    ram_limit_gb: float | None = None,
) -> dict:
    cap_process_memory(ram_limit_gb)
    set_seed(SEED)
    device = select_device()
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if horizons is None:
        horizons = [1, 2, 4, 6, 8, 10, 12]

    print(f"[eval] device={device} ckpt={ckpt_path} data={data_path}", flush=True)
    dataset = BilliardDataset(data_path)
    n_seq_full = len(dataset)

    # Stride-subsample BEFORE materializing any (N, T, 3, H, W) array. V1
    # called load_frames (transpose + ascontiguousarray = full copy, 1.2 GB
    # at 10K seqs) on the full dataset and then subsampled — paying for both
    # the dataset.frames original AND the transposed copy in memory.
    if n_seq_full > max_eval_sequences:
        stride = max(1, n_seq_full // max_eval_sequences)
        keep = np.arange(0, n_seq_full, stride)[:max_eval_sequences]
        print(
            f"[eval] subsampling {n_seq_full} -> {len(keep)} sequences "
            f"(stride={stride}) to fit memory budget; pass --max-eval-sequences "
            f"to override",
            flush=True,
        )
    else:
        keep = np.arange(n_seq_full)
    n_seq = len(keep)

    # Build the in-memory store from the subset only. Transpose (H,W,3) →
    # (3,H,W) on the subset, not the full dataset.
    sub_frames = np.ascontiguousarray(
        np.transpose(dataset.frames[keep], (0, 1, 4, 2, 3))
    )
    sub_positions = np.ascontiguousarray(dataset.positions[keep])
    sub_velocities = np.ascontiguousarray(dataset.velocities[keep])
    sub_strategies = None if dataset.strategies is None else dataset.strategies[keep]
    store = FrameStore(
        frames=sub_frames, positions=sub_positions, velocities=sub_velocities
    )
    # Drop the BilliardDataset reference so the full-dataset numpy arrays
    # (1.2 GB at 10K seqs) get reclaimed. The npz mmap is also released.
    del dataset
    gc.collect()

    train_idx, test_idx = shuffled_split_indices(n_seq, train_frac=0.8, seed=SEED)
    print(f"[eval] sequences={n_seq} train={len(train_idx)} test={len(test_idx)}", flush=True)
    context_frames = int(cfg["model"]["context_frames"])

    model = load_vjepa(ckpt_path, cfg, device)

    def _free_gpu():
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()

    # --- Phase A: EMA target encoder ---
    # Keep target_z around longer than the others: degradation, rollout, PCA
    # and t-SNE plots all need it.
    print("[eval] encoding frames with EMA target encoder...", flush=True)
    target_z = encode_all_frames_target(model, store, device)
    print(f"[eval] target_z shape={target_z.shape}", flush=True)
    pos_probe_target = linear_probe(target_z, store.positions, train_idx, test_idx)
    vel_probe_target = linear_probe(target_z, store.velocities, train_idx, test_idx)
    print(
        f"[eval]   target/positions R²={pos_probe_target['overall_r2']:.4f} "
        f"target/velocities R²={vel_probe_target['overall_r2']:.4f}",
        flush=True,
    )

    # --- Phase B: V-JEPA online encoder ---
    print("[eval] encoding frames with online encoder...", flush=True)
    online_z = encode_all_frames_online(model, store, device)
    pos_probe_online = linear_probe(online_z, store.positions, train_idx, test_idx)
    vel_probe_online = linear_probe(online_z, store.velocities, train_idx, test_idx)
    print(
        f"[eval]   online/positions R²={pos_probe_online['overall_r2']:.4f} "
        f"online/velocities R²={vel_probe_online['overall_r2']:.4f}",
        flush=True,
    )
    del online_z  # not needed for plotting or further probes
    _free_gpu()

    # --- Phase C: 4-frame context window (physically valid velocity probe) ---
    print(f"[eval] encoding 4-frame context windows (V-JEPA's native repr.)...",
          flush=True)
    ctx_z = encode_all_context_windows(model, store, device, context_frames=context_frames)
    print(f"[eval] ctx_z shape={ctx_z.shape}", flush=True)
    pos_probe_ctx = context_window_probe(
        ctx_z, store.positions, train_idx, test_idx, context_frames=context_frames
    )
    vel_probe_ctx = context_window_probe(
        ctx_z, store.velocities, train_idx, test_idx, context_frames=context_frames
    )
    print(
        f"[eval]   ctx-window/positions R²={pos_probe_ctx['overall_r2']:.4f} "
        f"ctx-window/velocities R²={vel_probe_ctx['overall_r2']:.4f} (α={vel_probe_ctx['alpha']:.2g})",
        flush=True,
    )

    # --- Phase D: held-out-by-strategy probes (reuse target_z + ctx_z) ---
    regime = regime_split_indices(sub_strategies)
    regime_metrics: dict | None = None
    if regime is not None:
        r_train, r_test = regime
        print(
            f"[eval] held-out-regime probe (test=random_velocities): "
            f"train={len(r_train)} test={len(r_test)}",
            flush=True,
        )
        regime_metrics = {
            "held_out": ["random_velocities"],
            "n_train": int(len(r_train)),
            "n_test": int(len(r_test)),
            "target": {
                "positions": linear_probe(target_z, store.positions, r_train, r_test),
                "velocities": linear_probe(target_z, store.velocities, r_train, r_test),
            },
            "ctx_window": {
                "positions": context_window_probe(
                    ctx_z, store.positions, r_train, r_test, context_frames=context_frames
                ),
                "velocities": context_window_probe(
                    ctx_z, store.velocities, r_train, r_test, context_frames=context_frames
                ),
            },
        }
        print(
            f"[eval]   regime/target/positions R²="
            f"{regime_metrics['target']['positions']['overall_r2']:.4f} "
            f"regime/ctx-window/velocities R²="
            f"{regime_metrics['ctx_window']['velocities']['overall_r2']:.4f}",
            flush=True,
        )
    else:
        print("[eval] held-out-regime probe: skipped (strategies metadata absent)",
              flush=True)

    # ctx_z (the largest single array) is no longer needed.
    del ctx_z
    _free_gpu()

    # --- Phase E: degradation + rollout — these use `model` directly ---
    print("[eval] degradation curve (cos sim, multi-t_start averaged)...", flush=True)
    deg = degradation_curve(model, store, test_idx, target_z, device, horizons, cfg)
    print("[eval] rollout MAE (probe-decoded positions, multi-t_start)...", flush=True)
    roll = rollout_position_error(
        model, store, train_idx, test_idx, target_z, device, horizons, cfg
    )

    metrics: dict = {
        "ckpt": str(ckpt_path),
        "data": str(data_path),
        "config": str(config_path),
        "seed": SEED,
        "n_sequences_dataset": int(n_seq_full),
        "n_sequences_evaluated": int(n_seq),
        "max_eval_sequences": int(max_eval_sequences),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "vjepa": {
            "positions": pos_probe_target,
            "velocities": vel_probe_target,
            "positions_online": pos_probe_online,
            "velocities_online": vel_probe_online,
            "positions_ctx_window": pos_probe_ctx,
            "velocities_ctx_window": vel_probe_ctx,
        },
        "degradation": deg,
        "rollout_mae": roll,
    }
    if regime_metrics is not None:
        metrics["regime_held_out"] = regime_metrics

    # --- Phase F: pixel baseline (own encoder lifecycle) ---
    pixel_probe_results = None
    pixel_z = None
    if pixel_ckpt_path is not None:
        print(f"[eval] loading pixel baseline ckpt={pixel_ckpt_path}", flush=True)
        pixel_model = load_pixel(pixel_ckpt_path, cfg, device)

        print("[eval] encoding frames with pixel baseline encoder...", flush=True)
        pixel_z = encode_all_frames_pixel(pixel_model, store, device)
        pix_pos = linear_probe(pixel_z, store.positions, train_idx, test_idx)
        pix_vel = linear_probe(pixel_z, store.velocities, train_idx, test_idx)

        print("[eval] encoding 4-frame context windows (pixel baseline)...", flush=True)
        pixel_ctx_z = encode_all_context_windows_pixel(
            pixel_model, store, device, context_frames=context_frames
        )
        pix_pos_ctx = context_window_probe(
            pixel_ctx_z, store.positions, train_idx, test_idx, context_frames=context_frames
        )
        pix_vel_ctx = context_window_probe(
            pixel_ctx_z, store.velocities, train_idx, test_idx, context_frames=context_frames
        )

        metrics["pixel"] = {
            "positions": pix_pos,
            "velocities": pix_vel,
            "positions_ctx_window": pix_pos_ctx,
            "velocities_ctx_window": pix_vel_ctx,
            "ckpt": str(pixel_ckpt_path),
        }
        if regime is not None:
            r_train, r_test = regime
            metrics["pixel"]["regime_held_out"] = {
                "per_frame": {
                    "positions": linear_probe(pixel_z, store.positions, r_train, r_test),
                    "velocities": linear_probe(pixel_z, store.velocities, r_train, r_test),
                },
                "ctx_window": {
                    "positions": context_window_probe(
                        pixel_ctx_z, store.positions, r_train, r_test, context_frames=context_frames
                    ),
                    "velocities": context_window_probe(
                        pixel_ctx_z, store.velocities, r_train, r_test, context_frames=context_frames
                    ),
                },
            }
        # Free the bigger one immediately; we still need pixel_z for the
        # side-by-side PCA plot below.
        del pixel_ctx_z
        _free_gpu()

        pixel_probe_results = {
            "vjepa": {"positions": pos_probe_target, "velocities": vel_probe_target},
            "pixel": {"positions": pix_pos, "velocities": pix_vel},
        }
        print(
            f"[eval]   pixel/per-frame/positions R²={pix_pos['overall_r2']:.4f} "
            f"pixel/ctx-window/velocities R²={pix_vel_ctx['overall_r2']:.4f}",
            flush=True,
        )

    print("[eval] writing visualizations...", flush=True)
    plot_latent_pca_trajectory(target_z, store, out / "latent_pca_trajectory.png")
    plot_degradation_curve(deg, out / "degradation_curve.png")
    plot_rollout_mae(roll, out / "rollout_mae.png")
    plot_tsne_by_speed(target_z, store, out / "tsne_by_speed.png")
    plot_training_curves(find_train_metrics_csv(Path(ckpt_path)), out / "training_curves.png")
    if pixel_z is not None and pixel_probe_results is not None:
        plot_jepa_vs_pixel(
            target_z, pixel_z, store, pixel_probe_results, out / "jepa_vs_pixel.png"
        )
        plot_probe_grid(metrics, out / "probe_grid.png")

    metrics_path = out / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] wrote {metrics_path}", flush=True)
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, type=str, help="V-JEPA checkpoint .pt path")
    p.add_argument("--data", required=True, type=str, help="NPZ dataset path")
    p.add_argument("--config", required=True, type=str,
                   help="YAML config matching the checkpoint's model architecture")
    p.add_argument("--out", default="assets/results/", type=str, help="Output directory")
    p.add_argument("--pixel-ckpt", default=None, type=str,
                   help="Optional pixel-baseline checkpoint for side-by-side comparison")
    p.add_argument("--max-eval-sequences", default=800, type=int,
                   help="Stride-subsample the dataset to at most this many sequences "
                        "before encoding. Bounds peak RAM (default 800 -> ~9 GB "
                        "peak with the lsqr context-window solver). Pass a larger "
                        "value to opt into more data and tighter R² estimates.")
    p.add_argument("--ram-limit-gb", default=None, type=float,
                   help="Hard-cap this process's virtual address space (RLIMIT_AS). "
                        "If exceeded the script raises MemoryError instead of "
                        "exhausting system RAM. Defense-in-depth only — pair with "
                        "--max-eval-sequences if your workload genuinely needs less.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run_evaluation(
        ckpt_path=args.ckpt,
        data_path=args.data,
        config_path=args.config,
        out_dir=args.out,
        pixel_ckpt_path=args.pixel_ckpt,
        max_eval_sequences=args.max_eval_sequences,
        ram_limit_gb=args.ram_limit_gb,
    )
    print(json.dumps(_summary(metrics), indent=2), flush=True)
    return 0


def _summary(metrics: dict) -> dict:
    vj = metrics["vjepa"]
    s: dict = {
        # Per-frame, EMA target — back-compat with the V1 headline numbers.
        "positions_R2_target_v1": vj["positions"]["overall_r2"],
        "velocities_R2_target_v1": vj["velocities"]["overall_r2"],
        # Per-frame, online encoder — apples-to-apples vs pixel baseline.
        "positions_R2_online": vj.get("positions_online", {}).get("overall_r2"),
        "velocities_R2_online": vj.get("velocities_online", {}).get("overall_r2"),
        # 4-frame context window — the physically valid velocity probe.
        "positions_R2_ctx_window": vj.get("positions_ctx_window", {}).get("overall_r2"),
        "velocities_R2_ctx_window": vj.get("velocities_ctx_window", {}).get("overall_r2"),
        # Rollout MAE (V-JEPA + probe), with reference baselines.
        "rollout_horizons": metrics["rollout_mae"]["horizons"],
        "rollout_vjepa_mae": metrics["rollout_mae"]["vjepa_mae"],
        "rollout_copy_last_mae": metrics["rollout_mae"]["copy_last_mae"],
        "rollout_identity_mae": metrics["rollout_mae"]["identity_mae"],
    }
    if "pixel" in metrics:
        px = metrics["pixel"]
        s["pixel_positions_R2"] = px["positions"]["overall_r2"]
        s["pixel_velocities_R2"] = px["velocities"]["overall_r2"]
        s["pixel_positions_R2_ctx_window"] = px.get("positions_ctx_window", {}).get("overall_r2")
        s["pixel_velocities_R2_ctx_window"] = px.get("velocities_ctx_window", {}).get("overall_r2")
    if "regime_held_out" in metrics:
        ro = metrics["regime_held_out"]
        s["regime_held_out_test_strategy"] = ro["held_out"]
        s["regime_positions_R2_ctx_window"] = ro["ctx_window"]["positions"]["overall_r2"]
        s["regime_velocities_R2_ctx_window"] = ro["ctx_window"]["velocities"]["overall_r2"]
    return s


if __name__ == "__main__":
    raise SystemExit(main())
