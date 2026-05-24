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
from sklearn.linear_model import Ridge  # noqa: E402
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


def linear_probe(
    z_seq: np.ndarray, targets: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray,
    alpha: float = 1.0,
    max_train_sequences: int = 2000,
) -> dict:
    """Fit Ridge from latents to targets; return overall + per-ball R^2 and MAE.

    On the 10000-sequence default dataset the flatten step would allocate
    ~8 GB per side; combined with target_z + pixel_z already in memory, this
    OOMs the 32GB-Mac eval. Cap train_idx at max_train_sequences so the probe
    runs in ~2 GB. Test set is always used in full (only ~2k sequences).
    """
    if len(train_idx) > max_train_sequences:
        # deterministic stride subsample preserves sequence diversity
        stride = max(1, len(train_idx) // max_train_sequences)
        train_idx = train_idx[::stride][:max_train_sequences]
    x_tr, y_tr = flatten_for_probe(z_seq, targets, train_idx)
    x_te, y_te = flatten_for_probe(z_seq, targets, test_idx)
    model = Ridge(alpha=alpha)
    model.fit(x_tr, y_tr)
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
    }


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
) -> dict:
    """For each horizon, average cosine sim of predicted vs target latent on test sequences.

    Three predictors:
      - V-JEPA: model.predict(context_tokens, dt)
      - copy-last: encoded last context frame (target encoder), tiled
      - random: gaussian noise matching the target shape
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

    results = {"horizons": list(horizons), "vjepa": [], "copy_last": [], "random": []}

    rng = np.random.default_rng(SEED)

    for h in horizons:
        max_t_start = seq_len - context_frames - h
        if max_t_start < 0:
            results["vjepa"].append(float("nan"))
            results["copy_last"].append(float("nan"))
            results["random"].append(float("nan"))
            continue

        vjepa_sims: list[float] = []
        copy_sims: list[float] = []
        rand_sims: list[float] = []

        for start in range(0, n_te, batch_size):
            chunk = test_frames[start : start + batch_size]  # (b, T, 3, H, W)
            b = chunk.shape[0]
            x = torch.from_numpy(chunk).to(device).float() / 255.0

            # Iterate possible t_starts but to keep cost low use one fixed t_start per horizon.
            t_start = max_t_start // 2
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

            sim_vj = cosine_sim_batched(z_pred, z_target).item()
            sim_cp = cosine_sim_batched(z_copy, z_target).item()
            z_rand = torch.from_numpy(
                rng.standard_normal((b, n_tokens, latent_dim)).astype(np.float32)
            ).to(device)
            sim_rd = cosine_sim_batched(z_rand, z_target).item()

            vjepa_sims.append(sim_vj * b)
            copy_sims.append(sim_cp * b)
            rand_sims.append(sim_rd * b)

        denom = float(n_te)
        results["vjepa"].append(float(sum(vjepa_sims) / denom))
        results["copy_last"].append(float(sum(copy_sims) / denom))
        results["random"].append(float(sum(rand_sims) / denom))

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
    ax.plot(h, curve["vjepa"], "o-", label="V-JEPA")
    ax.plot(h, curve["copy_last"], "s--", label="copy-last-frame")
    ax.plot(h, curve["random"], "x:", label="random")
    ax.set_xlabel("prediction horizon Δt")
    ax.set_ylabel("cosine similarity (z_pred, z_target)")
    ax.set_title("Latent prediction degradation vs horizon")
    ax.grid(True, alpha=0.3)
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


def run_evaluation(
    ckpt_path: str,
    data_path: str,
    config_path: str,
    out_dir: str,
    pixel_ckpt_path: str | None = None,
    horizons: list[int] | None = None,
) -> dict:
    set_seed(SEED)
    device = select_device()
    cfg = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if horizons is None:
        horizons = [1, 2, 4, 6, 8, 10, 12]

    print(f"[eval] device={device} ckpt={ckpt_path} data={data_path}", flush=True)
    dataset = BilliardDataset(data_path)
    store = load_frames(dataset)
    n_seq = len(dataset)
    train_idx, test_idx = split_indices(n_seq, train_frac=0.8)
    print(f"[eval] sequences={n_seq} train={len(train_idx)} test={len(test_idx)}", flush=True)

    model = load_vjepa(ckpt_path, cfg, device)

    print("[eval] encoding all frames with target encoder...", flush=True)
    target_z = encode_all_frames_target(model, store, device)
    print(f"[eval] target_z shape={target_z.shape}", flush=True)

    print("[eval] linear probe (positions)...", flush=True)
    pos_probe = linear_probe(target_z, store.positions, train_idx, test_idx)
    print(f"[eval]   positions R²={pos_probe['overall_r2']:.4f} mae={pos_probe['mae']:.4f}", flush=True)

    print("[eval] linear probe (velocities)...", flush=True)
    vel_probe = linear_probe(target_z, store.velocities, train_idx, test_idx)
    print(f"[eval]   velocities R²={vel_probe['overall_r2']:.4f} mae={vel_probe['mae']:.4f}", flush=True)

    print("[eval] degradation curve...", flush=True)
    deg = degradation_curve(model, store, test_idx, target_z, device, horizons, cfg)

    metrics: dict = {
        "ckpt": str(ckpt_path),
        "data": str(data_path),
        "config": str(config_path),
        "seed": SEED,
        "n_sequences": int(n_seq),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "vjepa": {
            "positions": pos_probe,
            "velocities": vel_probe,
        },
        "degradation": deg,
    }

    pixel_probe_results = None
    pixel_z = None
    if pixel_ckpt_path is not None:
        print(f"[eval] loading pixel baseline ckpt={pixel_ckpt_path}", flush=True)
        pixel_model = load_pixel(pixel_ckpt_path, cfg, device)
        print("[eval] encoding all frames with pixel baseline encoder...", flush=True)
        pixel_z = encode_all_frames_pixel(pixel_model, store, device)
        print(f"[eval] pixel_z shape={pixel_z.shape}", flush=True)
        pix_pos = linear_probe(pixel_z, store.positions, train_idx, test_idx)
        pix_vel = linear_probe(pixel_z, store.velocities, train_idx, test_idx)
        metrics["pixel"] = {
            "positions": pix_pos,
            "velocities": pix_vel,
            "ckpt": str(pixel_ckpt_path),
        }
        pixel_probe_results = {
            "vjepa": {"positions": pos_probe, "velocities": vel_probe},
            "pixel": {"positions": pix_pos, "velocities": pix_vel},
        }
        print(
            f"[eval]   pixel positions R²={pix_pos['overall_r2']:.4f} "
            f"velocities R²={pix_vel['overall_r2']:.4f}",
            flush=True,
        )

    print("[eval] writing visualizations...", flush=True)
    plot_latent_pca_trajectory(target_z, store, out / "latent_pca_trajectory.png")
    plot_degradation_curve(deg, out / "degradation_curve.png")
    plot_tsne_by_speed(target_z, store, out / "tsne_by_speed.png")
    plot_training_curves(find_train_metrics_csv(Path(ckpt_path)), out / "training_curves.png")
    if pixel_z is not None and pixel_probe_results is not None:
        plot_jepa_vs_pixel(
            target_z, pixel_z, store, pixel_probe_results, out / "jepa_vs_pixel.png"
        )

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
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics = run_evaluation(
        ckpt_path=args.ckpt,
        data_path=args.data,
        config_path=args.config,
        out_dir=args.out,
        pixel_ckpt_path=args.pixel_ckpt,
    )
    print(json.dumps(_summary(metrics), indent=2), flush=True)
    return 0


def _summary(metrics: dict) -> dict:
    s: dict = {
        "positions_R2": metrics["vjepa"]["positions"]["overall_r2"],
        "velocities_R2": metrics["vjepa"]["velocities"]["overall_r2"],
        "degradation_horizons": metrics["degradation"]["horizons"],
        "degradation_vjepa": metrics["degradation"]["vjepa"],
    }
    if "pixel" in metrics:
        s["pixel_positions_R2"] = metrics["pixel"]["positions"]["overall_r2"]
        s["pixel_velocities_R2"] = metrics["pixel"]["velocities"]["overall_r2"]
    return s


if __name__ == "__main__":
    raise SystemExit(main())
