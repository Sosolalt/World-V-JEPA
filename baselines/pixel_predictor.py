"""Pixel-prediction baseline for V-JEPA comparison (PLAN §5).

Reuses the V-JEPA ContextEncoder + Predictor verbatim so parameter budgets on
the shared parts match. The only added capacity is a transposed-conv decoder
that maps predicted tokens back to pixels. No EMA, no variance regularization;
loss is pixel MSE.
"""
from __future__ import annotations

import csv
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Must be set before torch is imported so the MPS fallback path is active for
# any caller that imports this module (e.g. a notebook) without going through
# scripts/train.py.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from mini_vjepa.dataset import BilliardDataset
from mini_vjepa.encoder import ContextEncoder
from mini_vjepa.predictor import Predictor


class PixelDecoder(nn.Module):
    """Mirror of the encoder: 64 tokens of dim 128 → 3x64x64 image in [0, 1]."""

    def __init__(self, latent_dim: int = 128, spatial_tokens: int = 64):
        super().__init__()
        assert spatial_tokens == 64, "decoder is fixed to 8x8 spatial tokens"
        self.latent_dim = latent_dim
        self.spatial_tokens = spatial_tokens

        self.deconv1 = nn.ConvTranspose2d(latent_dim, 256, 3, stride=2, padding=1, output_padding=1)
        self.gn1 = nn.GroupNorm(32, 256)
        self.deconv2 = nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
        self.gn2 = nn.GroupNorm(32, 128)
        self.deconv3 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
        self.gn3 = nn.GroupNorm(32, 64)
        self.to_rgb = nn.Conv2d(64, 3, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        if n != self.spatial_tokens or d != self.latent_dim:
            raise ValueError(
                f"expected tokens of shape (B, {self.spatial_tokens}, {self.latent_dim}), got {tokens.shape}"
            )
        h = tokens.transpose(1, 2).reshape(b, d, 8, 8)
        h = F.gelu(self.gn1(self.deconv1(h)))
        h = F.gelu(self.gn2(self.deconv2(h)))
        h = F.gelu(self.gn3(self.deconv3(h)))
        h = self.to_rgb(h)
        # Clamp rather than sigmoid: sigmoid saturates for logits far from 0,
        # producing near-zero gradient there and a huge gradient at the
        # unsaturated edge, which empirically NaN'd the pixel decoder under
        # MSE. Clamp gives the same [0, 1] range with bounded gradients.
        return torch.clamp(h, 0.0, 1.0)


class PixelPredictor(nn.Module):
    """Pixel-space variant of V-JEPA: same encoder+predictor, plus a pixel decoder."""

    def __init__(self, model_cfg: Mapping[str, Any]):
        super().__init__()
        self.latent_dim = int(model_cfg["latent_dim"])
        self.spatial_tokens = int(model_cfg["spatial_tokens"])
        self.context_frames = int(model_cfg["context_frames"])
        self.delta_t_max = int(model_cfg["delta_t_max"])

        n_context_tokens = self.context_frames * self.spatial_tokens

        self.encoder = ContextEncoder(
            latent_dim=self.latent_dim,
            spatial_tokens=self.spatial_tokens,
        )
        self.predictor = Predictor(
            dim=self.latent_dim,
            depth=int(model_cfg["predictor_layers"]),
            num_heads=int(model_cfg["predictor_heads"]),
            ffn_dim=int(model_cfg["predictor_ffn"]),
            n_context_tokens=n_context_tokens,
            n_output_tokens=self.spatial_tokens,
            delta_t_max=self.delta_t_max,
        )
        self.decoder = PixelDecoder(
            latent_dim=self.latent_dim,
            spatial_tokens=self.spatial_tokens,
        )

        self.temporal_pos = nn.Parameter(
            torch.zeros(self.context_frames, 1, self.latent_dim)
        )
        nn.init.normal_(self.temporal_pos, std=0.02)

    def encode_context(self, frames_4: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = frames_4.shape
        if t != self.context_frames:
            raise ValueError(
                f"expected {self.context_frames} context frames, got {t}"
            )
        flat = frames_4.reshape(b * t, c, h, w)
        tokens = self.encoder(flat)
        tokens = tokens.reshape(b, t, self.spatial_tokens, self.latent_dim)
        tokens = tokens + self.temporal_pos.unsqueeze(0)
        return tokens.reshape(b, t * self.spatial_tokens, self.latent_dim)

    def forward(self, context_frames: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        context_tokens = self.encode_context(context_frames)
        pred_tokens = self.predictor(context_tokens, delta_t)
        return self.decoder(pred_tokens)


def _count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def split_param_groups(model: PixelPredictor, wd_encoder: float, wd_predictor: float) -> list[dict]:
    encoder_params = list(model.encoder.parameters())
    predictor_params = list(model.predictor.parameters()) + [model.temporal_pos]
    decoder_params = list(model.decoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    predictor_ids = {id(p) for p in predictor_params}
    decoder_ids = {id(p) for p in decoder_params}
    trainable = [p for p in model.parameters() if p.requires_grad]
    covered = encoder_ids | predictor_ids | decoder_ids
    for p in trainable:
        if id(p) not in covered:
            raise RuntimeError("trainable parameter not assigned to a group")
    return [
        {"params": [p for p in encoder_params if p.requires_grad], "weight_decay": wd_encoder},
        {"params": [p for p in predictor_params if p.requires_grad], "weight_decay": wd_predictor},
        {"params": [p for p in decoder_params if p.requires_grad], "weight_decay": wd_predictor},
    ]


def make_lr_lambda(warmup_steps: int, total_steps: int, lr_peak: float, lr_final: float):
    final_scale = lr_final / lr_peak if lr_peak > 0 else 0.0

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        decay_steps = max(1, total_steps - warmup_steps)
        progress = (step - warmup_steps) / decay_steps
        progress = min(max(progress, 0.0), 1.0)
        cos = 0.5 * (1.0 + math.cos(math.pi * progress))
        return final_scale + (1.0 - final_scale) * cos

    return lr_lambda


def _set_seed(seed: int) -> None:
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _baseline_value(cfg: Mapping[str, Any], key: str, default):
    # Per-config baseline overrides: optional `baseline_overrides` block can
    # shadow any training key (commonly lr_peak + grad_clip, which the pixel
    # decoder is more sensitive to than the V-JEPA backbone).
    overrides = cfg.get("baseline_overrides") or {}
    if key in overrides:
        return overrides[key]
    return cfg["training"].get(key, default)


def train_pixel_baseline(cfg: Mapping[str, Any], data_path: str, epochs: int, repo_root: Path) -> int:
    _set_seed(int(cfg["training"]["seed"]))
    device = _select_device()
    print(f"[baseline-pixel] device={device}, data={data_path}, epochs={epochs}", flush=True)

    dataset = BilliardDataset(data_path)
    if len(dataset) == 0:
        raise RuntimeError(f"dataset is empty: {data_path}")
    batch_size = int(cfg["training"]["batch_size"])
    effective_bs = min(batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_bs,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = PixelPredictor(cfg["model"]).to(device)
    enc_params = _count(model.encoder)
    pred_params = _count(model.predictor) + model.temporal_pos.numel()
    dec_params = _count(model.decoder)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[baseline-pixel] params encoder={enc_params:,} predictor={pred_params:,} "
        f"decoder={dec_params:,} total={total_trainable:,}",
        flush=True,
    )

    param_groups = split_param_groups(
        model,
        wd_encoder=float(cfg["training"]["weight_decay_encoder"]),
        wd_predictor=float(cfg["training"]["weight_decay_predictor"]),
    )
    lr_peak = float(_baseline_value(cfg, "lr_peak", cfg["training"]["lr_peak"]))
    lr_final = float(_baseline_value(cfg, "lr_final", cfg["training"]["lr_final"]))
    optimizer = AdamW(param_groups, lr=lr_peak)
    print(f"[baseline-pixel] lr_peak={lr_peak:.2e} lr_final={lr_final:.2e}", flush=True)

    steps_per_epoch = max(1, math.ceil(len(dataset) / effective_bs))
    warmup_epochs = int(cfg["training"]["warmup_epochs"])
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = epochs * steps_per_epoch
    scheduler = LambdaLR(optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps, lr_peak, lr_final))

    sequence_length = int(cfg["data"]["sequence_length"])
    context_frames = int(cfg["model"]["context_frames"])
    delta_t_min = int(cfg["model"]["delta_t_min"])
    delta_t_max = int(cfg["model"]["delta_t_max"])
    grad_clip = float(_baseline_value(cfg, "grad_clip", cfg["training"]["grad_clip"]))
    grad_value_clip = float(_baseline_value(cfg, "grad_value_clip", 0.0))  # 0 disables
    checkpoint_every = int(cfg["training"]["checkpoint_every"])
    max_checkpoints = int(cfg["training"]["max_checkpoints"])

    max_t_start = sequence_length - context_frames - delta_t_max
    if max_t_start <= 0:
        raise ValueError(
            f"sequence_length={sequence_length} too short for context_frames={context_frames} + delta_t_max={delta_t_max}"
        )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = repo_root / "runs" / "baseline_pixel" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    metrics_columns = ["epoch", "lr", "loss", "pixel_mse", "time_s"]
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(metrics_columns)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    saved_ckpts: list[Path] = []
    metrics_history: list[dict] = []
    best_mse = float("inf")
    best_ckpt_path: Path | None = None

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()

        running_loss = 0.0
        running_mse = 0.0
        n_batches = 0

        for batch in loader:
            frames_u8 = batch["frames"].to(device, non_blocking=False)
            frames = frames_u8.float() / 255.0
            b = frames.shape[0]

            t_start = random.randint(0, max_t_start)
            delta_t = random.randint(delta_t_min, delta_t_max)

            context = frames[:, t_start: t_start + context_frames].contiguous()
            target_frame = frames[:, t_start + context_frames - 1 + delta_t].contiguous()
            dt_tensor = torch.full((b,), delta_t, device=device, dtype=torch.long)

            pred_pixels = model(context, dt_tensor)
            loss = F.mse_loss(pred_pixels, target_frame)

            # Numerical guard: a single NaN/Inf loss historically poisoned
            # every subsequent epoch (110 wasted epochs on the previous run).
            # Skip the bad batch and abort the run if it persists.
            if not torch.isfinite(loss):
                print(
                    f"[baseline-pixel] non-finite loss at epoch {epoch} batch {n_batches}; aborting run",
                    file=sys.stderr, flush=True,
                )
                return 2

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            clip_grad_norm_(trainable_params, grad_clip)
            if grad_value_clip > 0:
                clip_grad_value_(trainable_params, grad_value_clip)
            optimizer.step()
            scheduler.step()

            running_loss += float(loss.detach().item())
            running_mse += float(loss.detach().item())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        avg_mse = running_mse / max(1, n_batches)

        if device.type == "mps":
            torch.mps.synchronize()
        elapsed = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "loss": avg_loss,
            "pixel_mse": avg_mse,
            "time_s": elapsed,
        }
        metrics_history.append(row)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([row[k] for k in metrics_columns])
        print(
            f"[epoch {epoch:03d}] loss={avg_loss:.6f} pixel_mse={avg_mse:.6f} "
            f"lr={current_lr:.2e} t={elapsed:.1f}s",
            flush=True,
        )
        avg_epoch_s = sum(r["time_s"] for r in metrics_history) / len(metrics_history)
        remaining = (epochs - epoch - 1) * avg_epoch_s
        print(
            f"[ETA] avg={avg_epoch_s:.1f}s/epoch, {epochs - epoch - 1} epochs left, "
            f"~{remaining/60:.1f} min remaining (~{remaining/3600:.2f} h)",
            flush=True,
        )

        is_periodic = ((epoch + 1) % checkpoint_every == 0) or (epoch + 1 == epochs)
        if is_periodic:
            ckpt_path = run_dir / f"ckpt_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "metrics_history": metrics_history,
            }, ckpt_path)
            saved_ckpts.append(ckpt_path)
            while len(saved_ckpts) > max_checkpoints:
                oldest = saved_ckpts.pop(0)
                if oldest != best_ckpt_path and oldest.exists():
                    oldest.unlink()

        if avg_mse < best_mse:
            best_mse = avg_mse
            new_best = run_dir / "ckpt_best.pt"
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "metrics_history": metrics_history,
                "best_pixel_mse": best_mse,
            }, new_best)
            best_ckpt_path = new_best

        if device.type == "mps" and (epoch + 1) % 20 == 0:
            torch.mps.empty_cache()

    print(f"[baseline-pixel] done. run_dir={run_dir}", flush=True)
    return 0
