from typing import Any, Mapping

import torch
import torch.nn as nn

from .encoder import ContextEncoder
from .ema import EMAEncoder
from .losses import (
    covariance_regularization,
    mse_loss,
    smooth_l1_loss,
    variance_regularization,
)
from .predictor import Predictor


class VJEPA(nn.Module):
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
            use_spatial_pos_embed=bool(model_cfg.get("encoder_spatial_pos_embed", False)),
        )
        self.target_encoder = EMAEncoder(self.encoder)
        self.predictor = Predictor(
            dim=self.latent_dim,
            depth=int(model_cfg["predictor_layers"]),
            num_heads=int(model_cfg["predictor_heads"]),
            ffn_dim=int(model_cfg["predictor_ffn"]),
            n_context_tokens=n_context_tokens,
            n_output_tokens=self.spatial_tokens,
            delta_t_max=self.delta_t_max,
        )

        self.temporal_pos = nn.Parameter(
            torch.zeros(self.context_frames, 1, self.latent_dim)
        )
        nn.init.normal_(self.temporal_pos, std=0.02)

        # Mask token used when training.mask_ratio > 0. Random context positions
        # are replaced with this learned token before the predictor sees them,
        # creating the online/target asymmetry that V-JEPA relies on.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.latent_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def apply_context_mask(self, context_tokens: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        if mask_ratio <= 0.0:
            return context_tokens
        b, n, d = context_tokens.shape
        n_mask = int(round(mask_ratio * n))
        if n_mask <= 0:
            return context_tokens
        noise = torch.rand(b, n, device=context_tokens.device)
        mask_idx = noise.argsort(dim=-1)[:, :n_mask]
        mask_tok = self.mask_token.expand(b, n_mask, d)
        out = context_tokens.clone()
        out.scatter_(1, mask_idx.unsqueeze(-1).expand(-1, -1, d), mask_tok)
        return out

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

    def predict(self, context_tokens: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        return self.predictor(context_tokens, delta_t)

    @torch.no_grad()
    def target_encode(self, frame: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(frame).detach()

    @torch.no_grad()
    def ema_update(self, tau: float) -> None:
        self.target_encoder.update(self.encoder, tau)

    def compute_loss(
        self,
        z_pred: torch.Tensor,
        z_target: torch.Tensor,
        context_tokens: torch.Tensor | None = None,
        var_weight: float = 1.0,
        cov_weight: float = 1.0,
        reg_target: str = "z_pred",
        loss_fn: str = "mse",
    ) -> dict:
        # The three monitors (avg_std, effective_rank, avg_cosine_sim) are
        # intentionally not computed here: they're expensive (effective_rank
        # falls back to CPU on MPS) and train.py recomputes them once per
        # epoch on the last batch's z_pred, not once per batch.
        z_target = z_target.detach()
        if loss_fn == "smooth_l1":
            sim = smooth_l1_loss(z_pred, z_target)
        else:
            sim = mse_loss(z_pred, z_target)

        # reg_target controls *where* var/cov regularization is applied:
        #   "z_pred"  → predictor output (original behavior; the predictor can
        #               satisfy var/cov cheaply via its own weights while the
        #               encoder still collapses).
        #   "context" → online encoder output (forces the encoder itself to
        #               stay diverse; this is the actual collapse-preventing
        #               signal at small data scale).
        #   "both"    → 0.5*z_pred + 0.5*context.
        if reg_target == "z_pred":
            var_reg = variance_regularization(z_pred)
            cov_reg = covariance_regularization(z_pred)
        elif reg_target == "context":
            if context_tokens is None:
                raise ValueError("reg_target='context' requires context_tokens")
            var_reg = variance_regularization(context_tokens)
            cov_reg = covariance_regularization(context_tokens)
        elif reg_target == "both":
            if context_tokens is None:
                raise ValueError("reg_target='both' requires context_tokens")
            var_reg = 0.5 * (
                variance_regularization(z_pred)
                + variance_regularization(context_tokens)
            )
            cov_reg = 0.5 * (
                covariance_regularization(z_pred)
                + covariance_regularization(context_tokens)
            )
        else:
            raise ValueError(f"unknown reg_target: {reg_target!r}")

        loss = sim + var_weight * var_reg + cov_weight * cov_reg
        return {
            "loss": loss,
            "mse": sim.detach(),
            "var_reg": var_reg.detach(),
            "cov_reg": cov_reg.detach(),
        }


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}
