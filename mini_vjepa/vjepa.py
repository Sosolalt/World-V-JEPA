from typing import Any, Mapping

import torch
import torch.nn as nn

from .encoder import ContextEncoder
from .ema import EMAEncoder
from .losses import (
    covariance_regularization,
    l1_loss,
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

    def encode_context(
        self,
        frames_4: torch.Tensor,
        pixel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode context frames; optionally apply pre-encoder pixel-patch mask.

        When `pixel_mask` (B, T, H, W bool, True = masked) is provided:
          1. masked pixels are zeroed in the input
          2. the encoder runs on the corrupted input
          3. tokens at fully-masked patch positions are replaced with the
             learned `mask_token` (raw, no temporal_pos) BEFORE temporal_pos
             is added (Phase 2 F8 resolution: masked positions end up as
             `mask_token + temporal_pos[t]`).
        """
        b, t, c, h, w = frames_4.shape
        if t != self.context_frames:
            raise ValueError(
                f"expected {self.context_frames} context frames, got {t}"
            )
        if pixel_mask is not None:
            masked_frames = frames_4 * (~pixel_mask).unsqueeze(2).to(frames_4.dtype)
        else:
            masked_frames = frames_4
        flat = masked_frames.reshape(b * t, c, h, w)
        tokens = self.encoder(flat)
        tokens = tokens.reshape(b, t, self.spatial_tokens, self.latent_dim)
        if pixel_mask is not None:
            from .masking import pixel_mask_to_token_mask
            tm = pixel_mask_to_token_mask(pixel_mask, self.spatial_tokens)
            tm_4d = tm.reshape(b, t, self.spatial_tokens, 1)
            tokens = torch.where(tm_4d, self.mask_token.view(1, 1, 1, -1), tokens)
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
        context_token_mask: torch.Tensor | None = None,
    ) -> dict:
        # `context_token_mask` is the (B, n_context_tokens) bool mask where
        # True = position was substituted with `mask_token`. When provided,
        # variance/covariance regularization on `context_tokens` excludes
        # these positions (they are near-constant by construction, would
        # collapse var/cov on a healthy encoder; see Phase-2 reviewer C1+C3).
        z_target = z_target.detach()
        if loss_fn == "l1":
            sim = l1_loss(z_pred, z_target)
        elif loss_fn == "smooth_l1":
            sim = smooth_l1_loss(z_pred, z_target)
        else:
            sim = mse_loss(z_pred, z_target)

        def _ctx_unmasked(t: torch.Tensor) -> torch.Tensor:
            if context_token_mask is None:
                return t
            keep = ~context_token_mask
            if not keep.any():
                return t
            return t[keep]

        if reg_target == "z_pred":
            var_reg = variance_regularization(z_pred)
            cov_reg = covariance_regularization(z_pred)
        elif reg_target == "context":
            if context_tokens is None:
                raise ValueError("reg_target='context' requires context_tokens")
            ctx_um = _ctx_unmasked(context_tokens)
            var_reg = variance_regularization(ctx_um)
            cov_reg = covariance_regularization(ctx_um)
        elif reg_target == "both":
            if context_tokens is None:
                raise ValueError("reg_target='both' requires context_tokens")
            ctx_um = _ctx_unmasked(context_tokens)
            var_reg = 0.5 * (
                variance_regularization(z_pred)
                + variance_regularization(ctx_um)
            )
            cov_reg = 0.5 * (
                covariance_regularization(z_pred)
                + covariance_regularization(ctx_um)
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
