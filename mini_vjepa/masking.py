"""Pixel-patch tubelet masking for canonical V-JEPA training.

Phase 2 (canonical V-JEPA pixel-patch tubelet masking + L1) replaces the
prior post-encoder token-replacement masking. The encoder now sees an
explicitly-corrupted view of the context (zeroed pixels in tubelets), and
the predictor receives encoded tokens with masked-position tokens replaced
by a learned `mask_token` before the temporal positional embedding is added.

The sampler implements V-JEPA paper-style multi-block masking:
  - 4 "short" blocks @ 15% spatial coverage each
  - 1 "long" block @ 50% spatial coverage
  - aspect ratio of each block ~ U(0.75, 1.5)
  - tubelet thickness = 2 frames (block applied to a contiguous 2-frame slab)
  - all block coordinates 8-pixel-grid-aligned so token correspondence is
    unambiguous (every 8x8 patch is either fully masked or fully visible).
"""
from __future__ import annotations

import math
import torch


_TOKEN = 8  # pixel size of one spatial token (8x downsampling)


def _sample_block_dims(coverage: float, H: int, W: int, rng: torch.Generator) -> tuple[int, int]:
    """Sample (h_pix, w_pix) for a single block at the requested fraction.

    Sizes are rounded to multiples of _TOKEN so block edges align to the
    8x8 token grid. Aspect ratio ~ U(0.75, 1.5).
    """
    aspect = 0.75 + 0.75 * torch.rand((), generator=rng).item()
    target_area = coverage * H * W
    h_f = math.sqrt(target_area / aspect)
    w_f = h_f * aspect
    h_pix = max(_TOKEN, int(round(h_f / _TOKEN)) * _TOKEN)
    w_pix = max(_TOKEN, int(round(w_f / _TOKEN)) * _TOKEN)
    h_pix = min(h_pix, H)
    w_pix = min(w_pix, W)
    return h_pix, w_pix


def _sample_block_origin(h_pix: int, w_pix: int, H: int, W: int, rng: torch.Generator) -> tuple[int, int]:
    """Pick top-left (y, x) snapped to the 8-pixel grid."""
    max_y = (H - h_pix) // _TOKEN
    max_x = (W - w_pix) // _TOKEN
    y_tok = torch.randint(0, max_y + 1, (), generator=rng).item() if max_y > 0 else 0
    x_tok = torch.randint(0, max_x + 1, (), generator=rng).item() if max_x > 0 else 0
    return y_tok * _TOKEN, x_tok * _TOKEN


def sample_pixel_mask(
    batch_size: int,
    T: int,
    H: int,
    W: int,
    n_short_blocks: int = 4,
    short_coverage: float = 0.15,
    n_long_blocks: int = 1,
    long_coverage: float = 0.50,
    tubelet_t: int = 2,
    device: torch.device | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample a (B, T, H, W) boolean mask. True = masked (pixel will be zeroed).

    Each block is placed in a contiguous tubelet of `tubelet_t` frames; the
    tubelet position is sampled per block. Blocks are independently sampled
    per batch element. Returns a CPU tensor; caller is responsible for moving
    it to device (cheap relative to a training step).
    """
    if T % tubelet_t != 0:
        raise ValueError(f"T={T} must be divisible by tubelet_t={tubelet_t}")
    if H % _TOKEN != 0 or W % _TOKEN != 0:
        raise ValueError(f"H,W must be multiples of {_TOKEN}; got {H}x{W}")

    rng = generator if generator is not None else torch.Generator()
    n_tubelets = T // tubelet_t
    mask = torch.zeros(batch_size, T, H, W, dtype=torch.bool)

    block_specs = [(short_coverage, _) for _ in range(n_short_blocks)] + [
        (long_coverage, _) for _ in range(n_long_blocks)
    ]
    for b in range(batch_size):
        for coverage, _ in block_specs:
            h_pix, w_pix = _sample_block_dims(coverage, H, W, rng)
            y, x = _sample_block_origin(h_pix, w_pix, H, W, rng)
            tub_idx = torch.randint(0, n_tubelets, (), generator=rng).item()
            t_start = tub_idx * tubelet_t
            mask[b, t_start : t_start + tubelet_t, y : y + h_pix, x : x + w_pix] = True

    if device is not None:
        mask = mask.to(device)
    return mask


def pixel_mask_to_token_mask(pixel_mask: torch.Tensor, spatial_tokens: int = 64) -> torch.Tensor:
    """Convert (B, T, H, W) pixel mask to (B, T*spatial_tokens) token mask.

    Because blocks are 8-pixel-grid-aligned, every 8x8 pixel patch is either
    fully masked or fully visible — we just sample the top-left pixel of each
    patch. Token at flat index `t*spatial_tokens + r*8 + c` corresponds to
    pixel patch `[r*8:(r+1)*8, c*8:(c+1)*8]` of frame t.
    """
    B, T, H, W = pixel_mask.shape
    h_tok = H // _TOKEN
    w_tok = W // _TOKEN
    if h_tok * w_tok != spatial_tokens:
        raise ValueError(
            f"spatial_tokens={spatial_tokens} does not match {h_tok}x{w_tok}=h_tok*w_tok"
        )
    sampled = pixel_mask[:, :, ::_TOKEN, ::_TOKEN]  # (B, T, h_tok, w_tok)
    return sampled.reshape(B, T * spatial_tokens)
