"""PyTorch Dataset that loads a generated billiard NPZ into RAM."""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class BilliardDataset(Dataset):
    """Loads an NPZ produced by simulation.generator.generate_dataset.

    Frames are kept as uint8 in RAM and returned channel-first (T, 3, H, W) as
    a torch.uint8 tensor. The training code is responsible for casting to float
    and dividing by 255.
    """

    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=False)
        frames = data["frames"]
        positions = data["positions"]
        velocities = data["velocities"]

        if frames.dtype != np.uint8:
            raise TypeError(f"frames must be uint8, got {frames.dtype}")
        if frames.ndim != 5 or frames.shape[-1] != 3:
            raise ValueError(f"frames must be (N, T, H, W, 3), got {frames.shape}")
        if positions.ndim != 4 or positions.shape[-1] != 2:
            raise ValueError(f"positions must be (N, T, n_balls, 2), got {positions.shape}")
        if velocities.shape != positions.shape:
            raise ValueError("velocities shape must match positions shape")
        if positions.shape[0] != frames.shape[0] or positions.shape[1] != frames.shape[1]:
            raise ValueError("frames/positions N or T mismatch")

        self.frames = frames
        self.positions = positions.astype(np.float32, copy=False)
        self.velocities = velocities.astype(np.float32, copy=False)
        # Per-sequence init strategy label (string), or None if absent (older NPZs).
        # Used by the held-out regime probe in scripts/evaluate.py.
        self.strategies = np.asarray(data["strategies"]) if "strategies" in data.files else None
        self.npz_path = npz_path

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        f = self.frames[idx]
        f = np.ascontiguousarray(np.transpose(f, (0, 3, 1, 2)))
        return {
            "frames": torch.from_numpy(f),
            "positions": torch.from_numpy(self.positions[idx]),
            "velocities": torch.from_numpy(self.velocities[idx]),
        }
