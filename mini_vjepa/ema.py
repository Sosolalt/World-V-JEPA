import copy
import math

import torch
import torch.nn as nn


class EMAEncoder(nn.Module):
    def __init__(self, online_encoder: nn.Module):
        super().__init__()
        self.encoder = copy.deepcopy(online_encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False
        for b in self.encoder.buffers():
            b.requires_grad = False

    @torch.no_grad()
    def update(self, online_encoder: nn.Module, tau: float) -> None:
        for p_t, p_o in zip(self.encoder.parameters(), online_encoder.parameters()):
            p_t.data.mul_(tau).add_(p_o.data, alpha=1.0 - tau)
        for b_t, b_o in zip(self.encoder.buffers(), online_encoder.buffers()):
            if b_t.dtype.is_floating_point:
                b_t.data.mul_(tau).add_(b_o.data, alpha=1.0 - tau)
            else:
                b_t.data.copy_(b_o.data)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def tau_schedule(epoch: int, total_epochs: int, start: float = 0.996, end: float = 1.0) -> float:
    if total_epochs <= 1:
        return end
    progress = min(max(epoch / (total_epochs - 1), 0.0), 1.0)
    return end - (end - start) * 0.5 * (1.0 + math.cos(math.pi * progress))
