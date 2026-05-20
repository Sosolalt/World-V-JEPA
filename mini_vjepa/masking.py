import torch


def mask_tokens(x: torch.Tensor, mask_ratio: float):
    if x.dim() != 3:
        raise ValueError(f"expected [B, N, D] tensor, got shape {tuple(x.shape)}")
    b, n, _ = x.shape
    if not 0.0 <= mask_ratio < 1.0:
        raise ValueError(f"mask_ratio must be in [0, 1), got {mask_ratio}")

    n_mask = int(round(mask_ratio * n))
    if n_mask == 0:
        empty = torch.empty(b, 0, dtype=torch.long, device=x.device)
        return x, empty

    n_keep = n - n_mask
    noise = torch.rand(b, n, device=x.device)
    shuffle = torch.argsort(noise, dim=1)
    keep_idx = shuffle[:, :n_keep]
    mask_idx = shuffle[:, n_keep:]

    keep_idx_sorted, _ = torch.sort(keep_idx, dim=1)
    mask_idx_sorted, _ = torch.sort(mask_idx, dim=1)

    gather_idx = keep_idx_sorted.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    masked_x = torch.gather(x, dim=1, index=gather_idx)
    return masked_x, mask_idx_sorted
