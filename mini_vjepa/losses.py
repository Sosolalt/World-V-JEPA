import torch
import torch.nn.functional as F


def mse_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(z_pred, z_target)


def smooth_l1_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(z_pred, z_target)


def l1_loss(z_pred: torch.Tensor, z_target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(z_pred, z_target)


def variance_regularization(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    std = torch.sqrt(flat.var(dim=0, unbiased=False) + eps)
    return F.relu(1.0 - std).mean()


def covariance_regularization(z: torch.Tensor) -> torch.Tensor:
    # VICReg covariance term: penalize off-diagonal entries of the feature
    # covariance so dimensions become decorrelated. Without this, variance
    # regularization keeps per-dim std up but lets all dims encode the same
    # signal (dimensional collapse: effective_rank -> 1).
    flat = z.reshape(-1, z.shape[-1])
    n, d = flat.shape
    if n <= 1:
        return torch.zeros((), device=z.device)
    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = (centered.t() @ centered) / (n - 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    return (off_diag.pow(2).sum()) / d


@torch.no_grad()
def avg_std(z: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    return flat.std(dim=0).mean()


@torch.no_grad()
def effective_rank(z: torch.Tensor) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    flat = flat - flat.mean(dim=0, keepdim=True)
    n = flat.shape[0]
    if n <= 1:
        return torch.tensor(0.0, device=z.device)
    cov = (flat.t() @ flat) / (n - 1)
    try:
        eigvals = torch.linalg.eigvalsh(cov)
    except (torch._C._LinAlgError, NotImplementedError, RuntimeError):
        # Cov became ill-conditioned (typically because the representation has
        # collapsed), or MPS fallback isn't enabled. Caller treats this as
        # "rank ~ 1" via the collapse-detection logic.
        return torch.tensor(float("nan"), device=z.device)
    eigvals = torch.clamp(eigvals, min=0.0)
    total = eigvals.sum()
    if total <= 0:
        return torch.tensor(0.0, device=z.device)
    p = eigvals / total
    p = p[p > 0]
    entropy = -(p * torch.log(p)).sum()
    return torch.exp(entropy)


@torch.no_grad()
def avg_cosine_sim(z: torch.Tensor, max_tokens: int = 1024) -> torch.Tensor:
    flat = z.reshape(-1, z.shape[-1])
    n = flat.shape[0]
    if n > max_tokens:
        idx = torch.randperm(n, device=z.device)[:max_tokens]
        flat = flat[idx]
        n = max_tokens
    if n < 2:
        return torch.tensor(0.0, device=z.device)
    normed = F.normalize(flat, dim=-1)
    sim = normed @ normed.t()
    mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
    return sim[mask].mean()
