import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + out


class ContextEncoder(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        spatial_tokens: int = 64,
        use_spatial_pos_embed: bool = False,
    ):
        super().__init__()
        assert spatial_tokens == 64, "encoder is fixed to 8x8 spatial tokens"
        self.latent_dim = latent_dim
        self.spatial_tokens = spatial_tokens
        self.use_spatial_pos_embed = use_spatial_pos_embed

        self.conv1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.gn1 = nn.GroupNorm(32, 64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.gn2 = nn.GroupNorm(32, 128)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self.gn3 = nn.GroupNorm(32, 256)
        self.proj = nn.Conv2d(256, latent_dim, 1)
        self.token_norm = nn.LayerNorm(latent_dim)
        if use_spatial_pos_embed:
            # Learned positional embedding broken into the spatial_attn input.
            # Without it, the self-attention is permutation-equivariant on the
            # 64 spatial tokens and can collapse them to a global average.
            self.spatial_pos_embed = nn.Parameter(
                torch.zeros(1, spatial_tokens, latent_dim)
            )
            nn.init.normal_(self.spatial_pos_embed, std=0.02)
        else:
            self.spatial_pos_embed = None
        self.spatial_attn = SpatialSelfAttention(latent_dim, num_heads=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.gelu(self.gn1(self.conv1(x)))
        h = F.gelu(self.gn2(self.conv2(h)))
        h = F.gelu(self.gn3(self.conv3(h)))
        h = self.proj(h)
        b, c, hh, ww = h.shape
        tokens = h.reshape(b, c, hh * ww).transpose(1, 2)
        tokens = self.token_norm(tokens)
        if self.spatial_pos_embed is not None:
            tokens = tokens + self.spatial_pos_embed
        tokens = self.spatial_attn(tokens)
        return tokens
