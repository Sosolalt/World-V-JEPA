import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ffn_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x


class Predictor(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        depth: int = 3,
        num_heads: int = 4,
        ffn_dim: int = 256,
        n_context_tokens: int = 256,
        n_output_tokens: int = 64,
        delta_t_max: int = 12,
    ):
        super().__init__()
        self.dim = dim
        self.n_context_tokens = n_context_tokens
        self.n_output_tokens = n_output_tokens

        self.delta_t_embed = nn.Embedding(delta_t_max + 1, dim)
        self.query_tokens = nn.Parameter(torch.zeros(1, n_output_tokens, dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        total_tokens = n_context_tokens + 1 + n_output_tokens
        self.pos_embed = nn.Parameter(torch.zeros(1, total_tokens, dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            [TransformerBlock(dim, num_heads, ffn_dim) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, context_tokens: torch.Tensor, delta_t: torch.Tensor) -> torch.Tensor:
        b = context_tokens.shape[0]
        dt_tok = self.delta_t_embed(delta_t).unsqueeze(1)
        queries = self.query_tokens.expand(b, -1, -1)
        x = torch.cat([context_tokens, dt_tok, queries], dim=1)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, -self.n_output_tokens:, :]
