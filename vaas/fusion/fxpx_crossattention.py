import torch
import torch.nn as nn


class FxPxCrossAttention(nn.Module):
    def __init__(self, px_dim, fx_dim, gamma=0.1):
        super().__init__()

        self.register_buffer("gamma", torch.tensor(gamma))

        self.query_proj = nn.Linear(px_dim, px_dim)
        self.key_proj = nn.Linear(fx_dim, px_dim)
        self.value_proj = nn.Linear(fx_dim, px_dim)

        self.attn = nn.MultiheadAttention(px_dim, 8, batch_first=True)

    def forward(self, px_tokens, fx_tokens):
        q = self.query_proj(px_tokens)
        k = self.key_proj(fx_tokens)
        v = self.value_proj(fx_tokens)

        attn_out, attn_weights = self.attn(q, k, v)

        fused = px_tokens + self.gamma * attn_out

        return fused
