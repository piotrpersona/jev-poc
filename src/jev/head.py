"""Parallel sampler.

Every question reads the encoded state once through cross-attention, then
scores its own options by dot product against that read. All questions and all
options are resolved in a single pass - there is no decoding loop, and no
question can see another question's answer.

Padded option slots are masked to -inf, so after the per-question softmax they
carry exactly zero probability mass.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ParallelSampler(nn.Module):
    def __init__(self, hidden_size: int, attention_heads: int) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            batch_first=True,
        )
        self.read_norm = nn.LayerNorm(hidden_size)
        self.query = nn.Linear(hidden_size, hidden_size)
        self.scale = hidden_size**-0.5

    def forward(
        self,
        state_tokens: Tensor,
        state_mask: Tensor,
        question_emb: Tensor,
        option_emb: Tensor,
        option_mask: Tensor,
    ) -> Tensor:
        """
        state_tokens [B, T, H]   state_mask [B, T]
        question_emb [B, Q, H]   option_emb [B, Q, O, H]   option_mask [B, Q, O]
        -> logits    [B, Q, O]
        """
        read, _ = self.attention(
            query=question_emb,
            key=state_tokens,
            value=state_tokens,
            key_padding_mask=~state_mask,
            need_weights=False,
        )
        probe = self.query(self.read_norm(question_emb + read))
        logits = torch.einsum("bqh,bqoh->bqo", probe, option_emb) * self.scale
        return logits.masked_fill(~option_mask, float("-inf"))
