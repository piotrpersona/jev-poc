"""Shared text encoder.

One HuggingFace encoder serves both sides of the architecture: the program
state (kept at token resolution, because questions read it through attention)
and the schema text (pooled to one vector per question and per option).
"""

from __future__ import annotations

from collections.abc import Sequence

from torch import Tensor, nn
from transformers import AutoModel, AutoTokenizer

from jev.config import JevConfig


def mean_pool(hidden: Tensor, mask: Tensor) -> Tensor:
    weights = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weights).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-6)


class Backbone(nn.Module):
    def __init__(self, config: JevConfig) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.backbone)
        self.encoder = AutoModel.from_pretrained(config.backbone)

    @property
    def hidden_size(self) -> int:
        return int(self.encoder.config.hidden_size)

    def _encode(self, texts: Sequence[str], max_tokens: int) -> tuple[Tensor, Tensor]:
        batch = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        ).to(self.encoder.device)
        hidden = self.encoder(**batch).last_hidden_state
        return hidden, batch["attention_mask"].bool()

    def encode_state(self, states: Sequence[str]) -> tuple[Tensor, Tensor]:
        """-> token hidden states [B, T, H] and their padding mask [B, T]."""
        return self._encode(states, self.config.max_state_tokens)

    def encode_labels(self, labels: Sequence[str]) -> Tensor:
        """-> one pooled vector per label [N, H]."""
        hidden, mask = self._encode(labels, self.config.max_option_tokens)
        return mean_pool(hidden, mask)
