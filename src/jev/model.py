"""System One model: unstructured state in, typed decisions out, one pass."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import torch
from torch import Tensor, nn

from jev.config import JevConfig
from jev.encoder import Backbone
from jev.head import ParallelSampler
from jev.schema import QuestionSet, Verdict


@dataclass(frozen=True)
class SchemaEmbedding:
    """The encoded question set: fixed for a deployment, so encoded once."""

    question_emb: Tensor  # [Q, H]
    option_emb: Tensor  # [Q, O, H]
    option_mask: Tensor  # [Q, O]


class JevModel(nn.Module):
    """The backbone is a frozen feature extractor; only the sampler is trained.

    Freezing it keeps the schema embeddings constant, which is what lets the
    schema cache survive training, and keeps the autograd graph to the 3M-param
    head - the backbone pass builds no graph at all.
    """

    def __init__(self, config: JevConfig, backbone: Backbone | None = None) -> None:
        super().__init__()
        self.config = config
        self.backbone = backbone or Backbone(config)
        self.backbone.requires_grad_(False)
        self.sampler = ParallelSampler(self.backbone.hidden_size, config.attention_heads)
        self._schema_cache: dict[str, SchemaEmbedding] = {}
        self.to(config.torch_device)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        self.backbone.eval()
        return self

    @torch.no_grad()
    def embed_schema(self, questions: QuestionSet) -> SchemaEmbedding:
        cached = self._schema_cache.get(questions.fingerprint())
        if cached is not None:
            return cached

        arities = questions.arities
        width = max(arities)
        hidden = self.backbone.hidden_size
        device = self.config.torch_device

        prompts = [q.prompt for q in questions.questions]
        question_emb = self.backbone.encode_labels(prompts)

        option_emb = torch.zeros(len(arities), width, hidden, device=device)
        option_mask = torch.zeros(len(arities), width, dtype=torch.bool, device=device)
        for index, question in enumerate(questions.questions):
            labels = [f"{question.prompt} {label}" for label in question.labels]
            option_emb[index, : len(labels)] = self.backbone.encode_labels(labels)
            option_mask[index, : len(labels)] = True

        embedded = SchemaEmbedding(question_emb, option_emb, option_mask)
        self._schema_cache[questions.fingerprint()] = embedded
        return embedded

    def forward(self, states: Sequence[str], questions: QuestionSet) -> Tensor:
        """-> logits [B, Q, O]; padded option slots hold -inf."""
        schema = self.embed_schema(questions)
        state_tokens, state_mask = self.backbone.encode_state(states)
        batch = len(states)
        return self.sampler(
            state_tokens=state_tokens,
            state_mask=state_mask,
            question_emb=schema.question_emb.expand(batch, -1, -1),
            option_emb=schema.option_emb.expand(batch, -1, -1, -1),
            option_mask=schema.option_mask.expand(batch, -1, -1),
        )

    @torch.inference_mode()
    def probabilities(self, states: Sequence[str], questions: QuestionSet) -> Tensor:
        """-> [B, Q, O]; padded option slots hold exactly zero."""
        return self(states, questions).softmax(dim=-1)

    def decide(self, states: Sequence[str], questions: QuestionSet) -> list[Verdict]:
        verdicts: list[Verdict] = []
        for start in range(0, len(states), self.config.batch_size):
            chunk = states[start : start + self.config.batch_size]
            probs = self.probabilities(chunk, questions).float().cpu()
            for row in probs:
                verdicts.append(self._read(row, questions))
        return verdicts

    @staticmethod
    def _read(probs: Tensor, questions: QuestionSet) -> Verdict:
        return Verdict(
            root={
                key: question.decide(probs[index, : len(question.labels)].tolist())
                for index, (key, question) in enumerate(questions.root.items())
            }
        )
