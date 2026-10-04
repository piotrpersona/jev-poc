from __future__ import annotations

from torch import Tensor


def assert_schema_shaped(probs: Tensor, arities: list[int]) -> None:
    """Per question: mass 1 over its own options, exactly 0 on padded slots."""
    for index, arity in enumerate(arities):
        mass = probs[:, index, :arity].sum(-1)
        assert mass.allclose(probs.new_ones(probs.shape[0]), atol=1e-5)
        padding = probs[:, index, arity:]
        assert padding.numel() == 0 or padding.abs().max().item() == 0.0
