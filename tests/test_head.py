from __future__ import annotations

import torch
from conftest import assert_schema_shaped

from jev.head import ParallelSampler

HIDDEN = 16
HEADS = 4


def build(batch: int = 2, tokens: int = 7, questions: int = 3, options: int = 5):
    torch.manual_seed(0)
    state_tokens = torch.randn(batch, tokens, HIDDEN)
    state_mask = torch.ones(batch, tokens, dtype=torch.bool)
    state_mask[:, -2:] = False
    question_emb = torch.randn(batch, questions, HIDDEN)
    option_emb = torch.randn(batch, questions, options, HIDDEN)
    option_mask = torch.zeros(batch, questions, options, dtype=torch.bool)
    arities = [5, 3, 2]
    for index, arity in enumerate(arities):
        option_mask[:, index, :arity] = True
    return state_tokens, state_mask, question_emb, option_emb, option_mask, arities


def test_padded_options_get_no_probability_mass() -> None:
    sampler = ParallelSampler(HIDDEN, HEADS).eval()
    state_tokens, state_mask, question_emb, option_emb, option_mask, arities = build()

    with torch.no_grad():
        probs = sampler(state_tokens, state_mask, question_emb, option_emb, option_mask).softmax(-1)

    assert_schema_shaped(probs, arities)


def test_questions_are_answered_independently() -> None:
    """Changing one question's options cannot move another question's answer."""
    sampler = ParallelSampler(HIDDEN, HEADS).eval()
    state_tokens, state_mask, question_emb, option_emb, option_mask, _ = build()

    with torch.no_grad():
        before = sampler(state_tokens, state_mask, question_emb, option_emb, option_mask)
        perturbed = option_emb.clone()
        perturbed[:, 0] = torch.randn_like(perturbed[:, 0])
        after = sampler(state_tokens, state_mask, question_emb, perturbed, option_mask)

    assert torch.allclose(before[:, 1:], after[:, 1:], atol=1e-6)
    assert not torch.allclose(before[:, 0], after[:, 0], atol=1e-6)


def test_masked_state_tokens_are_not_read() -> None:
    sampler = ParallelSampler(HIDDEN, HEADS).eval()
    state_tokens, state_mask, question_emb, option_emb, option_mask, _ = build()

    with torch.no_grad():
        before = sampler(state_tokens, state_mask, question_emb, option_emb, option_mask)
        noisy = state_tokens.clone()
        noisy[:, -2:] = torch.randn_like(noisy[:, -2:]) * 100
        after = sampler(noisy, state_mask, question_emb, option_emb, option_mask)

    assert torch.allclose(before, after, atol=1e-5)
