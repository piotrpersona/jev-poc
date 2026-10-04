"""End-to-end over the real backbone. Enable with JEV_E2E=1 (downloads weights)."""

from __future__ import annotations

import os

import pytest
from conftest import assert_schema_shaped

from jev.config import JevConfig
from jev.model import JevModel
from jev.schema import ChoiceDecision, ChoiceQuestion, NoulQuestion, QuestionSet, ScoreQuestion

pytestmark = pytest.mark.skipif(
    not os.getenv("JEV_E2E"), reason="set JEV_E2E=1 to download weights"
)

STATES = [
    "Customer emailed twice this week about a failed refund on order 8821"
    " and is now asking for a chargeback.",
    "Hi, just checking whether my new card has been posted yet - no rush at all.",
]

QUESTIONS = QuestionSet(
    root={
        "category": ChoiceQuestion(
            prompt="Which queue should this land in?",
            options=["billing", "technical", "sales", "retention"],
        ),
        "urgency": ScoreQuestion(
            prompt="How urgent is this?",
            tiers=5,
            rubric=["none", "low", "normal", "high", "critical"],
        ),
        "needs_human": NoulQuestion(prompt="Does this need a human agent?"),
    }
)


@pytest.fixture(scope="module")
def model() -> JevModel:
    return JevModel(JevConfig.from_env()).eval()


def test_e2e_probabilities_are_schema_shaped(model: JevModel) -> None:
    probs = model.probabilities(STATES, QUESTIONS).float().cpu()
    assert tuple(probs.shape) == (len(STATES), 3, 5)
    assert_schema_shaped(probs, QUESTIONS.arities)


def test_e2e_decisions_are_typed_and_in_schema(model: JevModel) -> None:
    verdicts = model.decide(STATES, QUESTIONS)
    assert len(verdicts) == len(STATES)
    for verdict in verdicts:
        category = verdict.root["category"]
        assert isinstance(category, ChoiceDecision)
        assert category.value in QUESTIONS.root["category"].options
        assert 1 <= verdict.root["urgency"].value <= 5
        assert isinstance(verdict.root["needs_human"].value, bool)


def test_e2e_schema_is_encoded_once(model: JevModel) -> None:
    model._schema_cache.clear()
    model.probabilities(STATES, QUESTIONS)
    model.probabilities(STATES, QUESTIONS)
    assert len(model._schema_cache) == 1
