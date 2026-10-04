from __future__ import annotations

import pytest
from pydantic import ValidationError

from jev.schema import (
    MAX_OPTIONS,
    ChoiceQuestion,
    NoulQuestion,
    QuestionSet,
    ScoreQuestion,
)


@pytest.mark.parametrize(
    ("name", "build"),
    [
        ("one option", lambda: ChoiceQuestion(prompt="p", options=["a"])),
        ("duplicate options", lambda: ChoiceQuestion(prompt="p", options=["a", "a"])),
        ("empty prompt", lambda: ChoiceQuestion(prompt="", options=["a", "b"])),
        (
            "too many options",
            lambda: ChoiceQuestion(prompt="p", options=[str(i) for i in range(MAX_OPTIONS + 1)]),
        ),
        ("one tier", lambda: ScoreQuestion(prompt="p", tiers=1)),
        ("eleven tiers", lambda: ScoreQuestion(prompt="p", tiers=11)),
        ("rubric mismatch", lambda: ScoreQuestion(prompt="p", tiers=3, rubric=["low", "high"])),
        ("empty question set", lambda: QuestionSet(root={})),
    ],
)
def test_invalid_schema_is_rejected(name: str, build) -> None:
    with pytest.raises(ValidationError):
        build()


def test_choice_decision_reads_the_option_set() -> None:
    question = ChoiceQuestion(prompt="p", options=["billing", "technical", "sales"])
    decision = question.decide([0.2, 0.7, 0.1])
    assert decision.value == "technical"
    assert decision.confidence == pytest.approx(0.7)
    assert decision.distribution["sales"] == pytest.approx(0.1)


def test_score_decision_is_ordinal() -> None:
    question = ScoreQuestion(prompt="p", tiers=3, rubric=["low", "mid", "high"])
    decision = question.decide([0.5, 0.0, 0.5])
    assert decision.value == 1
    assert decision.expected == pytest.approx(2.0)
    assert list(decision.distribution) == ["low", "mid", "high"]


def test_noul_decision_is_binary() -> None:
    question = NoulQuestion(prompt="p")
    assert question.labels == ["no", "yes"]
    assert question.decide([0.8, 0.2]).value is False
    assert question.decide([0.2, 0.8]).p_yes == pytest.approx(0.8)
    assert question.decide([0.2, 0.8]).confidence == pytest.approx(0.8)


def test_arities_follow_the_schema() -> None:
    questions = QuestionSet(
        root={
            "category": ChoiceQuestion(prompt="p", options=["a", "b", "c"]),
            "urgency": ScoreQuestion(prompt="p", tiers=5),
            "escalate": NoulQuestion(prompt="p"),
        }
    )
    assert questions.arities == [3, 5, 2]
    assert questions.keys == ["category", "urgency", "escalate"]
