from __future__ import annotations

import pytest

from jev.datasets import (
    TASKS_BY_KEY,
    build_question,
    gold_value,
    humanise,
    resolve_limit,
    stratified_take,
)
from jev.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion


@pytest.mark.parametrize(
    ("spec", "total", "expected"),
    [
        ("100", 3080, 100),
        ("all", 3080, 3080),
        ("10%", 3080, 308),
        ("1%", 3080, 31),
        ("5000", 3080, 3080),
    ],
)
def test_resolve_limit(spec: str, total: int, expected: int) -> None:
    assert resolve_limit(spec, total) == expected


def test_stratified_take_covers_every_class_first() -> None:
    labels = [0] * 50 + [1] * 5 + [2] * 2
    taken = stratified_take(labels, 6, seed=1)
    assert len(taken) == 6
    assert {labels[i] for i in taken} == {0, 1, 2}


def test_stratified_take_is_seeded() -> None:
    labels = [i % 4 for i in range(40)]
    assert stratified_take(labels, 8, seed=3) == stratified_take(labels, 8, seed=3)
    assert stratified_take(labels, 8, seed=3) != stratified_take(labels, 8, seed=4)


def test_stratified_take_stops_at_available_rows() -> None:
    labels = [0, 1, 1]
    assert len(stratified_take(labels, 10, seed=0)) == 3


def test_label_names_become_question_types() -> None:
    intent = build_question(TASKS_BY_KEY["intent"], ["card_arrival", "card_linking"])
    sentiment = build_question(TASKS_BY_KEY["sentiment"], ["negative", "neutral", "positive"])
    irony = build_question(TASKS_BY_KEY["irony"], ["non_irony", "irony"])

    assert isinstance(intent, ChoiceQuestion) and intent.options == ["card arrival", "card linking"]
    assert isinstance(sentiment, ScoreQuestion) and sentiment.tiers == 3
    assert isinstance(irony, NoulQuestion)

    assert gold_value(intent, 1) == "card linking"
    assert gold_value(sentiment, 2) == 3
    assert gold_value(irony, 1) is True


def test_noul_rejects_a_non_binary_label_column() -> None:
    with pytest.raises(ValueError, match="exactly two labels"):
        build_question(TASKS_BY_KEY["irony"], ["a", "b", "c"])


def test_humanise() -> None:
    assert humanise("card_arrival") == "card arrival"
