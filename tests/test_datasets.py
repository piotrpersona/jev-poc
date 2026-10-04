from __future__ import annotations

import pytest

import jev.datasets as datasets_module
from jev.datasets import (
    TASKS_BY_KEY,
    build_question,
    gold_value,
    humanise,
    resolve_limit,
    split_rows,
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


def test_carved_validation_never_overlaps_train(monkeypatch: pytest.MonkeyPatch) -> None:
    """banking77 has no validation split, so one is carved out of train."""
    from datasets import ClassLabel, Dataset, Features, Value

    rows = {"text": [f"row {i}" for i in range(100)], "label": [i % 5 for i in range(100)]}
    features = Features(
        {"text": Value("string"), "label": ClassLabel(names=[f"c{i}" for i in range(5)])}
    )
    monkeypatch.setattr(
        datasets_module, "load_dataset", lambda *_, **__: Dataset.from_dict(rows, features=features)
    )

    spec = TASKS_BY_KEY["intent"]
    assert spec.has_validation is False
    _, train_rows = split_rows(spec, "train", seed=7)
    _, validation_rows = split_rows(spec, "validation", seed=7)

    assert len(validation_rows) == 10
    assert set(train_rows).isdisjoint(validation_rows)
    assert sorted(train_rows + validation_rows) == list(range(100))
    assert len({rows["label"][row] for row in validation_rows}) == 5
