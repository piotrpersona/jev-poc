"""Training over a stub backbone: no weights downloaded, no network."""

from __future__ import annotations

import random

import pytest
import torch
from torch import Tensor, nn
from torch.optim import AdamW

from jev.config import JevConfig
from jev.datasets import TASKS_BY_KEY, LoadedTask, Sample, build_question, question_set
from jev.evaluation import TaskOutcome
from jev.model import JevModel
from jev.schema import ChoiceQuestion, NoulQuestion, QuestionSet
from jev.training import Example, examples, metrics_of, question_rows, run_epoch

HIDDEN = 16
HEADS = 4

QUESTIONS = QuestionSet(
    root={
        "colour": ChoiceQuestion(prompt="colour", options=["red", "blue", "green"]),
        "warm": NoulQuestion(prompt="warm"),
    }
)


class StubBackbone(nn.Module):
    """A frozen word lookup: deterministic per word, so the head has real signal."""

    def __init__(self) -> None:
        super().__init__()
        self.table: dict[str, Tensor] = {}
        self.generator = torch.Generator().manual_seed(0)

    @property
    def hidden_size(self) -> int:
        return HIDDEN

    def vector(self, word: str) -> Tensor:
        if word not in self.table:
            self.table[word] = torch.randn(HIDDEN, generator=self.generator)
        return self.table[word]

    def encode_state(self, states: list[str]) -> tuple[Tensor, Tensor]:
        width = max(len(state.split()) for state in states)
        tokens = torch.zeros(len(states), width, HIDDEN)
        mask = torch.zeros(len(states), width, dtype=torch.bool)
        for row, state in enumerate(states):
            words = state.split()
            for column, word in enumerate(words):
                tokens[row, column] = self.vector(word)
            mask[row, : len(words)] = True
        return tokens, mask

    def encode_labels(self, labels: list[str]) -> Tensor:
        return torch.stack(
            [torch.stack([self.vector(word) for word in label.split()]).mean(0) for label in labels]
        )


@pytest.fixture
def model() -> JevModel:
    torch.manual_seed(0)
    config = JevConfig(device="cpu", batch_size=4, attention_heads=HEADS)
    return JevModel(config, backbone=StubBackbone())


def dataset() -> list[Example]:
    colour = [
        Example(state=f"the {name} one", question=0, label=index)
        for index, name in enumerate(["red", "blue", "green"])
    ]
    warm = [
        Example(state="the red one", question=1, label=1),
        Example(state="the blue one", question=1, label=0),
    ]
    return (colour + warm) * 4


def test_masked_options_do_not_change_the_loss() -> None:
    """A 2-option question padded out to width 3 must score as a 2-option question."""
    torch.manual_seed(0)
    logits = torch.randn(4, 3)
    labels = torch.tensor([0, 1, 1, 0])
    padded = logits.clone()
    padded[:, 2] = float("-inf")

    over_padded = nn.functional.cross_entropy(padded, labels)
    over_sliced = nn.functional.cross_entropy(logits[:, :2], labels)

    assert torch.allclose(over_padded, over_sliced)


def test_question_rows_picks_the_owning_question() -> None:
    logits = torch.arange(2 * 3 * 4, dtype=torch.float).reshape(2, 3, 4)
    rows = question_rows(logits, torch.tensor([2, 0]))
    assert torch.equal(rows[0], logits[0, 2])
    assert torch.equal(rows[1], logits[1, 0])


def test_examples_point_at_their_own_question() -> None:
    tasks = [
        LoadedTask(
            spec=TASKS_BY_KEY["irony"],
            split="train",
            question=build_question(TASKS_BY_KEY["irony"], ["non_irony", "irony"]),
            samples=[Sample(state="s", label=1, gold=True)],
        ),
        LoadedTask(
            spec=TASKS_BY_KEY["emotion"],
            split="train",
            question=build_question(TASKS_BY_KEY["emotion"], ["joy", "anger"]),
            samples=[Sample(state="t", label=0, gold="joy")],
        ),
    ]
    questions = question_set(tasks)
    assert examples(tasks, questions) == [
        Example(state="s", question=0, label=1),
        Example(state="t", question=1, label=0),
    ]


def test_metrics_summarise_every_question(model: JevModel) -> None:
    outcomes = [
        TaskOutcome(key="a", option_labels=["x", "y"], gold=[0, 1], predicted=[0, 1], loss=0.5),
        TaskOutcome(key="b", option_labels=["x", "y"], gold=[0, 1], predicted=[1, 0], loss=1.5),
    ]
    metrics = metrics_of(outcomes, "validation")

    assert metrics["validation/a/accuracy"] == 1.0
    assert metrics["validation/b/accuracy"] == 0.0
    assert metrics["validation/loss"] == pytest.approx(1.0)
    assert metrics["validation/mean_f1_macro"] == pytest.approx(0.5)


def test_an_epoch_trains_the_head_and_leaves_the_backbone_alone(model: JevModel) -> None:
    data = dataset()
    optimizer = AdamW(model.sampler.parameters(), lr=0.05)
    before = run_epoch(model, QUESTIONS, data, batch_size=4)

    for _ in range(40):
        run_epoch(
            model,
            QUESTIONS,
            data,
            batch_size=4,
            optimizer=optimizer,
            shuffle=random.Random(1),
        )
    after = run_epoch(model, QUESTIONS, data, batch_size=4)

    assert {outcome.key for outcome in after} == {"colour", "warm"}
    for start, end in zip(before, after, strict=True):
        assert end.loss < start.loss
        assert end.scores()["accuracy"] == 1.0
    assert all(parameter.grad is None for parameter in model.backbone.parameters())


def test_evaluation_leaves_no_gradients_behind(model: JevModel) -> None:
    run_epoch(model, QUESTIONS, dataset(), batch_size=4)
    assert all(parameter.grad is None for parameter in model.sampler.parameters())
