"""Real HuggingFace classification sets, re-cast as Jev question types.

Each set carries one label column, which maps onto exactly one question type:
a flat taxonomy becomes a Choice, an ordered sentiment scale becomes a Score,
and a binary flag becomes a Noul. The states are the datasets' own texts, so
the demo runs on real unstructured input rather than templates.
"""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from datasets import Dataset, load_dataset

from jev.schema import ChoiceQuestion, NoulQuestion, Question, QuestionSet, ScoreQuestion

Kind = Literal["choice", "score", "noul"]
Split = Literal["train", "validation", "test"]
Gold = str | int | bool

VALIDATION_FRACTION = 0.1


@dataclass(frozen=True)
class TaskSpec:
    key: str
    kind: Kind
    prompt: str
    path: str
    config: str | None
    text_field: str
    has_validation: bool = True


TASKS: tuple[TaskSpec, ...] = (
    TaskSpec(
        key="intent",
        kind="choice",
        prompt="Which banking intent does this customer message express?",
        path="legacy-datasets/banking77",
        config=None,
        text_field="text",
        has_validation=False,
    ),
    TaskSpec(
        key="emotion",
        kind="choice",
        prompt="Which emotion does the author of this text feel?",
        path="dair-ai/emotion",
        config=None,
        text_field="text",
    ),
    TaskSpec(
        key="sentiment",
        kind="score",
        prompt="How positive is the tone of this text?",
        path="cardiffnlp/tweet_eval",
        config="sentiment",
        text_field="text",
    ),
    TaskSpec(
        key="irony",
        kind="noul",
        prompt="Is this text ironic?",
        path="cardiffnlp/tweet_eval",
        config="irony",
        text_field="text",
    ),
)

TASKS_BY_KEY = {task.key: task for task in TASKS}


@dataclass(frozen=True)
class Sample:
    state: str
    label: int
    gold: Gold


@dataclass(frozen=True)
class LoadedTask:
    spec: TaskSpec
    split: Split
    question: Question
    samples: list[Sample]

    def breakdown(self) -> dict[Gold, int]:
        return dict(Counter(sample.gold for sample in self.samples).most_common())


def humanise(name: str) -> str:
    return name.replace("_", " ").strip()


def build_question(spec: TaskSpec, label_names: Sequence[str]) -> Question:
    labels = [humanise(name) for name in label_names]
    match spec.kind:
        case "choice":
            return ChoiceQuestion(prompt=spec.prompt, options=labels)
        case "score":
            return ScoreQuestion(prompt=spec.prompt, tiers=len(labels), rubric=labels)
        case "noul":
            if len(labels) != 2:
                raise ValueError(f"{spec.key}: a noul needs exactly two labels, got {labels}")
            return NoulQuestion(prompt=spec.prompt)


def gold_value(question: Question, label: int) -> Gold:
    match question:
        case ChoiceQuestion():
            return question.options[label]
        case ScoreQuestion():
            return label + 1
        case NoulQuestion():
            return bool(label)


def resolve_limit(spec: str, total: int) -> int:
    if spec == "all":
        return total
    if spec.endswith("%"):
        return max(1, round(total * float(spec[:-1]) / 100))
    return min(total, int(spec))


def stratified_take(labels: Sequence[int], count: int, seed: int) -> list[int]:
    """Round-robin over labels so a small canary still sees every class."""
    buckets: dict[int, list[int]] = {}
    for index, label in enumerate(labels):
        buckets.setdefault(label, []).append(index)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    taken: list[int] = []
    order = sorted(buckets)
    while len(taken) < count and any(buckets[label] for label in order):
        for label in order:
            if not buckets[label]:
                continue
            taken.append(buckets[label].pop())
            if len(taken) == count:
                break
    return taken


def split_rows(spec: TaskSpec, split: Split, seed: int) -> tuple[Dataset, list[int]]:
    """Rows of `split`; a set without a validation split lends one from train.

    The carve is the same stratified round-robin used for sampling and is keyed
    on `seed`, so train and validation stay disjoint across calls.
    """
    if spec.has_validation or split == "test":
        dataset = load_dataset(spec.path, spec.config, split=split)
        return dataset, list(range(len(dataset)))

    dataset = load_dataset(spec.path, spec.config, split="train")
    labels: list[int] = list(dataset["label"])
    held = stratified_take(labels, round(len(labels) * VALIDATION_FRACTION), seed)
    if split == "validation":
        return dataset, sorted(held)
    return dataset, sorted(set(range(len(labels))) - set(held))


def load_task(
    spec: TaskSpec,
    split: Split = "test",
    limit: str = "100",
    seed: int = 7,
) -> LoadedTask:
    dataset, rows = split_rows(spec, split, seed)
    question = build_question(spec, dataset.features["label"].names)
    all_labels: list[int] = list(dataset["label"])
    labels = [all_labels[row] for row in rows]
    taken = stratified_take(labels, resolve_limit(limit, len(labels)), seed)
    texts = dataset.select([rows[index] for index in taken])[spec.text_field]
    samples = [
        Sample(
            state=text,
            label=labels[index],
            gold=gold_value(question, labels[index]),
        )
        for text, index in zip(texts, taken, strict=True)
    ]
    return LoadedTask(spec=spec, split=split, question=question, samples=samples)


def load_tasks(split: Split, limit: str, seed: int) -> list[LoadedTask]:
    return [load_task(spec, split=split, limit=limit, seed=seed) for spec in TASKS]


def question_set(tasks: Sequence[LoadedTask]) -> QuestionSet:
    return QuestionSet(root={task.spec.key: task.question for task in tasks})
