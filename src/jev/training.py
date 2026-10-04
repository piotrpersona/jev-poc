"""Training the parallel sampler.

The backbone and therefore the schema embeddings are frozen, so what is learned
is the read: how a question attends over the state, and how that read scores the
option vectors it was handed. 3M parameters, one optimiser, one loss.

Every state is asked all four questions in the same pass - that is the Jev
shape - but a state only carries a gold label for the task it came from, so the
loss supervises that one question's row and leaves the other three untouched.
The questions share the sampler, so they still train each other.

Padded option slots arrive as -inf, which makes cross-entropy over the full
option width identical to cross-entropy over the question's own options.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import mlflow
import structlog
import torch
from mlflow.exceptions import MlflowException
from torch import Tensor, nn
from torch.optim import AdamW

from jev.checkpoint import load_head, save_head
from jev.config import JevConfig
from jev.datasets import LoadedTask, Split, load_tasks, question_set
from jev.evaluation import TaskOutcome, write_confusion_figures
from jev.model import JevModel
from jev.schema import QuestionSet

log = structlog.get_logger()

SPLITS: tuple[Split, ...] = ("train", "validation", "test")
CRITERION = "validation/mean_f1_macro"
REGISTERED_MODEL = "jev-head"


@dataclass(frozen=True)
class TrainConfig:
    limit: str = "100"
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    patience: int = 3
    seed: int = 7
    experiment: str = "jev-poc"
    run_name: str | None = None
    out_dir: Path = Path("out")

    @property
    def canary(self) -> bool:
        return self.limit != "all"


@dataclass(frozen=True)
class Example:
    """One state, the question that owns its gold label, and that label."""

    state: str
    question: int
    label: int


@dataclass
class Tally:
    gold: list[int] = field(default_factory=list)
    predicted: list[int] = field(default_factory=list)
    loss_sum: float = 0.0

    def outcome(self, key: str, option_labels: list[str]) -> TaskOutcome:
        return TaskOutcome(
            key=key,
            option_labels=option_labels,
            gold=self.gold,
            predicted=self.predicted,
            loss=self.loss_sum / len(self.gold),
        )


def examples(tasks: Sequence[LoadedTask], questions: QuestionSet) -> list[Example]:
    position = {key: index for index, key in enumerate(questions.keys)}
    return [
        Example(state=sample.state, question=position[task.spec.key], label=sample.label)
        for task in tasks
        for sample in task.samples
    ]


def batches(
    data: Sequence[Example], size: int, shuffle: random.Random | None
) -> Iterator[list[Example]]:
    order = list(range(len(data)))
    if shuffle is not None:
        shuffle.shuffle(order)
    for start in range(0, len(order), size):
        yield [data[index] for index in order[start : start + size]]


def question_rows(logits: Tensor, questions: Tensor) -> Tensor:
    """Each state's own question row: [B, Q, O] -> [B, O]."""
    return logits[torch.arange(logits.shape[0], device=logits.device), questions]


def run_epoch(
    model: JevModel,
    questions: QuestionSet,
    data: Sequence[Example],
    batch_size: int,
    optimizer: AdamW | None = None,
    shuffle: random.Random | None = None,
) -> list[TaskOutcome]:
    training = optimizer is not None
    model.train(training)
    device = model.config.torch_device
    tallies = {index: Tally() for index in range(len(questions.keys))}

    for batch in batches(data, batch_size, shuffle):
        states = [example.state for example in batch]
        owners = torch.tensor([example.question for example in batch], device=device)
        labels = torch.tensor([example.label for example in batch], device=device)

        with torch.set_grad_enabled(training):
            rows = question_rows(model(states, questions), owners)
            losses = nn.functional.cross_entropy(rows, labels, reduction="none")
            loss = losses.mean()

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.sampler.parameters(), 1.0)
            optimizer.step()

        predicted = rows.argmax(dim=-1).tolist()
        for example, guess, item in zip(batch, predicted, losses.tolist(), strict=True):
            tally = tallies[example.question]
            tally.gold.append(example.label)
            tally.predicted.append(guess)
            tally.loss_sum += item

    return [
        tallies[index].outcome(key, question.labels)
        for index, (key, question) in enumerate(questions.root.items())
        if tallies[index].gold
    ]


def metrics_of(outcomes: Sequence[TaskOutcome], split: str) -> dict[str, float]:
    """Per-question scores, plus the two numbers that summarise the split."""
    metrics: dict[str, float] = {}
    scored = [(outcome, outcome.scores()) for outcome in outcomes]
    for outcome, scores in scored:
        for name, value in scores.items():
            metrics[f"{split}/{outcome.key}/{name}"] = value
    support = sum(scores["support"] for _, scores in scored)
    metrics[f"{split}/loss"] = (
        sum(scores["loss"] * scores["support"] for _, scores in scored) / support
    )
    mean_f1 = sum(scores["f1_macro"] for _, scores in scored) / len(scored)
    metrics[f"{split}/mean_f1_macro"] = mean_f1
    return metrics


def report_epoch(outcomes: Sequence[TaskOutcome], split: str, epoch: int, directory: Path) -> Path:
    """Per-class precision/recall/f1/support for every question, as an artifact."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}_epoch{epoch:02d}_per_class.json"
    path.write_text(
        json.dumps({outcome.key: outcome.per_class() for outcome in outcomes}, indent=2) + "\n"
    )
    return path


def log_epoch(outcomes: Sequence[TaskOutcome], split: str, epoch: int) -> None:
    for outcome in outcomes:
        scores = outcome.scores()
        log.info(
            split,
            epoch=epoch,
            task=outcome.key,
            **{name: round(value, 4) for name, value in scores.items()},
        )


def breakdown(splits: dict[Split, list[LoadedTask]]) -> dict[str, dict[str, object]]:
    return {
        split: {
            task.spec.key: {
                "dataset": task.spec.path,
                "config": task.spec.config,
                "samples": len(task.samples),
                "classes": len(task.breakdown()),
                "counts": {str(gold): count for gold, count in task.breakdown().items()},
            }
            for task in tasks
        }
        for split, tasks in splits.items()
    }


def device_report(device: torch.device) -> dict[str, object]:
    report: dict[str, object] = {"device": str(device), "torch": torch.__version__}
    if device.type == "cuda":
        free, total = torch.cuda.mem_get_info()
        report |= {
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count": torch.cuda.device_count(),
            "memory_total_gb": round(total / 2**30, 2),
            "memory_free_gb": round(free / 2**30, 2),
        }
    return report


def nvidia_smi() -> str | None:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() or None


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def start_tracking(plan: TrainConfig) -> None:
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", plan.experiment))


def register_head(model: JevModel) -> None:
    """Log the head under the pytorch flavor and register it.

    Pickled rather than pt2: tracing the sampler's forward would need an example
    of all five tensors, and the graph is not what is being shipped - the three
    tensor sets a schema produces are.
    """
    info = mlflow.pytorch.log_model(model.sampler, name="head", serialization_format="pickle")
    try:
        mlflow.register_model(info.model_uri, REGISTERED_MODEL)
    except MlflowException as error:
        log.warning("registry unavailable", detail=str(error).splitlines()[0])


def train(config: JevConfig, plan: TrainConfig) -> dict[str, float]:
    device = config.torch_device
    log.info("device", **device_report(device))
    if device.type == "cpu":
        log.warning("no accelerator", detail="running on cpu, set JEV_DEVICE=cuda|mps for a GPU")
    if plan.canary:
        log.info("canary", limit=plan.limit, detail="per task per split, rerun with --limit all")

    splits = {split: load_tasks(split, plan.limit, plan.seed) for split in SPLITS}
    questions = question_set(splits["train"])
    data = {split: examples(tasks, questions) for split, tasks in splits.items()}
    log.info(
        "schema",
        questions=questions.keys,
        arities=questions.arities,
        types=[question.type for question in questions.questions],
    )
    for split, tasks in splits.items():
        for task in tasks:
            log.info(
                "split",
                split=split,
                task=task.spec.key,
                dataset=task.spec.path,
                samples=len(task.samples),
                classes=len(task.breakdown()),
                min_class=min(task.breakdown().values()),
                max_class=max(task.breakdown().values()),
            )

    start_tracking(plan)
    with mlflow.start_run(run_name=plan.run_name) as run:
        artifacts = plan.out_dir / run.info.run_id
        reports = artifacts / "per_class"
        checkpoint = artifacts / "best_head.pt"
        artifacts.mkdir(parents=True, exist_ok=True)

        model = JevModel(config)
        head_parameters = sum(p.numel() for p in model.sampler.parameters())
        mlflow.set_tags(
            {
                "run_kind": "canary" if plan.canary else "full",
                "git_commit": git_commit(),
                "dataset_version": json.dumps(
                    {task.spec.key: task.spec.path for task in splits["train"]}
                ),
                "device": str(device),
                "trainable": "head-only (backbone frozen)",
            }
        )
        mlflow.log_params(
            {
                "backbone": config.backbone,
                "device": str(device),
                "batch_size": config.batch_size,
                "max_state_tokens": config.max_state_tokens,
                "max_option_tokens": config.max_option_tokens,
                "attention_heads": config.attention_heads,
                "epochs": plan.epochs,
                "learning_rate": plan.learning_rate,
                "weight_decay": plan.weight_decay,
                "patience": plan.patience,
                "seed": plan.seed,
                "limit": plan.limit,
                "selection_criterion": CRITERION,
                "head_parameters": head_parameters,
                "backbone_parameters": sum(p.numel() for p in model.backbone.parameters()),
                "questions": json.dumps(dict(zip(questions.keys, questions.arities, strict=True))),
                **{f"samples_{split}": len(rows) for split, rows in data.items()},
            }
        )
        breakdown_path = artifacts / "dataset_breakdown.json"
        breakdown_path.write_text(json.dumps(breakdown(splits), indent=2) + "\n")
        mlflow.log_artifact(str(breakdown_path))

        torch.manual_seed(plan.seed)
        optimizer = AdamW(
            model.sampler.parameters(), lr=plan.learning_rate, weight_decay=plan.weight_decay
        )
        shuffle = random.Random(plan.seed)
        best = -math.inf
        best_epoch = 0
        stale = 0

        for epoch in range(1, plan.epochs + 1):
            trained = run_epoch(
                model,
                questions,
                data["train"],
                config.batch_size,
                optimizer=optimizer,
                shuffle=shuffle,
            )
            validated = run_epoch(model, questions, data["validation"], config.batch_size)
            metrics = metrics_of(trained, "train") | metrics_of(validated, "validation")
            mlflow.log_metrics(metrics, step=epoch)
            for outcomes, split in ((trained, "train"), (validated, "validation")):
                log_epoch(outcomes, split, epoch)
                mlflow.log_artifact(str(report_epoch(outcomes, split, epoch, reports)), "per_class")

            score = metrics[CRITERION]
            if score > best:
                best, best_epoch, stale = score, epoch, 0
                save_head(checkpoint, model, questions, metrics)
                log.info("checkpoint", epoch=epoch, criterion=CRITERION, score=round(score, 4))
            else:
                stale += 1
                log.info(
                    "no improvement",
                    epoch=epoch,
                    score=round(score, 4),
                    best=round(best, 4),
                    patience_left=plan.patience - stale,
                )
                if stale >= plan.patience:
                    log.info("early stop", epoch=epoch, best_epoch=best_epoch)
                    break

        mlflow.log_metrics({"best_epoch": best_epoch, f"best_{CRITERION}": best})
        load_head(checkpoint, model)
        tested = run_epoch(model, questions, data["test"], config.batch_size)
        test_metrics = metrics_of(tested, "test")
        mlflow.log_metrics(test_metrics, step=best_epoch)
        log_epoch(tested, "test", best_epoch)
        for figure in write_confusion_figures(tested, "test", artifacts / "figures"):
            mlflow.log_artifact(str(figure), "figures")
        mlflow.log_artifact(str(report_epoch(tested, "test", best_epoch, reports)), "per_class")
        mlflow.log_artifact(str(checkpoint))
        register_head(model)

        if device.type == "cuda":
            log.info(
                "gpu usage",
                peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                nvidia_smi=nvidia_smi(),
            )
        log.info(
            "done",
            run_id=run.info.run_id,
            best_epoch=best_epoch,
            **{CRITERION: round(best, 4)},
            test_mean_f1_macro=round(test_metrics["test/mean_f1_macro"], 4),
            checkpoint=str(checkpoint),
        )
        return test_metrics
