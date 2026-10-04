"""Scoring one question's predictions.

A state only carries a gold label for the task it came from, so every question
is scored on its own subset: `intent` against banking77 rows, `irony` against
the irony rows, and so on. Nothing is averaged across questions except the
selection criterion itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

# Cold for a class the model gets wrong, hot for one it gets right.
HOT_COLD = (
    (0.0, "#4c78a8"),
    (0.25, "#9ecae9"),
    (0.5, "#f2f2f2"),
    (0.75, "#f4a582"),
    (1.0, "#d6604d"),
)


@dataclass(frozen=True)
class TaskOutcome:
    """Everything one question produced over one split."""

    key: str
    option_labels: list[str]
    gold: list[int]
    predicted: list[int]
    loss: float

    @property
    def classes(self) -> list[int]:
        return list(range(len(self.option_labels)))

    def scores(self) -> dict[str, float]:
        macro = precision_recall_fscore_support(
            self.gold, self.predicted, labels=self.classes, average="macro", zero_division=0
        )
        weighted = precision_recall_fscore_support(
            self.gold, self.predicted, labels=self.classes, average="weighted", zero_division=0
        )
        return {
            "loss": self.loss,
            "accuracy": float(accuracy_score(self.gold, self.predicted)),
            "precision_macro": float(macro[0]),
            "recall_macro": float(macro[1]),
            "f1_macro": float(macro[2]),
            "f1_weighted": float(weighted[2]),
            "support": float(len(self.gold)),
        }

    def per_class(self) -> dict[str, dict[str, float]]:
        return classification_report(
            self.gold,
            self.predicted,
            labels=self.classes,
            target_names=self.option_labels,
            output_dict=True,
            zero_division=0,
        )

    def confusion_figure(self, split: str) -> go.Figure:
        matrix = confusion_matrix(self.gold, self.predicted, labels=self.classes)
        report = self.per_class()
        scores = self.scores()
        rows = []
        for counts in matrix:
            total = int(counts.sum())
            rows.append([count / total if total else 0.0 for count in counts])

        hover = [
            [
                (
                    f"true: {self.option_labels[true]}<br>"
                    f"predicted: {self.option_labels[guess]}<br>"
                    f"count: {int(matrix[true][guess])}<br>"
                    f"recall(true)={report[self.option_labels[true]]['recall']:.3f} "
                    f"f1(true)={report[self.option_labels[true]]['f1-score']:.3f} "
                    f"support(true)={int(report[self.option_labels[true]]['support'])}<br>"
                    f"precision(predicted)={report[self.option_labels[guess]]['precision']:.3f}"
                )
                for guess in self.classes
            ]
            for true in self.classes
        ]

        figure = go.Figure(
            go.Heatmap(
                z=rows,
                x=self.option_labels,
                y=self.option_labels,
                text=hover,
                hovertemplate="%{text}<extra></extra>",
                colorscale=[list(stop) for stop in HOT_COLD],
                zmin=0.0,
                zmax=1.0,
                colorbar={"title": "share of true class"},
            )
        )
        figure.update_layout(
            title=(
                f"{self.key} - {split} confusion matrix<br>"
                f"<sup>accuracy={scores['accuracy']:.3f} "
                f"f1_macro={scores['f1_macro']:.3f} "
                f"f1_weighted={scores['f1_weighted']:.3f} "
                f"precision_macro={scores['precision_macro']:.3f} "
                f"recall_macro={scores['recall_macro']:.3f} "
                f"support={int(scores['support'])}</sup>"
            ),
            xaxis_title="predicted option",
            yaxis_title="true option",
            template="plotly_white",
            height=max(420, 18 * len(self.option_labels) + 220),
            width=max(560, 18 * len(self.option_labels) + 320),
        )
        return figure


def write_confusion_figures(outcomes: list[TaskOutcome], split: str, directory: Path) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for outcome in outcomes:
        path = directory / f"{split}_{outcome.key}_confusion.html"
        outcome.confusion_figure(split).write_html(path, include_plotlyjs="cdn")
        written.append(path)
    return written
