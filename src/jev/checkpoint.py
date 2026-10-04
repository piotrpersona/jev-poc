"""Head checkpoints.

Only the sampler is trained, so a checkpoint is the head's weights plus the
question set they were trained against: a head learned to score *these* option
embeddings, produced by *this* backbone, and means nothing against others.
"""

from __future__ import annotations

from pathlib import Path

import torch

from jev.model import JevModel
from jev.schema import QuestionSet


def save_head(
    path: Path,
    model: JevModel,
    questions: QuestionSet,
    metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sampler": {key: value.cpu() for key, value in model.sampler.state_dict().items()},
            "backbone": model.config.backbone,
            "questions": questions.fingerprint(),
            "metrics": metrics,
        },
        path,
    )


def load_head(path: Path, model: JevModel) -> QuestionSet:
    """Load head weights into `model` and return the schema they were trained on."""
    payload = torch.load(path, map_location=model.config.torch_device, weights_only=True)
    if payload["backbone"] != model.config.backbone:
        raise ValueError(
            f"checkpoint was trained on {payload['backbone']},"
            f" this run uses {model.config.backbone}"
        )
    model.sampler.load_state_dict(payload["sampler"])
    return QuestionSet.model_validate_json(payload["questions"])
