"""Demo entry point: run real dataset states through the untrained architecture.

The head is randomly initialised, so the decisions carry no accuracy - what the
demo shows is the shape of the architecture and the invariants that hold by
construction: every answer is schema-valid, every question's probabilities are
normalised over its own options only, and padded option slots stay at zero.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import structlog
import torch

from jev.config import JevConfig, configure_logging
from jev.datasets import TASKS, TASKS_BY_KEY, load_task, question_set
from jev.model import JevModel
from jev.schema import (
    ChoiceDecision,
    ChoiceQuestion,
    NoulDecision,
    NoulQuestion,
    QuestionSet,
    ScoreDecision,
    ScoreQuestion,
    Verdict,
)

log = structlog.get_logger()


def check_invariants(probs: torch.Tensor, questions: QuestionSet) -> list[str]:
    report: list[str] = []
    arities = questions.arities
    for index, (key, arity) in enumerate(zip(questions.keys, arities, strict=True)):
        mass = probs[:, index, :arity].sum(dim=-1)
        padding = probs[:, index, arity:]
        report.append(
            f"{key}: mass={mass.min():.6f}..{mass.max():.6f} "
            f"padding_max={padding.max().item() if padding.numel() else 0.0:.1e} "
            f"options={arity}"
        )
    return report


def check_typesafe(verdict: Verdict, questions: QuestionSet) -> None:
    for key, question in questions.root.items():
        decision = verdict.root[key]
        match question, decision:
            case ChoiceQuestion(), ChoiceDecision():
                assert decision.value in question.options
            case ScoreQuestion(), ScoreDecision():
                assert 1 <= decision.value <= question.tiers
            case NoulQuestion(), NoulDecision():
                assert isinstance(decision.value, bool)
            case _:
                raise AssertionError(f"{key}: decision type does not match question type")


def run_demo(args: argparse.Namespace) -> None:
    configure_logging()
    config = JevConfig.from_env()
    log.info("device", device=str(config.torch_device), backbone=config.backbone)

    tasks = [load_task(spec, limit=args.limit, seed=args.seed) for spec in TASKS]
    questions = question_set(tasks)
    source = next(task for task in tasks if task.spec.key == args.task)
    states = [sample.state for sample in source.samples]

    log.info(
        "schema",
        questions=questions.keys,
        arities=questions.arities,
        types=[question.type for question in questions.questions],
    )
    log.info(
        "states",
        task=source.spec.key,
        dataset=source.spec.path,
        split=source.spec.split,
        count=len(states),
        classes=len(source.breakdown()),
    )
    log.info("breakdown", **{str(k): v for k, v in list(source.breakdown().items())[:8]})

    model = JevModel(config).eval()
    log.info(
        "parameters",
        backbone=sum(p.numel() for p in model.backbone.parameters()),
        head=sum(p.numel() for p in model.sampler.parameters()),
    )

    started = time.perf_counter()
    probs = model.probabilities(states[: config.batch_size], questions)
    elapsed = time.perf_counter() - started
    log.info(
        "forward",
        logits=tuple(probs.shape),
        states=probs.shape[0],
        ms_per_state=round(1000 * elapsed / probs.shape[0], 1),
    )
    for line in check_invariants(probs.float().cpu(), questions):
        log.info("invariant", detail=line)

    verdicts = model.decide(states, questions)
    for verdict in verdicts:
        check_typesafe(verdict, questions)
    log.info("typesafe", verdicts=len(verdicts), type_errors=0)

    for sample, verdict in list(zip(source.samples, verdicts, strict=True))[: args.show]:
        print()
        print(f"state: {sample.state[:200]}")
        print(f"gold[{source.spec.key}]: {sample.gold}")
        print(json.dumps(trim(verdict), indent=2))


def trim(verdict: Verdict, top: int = 3) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, decision in verdict.root.items():
        payload = decision.model_dump()
        distribution = payload.get("distribution")
        if isinstance(distribution, dict):
            ranked = sorted(distribution.items(), key=lambda item: -item[1])[:top]
            payload["distribution"] = {label: round(p, 4) for label, p in ranked}
        out[key] = payload
    return out


def run_gen_schema(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "questions": QuestionSet.model_json_schema(),
                "verdict": Verdict.model_json_schema(),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="jev")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run dataset states through the model")
    demo.add_argument("--task", choices=sorted(TASKS_BY_KEY), default="intent")
    demo.add_argument("--limit", default="100", help="count, percentage (10%%) or 'all'")
    demo.add_argument("--show", type=int, default=3)
    demo.add_argument("--seed", type=int, default=7)
    demo.set_defaults(handler=run_demo)

    gen = sub.add_parser("gen-schema", help="export the JSON Schema of the typed API")
    gen.add_argument("--out", default="schema/jev.schema.json")
    gen.set_defaults(handler=run_gen_schema)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
