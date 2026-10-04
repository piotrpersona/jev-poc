"""Typed decision schema.

A question owns three things: the option set the model may pick from, the text
that verbalises those options for the encoder, and the mapping from a
probability vector back to a typed decision. Keeping all three on the question
class is what makes the output type-safe by construction - the sampler can only
ever return indices into `options`, and only the question knows how to read
them.
"""

from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, Field, RootModel, model_validator

MAX_OPTIONS = 255
MIN_TIERS = 2
MAX_TIERS = 10

NOUL_OPTIONS = ("no", "yes")


class ChoiceQuestion(BaseModel):
    type: Literal["choice"] = "choice"
    prompt: str = Field(min_length=1)
    options: list[str] = Field(min_length=2, max_length=MAX_OPTIONS)

    @model_validator(mode="after")
    def _distinct_options(self) -> Self:
        if len(set(self.options)) != len(self.options):
            raise ValueError("choice options must be distinct")
        return self

    @property
    def labels(self) -> list[str]:
        return list(self.options)

    def decide(self, probs: list[float]) -> ChoiceDecision:
        best = max(range(len(probs)), key=probs.__getitem__)
        return ChoiceDecision(
            value=self.options[best],
            confidence=probs[best],
            distribution=dict(zip(self.options, probs, strict=True)),
        )


class ScoreQuestion(BaseModel):
    type: Literal["score"] = "score"
    prompt: str = Field(min_length=1)
    tiers: int = Field(ge=MIN_TIERS, le=MAX_TIERS)
    rubric: list[str] | None = None

    @model_validator(mode="after")
    def _rubric_matches_tiers(self) -> Self:
        if self.rubric is not None and len(self.rubric) != self.tiers:
            raise ValueError("rubric must name every tier")
        return self

    @property
    def labels(self) -> list[str]:
        if self.rubric is not None:
            return list(self.rubric)
        return [f"level {i + 1} of {self.tiers}" for i in range(self.tiers)]

    def decide(self, probs: list[float]) -> ScoreDecision:
        best = max(range(len(probs)), key=probs.__getitem__)
        return ScoreDecision(
            value=best + 1,
            expected=sum((i + 1) * p for i, p in enumerate(probs)),
            confidence=probs[best],
            distribution=dict(zip(self.labels, probs, strict=True)),
        )


class NoulQuestion(BaseModel):
    type: Literal["noul"] = "noul"
    prompt: str = Field(min_length=1)

    @property
    def labels(self) -> list[str]:
        return list(NOUL_OPTIONS)

    def decide(self, probs: list[float]) -> NoulDecision:
        p_yes = probs[1]
        return NoulDecision(value=p_yes >= 0.5, p_yes=p_yes, confidence=max(p_yes, 1.0 - p_yes))


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class ChoiceDecision(BaseModel):
    type: Literal["choice"] = "choice"
    value: str
    confidence: float
    distribution: dict[str, float]


class ScoreDecision(BaseModel):
    type: Literal["score"] = "score"
    value: int
    expected: float
    confidence: float
    distribution: dict[str, float]


class NoulDecision(BaseModel):
    type: Literal["noul"] = "noul"
    value: bool
    p_yes: float
    confidence: float


Decision = Annotated[ChoiceDecision | ScoreDecision | NoulDecision, Field(discriminator="type")]


class QuestionSet(RootModel[dict[str, Question]]):
    """The schema of one request: every question is answered in the same pass."""

    @model_validator(mode="after")
    def _not_empty(self) -> Self:
        if not self.root:
            raise ValueError("a question set needs at least one question")
        return self

    @property
    def keys(self) -> list[str]:
        return list(self.root)

    @property
    def questions(self) -> list[Question]:
        return list(self.root.values())

    @property
    def arities(self) -> list[int]:
        return [len(q.labels) for q in self.questions]

    def fingerprint(self) -> str:
        return self.model_dump_json()


class Verdict(RootModel[dict[str, Decision]]):
    pass
