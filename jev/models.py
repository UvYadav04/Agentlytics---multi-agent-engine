from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence, Union

from .errors import JevValidationError

MAX_CHOICE_OPTIONS = 255
MIN_CHOICE_OPTIONS = 2
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10

Instructions = Union[str, Mapping[str, Any], Sequence[Any]]
State = Union[str, Mapping[str, Any], Sequence[Any]]


def _check_instructions(instructions: Instructions) -> None:
    if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
        raise JevValidationError("A question needs non-empty instructions.")
    if not isinstance(instructions, str) and not instructions:
        raise JevValidationError("A question needs non-empty instructions.")


@dataclass(frozen=True)
class Choice:
    instructions: Instructions
    options: Mapping[str, Any]

    def __init__(self, instructions: Instructions, options: Union[Mapping[str, Any], Sequence[str]]):
        if isinstance(options, Mapping):
            normalized = {str(key): value for key, value in options.items()}
        else:
            normalized = {str(option): None for option in options}
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "options", normalized)
        self.validate()

    def validate(self) -> None:
        _check_instructions(self.instructions)
        if not MIN_CHOICE_OPTIONS <= len(self.options) <= MAX_CHOICE_OPTIONS:
            raise JevValidationError(
                f"A Choice needs between {MIN_CHOICE_OPTIONS} and {MAX_CHOICE_OPTIONS} options, got {len(self.options)}."
            )
        if any(not key.strip() for key in self.options):
            raise JevValidationError("Choice options must be non-empty strings.")

    def to_payload(self) -> dict:
        return {"type": "choice", "instructions": self.instructions, "criteria": dict(self.options)}


@dataclass(frozen=True)
class Score:
    instructions: Instructions
    levels: tuple

    def __init__(self, instructions: Instructions, levels: Sequence[Any]):
        object.__setattr__(self, "instructions", instructions)
        object.__setattr__(self, "levels", tuple(levels))
        self.validate()

    def validate(self) -> None:
        _check_instructions(self.instructions)
        if not MIN_SCORE_LEVELS <= len(self.levels) <= MAX_SCORE_LEVELS:
            raise JevValidationError(
                f"A Score needs between {MIN_SCORE_LEVELS} and {MAX_SCORE_LEVELS} levels, got {len(self.levels)}."
            )

    def to_payload(self) -> dict:
        return {"type": "score", "instructions": self.instructions, "criteria": list(self.levels)}


@dataclass(frozen=True)
class Noul:
    instructions: Instructions
    yes: Any = None
    no: Any = None

    def __post_init__(self):
        _check_instructions(self.instructions)

    def to_payload(self) -> dict:
        payload: dict = {"type": "noul", "instructions": self.instructions}
        if self.yes is not None or self.no is not None:
            criteria = {}
            if self.yes is not None:
                criteria["true"] = self.yes
            if self.no is not None:
                criteria["false"] = self.no
            payload["criteria"] = criteria
        return payload


Question = Union[Choice, Score, Noul]


@dataclass(frozen=True)
class ChoiceAnswer:
    choice: str
    probabilities: dict
    confidence: float | None
    type: str = "choice"

    def probability(self, option: str) -> float:
        return float(self.probabilities.get(option, 0.0))

    def ranked(self) -> list[tuple[str, float]]:
        return sorted(((k, float(v)) for k, v in self.probabilities.items()), key=lambda item: item[1], reverse=True)

    def is_confident(self, min_confidence: float) -> bool:
        return self.confidence is not None and self.confidence >= min_confidence

    def pick(self, min_confidence: float, fallback: str | None = None) -> str | None:
        return self.choice if self.is_confident(min_confidence) else fallback


@dataclass(frozen=True)
class ScoreAnswer:
    score: float
    legend: dict
    probabilities: dict
    confidence: float | None
    type: str = "score"

    def nearest_level(self) -> int:
        return int(round(self.score))

    def nearest_label(self):
        return self.legend.get(str(self.nearest_level()))

    def is_confident(self, min_confidence: float) -> bool:
        return self.confidence is not None and self.confidence >= min_confidence

    def at_least(self, level: float) -> bool:
        return self.score >= level


@dataclass(frozen=True)
class NoulAnswer:
    noul: float
    type: str = "noul"

    @property
    def probability(self) -> float:
        return self.noul

    def is_yes(self, threshold: float = 0.5) -> bool:
        return self.noul >= threshold

    def is_no(self, threshold: float = 0.5) -> bool:
        return self.noul <= 1 - threshold

    def is_uncertain(self, margin: float = 0.2) -> bool:
        return abs(self.noul - 0.5) < margin


Answer = Union[ChoiceAnswer, ScoreAnswer, NoulAnswer]


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass(frozen=True)
class DecisionResult:
    model: str
    answers: dict
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0

    def __getitem__(self, question_id: str) -> Answer:
        return self.answers[question_id]

    def get(self, question_id: str, default=None):
        return self.answers.get(question_id, default)

    def choice(self, question_id: str) -> ChoiceAnswer:
        return self._typed(question_id, ChoiceAnswer)

    def score(self, question_id: str) -> ScoreAnswer:
        return self._typed(question_id, ScoreAnswer)

    def noul(self, question_id: str) -> NoulAnswer:
        return self._typed(question_id, NoulAnswer)

    def _typed(self, question_id: str, kind):
        answer = self.answers.get(question_id)
        if not isinstance(answer, kind):
            raise KeyError(f"No {kind.__name__} for question {question_id!r}.")
        return answer


def parse_answer(raw: Mapping[str, Any]) -> Answer:
    kind = raw.get("type")
    if kind == "choice":
        return ChoiceAnswer(
            choice=str(raw.get("choice")),
            probabilities={str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()},
            confidence=_maybe_float(raw.get("confidence")),
        )
    if kind == "score":
        return ScoreAnswer(
            score=float(raw.get("score", 0.0)),
            legend={str(k): v for k, v in (raw.get("legend") or {}).items()},
            probabilities={str(k): float(v) for k, v in (raw.get("probabilities") or {}).items()},
            confidence=_maybe_float(raw.get("confidence")),
        )
    if kind == "noul":
        return NoulAnswer(noul=float(raw.get("noul", 0.0)))
    raise JevValidationError(f"Unknown answer type {kind!r} in Jev response.")


def _maybe_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
