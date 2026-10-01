import logging
from typing import Any, Mapping, Sequence

from .client import JevClient
from .errors import JevError
from .models import Choice, ChoiceAnswer, DecisionResult, Noul, NoulAnswer, Question, Score, ScoreAnswer, State
from .settings import JevSettings

logger = logging.getLogger("jev")

_default_client: JevClient | None = None

Questions = Mapping[str, Question | Mapping[str, Any]]


def get_jev() -> JevClient:
    global _default_client
    if _default_client is None:
        _default_client = JevClient()
    return _default_client


def configure_jev(settings: JevSettings | None = None, **kwargs) -> JevClient:
    global _default_client
    _default_client = JevClient(settings, **kwargs)
    return _default_client


def jev_available() -> bool:
    return get_jev().available


async def decide(state: State, questions: Questions, *, model: str | None = None, timeout: float | None = None) -> DecisionResult:
    return await get_jev().decide(state, questions, model=model, timeout=timeout)


def decide_sync(state: State, questions: Questions, *, model: str | None = None, timeout: float | None = None) -> DecisionResult:
    return get_jev().decide_sync(state, questions, model=model, timeout=timeout)


async def try_decide(
    state: State, questions: Questions, *, model: str | None = None, timeout: float | None = None,
) -> DecisionResult | None:
    client = get_jev()
    if not client.available:
        return None
    try:
        return await client.decide(state, questions, model=model, timeout=timeout)
    except JevError as exc:
        logger.warning("jev: falling back after error: %s", exc)
        return None


def try_decide_sync(
    state: State, questions: Questions, *, model: str | None = None, timeout: float | None = None,
) -> DecisionResult | None:
    client = get_jev()
    if not client.available:
        return None
    try:
        return client.decide_sync(state, questions, model=model, timeout=timeout)
    except JevError as exc:
        logger.warning("jev: falling back after error: %s", exc)
        return None


_SINGLE = "decision"


async def choose(
    state: State, instructions, options: Mapping[str, Any] | Sequence[str], *,
    model: str | None = None, timeout: float | None = None,
) -> ChoiceAnswer:
    result = await decide(state, {_SINGLE: Choice(instructions, options)}, model=model, timeout=timeout)
    return result.choice(_SINGLE)


async def rate(
    state: State, instructions, levels: Sequence[Any], *, model: str | None = None, timeout: float | None = None,
) -> ScoreAnswer:
    result = await decide(state, {_SINGLE: Score(instructions, levels)}, model=model, timeout=timeout)
    return result.score(_SINGLE)


async def yes_no(
    state: State, instructions, *, yes: Any = None, no: Any = None,
    model: str | None = None, timeout: float | None = None,
) -> NoulAnswer:
    result = await decide(state, {_SINGLE: Noul(instructions, yes=yes, no=no)}, model=model, timeout=timeout)
    return result.noul(_SINGLE)


def choose_sync(
    state: State, instructions, options: Mapping[str, Any] | Sequence[str], *,
    model: str | None = None, timeout: float | None = None,
) -> ChoiceAnswer:
    return decide_sync(state, {_SINGLE: Choice(instructions, options)}, model=model, timeout=timeout).choice(_SINGLE)


def rate_sync(
    state: State, instructions, levels: Sequence[Any], *, model: str | None = None, timeout: float | None = None,
) -> ScoreAnswer:
    return decide_sync(state, {_SINGLE: Score(instructions, levels)}, model=model, timeout=timeout).score(_SINGLE)


def yes_no_sync(
    state: State, instructions, *, yes: Any = None, no: Any = None,
    model: str | None = None, timeout: float | None = None,
) -> NoulAnswer:
    return decide_sync(state, {_SINGLE: Noul(instructions, yes=yes, no=no)}, model=model, timeout=timeout).noul(_SINGLE)
