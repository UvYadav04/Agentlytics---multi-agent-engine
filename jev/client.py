import asyncio
import logging
import random
import threading
import time
from typing import Any, Mapping

import httpx

from .errors import (
    JevAPIError,
    JevError,
    JevAuthenticationError,
    JevConfigError,
    JevRequestError,
    JevTimeoutError,
    JevUnavailableError,
    JevValidationError,
)
from .models import Choice, ChoiceAnswer, DecisionResult, Noul, Question, Score, State, Usage, parse_answer
from .settings import JevSettings

logger = logging.getLogger("jev")

ENDPOINT = "/v1/systemone"
RETRYABLE_STATUS = {408, 429, 500, 502, 503, 504, 529}
MAX_RETRY_AFTER_SECONDS = 10.0
_QUESTION_TYPES = {"choice", "score", "noul"}


def _question_payload(question_id: str, question: Question | Mapping[str, Any]) -> dict:
    if isinstance(question, (Choice, Score, Noul)):
        return question.to_payload()
    if isinstance(question, Mapping):
        kind = question.get("type")
        if kind not in _QUESTION_TYPES:
            raise JevValidationError(f"Question {question_id!r} has unknown type {kind!r}.")
        if not question.get("instructions"):
            raise JevValidationError(f"Question {question_id!r} needs instructions.")
        if kind in ("choice", "score") and not question.get("criteria"):
            raise JevValidationError(f"Question {question_id!r} of type {kind} needs criteria.")
        return dict(question)
    raise JevValidationError(f"Question {question_id!r} must be a Choice, Score, Noul or dict.")


def build_payload(state: State, questions: Mapping[str, Question | Mapping[str, Any]], model: str) -> dict:
    if state is None or (isinstance(state, str) and not state.strip()):
        raise JevValidationError("Jev needs a non-empty state.")
    if not questions:
        raise JevValidationError("Jev needs at least one question.")
    payload_questions = {}
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id.strip():
            raise JevValidationError("Question ids must be non-empty strings.")
        payload_questions[question_id] = _question_payload(question_id, question)
    return {"state": state, "model": model, "questions": payload_questions}


def _error_detail(response: httpx.Response):
    try:
        return response.json()
    except ValueError:
        return response.text[:500]


def _raise_for_status(response: httpx.Response) -> None:
    status = response.status_code
    if status < 400:
        return
    detail = _error_detail(response)
    if status in (401, 403):
        raise JevAuthenticationError("Jev rejected the API key.", status, detail)
    if status in RETRYABLE_STATUS:
        raise JevUnavailableError(f"Jev is temporarily unavailable (HTTP {status}).", status, detail)
    if 400 <= status < 500:
        raise JevRequestError(f"Jev rejected the request (HTTP {status}).", status, detail)
    raise JevAPIError(f"Jev returned HTTP {status}.", status, detail)


def _retry_delay(attempt: int, backoff: float, response: httpx.Response | None) -> float:
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return min(MAX_RETRY_AFTER_SECONDS, max(0.0, float(retry_after)))
            except ValueError:
                pass
    return backoff * (2 ** attempt) + random.uniform(0, backoff)


def parse_result(body: Mapping[str, Any], payload: dict, latency_ms: float) -> DecisionResult:
    raw_answers = body.get("answers")
    if not isinstance(raw_answers, Mapping):
        raise JevAPIError("Jev response has no answers.", detail=body)
    answers = {}
    for question_id, question in payload["questions"].items():
        raw = raw_answers.get(question_id)
        if not isinstance(raw, Mapping):
            raise JevAPIError(f"Jev response is missing an answer for {question_id!r}.", detail=body)
        try:
            answer = parse_answer(raw)
        except (JevValidationError, TypeError, ValueError) as exc:
            raise JevAPIError(f"Jev returned an unreadable answer for {question_id!r}: {exc}", detail=raw) from exc
        if answer.type != question["type"]:
            raise JevAPIError(f"Jev answered {question_id!r} with type {answer.type}, expected {question['type']}.")
        if isinstance(answer, ChoiceAnswer) and answer.choice not in question["criteria"]:
            raise JevAPIError(f"Jev chose {answer.choice!r} for {question_id!r}, which is not one of the options.")
        answers[question_id] = answer
    usage = body.get("usage") or {}
    return DecisionResult(
        model=str(body.get("model") or payload["model"]),
        answers=answers,
        usage=Usage(
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        ),
        latency_ms=latency_ms,
    )


class JevClient:
    def __init__(
        self,
        settings: JevSettings | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sync_transport: httpx.BaseTransport | None = None,
    ):
        self.settings = settings or JevSettings.from_env()
        self._transport = transport
        self._sync_transport = sync_transport
        self._async_client: httpx.AsyncClient | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._sync_client: httpx.Client | None = None
        self._lock = threading.Lock()

    @property
    def available(self) -> bool:
        return self.settings.configured

    def _headers(self) -> dict:
        if not self.settings.enabled:
            raise JevConfigError("Jev is disabled (JEV_ENABLED=false).")
        if not self.settings.api_key:
            raise JevConfigError("TYPESAFE_API_KEY is not configured.")
        return {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _async_http(self) -> httpx.AsyncClient:
        loop = asyncio.get_running_loop()
        if self._async_client is None or self._async_loop is not loop or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.settings.base_url, timeout=self.settings.timeout_seconds, transport=self._transport,
            )
            self._async_loop = loop
        return self._async_client

    def _sync_http(self) -> httpx.Client:
        with self._lock:
            if self._sync_client is None or self._sync_client.is_closed:
                self._sync_client = httpx.Client(
                    base_url=self.settings.base_url, timeout=self.settings.timeout_seconds,
                    transport=self._sync_transport,
                )
            return self._sync_client

    def _prepare(self, state, questions, model, timeout):
        headers = self._headers()
        payload = build_payload(state, questions, model or self.settings.model)
        return headers, payload, timeout if timeout is not None else self.settings.timeout_seconds

    def _log(self, payload: dict, result: DecisionResult | None, attempts: int, error: Exception | None = None):
        question_ids = ",".join(payload["questions"])
        if result is not None:
            logger.info(
                "jev: questions=%s model=%s attempts=%d latency_ms=%.1f input_tokens=%d output_tokens=%d",
                question_ids, result.model, attempts, result.latency_ms,
                result.usage.input_tokens, result.usage.output_tokens,
            )
        else:
            logger.warning(
                "jev: questions=%s model=%s attempts=%d failed: %s",
                question_ids, payload["model"], attempts, error,
            )

    async def decide(
        self,
        state: State,
        questions: Mapping[str, Question | Mapping[str, Any]],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> DecisionResult:
        headers, payload, request_timeout = self._prepare(state, questions, model, timeout)
        http = self._async_http()
        attempts = 0
        start = time.perf_counter()
        while True:
            attempts += 1
            response = None
            try:
                response = await http.post(ENDPOINT, json=payload, headers=headers, timeout=request_timeout)
                _raise_for_status(response)
                result = parse_result(response.json(), payload, (time.perf_counter() - start) * 1000)
                self._log(payload, result, attempts)
                return result
            except (httpx.TimeoutException, httpx.TransportError, JevUnavailableError) as exc:
                if attempts > self.settings.max_retries:
                    error = _final_error(exc)
                    self._log(payload, None, attempts, error)
                    raise error from exc
                await asyncio.sleep(_retry_delay(attempts - 1, self.settings.backoff_seconds, response))
            except JevAPIError as exc:
                self._log(payload, None, attempts, exc)
                raise
            except ValueError as exc:
                error = JevAPIError("Jev returned a response that could not be read.")
                self._log(payload, None, attempts, error)
                raise error from exc

    def decide_sync(
        self,
        state: State,
        questions: Mapping[str, Question | Mapping[str, Any]],
        *,
        model: str | None = None,
        timeout: float | None = None,
    ) -> DecisionResult:
        headers, payload, request_timeout = self._prepare(state, questions, model, timeout)
        http = self._sync_http()
        attempts = 0
        start = time.perf_counter()
        while True:
            attempts += 1
            response = None
            try:
                response = http.post(ENDPOINT, json=payload, headers=headers, timeout=request_timeout)
                _raise_for_status(response)
                result = parse_result(response.json(), payload, (time.perf_counter() - start) * 1000)
                self._log(payload, result, attempts)
                return result
            except (httpx.TimeoutException, httpx.TransportError, JevUnavailableError) as exc:
                if attempts > self.settings.max_retries:
                    error = _final_error(exc)
                    self._log(payload, None, attempts, error)
                    raise error from exc
                time.sleep(_retry_delay(attempts - 1, self.settings.backoff_seconds, response))
            except JevAPIError as exc:
                self._log(payload, None, attempts, exc)
                raise
            except ValueError as exc:
                error = JevAPIError("Jev returned a response that could not be read.")
                self._log(payload, None, attempts, error)
                raise error from exc

    async def aclose(self) -> None:
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
        self._async_client = None

    def close(self) -> None:
        if self._sync_client is not None and not self._sync_client.is_closed:
            self._sync_client.close()
        self._sync_client = None


def _final_error(exc: Exception) -> JevError:
    if isinstance(exc, JevUnavailableError):
        return exc
    if isinstance(exc, httpx.TimeoutException):
        return JevTimeoutError("Jev did not respond in time.")
    return JevUnavailableError(f"Could not reach Jev: {exc.__class__.__name__}.")
