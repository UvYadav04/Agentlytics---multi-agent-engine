import logging
import time
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import requests

from config import get_settings

DEFAULT_MODEL = "BAAI/bge-reranker-base"
HF_INFERENCE_URL = "https://api-inference.huggingface.co/models/{model}"

logger = logging.getLogger("vectordb.reranker")

try:
    from monitoring.wrappers.rag import trace_reranker
except Exception:
    def trace_reranker(*args, **kwargs):
        return lambda fn: fn


def compute_ensemble_cutoff(scores: list[float], min_top_score: float = 0.005) -> float:
    """Computes an ensemble cutoff line averaging 3 distribution-aware methods:
    1. Maximum Gap: Midpoint of the largest adjacent score drop.
    2. Otsu's 1D Threshold: Maximizes between-class variance between high and low clusters.
    3. Power-Weighted Threshold: Quadratic center-of-mass weighting higher scores more heavily.

    Returns the consensus average cutoff line.
    """
    if not scores or len(scores) <= 1:
        return 0.0

    arr = np.array(scores, dtype=float)
    if arr[0] < min_top_score:
        return float(arr[0] + 1e-4)

    # Uniform high plateau: keep all
    spread = arr[0] - arr[-1]
    if spread < 0.05 and arr[0] > 0.5:
        return float(arr[-1] - 0.01)

    # 1. Max Gap Threshold
    gaps = arr[:-1] - arr[1:]
    max_idx = int(np.argmax(gaps))
    t_gap = (arr[max_idx] + arr[max_idx + 1]) / 2.0

    # 2. Otsu 1D Threshold (maximizes between-class variance)
    midpoints = (arr[:-1] + arr[1:]) / 2.0
    best_t_otsu = midpoints[0]
    best_var = -1.0
    for m in midpoints:
        high = arr[arr >= m]
        low = arr[arr < m]
        if len(high) == 0 or len(low) == 0:
            continue
        w0 = len(high) / len(arr)
        w1 = len(low) / len(arr)
        mu0 = np.mean(high)
        mu1 = np.mean(low)
        var_between = w0 * w1 * ((mu0 - mu1) ** 2)
        if var_between > best_var:
            best_var = var_between
            best_t_otsu = m
    t_otsu = best_t_otsu

    # 3. Power-Weighted Threshold (quadratic weighting towards higher scores)
    norm_arr = arr - arr.min()
    if np.all(norm_arr == 0):
        t_power = arr[0]
    else:
        weights = (norm_arr + 1e-6) ** 2
        weighted_mean = np.sum(arr * weights) / np.sum(weights)
        t_power = 0.5 * (weighted_mean + arr.min())

    # Consensus Average Line
    t_ensemble = (t_gap + t_otsu + t_power) / 3.0
    return float(t_ensemble)


class BaseReranker(ABC):
    @abstractmethod
    def rank(self, query: str, chunks: list, top_k: int = None, dynamic_cutoff: bool = None) -> list:
        raise NotImplementedError

    @staticmethod
    def _apply_dynamic_cutoff(ranked: list, min_top_score: float = 0.005) -> list:
        if len(ranked) <= 1:
            return ranked
        scores = [getattr(c, "score", 0.0) for c in ranked]
        cutoff = compute_ensemble_cutoff(scores, min_top_score=min_top_score)
        filtered = [c for c in ranked if getattr(c, "score", 0.0) >= cutoff]
        if not filtered and ranked and getattr(ranked[0], "score", 0.0) >= min_top_score:
            filtered = [ranked[0]]
        logger.info(
            "dynamic ensemble cutoff line: %.4f (kept %d/%d chunks)",
            cutoff, len(filtered), len(ranked),
        )
        return filtered


_CROSS_ENCODER_CACHE: dict[str, Any] = {}


def get_cross_encoder(model_name: str) -> Any:
    """Module-level singleton cache for CrossEncoder models to avoid redundant reloading."""
    if model_name in _CROSS_ENCODER_CACHE:
        return _CROSS_ENCODER_CACHE[model_name]
    from sentence_transformers import CrossEncoder

    try:
        model = CrossEncoder(model_name, local_files_only=True)
    except Exception:
        model = CrossEncoder(model_name, local_files_only=False)
    _CROSS_ENCODER_CACHE[model_name] = model
    return model


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str = None, timeout: float = 30.0, use_local: bool = True, use_dynamic_cutoff: bool = None):
        settings = get_settings()
        self.model_name = model_name or settings.get("RERANKER_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
        self.timeout = timeout
        self.use_local = use_local
        self._local_model = None

        if use_dynamic_cutoff is None:
            raw_dyn = str(settings.get("RERANKER_DYNAMIC_CUTOFF", "true")).strip().lower()
            use_dynamic_cutoff = raw_dyn in ("true", "1", "yes")
        self.use_dynamic_cutoff = use_dynamic_cutoff

        if self.use_local:
            try:
                self._local_model = get_cross_encoder(self.model_name)
            except Exception as exc:
                logger.warning("Failed to load local CrossEncoder (%s), falling back to HF API", exc)
                self._local_model = None

        self.api_token = settings.get("HF_API_TOKEN")
        self.url = HF_INFERENCE_URL.format(model=self.model_name)

    @trace_reranker(name="cross_encoder_rerank", component="rag.reranker")
    def rank(self, query: str, chunks: list, top_k: int = None, dynamic_cutoff: bool = None) -> list:
        if not chunks:
            return []

        start = time.perf_counter()

        if self._local_model is not None:
            pairs = [(query, chunk.text) for chunk in chunks]
            raw_scores = self._local_model.predict(pairs)
            scores = [float(s) for s in raw_scores]
        else:
            if not self.api_token:
                raise RuntimeError("HF_API_TOKEN is required when local CrossEncoder is unavailable")
            payload = {
                "inputs": [{"text": query, "text_pair": chunk.text} for chunk in chunks],
                "options": {"wait_for_model": True},
            }
            response = requests.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_token}"},
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            results = response.json()
            scores = [self._extract_score(r) for r in results]

        for s, c in zip(scores, chunks):
            setattr(c, "score", float(s))
        ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]

        should_cutoff = self.use_dynamic_cutoff if dynamic_cutoff is None else dynamic_cutoff
        if should_cutoff:
            ranked = self._apply_dynamic_cutoff(ranked)

        logger.info(
            "reranker call took %.3fs (%d chunks in, %d returned)",
            time.perf_counter() - start, len(chunks), len(ranked[:top_k] if top_k else ranked),
        )
        return ranked[:top_k] if top_k else ranked

    @staticmethod
    def _extract_score(result) -> float:
        entry = result[0] if isinstance(result, list) else result
        return float(entry["score"])


class DeepInfraReranker(BaseReranker):
    DEFAULT_MODEL = "Qwen/Qwen3-Reranker-4B"
    _URL = "https://api.deepinfra.com/v1/inference/{model}"

    def __init__(self, model_name: str = None, timeout: float = 30.0, use_dynamic_cutoff: bool = None):
        settings = get_settings()
        self.model_name = model_name or settings.get("RERANKER_MODEL", self.DEFAULT_MODEL) or self.DEFAULT_MODEL
        self.api_token = settings.get("DEEPINFRA_API_KEY")
        if not self.api_token:
            raise RuntimeError(
                "DEEPINFRA_API_KEY is required to use the DeepInfra reranker"
            )
        self.url = self._URL.format(model=self.model_name)
        self.timeout = timeout

        if use_dynamic_cutoff is None:
            raw_dyn = str(settings.get("RERANKER_DYNAMIC_CUTOFF", "true")).strip().lower()
            use_dynamic_cutoff = raw_dyn in ("true", "1", "yes")
        self.use_dynamic_cutoff = use_dynamic_cutoff

    @trace_reranker(name="deepinfra_rerank", component="rag.reranker")
    def rank(self, query: str, chunks: list, top_k: int = None, dynamic_cutoff: bool = None) -> list:
        if not chunks:
            return []

        start = time.perf_counter()
        payload = {
            "queries": [query] * len(chunks),
            "documents": [chunk.text for chunk in chunks],
        }
        response = requests.post(
            self.url,
            headers={"Authorization": f"Bearer {self.api_token}"},
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        scores = response.json()["scores"]
        for s, c in zip(scores, chunks):
            setattr(c, "score", float(s))
        ranked = [c for _, c in sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)]

        should_cutoff = self.use_dynamic_cutoff if dynamic_cutoff is None else dynamic_cutoff
        if should_cutoff:
            ranked = self._apply_dynamic_cutoff(ranked)

        logger.info(
            "deepinfra reranker call took %.3fs (%d chunks in, %d returned)",
            time.perf_counter() - start, len(chunks), len(ranked[:top_k] if top_k else ranked),
        )
        return ranked[:top_k] if top_k else ranked
