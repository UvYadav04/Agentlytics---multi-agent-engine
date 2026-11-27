import logging
import threading
import time

import chromadb

from config import get_settings
from shared.observability import get_tracer
from vectordb.base import BaseVectorStore, VectorStoreError
from vectordb.schema import ChunkRecord

logger = logging.getLogger("vectordb.chroma")

_TRANSIENT_KEYWORDS = (
    "500",
    "502",
    "503",
    "504",
    "429",
    "internalerror",
    "internal server error",
    "rate limit",
    "timed out",
    "timeout",
    "temporarily unavailable",
    "connection",
    "reset by peer",
    "broken pipe",
)


def _extract_status_code(exc: Exception) -> int | None:
    for attr in ("status_code", "code"):
        val = getattr(exc, attr, None)
        if isinstance(val, int):
            return val
    resp = getattr(exc, "response", None)
    if resp is not None and isinstance(getattr(resp, "status_code", None), int):
        return resp.status_code
    return None


def _is_transient_error(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
        return True
    status = _extract_status_code(exc)
    if status is not None:
        return status == 429 or status >= 500
    msg = f"{type(exc).__name__} {exc}".lower()
    if any(code in msg for code in ("400", "401", "403", "404", "413", "422")):
        return False
    return any(kw in msg for kw in _TRANSIENT_KEYWORDS)

_tracer = get_tracer("analyzerEngine.rag")

try:
    from monitoring.wrappers.rag import trace_retriever
except Exception:
    def trace_retriever(*args, **kwargs):
        return lambda fn: fn


class ChromaVectorStore(BaseVectorStore):
    DEFAULT_MAX_RETRIES = 2
    DEFAULT_BACKOFF_SECONDS = 0.25

    def __init__(
        self,
        collection_name: str = "chunks",
        client=None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
    ):
        self.collection_name = collection_name
        self._client = client
        self._external_client = client is not None
        self._collection = None
        self._init_lock = threading.RLock()
        self._max_retries = max(0, max_retries)
        self._retry_backoff_seconds = max(0.0, retry_backoff_seconds)

    @property
    def client(self):
        if self._client is None:
            with self._init_lock:
                if self._client is None:
                    settings = get_settings()
                    missing = [
                        key for key in ("CHROMA_API_KEY", "CHROMA_TENANT", "CHROMA_DATABASE")
                        if not settings.get(key)
                    ]
                    if missing:
                        raise VectorStoreError(
                            f"Chroma Cloud is not configured (missing {', '.join(missing)})"
                        )
                    self._client = chromadb.CloudClient(
                        tenant=settings.CHROMA_TENANT,
                        database=settings.CHROMA_DATABASE,
                        api_key=settings.CHROMA_API_KEY,
                    )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            with self._init_lock:
                if self._collection is None:
                    client = self.client
                    self._collection = client.get_or_create_collection(name=self.collection_name)
        return self._collection

    def _reset_uninjected_connection(self) -> None:
        with self._init_lock:
            self._collection = None
            if not self._external_client:
                self._client = None

    def _run_with_retry(self, op_name: str, fn):
        last_exc: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                return fn()
            except VectorStoreError:
                raise
            except Exception as exc:
                last_exc = exc
                status_code = _extract_status_code(exc)
                transient = _is_transient_error(exc)
                if transient and attempt < self._max_retries:
                    delay = self._retry_backoff_seconds * (2 ** attempt)
                    logger.warning(
                        "chroma %s failed (attempt %d/%d, retrying in %.2fs): %s",
                        op_name, attempt + 1, self._max_retries + 1, delay, exc,
                    )
                    if self._collection is None:
                        self._reset_uninjected_connection()
                    if delay > 0:
                        time.sleep(delay)
                    continue
                if self._collection is None:
                    self._reset_uninjected_connection()
                logger.warning("chroma %s failed: %s", op_name, exc)
                raise VectorStoreError(
                    f"Chroma {op_name} failed: {exc}",
                    status_code=status_code,
                ) from exc

        raise VectorStoreError(f"Chroma {op_name} failed: {last_exc}") from last_exc

    def warmup(self) -> None:
        """Eagerly establish the Chroma Cloud client and collection (safe to run in a background thread)."""
        self._run_with_retry("warmup", lambda: self.collection)

    def upsert(self, chunks: list) -> list:
        if not chunks:
            return []
        start = time.perf_counter()
        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [
            {"file_id": c.file_id, "workspace_id": c.workspace_id, **c.metadata}
            for c in chunks
        ]

        self._run_with_retry(
            "upsert",
            lambda: self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas),
        )
        logger.info("chroma upsert took %.3fs (%d chunks)", time.perf_counter() - start, len(ids))
        return ids

    @trace_retriever(name="chroma_query", component="rag.chroma")
    def query(self, query_text: str, top_k: int, filters: dict = None) -> list:
        start = time.perf_counter()
        with _tracer.start_as_current_span(
            "rag.retrieval", attributes={"rag.top_k": top_k, "rag.query_chars": len(query_text or "")},
        ) as span:
            try:
                result = self._run_with_retry(
                    "query",
                    lambda: self.collection.query(
                        query_texts=[query_text],
                        n_results=top_k,
                        where=filters,
                        include=["documents", "metadatas", "distances"],
                    ),
                )
            except Exception as exc:
                span.record_exception(exc)
                raise
            chunks = self._to_chunks(result, batched=True)
            span.set_attribute("rag.results_returned", len(chunks))
            logger.info(
                "chroma query took %.3fs (top_k=%d, returned=%d)",
                time.perf_counter() - start, top_k, len(chunks),
            )
            return chunks

    def get_by_id(self, ids: list) -> list:
        if not ids:
            return []
        start = time.perf_counter()
        result = self._run_with_retry("get_by_id", lambda: self.collection.get(ids=ids))
        chunks = self._to_chunks(result, batched=False)
        logger.info("chroma get_by_id took %.3fs (%d ids)", time.perf_counter() - start, len(ids))
        return chunks

    def get_by_filter(self, filters: dict) -> list:
        start = time.perf_counter()
        result = self._run_with_retry("get_by_filter", lambda: self.collection.get(where=filters))
        chunks = self._to_chunks(result, batched=False)
        logger.info(
            "chroma get_by_filter took %.3fs (returned=%d)", time.perf_counter() - start, len(chunks),
        )
        return chunks

    def delete(self, ids: list) -> None:
        if not ids:
            return
        start = time.perf_counter()
        self._run_with_retry("delete", lambda: self.collection.delete(ids=ids))
        logger.info("chroma delete took %.3fs (%d ids)", time.perf_counter() - start, len(ids))

    def _to_chunks(self, result: dict, batched: bool) -> list:
        if not result:
            return []
        ids = (result.get("ids") or [[]])[0] if batched else (result.get("ids") or [])
        documents = (result.get("documents") or [[]])[0] if batched else (result.get("documents") or [])
        metadatas = (result.get("metadatas") or [[]])[0] if batched else (result.get("metadatas") or [])
        distances = result.get("distances")
        distances = (distances[0] if batched else distances) if distances else None

        chunks = []
        for i, chunk_id in enumerate(ids):
            meta = dict(metadatas[i] or {}) if i < len(metadatas) else {}
            doc = (documents[i] or "") if i < len(documents) else ""
            file_id = meta.pop("file_id", "")
            workspace_id = meta.pop("workspace_id", "")
            if distances and i < len(distances):
                meta["score"] = distances[i]
            chunks.append(ChunkRecord(
                chunk_id=chunk_id,
                file_id=file_id,
                workspace_id=workspace_id,
                text=doc,
                metadata=meta,
            ))
        return chunks
