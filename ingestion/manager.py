import logging
from typing import Callable, Optional

from ingestion import registry
from ingestion.models import IngestionResult
from ingestion.storage.base import BaseObjectStore
from vectordb.base import BaseVectorStore

logger = logging.getLogger("ingestion.manager")


class IngestionManager:
    def __init__(self, storage: BaseObjectStore, vector_store: BaseVectorStore):
        self.storage = storage
        self.vector_store = vector_store

    def ingest_file(
        self, file_path: str, workspace_id: str, file_id: str,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> IngestionResult:
        try:
            ingestor_cls = registry.get_ingestor_for(file_path)
        except ValueError as exc:
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=[str(exc)],
            )

        ingestor = ingestor_cls(storage=self.storage, vector_store=self.vector_store)

        if not ingestor.validate(file_path):
            errors = getattr(ingestor, "errors", None) or ["validation failed"]
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=errors,
            )

        try:
            return ingestor.ingest(file_path, workspace_id, file_id, progress_callback=progress_callback)
        except Exception as exc:
            logger.exception("ingest_file: %s ingestor raised for file %s", ingestor_cls.__name__, file_id)
            return IngestionResult(
                file_id=file_id,
                workspace_id=workspace_id,
                status="failed",
                output_ref="",
                schema_summary={},
                errors=[f"Unexpected error during ingestion: {exc}"],
            )
