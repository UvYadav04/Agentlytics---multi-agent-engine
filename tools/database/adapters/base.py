from abc import ABC, abstractmethod
from typing import Any, Optional


class BaseDatabaseAdapter(ABC):
    def __init__(
        self,
        connection_string: str,
        database: str | None = None,
        cached_inspection: Optional[Any] = None,
    ):
        self.connection_string = connection_string
        self.database = database
        if isinstance(cached_inspection, dict):
            from shared.external_db.models import InspectionResult
            self._cached_inspection = InspectionResult(
                database=cached_inspection.get("database") or database,
                server_version=cached_inspection.get("server_version"),
                objects=cached_inspection.get("objects", []),
                warnings=cached_inspection.get("warnings", []),
                truncated=cached_inspection.get("truncated", False),
            )
        else:
            self._cached_inspection = cached_inspection

    @abstractmethod
    async def inspect_schema(self, table_or_collection: str | None = None) -> dict[str, Any]:
        """Inspect and return schema metadata (tables/collections, columns/fields)."""
        pass

    @abstractmethod
    async def execute_query(self, query: dict) -> dict[str, Any]:
        """Execute a query and return normalized results."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Clean up connection resources."""
        pass
