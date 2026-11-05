from abc import ABC, abstractmethod
from typing import Any


class BaseDatabaseAdapter(ABC):
    def __init__(self, connection_string: str, database: str | None = None):
        self.connection_string = connection_string
        self.database = database

    @abstractmethod
    async def inspect_schema(self) -> dict[str, Any]:
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

