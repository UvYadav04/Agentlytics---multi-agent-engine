import logging
from typing import Optional

from tools.database.adapters.base import BaseDatabaseAdapter
from tools.database.adapters.mongodb import MongoAdapter
from tools.database.adapters.postgres import PostgresAdapter

logger = logging.getLogger("tools.database.connection_manager")


class ConnectionManager:
    """
    Manages database adapters and resolves connection credentials safely
    within the current investigation execution context.
    """

    def __init__(self, database_access):
        self.database_access = database_access
        self._adapters: dict[str, BaseDatabaseAdapter] = {}
        self._connection_info: dict[str, dict] = {}

    async def get_adapter(self, connection_id: str) -> tuple[dict, BaseDatabaseAdapter]:
        if connection_id in self._adapters:
            return self._connection_info[connection_id], self._adapters[connection_id]

        if self.database_access is None:
            raise LookupError("No database access provider is configured for this workspace.")

        connection_info, connection_string = await self.database_access.resolve(connection_id)
        db_type = connection_info.get("db_type")
        database = connection_info.get("database")

        if db_type == "postgresql":
            adapter = PostgresAdapter(connection_string=connection_string, database=database)
        elif db_type == "mongodb":
            adapter = MongoAdapter(connection_string=connection_string, database=database)
        else:
            raise LookupError(f"Unsupported database type: '{db_type}'. Supported: 'postgresql', 'mongodb'.")

        self._adapters[connection_id] = adapter
        self._connection_info[connection_id] = connection_info
        return connection_info, adapter

    async def close_all(self) -> None:
        for conn_id, adapter in self._adapters.items():
            try:
                await adapter.close()
            except Exception as exc:
                logger.debug("Error closing adapter for connection %s: %s", conn_id, exc)
        self._adapters.clear()
        self._connection_info.clear()

