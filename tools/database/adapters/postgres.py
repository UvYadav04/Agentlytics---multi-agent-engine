import asyncio
import logging
import time
from typing import Any

from shared.external_db.errors import ConnectionStringError, DatabaseInspectionError
from shared.external_db.queries import QueryResult, QueryValidationError, run_postgres_queries
from shared.external_db.service import inspect_database
from tools.database.adapters.base import BaseDatabaseAdapter

logger = logging.getLogger("tools.database.adapters.postgres")


class PostgresAdapter(BaseDatabaseAdapter):
    async def inspect_schema(self) -> dict[str, Any]:
        try:
            _, result = await inspect_database("postgresql", self.connection_string)
            return {
                "success": True,
                "database_type": "postgresql",
                "database": result.database,
                "schema": result.schema_document(),
                "warnings": result.warnings,
            }
        except (ConnectionStringError, DatabaseInspectionError, Exception) as exc:
            logger.warning("Postgres schema inspection error: %s", exc)
            return {
                "success": False,
                "database_type": "postgresql",
                "error": str(exc),
            }

    async def execute_query(self, query: dict) -> dict[str, Any]:
        sql = query.get("statement") or query.get("sql")
        if not sql or not isinstance(sql, str):
            return {
                "success": False,
                "database_type": "postgresql",
                "error": "The query must include a SQL 'statement' string, e.g. {'statement': 'SELECT * FROM users LIMIT 10'}.",
            }

        start = time.perf_counter()
        try:
            results = await asyncio.to_thread(
                run_postgres_queries, self.connection_string, [{"sql": sql}], None
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("Postgres query execution exception: %s", exc)
            return {
                "success": False,
                "database_type": "postgresql",
                "error": str(exc),
                "execution_time_ms": round(elapsed_ms, 2),
            }

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not results:
            return {
                "success": False,
                "database_type": "postgresql",
                "error": "No result returned from PostgreSQL.",
                "execution_time_ms": round(elapsed_ms, 2),
            }

        first_res = results[0]
        if isinstance(first_res, (QueryValidationError, DatabaseInspectionError, Exception)):
            return {
                "success": False,
                "database_type": "postgresql",
                "error": str(first_res),
                "execution_time_ms": round(elapsed_ms, 2),
            }

        if isinstance(first_res, QueryResult):
            return {
                "success": True,
                "database_type": "postgresql",
                "columns": [{"name": col, "data_type": "unknown"} for col in first_res.columns],
                "rows": first_res.rows,
                "row_count": len(first_res.rows),
                "truncated": first_res.truncated,
                "execution_time_ms": round(elapsed_ms, 2),
            }

        return {
            "success": False,
            "database_type": "postgresql",
            "error": "Unexpected result structure from query execution.",
            "execution_time_ms": round(elapsed_ms, 2),
        }

    async def close(self) -> None:
        pass

