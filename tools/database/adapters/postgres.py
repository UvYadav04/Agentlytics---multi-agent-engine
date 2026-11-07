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
    async def inspect_schema(self, table_or_collection: str | None = None) -> dict[str, Any]:
        try:
            _, result = await inspect_database("postgresql", self.connection_string)
            if not result.objects:
                return {
                    "success": True,
                    "database_type": "postgresql",
                    "database": result.database,
                    "tables": [],
                    "message": "Connected, but no tables were found in this database.",
                }

            if table_or_collection:
                target = table_or_collection.strip().lower()
                bare_target = target.split(".")[-1]
                matched = next(
                    (
                        obj for obj in result.objects
                        if obj.get("name", "").lower() in (target, bare_target)
                        or f"{obj.get('schema', '').lower()}.{obj.get('name', '').lower()}" == target
                    ),
                    None,
                )
                if not matched:
                    available = [
                        f"{obj.get('schema')}.{obj.get('name')}" if obj.get("schema") else obj.get("name")
                        for obj in result.objects
                    ]
                    return {
                        "success": False,
                        "database_type": "postgresql",
                        "error": f"Table '{table_or_collection}' not found. Available tables: {available}",
                    }

                raw_fields = matched.get("fields", [])
                compact_fields = [
                    {
                        "name": f.get("name"),
                        "type": f.get("type"),
                        "nullable": f.get("nullable"),
                        "primary_key": f.get("primary_key", False),
                    }
                    for f in raw_fields[:100]
                ]
                return {
                    "success": True,
                    "database_type": "postgresql",
                    "database": result.database,
                    "schema": matched.get("schema"),
                    "table": matched.get("name"),
                    "row_count": matched.get("row_count"),
                    "fields": compact_fields,
                    "foreign_keys": matched.get("foreign_keys", []),
                    "total_columns": len(raw_fields),
                }

            if len(result.objects) <= 2:
                tables = []
                for obj in result.objects:
                    raw_fields = obj.get("fields", [])
                    tables.append({
                        "table": f"{obj.get('schema')}.{obj.get('name')}" if obj.get("schema") else obj.get("name"),
                        "row_count": obj.get("row_count"),
                        "fields": [
                            {
                                "name": f.get("name"),
                                "type": f.get("type"),
                                "primary_key": f.get("primary_key", False),
                            }
                            for f in raw_fields[:100]
                        ],
                        "foreign_keys": obj.get("foreign_keys", []),
                    })
                return {
                    "success": True,
                    "database_type": "postgresql",
                    "database": result.database,
                    "tables": tables,
                }

            tables_summary = [
                {
                    "table": f"{obj.get('schema')}.{obj.get('name')}" if obj.get("schema") else obj.get("name"),
                    "row_count": obj.get("row_count"),
                    "key_columns": [f.get("name") for f in obj.get("fields", [])[:10]],
                    "total_columns": len(obj.get("fields", [])),
                }
                for obj in result.objects
            ]
            return {
                "success": True,
                "database_type": "postgresql",
                "database": result.database,
                "tables": tables_summary,
                "total_tables": len(result.objects),
                "instruction": "Call inspect_schema with table_or_collection='<table_name>' to view detailed columns and keys for a specific table.",
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

