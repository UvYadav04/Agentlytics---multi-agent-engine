import asyncio
import logging
import time
from typing import Any

from shared.external_db.errors import ConnectionStringError, DatabaseInspectionError
from shared.external_db.queries import QueryResult, QueryValidationError, run_mongo_queries
from shared.external_db.service import inspect_database
from tools.database.adapters.base import BaseDatabaseAdapter

logger = logging.getLogger("tools.database.adapters.mongodb")


class MongoAdapter(BaseDatabaseAdapter):
    async def inspect_schema(self, table_or_collection: str | None = None) -> dict[str, Any]:
        try:
            _, result = await inspect_database("mongodb", self.connection_string)
            if not result.objects:
                return {
                    "success": True,
                    "database_type": "mongodb",
                    "database": result.database,
                    "collections": [],
                    "message": "Connected, but no collections were found in this database.",
                }

            if table_or_collection:
                target = table_or_collection.strip().lower()
                bare_target = target.split(".")[-1]
                db_name = str(result.database or "").lower()
                matched = next(
                    (
                        obj for obj in result.objects
                        if obj.get("name", "").lower() in (target, bare_target)
                        or (db_name and f"{db_name}.{obj.get('name', '').lower()}" == target)
                        or (len(bare_target) > 3 and obj.get("name", "").lower().rstrip("s") == bare_target.rstrip("s"))
                    ),
                    None,
                )
                if not matched:
                    available = [obj.get("name") for obj in result.objects]
                    return {
                        "success": False,
                        "database_type": "mongodb",
                        "error": f"Collection '{table_or_collection}' not found. Available collections: {available}",
                    }

                raw_fields = matched.get("fields", [])
                compact_fields = [
                    {
                        "name": f.get("name"),
                        "type": f.get("type"),
                    }
                    for f in raw_fields[:80]
                ]
                return {
                    "success": True,
                    "database_type": "mongodb",
                    "database": result.database,
                    "collection": matched.get("name"),
                    "row_count": matched.get("row_count"),
                    "fields": compact_fields,
                    "total_fields": len(raw_fields),
                }

            if len(result.objects) == 1:
                obj = result.objects[0]
                raw_fields = obj.get("fields", [])
                return {
                    "success": True,
                    "database_type": "mongodb",
                    "database": result.database,
                    "collection": obj.get("name"),
                    "row_count": obj.get("row_count"),
                    "fields": [
                        {"name": f.get("name"), "type": f.get("type")}
                        for f in raw_fields[:80]
                    ],
                    "total_fields": len(raw_fields),
                }

            collections_summary = [
                {
                    "name": obj.get("name"),
                    "row_count": obj.get("row_count"),
                    "key_fields": [f.get("name") for f in obj.get("fields", [])[:10]],
                    "total_fields": len(obj.get("fields", [])),
                }
                for obj in result.objects
            ]
            return {
                "success": True,
                "database_type": "mongodb",
                "database": result.database,
                "collections": collections_summary,
                "total_collections": len(result.objects),
                "instruction": "Call inspect_schema with table_or_collection='<collection_name>' to view detailed fields for a specific collection.",
            }
        except (ConnectionStringError, DatabaseInspectionError, Exception) as exc:
            logger.warning("MongoDB schema inspection error: %s", exc)
            return {
                "success": False,
                "database_type": "mongodb",
                "error": str(exc),
            }

    async def execute_query(self, query: dict) -> dict[str, Any]:
        collection = query.get("collection")
        pipeline = query.get("pipeline")
        database = query.get("database") or self.database

        if not collection or not isinstance(collection, str):
            return {
                "success": False,
                "database_type": "mongodb",
                "error": "The MongoDB query must specify a 'collection' name.",
            }

        # Normalize collection name: strip database prefix if passed like "data_analyzer.messages" or "db.messages"
        collection = collection.strip()
        if "." in collection:
            collection = collection.split(".")[-1]
        if pipeline is None or not isinstance(pipeline, list):
            return {
                "success": False,
                "database_type": "mongodb",
                "error": "The MongoDB query must include a 'pipeline' list of aggregation stages, e.g. [{'$match': ...}, {'$limit': 10}].",
            }

        start = time.perf_counter()
        query_payload = {"collection": collection, "pipeline": pipeline}
        if database:
            query_payload["database"] = database

        try:
            results = await asyncio.to_thread(
                run_mongo_queries, self.connection_string, database, [query_payload], None
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.warning("MongoDB query execution exception: %s", exc)
            return {
                "success": False,
                "database_type": "mongodb",
                "error": str(exc),
                "execution_time_ms": round(elapsed_ms, 2),
            }

        elapsed_ms = (time.perf_counter() - start) * 1000

        if not results:
            return {
                "success": False,
                "database_type": "mongodb",
                "error": "No result returned from MongoDB.",
                "execution_time_ms": round(elapsed_ms, 2),
            }

        first_res = results[0]
        if isinstance(first_res, (QueryValidationError, DatabaseInspectionError, Exception)):
            return {
                "success": False,
                "database_type": "mongodb",
                "error": str(first_res),
                "execution_time_ms": round(elapsed_ms, 2),
            }

        if isinstance(first_res, QueryResult):
            return {
                "success": True,
                "database_type": "mongodb",
                "columns": [{"name": col, "data_type": "unknown"} for col in first_res.columns],
                "rows": first_res.rows,
                "row_count": len(first_res.rows),
                "truncated": first_res.truncated,
                "execution_time_ms": round(elapsed_ms, 2),
            }

        return {
            "success": False,
            "database_type": "mongodb",
            "error": "Unexpected result structure from query execution.",
            "execution_time_ms": round(elapsed_ms, 2),
        }

    async def close(self) -> None:
        pass

