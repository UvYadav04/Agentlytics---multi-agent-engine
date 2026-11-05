import logging
from typing import Any, Optional

from shared.services import DashboardService, VisualizationService
from tools.ask_user import AskUserMixin
from tools.database.connection_manager import ConnectionManager

logger = logging.getLogger("tools.database.database_tools")


class DatabaseTools(AskUserMixin):
    def __init__(
        self,
        connection_manager: ConnectionManager,
        available_connections: Optional[list[dict]] = None,
        on_event=None,
        visualization_service: Optional[VisualizationService] = None,
        dashboard_service: Optional[DashboardService] = None,
        workspace_id: str = "default",
    ):
        self.connection_manager = connection_manager
        self.available_connections = available_connections or []
        self.on_event = on_event
        self.charts_proposed: list[dict] = []
        self.query_history: list[dict] = []
        self.ask_user_source = "database_agent"
        self.visualization_service = visualization_service
        self.dashboard_service = dashboard_service
        self.workspace_id = workspace_id

    def list_available_connections(self) -> list[dict]:
        """
        List all authorized database connections available in this investigation.
        Returns safe metadata: connection_id, name, db_type, database, and top tables/collections.
        Never returns connection strings or passwords.
        """
        return [
            {
                "connection_id": c["connection_id"],
                "name": c.get("name"),
                "db_type": c.get("db_type"),
                "database": c.get("database"),
                "objects": c.get("objects", [])[:40],
            }
            for c in self.available_connections
        ]

    async def inspect_schema(self, connection_id: str) -> dict[str, Any]:
        """
        Inspect detailed schema for a connected database.
        - For PostgreSQL: returns tables, columns, and primary/foreign keys.
        - For MongoDB: returns collections, sampled field names, and data types.
        """
        try:
            _, adapter = await self.connection_manager.get_adapter(connection_id)
            return await adapter.inspect_schema()
        except LookupError as exc:
            return {"success": False, "error": str(exc)}
        except Exception as exc:
            logger.exception("Failed to inspect schema for connection %s", connection_id)
            return {"success": False, "error": f"Failed to inspect schema: {exc}"}

    async def execute_query(self, connection_id: str, query: dict) -> dict[str, Any]:
        """
        Execute a read-only query directly against an authorized database connection.

        - For PostgreSQL:
            Provide a query object with SQL statement:
            {"statement": "SELECT category, SUM(revenue) AS total_rev FROM sales GROUP BY category LIMIT 50"}
            (Query must be a single read-only SELECT or WITH ... SELECT statement).

        - For MongoDB:
            Provide a query object with collection and aggregation pipeline:
            {
                "collection": "orders",
                "pipeline": [
                    {"$match": {"status": "completed"}},
                    {"$group": {"_id": "$customer_id", "total": {"$sum": "$amount"}}},
                    {"$sort": {"total": -1}},
                    {"$limit": 20}
                ]
            }

        Returns normalized results:
        {
            "success": True/False,
            "columns": [{"name": "..."}, ...],
            "rows": [{...}, ...],
            "row_count": N,
            "truncated": True/False,
            "execution_time_ms": N
        }
        """
        try:
            conn_info, adapter = await self.connection_manager.get_adapter(connection_id)
        except LookupError as exc:
            return {"success": False, "error": str(exc)}

        db_type = conn_info.get("db_type")

        # Type validation: ensure query representation matches connection type
        if db_type == "postgresql":
            if "pipeline" in query and "statement" not in query and "sql" not in query:
                return {
                    "success": False,
                    "database_type": "postgresql",
                    "error": "This is a PostgreSQL database. Please provide a SQL statement, not a MongoDB pipeline.",
                }
        elif db_type == "mongodb":
            if ("statement" in query or "sql" in query) and "pipeline" not in query:
                return {
                    "success": False,
                    "database_type": "mongodb",
                    "error": "This is a MongoDB database. Please provide a collection and aggregation pipeline, not SQL.",
                }

        result = await adapter.execute_query(query)

        # Record query history for the agent's findings
        self.query_history.append({
            "connection_id": connection_id,
            "db_type": db_type,
            "query": query,
            "success": result.get("success", False),
            "row_count": result.get("row_count", 0),
        })

        if self.on_event:
            await self.on_event({
                "type": "database_query_executed",
                "connection_id": connection_id,
                "db_type": db_type,
                "success": result.get("success", False),
                "row_count": result.get("row_count", 0),
            })

        return result

    def propose_visualization(
        self, connection_id: str, chart_type: str, title: str, x_axis: str, y_axis: str,
        category: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Propose a chart specification based on actual data returned by a query.
        - chart_type: "bar", "line", "scatter", "pie", or "donut"
        - title: Plain descriptive title for the chart
        - x_axis: Column/field name representing the independent axis
        - y_axis: Column/field name representing the numerical measure
        - category: Optional series/category column name
        """
        spec = {
            "connection_id": connection_id,
            "chart_type": chart_type,
            "title": title,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "category": category,
        }
        self.charts_proposed.append(spec)
        if self.visualization_service:
            staged = self.visualization_service.stage_chart(
                workspace_id=self.workspace_id,
                title=title,
                chart_type=chart_type,
                location=f"db_query_chart_{title}",
                spec=spec,
            )
            return {"status": "accepted", "specification": staged}
        return {"status": "accepted", "specification": spec}

