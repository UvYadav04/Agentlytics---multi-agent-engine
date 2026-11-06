import logging
import os
import re
import uuid
from typing import Any, Optional

import pandas as pd

from shared.services import DashboardService, VisualizationService
from tools.ask_user import AskUserMixin
from tools.database.connection_manager import ConnectionManager

try:
    from monitoring.wrappers.tools import trace_tool
except Exception:
    def trace_tool(*args, **kwargs):
        return lambda fn: fn

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
        reports_dir: str = "data/reports",
    ):
        self.connection_manager = connection_manager
        self.available_connections = available_connections or []
        self.on_event = on_event
        self.charts_proposed: list[dict] = []
        self.dashboards_created: list[dict] = []
        self.query_history: list[dict] = []
        self.ask_user_source = "database_agent"
        self.visualization_service = visualization_service or VisualizationService(output_dir=reports_dir)
        self.dashboard_service = dashboard_service or DashboardService(output_dir=reports_dir)
        self.workspace_id = workspace_id
        self.reports_dir = reports_dir
        self.last_query_rows: list[dict] = []
        self.last_query_columns: list[Any] = []

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

    @trace_tool(name="inspect_database_schema", component="tool.database.inspect_schema", step_type="database")
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

    @trace_tool(name="execute_database_query", component="tool.database.execute_query", step_type="database")
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
        """
        try:
            conn_info, adapter = await self.connection_manager.get_adapter(connection_id)
        except LookupError as exc:
            return {"success": False, "error": str(exc)}

        db_type = conn_info.get("db_type")

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

        if result.get("success"):
            self.last_query_rows = list(result.get("rows") or [])
            self.last_query_columns = list(result.get("columns") or [])

        self.query_history.append({
            "connection_id": connection_id,
            "db_type": db_type,
            "query": query,
            "success": result.get("success", False),
            "row_count": result.get("row_count", 0),
        })

        if self.on_event:
            count = result.get("row_count", 0)
            status_msg = f"Executed database query ({count} rows)" if result.get("success") else "Database query failed"
            await self.on_event({
                "type": "database_query_executed",
                "message": status_msg,
                "connection_id": connection_id,
                "db_type": db_type,
                "success": result.get("success", False),
                "row_count": count,
            })

        return result

    @trace_tool(name="propose_visualization", component="tool.database.propose_visualization")
    def propose_visualization(
        self, connection_id: str, chart_type: str, title: str, x_axis: str, y_axis: str,
        category: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Propose a visualization specification based on database query data.
        Returns a validated specification and chart_id before rendering.
        - chart_type: "bar", "line", "scatter3d", "surface", "pie", "histogram", "box", "heatmap"
        - title: Plain descriptive title for the chart
        - x_axis: Column/field name representing the independent axis
        - y_axis: Column/field name representing the numerical measure
        - category: Optional series/category column name
        """
        cols = [c["name"] if isinstance(c, dict) else str(c) for c in (self.last_query_columns or [])]
        if not cols and self.last_query_rows:
            cols = list(self.last_query_rows[0].keys())

        proposal = self.visualization_service.propose_visualization(
            chart_type=chart_type,
            title=title,
            x_axis=x_axis,
            y_axis=y_axis,
            category=category,
            connection_id=connection_id,
            available_columns=cols or None,
        )
        if proposal.get("status") == "error":
            return proposal

        # Automatically render and stage if query data is ready
        if self.last_query_rows:
            created = self.visualization_service.create_visualization(
                spec=proposal["specification"],
                data=self.last_query_rows,
                workspace_id=self.workspace_id,
            )
            chart_record = {**created, "location": created["location"], "chart_id": created["chart_id"]}
            self.charts_proposed.append(chart_record)
            return {
                "status": "accepted",
                "chart_id": created["chart_id"],
                "specification": created["specification"],
                "message": f"Visualization '{title}' ({created['chart_type']}) proposed and staged as artifact with chart_id '{created['chart_id']}'.",
            }

        self.charts_proposed.append(proposal["specification"])
        return proposal

    @trace_tool(name="create_visualization", component="tool.database.create_visualization")
    def create_visualization(
        self,
        chart_id: Optional[str] = None,
        chart_type: Optional[str] = None,
        title: Optional[str] = None,
        x_axis: Optional[str] = None,
        y_axis: Optional[str] = None,
        category: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Create, render, and stage an artifact visualization from the query results.
        Pass the proposed chart_id or specify chart_type, title, and axis mappings.
        """
        if not self.last_query_rows:
            return {"status": "error", "error": "No query data available to visualize. Execute a query first."}

        spec = {}
        if chart_id:
            staged = self.visualization_service.get_by_id(chart_id)
            if staged and staged.spec:
                spec = dict(staged.spec)
            else:
                for c in self.charts_proposed:
                    if c.get("chart_id") == chart_id:
                        spec = dict(c.get("specification", c))
                        break

        created = self.visualization_service.create_visualization(
            spec=spec or None,
            data=self.last_query_rows,
            workspace_id=self.workspace_id,
            chart_id=chart_id,
            title=title,
            chart_type=chart_type,
            x_axis=x_axis,
            y_axis=y_axis,
            category=category,
        )
        chart_record = {**created, "location": created["location"], "chart_id": created["chart_id"]}
        self.charts_proposed.append(chart_record)
        return {
            "status": "created",
            "chart_id": created["chart_id"],
            "title": created["title"],
            "chart_type": created["chart_type"],
            "specification": created.get("specification"),
        }

    @trace_tool(name="build_live_dashboard", component="tool.database.build_live_dashboard")
    async def build_live_dashboard(self, connection_id: str, request: str) -> dict[str, Any]:
        """
        Build a persistent LIVE dashboard with real-time charts on one of the user's connected databases (PostgreSQL or MongoDB).
        Use ONLY when the user explicitly asks to build or create a dashboard.
        """
        if self.connection_manager.database_access is None or not self.available_connections:
            return {"error": "No databases are connected to this workspace. Ask the user to connect one first."}

        try:
            conn_info, conn_str = await self.connection_manager.database_access.resolve(connection_id)
        except LookupError as exc:
            return {"error": str(exc)}

        result = await self.dashboard_service.build_live_dashboard(
            connection_id=connection_id,
            request=request,
            connection_info=conn_info,
            connection_string=conn_str,
            workspace_id=self.workspace_id,
            on_event=self.on_event,
        )
        if result.get("status") == "created":
            self.dashboards_created.append(result)
        return result
