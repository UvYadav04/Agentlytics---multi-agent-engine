import logging
import uuid
from typing import Optional

from tools.dashboard.live_builder import LiveDashboardBuildError, LiveDashboardBuilder

logger = logging.getLogger("tools.dashboard.live_tool")

BUILD_LIVE_DASHBOARD_DESCRIPTION = """Build a persistent LIVE dashboard with real-time charts on one of the user's connected databases (PostgreSQL or MongoDB).

CRITICAL USAGE RESTRICTION:
- Use this tool ONLY AND EXCLUSIVELY when the user explicitly asks to CREATE or BUILD a DASHBOARD (e.g. "build a dashboard", "create a live dashboard", "set up a real-time monitoring view").
- NEVER use this tool for general database queries, data fetching, questions, summaries, or investigations (e.g. "fetch me my database", "show me data", "query my users").
- For ANY data fetching, querying, record inspection, or analysis on connected databases, you MUST use `invoke_database_agent` instead.

- connection_id: the connection_id of the database to use, exactly as listed in your context.
- request: the user's dashboard request in plain language, including any time period
  ("past 3 months") and breakdowns ("region wise") they asked for.

This tool reads the live schema, writes and tests read-only queries for each chart (retrying failed
ones), and saves a dashboard that re-runs those queries every time it is opened - no data is copied.
It returns the dashboard title and its charts. In your final answer, mention the dashboard by its
title and the charts it contains; the user gets a link to open it. Do not invent charts it did not
return, and do not call it again for the same request. If it returns an error, explain it plainly."""


class LiveDashboardMixin:
    database_access = None
    database_connections: Optional[list] = None

    async def build_live_dashboard(self, connection_id: str, request: str) -> dict:
        if self.database_access is None or not self.database_connections:
            return {"error": "No databases are connected to this workspace. Ask the user to connect one first."}
        try:
            connection, connection_string = await self.database_access.resolve(connection_id)
        except LookupError as exc:
            return {"error": str(exc)}

        builder = LiveDashboardBuilder(on_event=getattr(self, "on_event", None))
        try:
            spec = await builder.build(request, connection, connection_string)
        except LiveDashboardBuildError as exc:
            logger.info("live dashboard build failed for connection %s: %s", connection_id, exc)
            return {"error": str(exc)}

        reference = f"live-dashboard-{uuid.uuid4().hex[:12]}"
        dashboard_service = getattr(self, "dashboard_service", None)
        if dashboard_service:
            dashboard_service.stage_live_dashboard(
                workspace_id=getattr(self, "workspace_id", "default"),
                title=spec.get("title", "Live Dashboard"),
                spec=spec,
                connection_id=connection["id"],
                dashboard_id=reference,
            )
        self.result_collector.add_artifact(
            reference, kind="live_dashboard", spec=spec, connection_id=connection["id"],
        )
        return {
            "status": "created",
            "title": spec["title"],
            "database": connection["name"],
            "time_window": spec["time_window"],
            "charts": [
                {"title": chart["title"], "chart_type": chart["chart_type"], "rows_now": chart.get("row_count")}
                for chart in spec["charts"]
            ],
            "skipped_charts": [d["title"] for d in spec["dropped"]],
        }


LiveDashboardMixin.build_live_dashboard.__doc__ = BUILD_LIVE_DASHBOARD_DESCRIPTION
