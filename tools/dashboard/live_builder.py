import json
import logging
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from config import get_settings
from llm_provider import LLMProvider
from shared.external_db.errors import ConnectionStringError, DatabaseInspectionError
from shared.external_db.queries import QueryResult, QueryValidationError, query_placeholders, validate_query
from shared.external_db.service import inspect_database, run_dashboard_queries
from shared.external_db.time_window import resolve_window, validate_window
from shared.models.dashboard import LIVE_CHART_TYPES
from tools.dashboard.prompts import CHART_TYPES_TEXT, MONGO_RULES, PLAN_PROMPT, POSTGRES_RULES, REPAIR_PROMPT
from tools.llm_call import ask_llm_async

logger = logging.getLogger("tools.dashboard.live_builder")

MAX_ATTEMPTS = 3
MAX_CHARTS = 6
MAX_SCHEMA_CHARS = 14000
MAX_SCHEMA_OBJECTS = 60
MAX_FIELDS_PER_OBJECT = 40
ENGINE_LABELS = {"postgresql": "PostgreSQL", "mongodb": "MongoDB"}


class LiveDashboardBuildError(RuntimeError):
    pass


class ChartCheckError(ValueError):
    pass


def get_model_config() -> dict:
    settings = get_settings()
    return {
        "provider": settings.get("DASHBOARD_BUILDER_PROVIDER", "") or settings.get("TABULAR_AGENT_PROVIDER", "") or None,
        "model": settings.get("DASHBOARD_BUILDER_MODEL", "") or settings.get("TABULAR_AGENT_MODEL", "") or None,
    }


def render_schema(schema: dict) -> str:
    lines: list[str] = []
    for obj in schema.get("objects", [])[:MAX_SCHEMA_OBJECTS]:
        name = f"{obj['schema']}.{obj['name']}" if obj.get("schema") else obj["name"]
        count = obj.get("row_count")
        header = f"{name} ({obj.get('kind', 'table')}"
        header += f", ~{count} rows)" if count is not None else ")"
        fields = []
        for field in obj.get("fields", [])[:MAX_FIELDS_PER_OBJECT]:
            text = f"{field['name']} {field['type']}"
            if field.get("primary_key"):
                text += " pk"
            fields.append(text)
        lines.append(f"- {header}: {', '.join(fields)}")
        for fk in obj.get("foreign_keys", []):
            lines.append(f"    fk {', '.join(fk['columns'])} -> {fk['references']}({', '.join(fk['referenced_columns'])})")
    text = "\n".join(lines)
    if len(text) > MAX_SCHEMA_CHARS:
        text = text[:MAX_SCHEMA_CHARS] + "\n... (schema truncated)"
    return text or "(no readable tables)"


def _extract_json(raw: str) -> dict:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ChartCheckError("The model did not return a JSON object.")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        raise ChartCheckError(f"The model returned invalid JSON: {exc.msg}.")
    if not isinstance(data, dict):
        raise ChartCheckError("The model did not return a JSON object.")
    return data


def _as_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(v) for v in value if isinstance(v, (str, int)) and str(v).strip()]


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _normalize_chart(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ChartCheckError("Each chart must be a JSON object.")
    chart_type = str(raw.get("chart_type") or "bar").strip().lower()
    if chart_type not in LIVE_CHART_TYPES:
        raise ChartCheckError(f"chart_type must be one of {', '.join(LIVE_CHART_TYPES)}.")
    axis = raw.get("axis") if isinstance(raw.get("axis"), dict) else {}
    fmt = str(axis.get("format") or "number").lower()
    labels = raw.get("labels") if isinstance(raw.get("labels"), dict) else {}
    return {
        "title": str(raw.get("title") or "Untitled chart")[:120],
        "description": str(raw.get("description") or "")[:300] or None,
        "chart_type": chart_type,
        "tables": _as_list(raw.get("tables")),
        "query": raw.get("query"),
        "x_column": (str(raw["x_column"]) if raw.get("x_column") else None),
        "value_columns": _as_list(raw.get("value_columns")),
        "series_column": (str(raw["series_column"]) if raw.get("series_column") else None),
        "aggregation": (str(raw["aggregation"])[:20] if raw.get("aggregation") else None),
        "axis": {
            "x_label": axis.get("x_label"),
            "y_label": axis.get("y_label"),
            "format": fmt if fmt in ("number", "currency", "percent") else "number",
        },
        "labels": {str(k): str(v)[:60] for k, v in labels.items()},
    }


def check_result(chart: dict, result: QueryResult) -> None:
    columns = set(result.columns)
    chart_type = chart["chart_type"]
    values = chart["value_columns"]
    if chart_type != "table" and not values:
        raise ChartCheckError("value_columns must list at least one numeric output column.")
    if chart_type not in ("metric", "table") and not chart["x_column"]:
        raise ChartCheckError("x_column is required for this chart type.")
    if not result.rows:
        return
    referenced = [c for c in [chart["x_column"], chart["series_column"], *values] if c]
    missing = [c for c in referenced if c not in columns]
    if missing:
        raise ChartCheckError(
            f"The chart references column(s) {missing} but the query returned {sorted(columns)}. "
            "Alias the query outputs to match, or fix the chart columns."
        )
    if chart_type != "table":
        for column in values:
            sample = next((row[column] for row in result.rows if row.get(column) is not None), None)
            if sample is not None and not _is_number(sample):
                raise ChartCheckError(
                    f"Value column '{column}' returned {type(sample).__name__} values, not numbers. Cast it to a number."
                )
    if chart_type == "metric" and len(result.rows) != 1:
        raise ChartCheckError(f"A metric chart must return exactly one row, got {len(result.rows)}.")
    if chart_type == "pie" and (len(values) != 1 or len(result.rows) > 12):
        raise ChartCheckError("A pie chart needs exactly one value column and at most 12 rows.")


class LiveDashboardBuilder:
    def __init__(self, on_event: Optional[Callable[[dict], Awaitable[None]]] = None, max_attempts: int = MAX_ATTEMPTS):
        self.on_event = on_event
        self.max_attempts = max_attempts
        self.model_config = get_model_config()

    async def _status(self, message: str) -> None:
        if self.on_event is not None:
            await self.on_event({"type": "status", "message": message})

    async def _ask(self, prompt: str) -> dict:
        client = LLMProvider(self.model_config["provider"]).get_client(self.model_config["model"])
        return _extract_json(await ask_llm_async(client, prompt))

    def _context(self, db_type: str, schema: dict, database: Optional[str]) -> dict:
        return {
            "engine": ENGINE_LABELS[db_type],
            "schema": render_schema(schema),
            "database_label": f"database {database}" if database else "all readable databases",
            "rules": POSTGRES_RULES if db_type == "postgresql" else MONGO_RULES,
            "chart_types": CHART_TYPES_TEXT,
        }

    async def _plan(self, request: str, context: dict) -> tuple[dict, Optional[dict]]:
        last_error = None
        for attempt in range(1, self.max_attempts + 1):
            prompt = PLAN_PROMPT.format(request=request, **context)
            if last_error:
                prompt += f"\n\nYour previous answer was rejected: {last_error}\nReturn a corrected JSON object."
            try:
                plan = await self._ask(prompt)
                window = validate_window(plan.get("time_window")) if plan.get("time_window") else None
                charts = plan.get("charts")
                if not isinstance(charts, list) or not charts:
                    raise ChartCheckError("The plan must include at least one chart.")
                return plan, window
            except (ChartCheckError, ValueError) as exc:
                last_error = str(exc)
                logger.info("live dashboard plan attempt %d rejected: %s", attempt, last_error)
        raise LiveDashboardBuildError(f"Couldn't design the dashboard: {last_error}")

    async def _check_chart(self, db_type: str, connection_string: str, database: Optional[str], chart: dict, window) -> QueryResult:
        try:
            query = validate_query(db_type, chart["query"] or {})
        except QueryValidationError as exc:
            raise ChartCheckError(str(exc))
        uses_placeholders = query_placeholders(db_type, query)
        if window and uses_placeholders != {"start", "end"}:
            raise ChartCheckError(
                f"The dashboard has a time window on '{window['column']}', so the query must filter on it "
                "with BOTH the start and end placeholders."
            )
        if not window and uses_placeholders:
            raise ChartCheckError("The dashboard has no time window, so the query must not use start/end placeholders.")
        chart["query"] = query
        results = await run_dashboard_queries(db_type, connection_string, database, [query], resolve_window(window))
        outcome = results[0]
        if isinstance(outcome, Exception):
            raise ChartCheckError(str(outcome))
        check_result(chart, outcome)
        return outcome

    async def _build_chart(
        self, request: str, context: dict, db_type: str, connection_string: str, database: Optional[str],
        raw_chart: dict, window: Optional[dict],
    ) -> tuple[Optional[dict], Optional[str]]:
        candidate = raw_chart
        last_error = None
        title = str((raw_chart or {}).get("title") or "chart") if isinstance(raw_chart, dict) else "chart"
        for attempt in range(1, self.max_attempts + 1):
            try:
                chart = _normalize_chart(candidate)
                title = chart["title"]
                result = await self._check_chart(db_type, connection_string, database, chart, window)
                chart["validated_at"] = datetime.now(timezone.utc)
                chart["row_count"] = len(result.rows)
                return chart, None
            except ChartCheckError as exc:
                last_error = str(exc)
                logger.info("live dashboard chart %r attempt %d failed: %s", title, attempt, last_error)
                if attempt == self.max_attempts:
                    break
                await self._status(f"Fixing the query for \"{title}\" (attempt {attempt + 1})")
                try:
                    candidate = await self._ask(REPAIR_PROMPT.format(
                        request=request,
                        time_window=json.dumps(window) if window else "none",
                        chart=json.dumps(candidate, default=str)[:6000],
                        error=last_error,
                        **context,
                    ))
                except ChartCheckError as repair_exc:
                    last_error = str(repair_exc)
        return None, last_error

    async def build(self, request: str, connection: dict, connection_string: str) -> dict:
        db_type = connection["db_type"]
        await self._status(f"Reading the live schema of {connection['name']}")
        try:
            _, inspection = await inspect_database(db_type, connection_string)
        except (ConnectionStringError, DatabaseInspectionError) as exc:
            raise LiveDashboardBuildError(str(exc))
        schema = inspection.schema_document()
        database = connection.get("database") or inspection.database
        context = self._context(db_type, schema, database)

        await self._status("Designing the dashboard and writing its queries")
        plan, window = await self._plan(request, context)

        known_objects = {
            (f"{o['schema']}.{o['name']}" if o.get("schema") else o["name"]) for o in schema.get("objects", [])
        } | {o["name"] for o in schema.get("objects", [])}

        charts: list[dict] = []
        dropped: list[dict] = []
        for raw_chart in plan["charts"][:MAX_CHARTS]:
            title = raw_chart.get("title") if isinstance(raw_chart, dict) else None
            await self._status(f"Testing the live query for \"{title or 'chart'}\"")
            try:
                chart, error = await self._build_chart(
                    request, context, db_type, connection_string, database, raw_chart, window,
                )
            except (ConnectionStringError, DatabaseInspectionError) as exc:
                raise LiveDashboardBuildError(str(exc))
            if chart is None:
                dropped.append({"title": title or "chart", "error": error})
            else:
                charts.append(chart)

        if not charts:
            reasons = "; ".join(f"{d['title']}: {d['error']}" for d in dropped)[:800]
            raise LiveDashboardBuildError(f"None of the chart queries could be made to work. {reasons}")

        tables = sorted({t for chart in charts for t in chart.pop("tables", []) if t in known_objects})
        return {
            "title": str(plan.get("title") or "Live dashboard")[:120],
            "description": (str(plan.get("description"))[:300] if plan.get("description") else None),
            "db_type": db_type,
            "database": database,
            "time_window": window,
            "tables": tables,
            "charts": charts,
            "dropped": dropped,
        }
