CHART_TYPES_TEXT = """
- bar: categories on x_column, one or more numeric value_columns (or one value column + series_column)
- stacked_bar: like bar, parts of a whole per category (needs series_column or 2+ value_columns)
- line / area: a time or ordered x_column with numeric value_columns (or one value column + series_column)
- pie: one category x_column and exactly one value column, at most 8 rows
- metric: a single headline number - the query returns ONE row, value_columns has exactly one column
- table: any tabular result; value_columns lists the columns to show
"""

POSTGRES_RULES = """
PostgreSQL query rules:
- Write exactly ONE read-only statement: SELECT, or WITH ... SELECT. No comments, no semicolons in the middle.
- Qualify tables with their schema (e.g. sales.orders) and only use tables/columns from the schema below.
- If the dashboard has a time window, filter on the window column with `>= :start AND < :end`
  (these placeholders are filled in on every load - never write literal dates for the window).
- Give every output column a simple snake_case alias.
- For trends, bucket dates with date_trunc('day'|'week'|'month', column) AS period and ORDER BY period.
- For category breakdowns, ORDER BY the value DESC and LIMIT 12 when there could be many categories.
- Cast aggregates to numeric/float (e.g. SUM(amount)::float) so they come back as numbers.
- The query spec is: {"sql": "<the SQL>"}
"""

MONGO_RULES = """
MongoDB query rules:
- Write an aggregation pipeline. Allowed stages: $match, $group, $project, $sort, $limit, $skip, $unwind,
  $addFields, $set, $unset, $count, $bucket, $bucketAuto, $facet, $lookup (same database only),
  $sortByCount, $replaceRoot, $replaceWith, $densify, $fill, $setWindowFields.
- Never use $out, $merge, $function, $accumulator, $where or $unionWith.
- If the dashboard has a time window, filter on the window field with
  {"<field>": {"$gte": "{{start}}", "$lt": "{{end}}"}} - these exact strings are replaced with real dates
  on every load. Never write literal dates for the window.
- Group keys go in _id as an object, e.g. {"_id": {"region": "$region"}} - they are flattened into
  columns named region etc. Name every other output field in snake_case.
- For trends, bucket dates with {"$dateTrunc": {"date": "$<field>", "unit": "month"}} and $sort by it.
- For category breakdowns, $sort by the value descending and $limit 12 when there could be many categories.
- The query spec is: {"collection": "<collection>", "database": "<database, only if the schema lists several>",
  "pipeline": [...]}
"""

PLAN_PROMPT = """You design live dashboards on top of a {engine} database. The dashboard re-runs your
queries every time it is opened, so the queries must be correct and read-only.

User request:
{request}

Live database schema ({database_label}):
{schema}

{rules}
Chart types:
{chart_types}

Decide on 2-6 charts that answer the request (fewer if the request is narrow). If the user asks for a
relative period ("past 3 months", "last 30 days", "this year so far"), set time_window to
{{"column": "<real date/timestamp column or field>", "last": <number>, "unit": "day|week|month|quarter|year"}}
and make EVERY chart filter on that same column with the start/end placeholders. If no period is asked
for, set time_window to null and use no placeholders.

Return ONLY a JSON object, no markdown fences:
{{
  "title": "short dashboard title",
  "description": "one sentence",
  "time_window": null or {{"column": "...", "last": 3, "unit": "month"}},
  "charts": [
    {{
      "title": "chart title",
      "description": "what it shows",
      "chart_type": "bar|stacked_bar|line|area|pie|metric|table",
      "tables": ["schema.table or collection names this query reads"],
      "query": <query spec>,
      "x_column": "output column for the x axis / category (null for metric)",
      "value_columns": ["numeric output columns"],
      "series_column": null or "output column that splits series",
      "aggregation": "sum|avg|count|min|max|none",
      "axis": {{"x_label": "...", "y_label": "...", "format": "number|currency|percent"}},
      "labels": {{"output_column": "Human label"}}
    }}
  ]
}}
"""

REPAIR_PROMPT = """A chart query for a live {engine} dashboard failed. Fix it.

User request: {request}
Dashboard time window: {time_window}

Live database schema ({database_label}):
{schema}

{rules}
Chart types:
{chart_types}

The chart that failed (JSON):
{chart}

Error:
{error}

Return ONLY the corrected chart as a JSON object with the same keys (title, description, chart_type,
tables, query, x_column, value_columns, series_column, aggregation, axis, labels). No markdown fences.
"""
