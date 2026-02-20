import math
import os
import traceback

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dash_table, dcc, html
from databricks import sql
from databricks.sdk.core import Config

DEFAULT_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
MAX_ROWS = 10_000

cfg = Config()

HIST_COLORS = [
    "#636EFA", "#00CC96", "#AB63FA", "#19D3F3", "#B6E880",
    "#3366CC", "#009988", "#7744BB", "#1199AA", "#88AA44",
]
FCST_COLORS = [
    "#EF553B", "#FFA15A", "#FF6692", "#FF97FF", "#FECB52",
    "#DD3333", "#DD8822", "#DD4477", "#DD77DD", "#DDAA22",
]
ACTUAL_COLORS = [
    "#2CA02C", "#17BECF", "#8C564B", "#BCBD22", "#9467BD",
    "#228B22", "#008080", "#A0522D", "#808000", "#6A5ACD",
]


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def get_connection(warehouse_id: str):
    wid = warehouse_id.strip() or DEFAULT_WAREHOUSE_ID
    if not wid:
        raise ValueError(
            "No SQL warehouse ID provided. Either enter one in the form "
            "or add a sql-warehouse resource in the Databricks Apps UI."
        )
    return sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{wid}",
        credentials_provider=lambda: cfg.authenticate,
    )


def list_tables(catalog: str, schema: str, warehouse_id: str) -> list[str]:
    conn = get_connection(warehouse_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW TABLES IN `{catalog}`.`{schema}`")
            rows = cursor.fetchall()
        return sorted(row[1] for row in rows)
    finally:
        conn.close()


def describe_table(catalog: str, schema: str, table: str, warehouse_id: str) -> list[tuple]:
    full_name = f"`{catalog}`.`{schema}`.`{table}`"
    conn = get_connection(warehouse_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"DESCRIBE TABLE {full_name}")
            rows = cursor.fetchall()
        result = []
        for row in rows:
            if row[0].startswith("#"):
                break
            result.append((row[0], row[1].upper()))
        return result
    finally:
        conn.close()


def detect_group_id_col(
    catalog: str, schema: str, table: str, warehouse_id: str
) -> str | None:
    cols = describe_table(catalog, schema, table, warehouse_id)
    return cols[0][0] if cols else None


def fetch_run_ids(
    catalog: str, schema: str, table: str, warehouse_id: str,
    run_date: str = "", col_map: dict | None = None,
) -> list[str]:
    full_name = f"`{catalog}`.`{schema}`.`{table}`"
    rd_col = _c(col_map, "run_date")
    rid_col = _c(col_map, "run_id")
    conn = get_connection(warehouse_id)
    try:
        where = ""
        if run_date.strip():
            where = f"WHERE CAST(`{rd_col}` AS DATE) = '{_esc(run_date)}'"
        query = f"SELECT DISTINCT `{rid_col}` FROM {full_name} {where} ORDER BY `{rid_col}`"
        with conn.cursor() as cursor:
            cursor.execute(query)
            return [row[0] for row in cursor.fetchall() if row[0]]
    finally:
        conn.close()


def fetch_model_names(
    catalog: str, schema: str, table: str, warehouse_id: str,
    run_date: str = "", run_ids=None,
    col_map: dict | None = None,
) -> list[str]:
    """Return sorted distinct model names from the evaluation table."""
    full_name = f"`{catalog}`.`{schema}`.`{table}`"
    model_col = _c(col_map, "model")
    rd_col = _c(col_map, "run_date")
    fc_col = _c(col_map, "forecast")
    mv_col = _c(col_map, "metric_value")
    conn = get_connection(warehouse_id)
    try:
        conds = []
        if run_date.strip():
            conds.append(f"CAST(`{rd_col}` AS DATE) = '{_esc(run_date)}'")
        rid_f = _run_id_filter(run_ids, col_map=col_map)
        if rid_f:
            conds.append(rid_f)
        conds.append(f"`{fc_col}`[0] IS NOT NULL")
        conds.append(f"`{mv_col}` IS NOT NULL")
        where = "WHERE " + " AND ".join(conds)
        query = f"SELECT DISTINCT `{model_col}` FROM {full_name} {where} ORDER BY `{model_col}`"
        with conn.cursor() as cursor:
            cursor.execute(query)
            return [str(row[0]) for row in cursor.fetchall() if row[0]]
    finally:
        conn.close()




def fetch_unique_ids(
    catalog: str, schema: str, table: str, warehouse_id: str,
    group_id_col: str, run_date: str = "", run_ids=None,
    col_map: dict | None = None,
) -> list[str]:
    full_name = f"`{catalog}`.`{schema}`.`{table}`"
    rd_col = _c(col_map, "run_date")
    conn = get_connection(warehouse_id)
    try:
        conditions = []
        if run_date.strip():
            conditions.append(f"CAST(`{rd_col}` AS DATE) = '{_esc(run_date)}'")
        rid_f = _run_id_filter(run_ids, col_map=col_map)
        if rid_f:
            conditions.append(rid_f)
        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = (
            f"SELECT DISTINCT `{group_id_col}` FROM {full_name} "
            f"{where} ORDER BY `{group_id_col}` LIMIT 5000"
        )
        with conn.cursor() as cursor:
            cursor.execute(query)
            return [str(row[0]) for row in cursor.fetchall() if row[0] is not None]
    finally:
        conn.close()


def _detect_score_array_cols(cols: list[tuple]) -> tuple[str | None, str | None]:
    date_col = None
    value_col = None
    for name, dtype in cols:
        if "ARRAY" not in dtype:
            continue
        if "TIMESTAMP" in dtype or "DATE" in dtype:
            date_col = date_col or name
        elif "DOUBLE" in dtype or "FLOAT" in dtype or "INT" in dtype or "DECIMAL" in dtype:
            value_col = value_col or name
    return date_col, value_col


def _detect_flat_date_value_cols(
    cols: list[tuple], group_col: str,
) -> tuple[str | None, str | None]:
    """Detect the scalar date and value columns from a training table, skipping the group column."""
    _DATE_TYPES = {"TIMESTAMP", "DATE"}
    _NUM_TYPES = {"DOUBLE", "FLOAT", "INT", "DECIMAL", "LONG", "SHORT", "BIGINT", "SMALLINT", "TINYINT"}
    _DATE_NAMES = {"ds", "date", "timestamp", "time", "datetime", "event_date", "order_date"}

    date_col = None
    value_col = None
    for name, dtype in cols:
        if name == group_col:
            continue
        if "ARRAY" in dtype or "BINARY" in dtype:
            continue
        upper = dtype.upper()
        if any(t in upper for t in _DATE_TYPES) and date_col is None:
            date_col = name
        elif any(t in upper for t in _NUM_TYPES) and value_col is None:
            value_col = name

    if date_col is None:
        for name, _ in cols:
            if name == group_col:
                continue
            if name.lower() in _DATE_NAMES:
                date_col = name
                break

    if value_col is None:
        for name, dtype in cols:
            if name == group_col or name == date_col:
                continue
            if "ARRAY" not in dtype and "BINARY" not in dtype:
                value_col = name
                break

    return date_col, value_col


def _infer_offset(dates: pd.Series):
    """Infer a pd.DateOffset from sorted datetime Series."""
    if len(dates) < 2:
        return pd.DateOffset(months=1)
    diffs = dates.diff().dropna()
    median_days = diffs.median().total_seconds() / 86400
    if median_days <= 1.5:
        return pd.DateOffset(days=1)
    elif median_days <= 8:
        return pd.DateOffset(weeks=1)
    elif median_days <= 35:
        return pd.DateOffset(months=1)
    elif median_days <= 100:
        return pd.DateOffset(months=3)
    else:
        return pd.DateOffset(years=1)


def _esc(val: str) -> str:
    return val.replace("'", "''")


_DEFAULT_ROLES = {
    "model": "model",
    "run_id": "run_id",
    "run_date": "run_date",
    "metric_name": "metric_name",
    "metric_value": "metric_value",
    "bt_start": "backtest_window_start_date",
    "forecast": "forecast",
    "actual": "actual",
}


def _c(col_map: dict | None, role: str) -> str:
    """Resolve an actual column name for a semantic *role*, with defaults."""
    if col_map:
        return col_map.get(role, _DEFAULT_ROLES.get(role, role))
    return _DEFAULT_ROLES.get(role, role)


def detect_column_roles(cols: list[tuple], group_col: str) -> dict:
    """Detect standard column roles from a table schema.

    Tries well-known names first, then falls back to type-based heuristics.
    """
    col_set = {c[0] for c in cols}
    mapping: dict[str, str] = {}

    for role, default_name in _DEFAULT_ROLES.items():
        if default_name in col_set:
            mapping[role] = default_name

    used = set(mapping.values()) | {group_col}
    for name, dtype in cols:
        if name in used:
            continue
        upper = dtype.upper()
        is_array_num = "ARRAY" in upper and any(
            t in upper for t in ("DOUBLE", "FLOAT", "INT", "DECIMAL")
        )
        if "forecast" not in mapping and is_array_num:
            mapping["forecast"] = name
            used.add(name)
        elif "actual" not in mapping and is_array_num:
            mapping["actual"] = name
            used.add(name)

    return mapping


def _run_id_filter(run_ids, prefix: str = "", col_map: dict | None = None) -> str | None:
    """Build a SQL filter for one or more run_ids. Returns None if empty."""
    if not run_ids:
        return None
    if isinstance(run_ids, str):
        run_ids = [run_ids]
    ids = [r for r in run_ids if r and str(r).strip()]
    if not ids:
        return None
    rid_col = _c(col_map, "run_id")
    col = f"{prefix}`{rid_col}`" if prefix else f"`{rid_col}`"
    if len(ids) == 1:
        return f"{col} = '{_esc(ids[0])}'"
    vals = ", ".join(f"'{_esc(r)}'" for r in ids)
    return f"{col} IN ({vals})"


def fetch_forecast_data(
    catalog: str, schema: str,
    eval_table: str, score_table: str,
    warehouse_id: str,
    run_ids=None, run_date: str = "",
    unique_ids: list[str] | None = None,
    model: str = "__best__",
    group_col: str = "",
    eval_cm: dict | None = None, score_cm: dict | None = None,
) -> tuple[pd.DataFrame, str, str, str]:
    eval_cols = describe_table(catalog, schema, eval_table, warehouse_id)
    score_cols = describe_table(catalog, schema, score_table, warehouse_id)

    eval_group = group_col.strip() or eval_cols[0][0]
    score_col_names = [c[0] for c in score_cols]
    score_group = eval_group if eval_group in score_col_names else score_cols[0][0]
    score_date, score_value = _detect_score_array_cols(score_cols)

    if not score_date or not score_value:
        raise ValueError(
            f"Could not detect array columns in scoring table. "
            f"Columns found: {[c[0] for c in score_cols]}"
        )

    e_model = _c(eval_cm, "model")
    e_rd = _c(eval_cm, "run_date")
    e_mv = _c(eval_cm, "metric_value")
    s_model = _c(score_cm, "model")
    s_rd = _c(score_cm, "run_date")

    eval_full = f"`{catalog}`.`{schema}`.`{eval_table}`"
    score_full = f"`{catalog}`.`{schema}`.`{score_table}`"

    eval_conds = []
    if run_date.strip():
        eval_conds.append(f"CAST(`{e_rd}` AS DATE) = '{_esc(run_date)}'")
    rid_f = _run_id_filter(run_ids, col_map=eval_cm)
    if rid_f:
        eval_conds.append(rid_f)
    if unique_ids:
        vals = ", ".join(f"'{_esc(v)}'" for v in unique_ids)
        eval_conds.append(f"`{eval_group}` IN ({vals})")
    eval_where = ("WHERE " + " AND ".join(eval_conds)) if eval_conds else ""

    score_conds = []
    if run_date.strip():
        score_conds.append(f"CAST(score.`{s_rd}` AS DATE) = '{_esc(run_date)}'")
    rid_sf = _run_id_filter(run_ids, prefix="score.", col_map=score_cm)
    if rid_sf:
        score_conds.append(rid_sf)
    if model != "__best__":
        score_conds.append(f"score.`{s_model}` = '{_esc(model)}'")
    score_extra = (" AND " + " AND ".join(score_conds)) if score_conds else ""

    if model == "__best__":
        eval_subquery = f"""
        SELECT `{eval_group}`, `{e_model}`, avg_metric,
            RANK() OVER (PARTITION BY `{eval_group}` ORDER BY avg_metric ASC NULLS LAST) AS rnk
        FROM (
            SELECT `{eval_group}`, `{e_model}`, AVG(`{e_mv}`) AS avg_metric
            FROM {eval_full}
            {eval_where}
            GROUP BY `{eval_group}`, `{e_model}`
            HAVING AVG(`{e_mv}`) IS NOT NULL
        )"""
        rank_filter = "WHERE eval.rnk = 1"
    else:
        model_cond = f"`{e_model}` = '{_esc(model)}'"
        if eval_where:
            model_where = f"{eval_where} AND {model_cond}"
        else:
            model_where = f"WHERE {model_cond}"
        eval_subquery = f"""
        SELECT `{eval_group}`, `{e_model}`, AVG(`{e_mv}`) AS avg_metric, 1 AS rnk
        FROM {eval_full}
        {model_where}
        GROUP BY `{eval_group}`, `{e_model}`
        HAVING AVG(`{e_mv}`) IS NOT NULL"""
        rank_filter = ""

    query = f"""
    SELECT
        eval.`{eval_group}`,
        eval.`{e_model}` AS model,
        eval.avg_metric,
        score.`{score_date}`  AS forecast_dates,
        score.`{score_value}` AS forecast_values
    FROM ({eval_subquery}) AS eval
    INNER JOIN {score_full} AS score
        ON eval.`{eval_group}` = score.`{score_group}`
        AND eval.`{e_model}` = score.`{s_model}`
        {score_extra}
    {rank_filter}
    ORDER BY eval.`{eval_group}`
    LIMIT {MAX_ROWS}
    """

    conn = get_connection(warehouse_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            result_cols = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=result_cols)
    finally:
        conn.close()

    return df, eval_group, score_date, score_value


def fetch_training_data(
    catalog: str, schema: str, train_table: str, warehouse_id: str,
    unique_ids: list[str],
    known_group_col: str = "", target_col: str = "",
) -> tuple[pd.DataFrame, str, str, str]:
    """Fetch historical data from the training table for the given group IDs."""
    train_cols = describe_table(catalog, schema, train_table, warehouse_id)
    col_names = [c[0] for c in train_cols]

    if known_group_col and known_group_col in col_names:
        group_col = known_group_col
    else:
        group_col = train_cols[0][0]

    date_col, auto_value_col = _detect_flat_date_value_cols(train_cols, group_col)
    value_col = target_col.strip() if target_col.strip() and target_col.strip() in col_names else auto_value_col

    if not date_col or not value_col:
        raise ValueError(
            f"Could not detect date/value columns in training table. "
            f"Columns found: {[c[0] for c in train_cols]}"
        )

    train_full = f"`{catalog}`.`{schema}`.`{train_table}`"
    vals = ", ".join(f"'{_esc(v)}'" for v in unique_ids)

    query = f"""
    SELECT `{group_col}`, `{date_col}`, `{value_col}`
    FROM {train_full}
    WHERE `{group_col}` IN ({vals})
    ORDER BY `{group_col}`, `{date_col}`
    LIMIT {MAX_ROWS}
    """

    conn = get_connection(warehouse_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            result_cols = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(rows, columns=result_cols)
    finally:
        conn.close()

    return df, group_col, date_col, value_col


def fetch_backtest_start_dates(
    catalog: str, schema: str, eval_table: str, warehouse_id: str,
    group_col: str, unique_ids: list[str] | None = None,
    run_date: str = "", run_ids=None,
    col_map: dict | None = None,
) -> list[str]:
    """Return sorted distinct backtest start-date values."""
    full = f"`{catalog}`.`{schema}`.`{eval_table}`"
    rd_col = _c(col_map, "run_date")
    bt_col = _c(col_map, "bt_start")
    fc_col = _c(col_map, "forecast")
    mv_col = _c(col_map, "metric_value")

    conds = []
    if run_date.strip():
        conds.append(f"CAST(`{rd_col}` AS DATE) = '{_esc(run_date)}'")
    rid_f = _run_id_filter(run_ids, col_map=col_map)
    if rid_f:
        conds.append(rid_f)
    if unique_ids:
        vals = ", ".join(f"'{_esc(v)}'" for v in unique_ids)
        conds.append(f"`{group_col}` IN ({vals})")
    conds.append(f"`{fc_col}`[0] IS NOT NULL")
    conds.append(f"`{mv_col}` IS NOT NULL")
    where = "WHERE " + " AND ".join(conds)
    query = f"""
    SELECT DISTINCT CAST(`{bt_col}` AS DATE) AS bt_date
    FROM {full} {where}
    ORDER BY bt_date
    """
    conn = get_connection(warehouse_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            return [str(row[0]) for row in cursor.fetchall() if row[0]]
    finally:
        conn.close()


def fetch_backtest_data(
    catalog: str, schema: str, eval_table: str, warehouse_id: str,
    group_col: str, unique_ids: list[str],
    backtest_date: str, run_date: str = "", run_ids=None,
    model: str = "__best__",
    col_map: dict | None = None,
) -> pd.DataFrame:
    """Fetch forecast & actual arrays for a model at a backtest window."""
    full = f"`{catalog}`.`{schema}`.`{eval_table}`"
    model_col = _c(col_map, "model")
    rd_col = _c(col_map, "run_date")
    mv_col = _c(col_map, "metric_value")
    bt_col = _c(col_map, "bt_start")
    fc_col = _c(col_map, "forecast")
    ac_col = _c(col_map, "actual")

    base_conds = []
    if run_date.strip():
        base_conds.append(f"CAST(`{rd_col}` AS DATE) = '{_esc(run_date)}'")
    rid_f = _run_id_filter(run_ids, col_map=col_map)
    if rid_f:
        base_conds.append(rid_f)
    vals = ", ".join(f"'{_esc(v)}'" for v in unique_ids)
    base_conds.append(f"`{group_col}` IN ({vals})")
    base_where = "WHERE " + " AND ".join(base_conds)

    e_conds = []
    if run_date.strip():
        e_conds.append(f"CAST(e.`{rd_col}` AS DATE) = '{_esc(run_date)}'")
    rid_ef = _run_id_filter(run_ids, prefix="e.", col_map=col_map)
    if rid_ef:
        e_conds.append(rid_ef)
    e_conds.append(f"e.`{group_col}` IN ({vals})")
    e_conds.append(
        f"CAST(e.`{bt_col}` AS DATE) = '{_esc(backtest_date)}'"
    )
    if model == "__best__":
        cte = f"""
        WITH best AS (
            SELECT `{group_col}`, `{model_col}`,
                RANK() OVER (
                    PARTITION BY `{group_col}`
                    ORDER BY AVG(`{mv_col}`) ASC NULLS LAST
                ) AS rnk
            FROM {full}
            {base_where}
            GROUP BY `{group_col}`, `{model_col}`
            HAVING AVG(`{mv_col}`) IS NOT NULL
        )"""
        join = f"""
        INNER JOIN best b
            ON e.`{group_col}` = b.`{group_col}`
            AND e.`{model_col}` = b.`{model_col}`
            AND b.rnk = 1"""
    else:
        cte = ""
        join = ""
        e_conds.append(f"e.`{model_col}` = '{_esc(model)}'")

    query = f"""
    {cte}
    SELECT e.`{group_col}`, e.`{model_col}` AS model,
           CAST(e.`{bt_col}` AS DATE) AS backtest_window_start_date,
           first(e.`{fc_col}`, true) AS forecast,
           first(e.`{ac_col}`, true) AS actual
    FROM {full} e
    {join}
    WHERE {" AND ".join(e_conds)}
      AND e.`{fc_col}` IS NOT NULL
      AND e.`{fc_col}`[0] IS NOT NULL
    GROUP BY e.`{group_col}`, e.`{model_col}`,
             CAST(e.`{bt_col}` AS DATE)
    ORDER BY e.`{group_col}`
    LIMIT {MAX_ROWS}
    """

    conn = get_connection(warehouse_id)
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
            cols = [desc[0] for desc in cursor.description]
        return pd.DataFrame(rows, columns=cols)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    title="Many Model Forecasting - Results Explorer",
)

header = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    html.I(
                        className="fa-solid fa-chart-line me-2",
                        style={"fontSize": "1.5rem"},
                    ),
                    width="auto",
                ),
                dbc.Col(
                    dbc.NavbarBrand(
                        "Many Model Forecasting — Results Explorer",
                        className="fs-5 fw-semibold mb-0",
                    ),
                ),
            ],
            align="center",
            className="g-0",
        ),
        fluid=True,
    ),
    color="dark",
    dark=True,
    className="mb-4",
)

warehouse_note = ""
if DEFAULT_WAREHOUSE_ID:
    warehouse_note = f"Auto-detected: {DEFAULT_WAREHOUSE_ID}"

input_card = dbc.Card(
    dbc.CardBody(
        [
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("SQL Warehouse ID", html_for="input-warehouse"),
                            dbc.Input(
                                id="input-warehouse",
                                placeholder="e.g. abcd1234ef567890",
                                type="text",
                                value=DEFAULT_WAREHOUSE_ID,
                            ),
                            dbc.FormText(warehouse_note) if warehouse_note else None,
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Catalog", html_for="input-catalog"),
                            dbc.Input(
                                id="input-catalog", placeholder="e.g. my_catalog",
                                type="text", debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Schema", html_for="input-schema"),
                            dbc.Input(
                                id="input-schema", placeholder="e.g. my_schema",
                                type="text", debounce=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="list-tables-feedback"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Training Table", html_for="select-train-table"),
                            dcc.Dropdown(
                                id="select-train-table",
                                placeholder="Select training table…",
                                clearable=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Evaluation Table", html_for="select-eval-table"),
                            dcc.Dropdown(
                                id="select-eval-table",
                                placeholder="Select evaluation table…",
                                clearable=True,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Scoring Table", html_for="select-score-table"),
                            dcc.Dropdown(
                                id="select-score-table",
                                placeholder="Select scoring table…",
                                clearable=True,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="col-mapping-feedback"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Group ID Column", html_for="select-group-col"),
                            dcc.Dropdown(
                                id="select-group-col",
                                placeholder="Auto-detected from eval table…",
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Target Column", html_for="select-target-col"),
                            dcc.Dropdown(
                                id="select-target-col",
                                placeholder="Auto-detected from training table…",
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Metric Value Column", html_for="select-metric-name"),
                            dcc.Dropdown(
                                id="select-metric-name",
                                placeholder="Auto-detected from eval table…",
                                clearable=False,
                            ),
                        ],
                        md=4,
                    ),
                ],
                className="g-3 mb-3",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Run Date", html_for="input-run-date"),
                            dcc.DatePickerSingle(
                                id="input-run-date",
                                placeholder="(optional)",
                                clearable=True,
                                style={"display": "block"},
                            ),
                        ],
                        md=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Run ID(s)", html_for="select-run-id"),
                            dcc.Dropdown(
                                id="select-run-id",
                                placeholder="(optional) select one or more run_id…",
                                clearable=True,
                                multi=True,
                            ),
                        ],
                        md=8,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="run-id-feedback"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id="unique-id-label"),
                            dcc.Dropdown(
                                id="select-unique-ids",
                                placeholder="(optional) select one or more…",
                                clearable=True,
                                multi=True,
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="unique-id-feedback"),
            dbc.Row(
                [
                    dbc.Col(
                        [
                            dbc.Label("Model", html_for="select-model"),
                            dcc.Dropdown(
                                id="select-model",
                                options=[{"label": "Best (auto)", "value": "__best__"}],
                                value="__best__",
                                placeholder="Select model…",
                                clearable=False,
                            ),
                        ],
                        md=12,
                    ),
                ],
                className="g-3 mb-3",
            ),
            html.Div(id="model-feedback"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Button(
                            [html.I(className="fa-solid fa-magnifying-glass me-2"), "Load"],
                            id="btn-load",
                            color="primary",
                            className="w-100",
                            size="lg",
                        ),
                        md={"size": 4, "offset": 4},
                    ),
                ],
                className="g-3",
            ),
        ]
    ),
    className="mb-4 shadow-sm",
)

alert_placeholder = html.Div(id="alert-placeholder")

forecast_section = html.Div(
    [
        html.H5(
            [html.I(className="fa-solid fa-chart-line me-2"), "Best-Model Forecasts"],
            className="mb-3",
        ),
        dbc.Spinner(
            html.Div(id="forecast-container"),
            color="primary",
            spinner_class_name="me-2",
        ),
    ],
    className="mb-5",
)

backtest_section = html.Div(
    id="backtest-section-wrapper",
    children=[
        html.Hr(),
        html.H5(
            [html.I(className="fa-solid fa-chart-bar me-2"), "Backtesting Results"],
            className="mb-3",
        ),
        dbc.Tabs(
            id="backtest-tabs",
            active_tab=None,
            className="mb-3",
        ),
        dbc.Spinner(
            html.Div(id="backtest-container"),
            color="primary",
            spinner_class_name="me-2",
        ),
    ],
    style={"display": "none"},
    className="mb-5",
)

app.layout = dbc.Container(
    [
        dcc.Store(id="eval-col-map", data={}),
        dcc.Store(id="score-col-map", data={}),
        header, input_card, alert_placeholder, forecast_section, backtest_section,
    ],
    fluid=True,
    className="pb-5",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_datatable(df: pd.DataFrame, table_id: str) -> dash_table.DataTable:
    return dash_table.DataTable(
        id=table_id,
        columns=[{"name": c, "id": c} for c in df.columns],
        data=df.to_dict("records"),
        page_size=25,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#f8f9fa",
            "fontWeight": "600",
            "border": "1px solid #dee2e6",
        },
        style_cell={
            "textAlign": "left",
            "padding": "8px 12px",
            "border": "1px solid #dee2e6",
            "maxWidth": "300px",
            "overflow": "hidden",
            "textOverflow": "ellipsis",
        },
        style_data_conditional=[
            {"if": {"row_index": "odd"}, "backgroundColor": "#f8f9fa"}
        ],
        export_format="csv",
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(
    Output("select-train-table", "options"),
    Output("select-eval-table", "options"),
    Output("select-score-table", "options"),
    Output("select-train-table", "value"),
    Output("select-eval-table", "value"),
    Output("select-score-table", "value"),
    Output("list-tables-feedback", "children"),
    Input("input-catalog", "value"),
    Input("input-schema", "value"),
    State("input-warehouse", "value"),
    prevent_initial_call=True,
)
def populate_table_dropdowns(catalog, schema, warehouse_id):
    catalog = (catalog or "").strip()
    schema = (schema or "").strip()
    warehouse_id = warehouse_id or ""

    empty = [], [], [], None, None, None

    if not catalog or not schema:
        return *empty, html.Div()

    if not warehouse_id.strip() and not DEFAULT_WAREHOUSE_ID:
        return (*empty,
                dbc.Alert("Enter a SQL Warehouse ID first.", color="danger",
                          dismissable=True, className="mt-2"))

    try:
        tables = list_tables(catalog, schema, warehouse_id)
    except Exception as exc:
        return (*empty,
                dbc.Alert([html.Strong("Error listing tables: "), html.Code(str(exc))],
                          color="danger", dismissable=True, className="mt-2"))

    if not tables:
        return (*empty,
                dbc.Alert(f"No tables found in `{catalog}`.`{schema}`.",
                          color="info", dismissable=True, className="mt-2"))

    options = [{"label": t, "value": t} for t in tables]
    return options, options, options, None, None, None, html.Div()


@callback(
    Output("select-group-col", "options"),
    Output("select-group-col", "value"),
    Output("select-target-col", "options"),
    Output("select-target-col", "value"),
    Output("select-metric-name", "options"),
    Output("select-metric-name", "value"),
    Output("eval-col-map", "data"),
    Output("score-col-map", "data"),
    Output("col-mapping-feedback", "children"),
    Input("select-eval-table", "value"),
    Input("select-train-table", "value"),
    Input("select-score-table", "value"),
    State("input-warehouse", "value"),
    State("input-catalog", "value"),
    State("input-schema", "value"),
    prevent_initial_call=True,
)
def populate_column_mapping(eval_table, train_table, score_table,
                            warehouse_id, catalog, schema):
    catalog = (catalog or "").strip()
    schema = (schema or "").strip()
    warehouse_id = (warehouse_id or "").strip()

    group_opts, group_val = [], None
    target_opts, target_val = [], None
    metric_opts, metric_val = [], None
    eval_cm, score_cm = {}, {}

    try:
        if eval_table and catalog and schema:
            eval_cols = describe_table(catalog, schema, eval_table, warehouse_id)
            group_opts = [{"label": c[0], "value": c[0]} for c in eval_cols]
            group_val = eval_cols[0][0] if eval_cols else None
            eval_cm = detect_column_roles(eval_cols, group_val or "")

            _NUMERIC = {"DOUBLE", "FLOAT", "DECIMAL", "INT", "BIGINT",
                        "SMALLINT", "TINYINT", "LONG", "SHORT"}
            mv_candidates = [
                name for name, dtype in eval_cols
                if name != (group_val or "")
                and "ARRAY" not in dtype.upper()
                and any(t in dtype.upper() for t in _NUMERIC)
            ]
            metric_opts = [{"label": c, "value": c} for c in mv_candidates]
            auto_mv = _c(eval_cm, "metric_value")
            if auto_mv in mv_candidates:
                metric_val = auto_mv
            elif mv_candidates:
                metric_val = mv_candidates[0]
            if metric_val:
                eval_cm["metric_value"] = metric_val

        if score_table and catalog and schema:
            score_cols = describe_table(catalog, schema, score_table, warehouse_id)
            score_cm = detect_column_roles(score_cols, group_val or "")

        if train_table and catalog and schema:
            train_cols = describe_table(catalog, schema, train_table, warehouse_id)
            known_group = group_val or ""
            _SKIP = {"ARRAY", "BINARY", "MAP", "STRUCT", "TIMESTAMP", "DATE"}
            candidates = [
                name for name, dtype in train_cols
                if name != known_group
                and not any(t in dtype.upper() for t in _SKIP)
            ]
            target_opts = [{"label": c, "value": c} for c in candidates]
            _, auto_val = _detect_flat_date_value_cols(train_cols, known_group)
            if auto_val and auto_val in candidates:
                target_val = auto_val
            elif candidates:
                target_val = candidates[0]
    except Exception as exc:
        return (group_opts, group_val, target_opts, target_val,
                metric_opts, metric_val, eval_cm, score_cm,
                dbc.Alert([html.Strong("Error: "), html.Code(str(exc))],
                          color="danger", dismissable=True, className="mt-2"))

    return (group_opts, group_val, target_opts, target_val,
            metric_opts, metric_val, eval_cm, score_cm, html.Div())


@callback(
    Output("select-run-id", "options"),
    Output("select-run-id", "value"),
    Output("run-id-feedback", "children"),
    Input("select-eval-table", "value"),
    Input("select-score-table", "value"),
    Input("input-run-date", "date"),
    State("input-warehouse", "value"),
    State("input-catalog", "value"),
    State("input-schema", "value"),
    State("eval-col-map", "data"),
    prevent_initial_call=True,
)
def populate_run_ids(eval_table, score_table, run_date, warehouse_id, catalog, schema,
                     eval_cm):
    catalog = (catalog or "").strip()
    schema = (schema or "").strip()
    warehouse_id = (warehouse_id or "").strip()
    run_date = run_date or ""
    table = eval_table or score_table

    if not table or not catalog or not schema:
        return [], None, html.Div()

    try:
        run_ids = fetch_run_ids(catalog, schema, table, warehouse_id, run_date,
                                col_map=eval_cm or {})
    except Exception as exc:
        return ([], None,
                dbc.Alert([html.Strong("Error fetching run IDs: "), html.Code(str(exc))],
                          color="danger", dismissable=True, className="mt-2"))

    if not run_ids:
        return [], None, html.Div()

    return [{"label": r, "value": r} for r in run_ids], None, html.Div()


@callback(
    Output("select-model", "options"),
    Output("select-model", "value"),
    Output("model-feedback", "children"),
    Input("select-eval-table", "value"),
    Input("input-run-date", "date"),
    Input("select-run-id", "value"),
    Input("select-metric-name", "value"),
    State("input-warehouse", "value"),
    State("input-catalog", "value"),
    State("input-schema", "value"),
    State("eval-col-map", "data"),
    prevent_initial_call=True,
)
def populate_models(eval_table, run_date, run_ids, selected_mv_col, warehouse_id,
                    catalog, schema, eval_cm):
    best_opt = {"label": "Best (auto)", "value": "__best__"}
    catalog = (catalog or "").strip()
    schema = (schema or "").strip()
    warehouse_id = (warehouse_id or "").strip()
    run_date = run_date or ""
    run_ids = run_ids or []
    eval_cm = {**(eval_cm or {})}
    if selected_mv_col:
        eval_cm["metric_value"] = selected_mv_col

    if not eval_table or not catalog or not schema:
        return [best_opt], "__best__", html.Div()

    try:
        models = fetch_model_names(
            catalog, schema, eval_table, warehouse_id,
            run_date=run_date, run_ids=run_ids,
            col_map=eval_cm,
        )
    except Exception as exc:
        return ([best_opt], "__best__",
                dbc.Alert([html.Strong("Error fetching models: "), html.Code(str(exc))],
                          color="danger", dismissable=True, className="mt-2"))

    options = [best_opt] + [{"label": m, "value": m} for m in models]
    return options, "__best__", html.Div()


@callback(
    Output("select-unique-ids", "options"),
    Output("select-unique-ids", "value"),
    Output("unique-id-label", "children"),
    Output("unique-id-feedback", "children"),
    Input("select-eval-table", "value"),
    Input("select-score-table", "value"),
    Input("input-run-date", "date"),
    Input("select-run-id", "value"),
    Input("select-group-col", "value"),
    State("input-warehouse", "value"),
    State("input-catalog", "value"),
    State("input-schema", "value"),
    State("eval-col-map", "data"),
    prevent_initial_call=True,
)
def populate_unique_ids(eval_table, score_table, run_date, run_ids, selected_group_col,
                        warehouse_id, catalog, schema, eval_cm):
    default_label = dbc.Label("Group ID(s)", className="mb-1")
    catalog = (catalog or "").strip()
    schema = (schema or "").strip()
    warehouse_id = (warehouse_id or "").strip()
    run_date = run_date or ""
    run_ids = run_ids or []
    table = eval_table or score_table

    if not table or not catalog or not schema:
        return [], None, default_label, html.Div()

    try:
        group_col = selected_group_col or detect_group_id_col(catalog, schema, table, warehouse_id)
        if not group_col:
            return [], None, default_label, html.Div()
        label = dbc.Label(f"{group_col}(s)", className="mb-1")
        uids = fetch_unique_ids(
            catalog, schema, table, warehouse_id, group_col,
            run_date=run_date, run_ids=run_ids,
            col_map=eval_cm or {},
        )
    except Exception as exc:
        return ([], None, default_label,
                dbc.Alert([html.Strong("Error fetching group IDs: "), html.Code(str(exc))],
                          color="danger", dismissable=True, className="mt-2"))

    if not uids:
        return [], None, label, html.Div()

    return [{"label": u, "value": u} for u in uids], None, label, html.Div()


@callback(
    Output("forecast-container", "children"),
    Output("alert-placeholder", "children"),
    Output("backtest-tabs", "children"),
    Output("backtest-tabs", "active_tab"),
    Output("backtest-section-wrapper", "style"),
    Input("btn-load", "n_clicks"),
    State("input-warehouse", "value"),
    State("input-catalog", "value"),
    State("input-schema", "value"),
    State("select-train-table", "value"),
    State("select-eval-table", "value"),
    State("select-score-table", "value"),
    State("input-run-date", "date"),
    State("select-run-id", "value"),
    State("select-model", "value"),
    State("select-unique-ids", "value"),
    State("select-group-col", "value"),
    State("select-target-col", "value"),
    State("select-metric-name", "value"),
    State("eval-col-map", "data"),
    State("score-col-map", "data"),
    prevent_initial_call=True,
)
def load_forecast(n_clicks, warehouse_id, catalog, schema, train_table,
                  eval_table, score_table, run_date, run_ids, selected_model,
                  unique_ids, selected_group_col, selected_target_col, selected_metric,
                  eval_cm, score_cm):
    warehouse_id = warehouse_id or ""
    run_ids = run_ids or []
    run_date = run_date or ""
    unique_ids = unique_ids or []
    selected_model = selected_model or "__best__"
    selected_group_col = selected_group_col or ""
    selected_target_col = selected_target_col or ""
    eval_cm = {**(eval_cm or {})}
    score_cm = score_cm or {}
    if selected_metric:
        eval_cm["metric_value"] = selected_metric
    NO_BT = [], None, {"display": "none"}

    if not all([catalog, schema]):
        return (dash.no_update, dbc.Alert(
            "Please provide Catalog and Schema.", color="warning", dismissable=True),
            *NO_BT)

    if not warehouse_id.strip() and not DEFAULT_WAREHOUSE_ID:
        return (dash.no_update, dbc.Alert(
            "No SQL Warehouse ID.", color="danger", dismissable=True),
            *NO_BT)

    if not eval_table or not score_table:
        return (dash.no_update, dbc.Alert(
            "Select both an Evaluation Table and a Scoring Table.",
            color="warning", dismissable=True),
            *NO_BT)

    # ---- Fetch forecast data ----
    try:
        df_raw, group_col, date_col, value_col = fetch_forecast_data(
            catalog, schema, eval_table, score_table, warehouse_id,
            run_ids=run_ids, run_date=run_date,
            unique_ids=unique_ids or None,
            model=selected_model,
            group_col=selected_group_col,
            eval_cm=eval_cm, score_cm=score_cm,
        )
    except Exception as exc:
        return (
            dbc.Alert(
                [
                    html.Strong("Error loading forecast data"),
                    html.Br(), html.Code(str(exc)),
                    html.Details(
                        [html.Summary("Traceback"), html.Pre(traceback.format_exc())],
                        className="mt-2",
                    ),
                ],
                color="danger",
            ),
            html.Div(),
            *NO_BT,
        )

    if df_raw.empty:
        return (dbc.Alert("No forecast data found for the selected filters.", color="info"),
                html.Div(), *NO_BT)

    # Explode arrays
    df_flat = df_raw.copy()
    df_flat = df_flat.explode(["forecast_dates", "forecast_values"])
    df_flat["forecast_dates"] = pd.to_datetime(df_flat["forecast_dates"], errors="coerce")
    df_flat["forecast_values"] = pd.to_numeric(df_flat["forecast_values"], errors="coerce")
    df_flat = df_flat.dropna(subset=["forecast_dates", "forecast_values"])
    df_flat = df_flat.sort_values([group_col, "forecast_dates"])

    # ---- If group IDs selected → plot with historical data ----
    if unique_ids and train_table:
        try:
            df_hist, hist_group, hist_date, hist_value = fetch_training_data(
                catalog, schema, train_table, warehouse_id, unique_ids,
                known_group_col=group_col, target_col=selected_target_col,
            )
            df_hist[hist_date] = pd.to_datetime(df_hist[hist_date], errors="coerce")
            df_hist[hist_value] = pd.to_numeric(df_hist[hist_value], errors="coerce")
            df_hist = df_hist.dropna(subset=[hist_date, hist_value])
            df_hist = df_hist.sort_values([hist_group, hist_date])
        except Exception as exc:
            return (
                dbc.Alert(
                    [
                        html.Strong("Error loading training data"),
                        html.Br(), html.Code(str(exc)),
                        html.Details(
                            [html.Summary("Traceback"), html.Pre(traceback.format_exc())],
                            className="mt-2",
                        ),
                    ],
                    color="danger",
                ),
                html.Div(),
                *NO_BT,
            )

        fig = go.Figure()
        groups = sorted(df_flat[group_col].unique())
        for i, gid in enumerate(groups):
            h_color = HIST_COLORS[i % len(HIST_COLORS)]
            f_color = FCST_COLORS[i % len(FCST_COLORS)]
            gid_str = str(gid)
            model_name = df_flat.loc[df_flat[group_col] == gid, "model"].iloc[0]
            legend_label = f"{gid_str} ({model_name})"

            hist_mask = df_hist[hist_group].astype(str) == gid_str
            h = df_hist.loc[hist_mask]
            if not h.empty:
                fig.add_trace(go.Scatter(
                    x=h[hist_date], y=h[hist_value],
                    mode="lines",
                    name=f"{gid_str} — historical",
                    legendgroup=gid_str,
                    line=dict(color=h_color, width=2),
                ))

            f_mask = df_flat[group_col] == gid
            f = df_flat.loc[f_mask]
            if not f.empty:
                x_vals = f["forecast_dates"].tolist()
                y_vals = f["forecast_values"].tolist()
                if not h.empty:
                    x_vals = [h[hist_date].iloc[-1]] + x_vals
                    y_vals = [h[hist_value].iloc[-1]] + y_vals
                fig.add_trace(go.Scatter(
                    x=x_vals, y=y_vals,
                    mode="lines",
                    name=f"{legend_label} — forecast",
                    legendgroup=gid_str,
                    line=dict(color=f_color, width=2, dash="dash"),
                ))

        fig.update_layout(
            template="plotly_white",
            title=f"Historical + Forecast ({len(groups)} series)",
            xaxis_title="",
            yaxis_title=hist_value,
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.08),
                type="date",
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
            margin=dict(t=50, b=150),
            hovermode="x unified",
        )

        content = html.Div([
            dbc.Badge(f"{len(groups)} series", color="success", className="mb-2 me-2"),
            dcc.Graph(figure=fig),
        ])

        # Populate backtest tabs
        try:
            bt_dates = fetch_backtest_start_dates(
                catalog, schema, eval_table, warehouse_id,
                group_col, unique_ids, run_date=run_date, run_ids=run_ids,
                col_map=eval_cm,
            )
            bt_tabs = [
                dbc.Tab(label=d, tab_id=d) for d in bt_dates
            ]
            bt_active = bt_dates[0] if bt_dates else None
            bt_style = {"display": "block"} if bt_dates else {"display": "none"}
        except Exception:
            bt_tabs, bt_active, bt_style = [], None, {"display": "none"}

        return content, html.Div(), bt_tabs, bt_active, bt_style

    # ---- No group IDs selected → raw table only ----
    df_flat = df_flat.rename(columns={
        "forecast_dates": date_col,
        "forecast_values": value_col,
        "avg_metric": "avg_metric_value",
    })

    content = html.Div(
        [
            dbc.Badge(
                f"{df_flat[group_col].nunique()} series, {len(df_flat):,} rows",
                color="success",
                className="mb-2",
            ),
            build_datatable(df_flat, "dt-forecast"),
        ]
    )
    return content, html.Div(), *NO_BT


@callback(
    Output("backtest-container", "children"),
    Input("backtest-tabs", "active_tab"),
    State("input-warehouse", "value"),
    State("input-catalog", "value"),
    State("input-schema", "value"),
    State("select-train-table", "value"),
    State("select-eval-table", "value"),
    State("input-run-date", "date"),
    State("select-run-id", "value"),
    State("select-model", "value"),
    State("select-unique-ids", "value"),
    State("select-group-col", "value"),
    State("select-target-col", "value"),
    State("select-metric-name", "value"),
    State("eval-col-map", "data"),
    prevent_initial_call=True,
)
def render_backtest(backtest_date, warehouse_id, catalog, schema, train_table,
                    eval_table, run_date, run_ids, selected_model, unique_ids,
                    selected_group_col, selected_target_col, selected_metric,
                    eval_cm):
    if not backtest_date or not unique_ids or not train_table or not eval_table:
        return html.Div()

    warehouse_id = warehouse_id or ""
    run_date = run_date or ""
    run_ids = run_ids or []
    selected_model = selected_model or "__best__"
    selected_group_col = selected_group_col or ""
    selected_target_col = selected_target_col or ""
    eval_cm = {**(eval_cm or {})}
    if selected_metric:
        eval_cm["metric_value"] = selected_metric

    try:
        if selected_group_col:
            group_col = selected_group_col
        else:
            eval_cols = describe_table(catalog, schema, eval_table, warehouse_id)
            group_col = eval_cols[0][0]

        df_hist, hist_group, hist_date, hist_value = fetch_training_data(
            catalog, schema, train_table, warehouse_id, unique_ids,
            known_group_col=group_col, target_col=selected_target_col,
        )
        df_hist[hist_date] = pd.to_datetime(df_hist[hist_date], errors="coerce")
        df_hist[hist_value] = pd.to_numeric(df_hist[hist_value], errors="coerce")
        df_hist = df_hist.dropna(subset=[hist_date, hist_value])
        df_hist = df_hist.sort_values([hist_group, hist_date])

        first_group = df_hist[hist_group].iloc[0] if not df_hist.empty else None
        if first_group is not None:
            g_dates = df_hist.loc[df_hist[hist_group] == first_group, hist_date].sort_values()
            offset = _infer_offset(g_dates)
        else:
            offset = pd.DateOffset(months=1)

        df_bt = fetch_backtest_data(
            catalog, schema, eval_table, warehouse_id,
            group_col, unique_ids, backtest_date,
            run_date=run_date, run_ids=run_ids,
            model=selected_model,
            col_map=eval_cm,
        )

        if df_bt.empty:
            return dbc.Alert("No backtest data found for the selected date.", color="info")

        bt_start = pd.to_datetime(backtest_date)
        if not df_hist.empty and hasattr(df_hist[hist_date].dt, "tz") and df_hist[hist_date].dt.tz is not None:
            bt_start = bt_start.tz_localize(df_hist[hist_date].dt.tz)

        fig = go.Figure()
        groups = sorted(df_bt[group_col].astype(str).unique())

        for i, gid in enumerate(groups):
            h_color = HIST_COLORS[i % len(HIST_COLORS)]
            f_color = FCST_COLORS[i % len(FCST_COLORS)]

            bt_row = df_bt.loc[df_bt[group_col].astype(str) == gid].iloc[0]
            model_name = bt_row["model"]

            def _to_float_list(val):
                if val is None:
                    return []
                result = []
                for x in val:
                    if x is None:
                        continue
                    try:
                        f = float(x)
                        if not math.isnan(f):
                            result.append(f)
                    except (TypeError, ValueError):
                        continue
                return result

            forecast_arr = _to_float_list(bt_row["forecast"])
            actual_arr = _to_float_list(bt_row["actual"])

            n_points = max(len(forecast_arr), len(actual_arr))
            bt_dates = [bt_start + offset * j for j in range(n_points)]

            # Historical + Actual (same color, single legend entry)
            hist_mask = (df_hist[hist_group].astype(str) == gid) & (df_hist[hist_date] < bt_start)
            h = df_hist.loc[hist_mask]

            hist_x = list(h[hist_date]) if not h.empty else []
            hist_y = [float(v) for v in h[hist_value]] if not h.empty else []

            if len(actual_arr) > 0:
                act_dates = bt_dates[:len(actual_arr)]
                hist_x = hist_x + list(act_dates)
                hist_y = hist_y + actual_arr

            if hist_x:
                fig.add_trace(go.Scatter(
                    x=hist_x, y=hist_y,
                    mode="lines",
                    name=f"{gid} — historical + actual",
                    legendgroup=gid,
                    line=dict(color=h_color, width=2),
                ))

            # Forecast
            if len(forecast_arr) > 0:
                fcst_dates = bt_dates[:len(forecast_arr)]
                fcst_x = list(fcst_dates)
                fcst_y = list(forecast_arr)
                if not h.empty:
                    fcst_x = [h[hist_date].iloc[-1]] + fcst_x
                    fcst_y = [float(h[hist_value].iloc[-1])] + fcst_y
                fig.add_trace(go.Scatter(
                    x=fcst_x, y=fcst_y,
                    mode="lines",
                    name=f"{gid} ({model_name}) — forecast",
                    legendgroup=gid,
                    line=dict(color=f_color, width=2, dash="dash"),
                ))

        fig.update_layout(
            template="plotly_white",
            title=f"Backtest @ {backtest_date} ({len(groups)} series)",
            xaxis_title="",
            yaxis_title=hist_value,
            xaxis=dict(
                rangeslider=dict(visible=True, thickness=0.08),
                type="date",
            ),
            legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5),
            margin=dict(t=50, b=150),
            hovermode="x unified",
        )

        return html.Div([
            dbc.Badge(f"{len(groups)} series", color="success", className="mb-2 me-2"),
            dcc.Graph(figure=fig),
        ])

    except Exception as exc:
        return dbc.Alert(
            [
                html.Strong("Error loading backtest data"),
                html.Br(), html.Code(str(exc)),
                html.Details(
                    [html.Summary("Traceback"), html.Pre(traceback.format_exc())],
                    className="mt-2",
                ),
            ],
            color="danger",
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
