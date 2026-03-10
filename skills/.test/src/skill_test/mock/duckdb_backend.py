"""DuckDB backend for Tier 1 tests.

Seeds an in-memory DuckDB database with test data matching the Databricks
catalog/schema/table structure, transpiles Databricks SQL to DuckDB SQL
via SQLGlot, and executes queries returning CSV-formatted results.
"""

from __future__ import annotations

import re

import duckdb
import sqlglot

# ---------------------------------------------------------------------------
# Type mapping: DuckDB types back to Databricks display types
# ---------------------------------------------------------------------------

_DUCKDB_TO_DATABRICKS_TYPE = {
    "VARCHAR": "STRING",
    "TEXT": "STRING",
    "BIGINT": "BIGINT",
    "INTEGER": "INT",
    "INT": "INT",
    "SMALLINT": "SMALLINT",
    "TINYINT": "TINYINT",
    "DOUBLE": "DOUBLE",
    "FLOAT": "FLOAT",
    "REAL": "FLOAT",
    "BOOLEAN": "BOOLEAN",
    "DATE": "DATE",
    "TIMESTAMP": "TIMESTAMP",
    "TIMESTAMP WITH TIME ZONE": "TIMESTAMP",
    "TIMESTAMPTZ": "TIMESTAMP",
}


def _to_databricks_type(duckdb_type: str) -> str:
    """Map a DuckDB column type string to its Databricks equivalent."""
    upper = duckdb_type.upper()
    return _DUCKDB_TO_DATABRICKS_TYPE.get(upper, upper)


# ---------------------------------------------------------------------------
# Database seeding
# ---------------------------------------------------------------------------


def create_test_database() -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB database seeded with test data.

    Mirrors the fixtures previously in databricks_responses.py:
    - test_catalog."default".raw_timeseries: 50 series x 365 days = 18,250 rows
    - test_catalog."default".raw_weekly_timeseries: 10 series x 52 weeks = 520 rows
    - test_catalog."default".raw_monthly_timeseries: 10 series x 24 months = 240 rows
    - test_catalog."default".metadata: 2 rows
    - test_catalog."default".lookup_table: 2 rows
    - test_catalog."default".mmf_train_data: daily, filtered nulls
    - test_catalog."default".mmf_train_data_weekly: aligned to Sundays
    - test_catalog."default".mmf_train_data_monthly: aligned to month-end

    Date ranges:
    - Daily: 2022-01-01 to 2023-12-31
    - Weekly: 2022-01-05 to 2022-12-28 (Wednesdays, needs alignment)
    - Monthly: 2022-01-15 to 2023-12-15 (15th of month, needs alignment)

    Notable series:
    - series_42 (5 nulls), series_17 (3 nulls), series_99 (2 negatives)
    - wseries_03 (3 nulls), wseries_07 (2 negatives)
    - mseries_04 (3 nulls), mseries_08 (2 negatives)
    """
    conn = duckdb.connect(":memory:")

    # Set up catalog and schema to match Databricks three-part naming
    conn.execute("ATTACH ':memory:' AS test_catalog")
    conn.execute('CREATE SCHEMA test_catalog."default"')

    # --- raw_timeseries: 50 series x 365 days ---
    conn.execute("""
        CREATE TABLE test_catalog."default".raw_timeseries (
            unique_id VARCHAR,
            ds TIMESTAMP,
            y DOUBLE,
            category VARCHAR,
            region VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO test_catalog."default".raw_timeseries
        SELECT
            'series_' || LPAD(CAST(s AS VARCHAR), 2, '0') AS unique_id,
            CAST(DATE '2022-01-01' + INTERVAL (d) DAY AS TIMESTAMP) AS ds,
            CASE
                WHEN s = 42 AND d IN (10, 50, 100, 200, 300) THEN NULL
                WHEN s = 17 AND d IN (20, 120, 250) THEN NULL
                WHEN s = 99 AND d IN (30, 180) THEN -(d + 1.0)
                ELSE (s * 10.0) + (d * 0.1) + (HASH(s * 1000 + d) % 500) / 100.0
            END AS y,
            CASE WHEN s % 3 = 0 THEN 'electronics'
                 WHEN s % 3 = 1 THEN 'clothing'
                 ELSE 'food' END AS category,
            CASE WHEN s % 2 = 0 THEN 'north' ELSE 'south' END AS region
        FROM generate_series(0, 49) AS t(s),
             generate_series(0, 364) AS u(d)
    """)

    # --- metadata ---
    conn.execute("""
        CREATE TABLE test_catalog."default".metadata (
            id INTEGER,
            name VARCHAR,
            created_at TIMESTAMP,
            is_active BOOLEAN
        )
    """)
    conn.execute("""
        INSERT INTO test_catalog."default".metadata VALUES
        (1, 'item_a', '2023-01-01', true),
        (2, 'item_b', '2023-06-15', false)
    """)

    # --- lookup_table ---
    conn.execute("""
        CREATE TABLE test_catalog."default".lookup_table (
            key VARCHAR,
            value VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO test_catalog."default".lookup_table VALUES
        ('k1', 'v1'), ('k2', 'v2')
    """)

    # --- raw_weekly_timeseries: 10 series x 52 weeks ---
    # Dates on Wednesdays (not end-of-week) so alignment is testable
    conn.execute("""
        CREATE TABLE test_catalog."default".raw_weekly_timeseries (
            unique_id VARCHAR,
            ds TIMESTAMP,
            y DOUBLE,
            category VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO test_catalog."default".raw_weekly_timeseries
        SELECT
            'wseries_' || LPAD(CAST(s AS VARCHAR), 2, '0') AS unique_id,
            CAST(DATE '2022-01-05' + INTERVAL (w * 7) DAY AS TIMESTAMP) AS ds,
            CASE
                WHEN s = 3 AND w IN (5, 20, 40) THEN NULL
                WHEN s = 7 AND w IN (10, 30) THEN -(w + 1.0)
                ELSE (s * 5.0) + (w * 0.5) + (HASH(s * 100 + w) % 200) / 100.0
            END AS y,
            CASE WHEN s % 2 = 0 THEN 'electronics' ELSE 'clothing' END AS category
        FROM generate_series(0, 9) AS t(s),
             generate_series(0, 51) AS u(w)
    """)

    # --- raw_monthly_timeseries: 10 series x 24 months ---
    # Dates on 15th of each month (not end-of-month) so alignment is testable
    conn.execute("""
        CREATE TABLE test_catalog."default".raw_monthly_timeseries (
            unique_id VARCHAR,
            ds TIMESTAMP,
            y DOUBLE,
            category VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO test_catalog."default".raw_monthly_timeseries
        SELECT
            'mseries_' || LPAD(CAST(s AS VARCHAR), 2, '0') AS unique_id,
            CAST(DATE '2022-01-15' + INTERVAL (m) MONTH AS TIMESTAMP) AS ds,
            CASE
                WHEN s = 4 AND m IN (3, 10, 20) THEN NULL
                WHEN s = 8 AND m IN (6, 18) THEN -(m + 1.0)
                ELSE (s * 20.0) + (m * 2.0) + (HASH(s * 100 + m) % 300) / 100.0
            END AS y,
            CASE WHEN s % 2 = 0 THEN 'food' ELSE 'clothing' END AS category
        FROM generate_series(0, 9) AS t(s),
             generate_series(0, 23) AS u(m)
    """)

    # --- mmf_train_data: pre-created so /run-mmf tests have data to query ---
    conn.execute("""
        CREATE TABLE test_catalog."default".mmf_train_data AS
        SELECT
            unique_id,
            ds,
            y
        FROM test_catalog."default".raw_timeseries
        WHERE y IS NOT NULL
    """)

    # --- mmf_train_data_weekly: aligned to Sundays ---
    conn.execute("""
        CREATE TABLE test_catalog."default".mmf_train_data_weekly AS
        SELECT
            CAST(unique_id AS VARCHAR) AS unique_id,
            CAST(DATE_TRUNC('week', ds) + INTERVAL 6 DAY AS TIMESTAMP) AS ds,
            SUM(y) AS y
        FROM test_catalog."default".raw_weekly_timeseries
        WHERE y IS NOT NULL
        GROUP BY unique_id, DATE_TRUNC('week', ds) + INTERVAL 6 DAY
    """)

    # --- mmf_train_data_monthly: aligned to month-end ---
    conn.execute("""
        CREATE TABLE test_catalog."default".mmf_train_data_monthly AS
        SELECT
            CAST(unique_id AS VARCHAR) AS unique_id,
            CAST(LAST_DAY(ds) AS TIMESTAMP) AS ds,
            SUM(y) AS y
        FROM test_catalog."default".raw_monthly_timeseries
        WHERE y IS NOT NULL
        GROUP BY unique_id, LAST_DAY(ds)
    """)

    return conn


# ---------------------------------------------------------------------------
# SQL transpilation and execution
# ---------------------------------------------------------------------------


def transpile_sql(databricks_sql: str) -> str:
    """Transpile a Databricks SQL statement to DuckDB dialect via SQLGlot."""
    try:
        results = sqlglot.transpile(databricks_sql, read="databricks", write="duckdb")
        return results[0]
    except sqlglot.errors.ParseError:
        # If SQLGlot can't parse it, return as-is and let DuckDB try
        return databricks_sql


def _format_as_csv(description: list, rows: list) -> str:
    """Format DuckDB query results as CSV string (Databricks SQL output format)."""
    col_names = [col[0] for col in description]
    lines = [",".join(col_names)]
    for row in rows:
        lines.append(",".join(_format_value(v) for v in row))
    return "\n".join(lines)


def _format_value(v: object) -> str:
    """Format a single value for CSV output, matching Databricks formatting."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float):
        # Format cleanly: 1.0 -> "1.0", 1.37 -> "1.37", avoid trailing zeros
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:g}"
    # Timestamps: format as date-only if time is midnight
    s = str(v)
    if " 00:00:00" in s:
        s = s.replace(" 00:00:00", "")
    # Strip timezone info if present (handles any offset, e.g. +00:00, -05:00)
    s = re.sub(r"[+-]\d{2}:\d{2}$", "", s)
    return s


def _is_show_tables(sql: str) -> bool:
    """Check if the SQL is a SHOW TABLES command."""
    return bool(re.match(r"\s*SHOW\s+TABLES\s", sql, re.IGNORECASE))


def _is_describe(sql: str) -> bool:
    """Check if the SQL is a DESCRIBE command."""
    return bool(re.match(r"\s*DESCRIBE\s", sql, re.IGNORECASE))


def execute_sql(conn: duckdb.DuckDBPyConnection, databricks_sql: str, show_tables_fixture: str) -> str:
    """Execute a Databricks SQL statement against DuckDB.

    - SHOW TABLES: returns the fixture (DuckDB doesn't support SHOW TABLES IN)
    - DESCRIBE: executes on DuckDB, remaps output to Databricks column format
    - Everything else: transpiles via SQLGlot and executes

    Args:
        conn: DuckDB connection with seeded test data.
        databricks_sql: The SQL statement as generated by the agent.
        show_tables_fixture: Pre-canned SHOW TABLES response.

    Returns:
        CSV-formatted result string.
    """
    sql = databricks_sql.strip()

    # SHOW TABLES: DuckDB doesn't support "SHOW TABLES IN catalog.schema"
    if _is_show_tables(sql):
        return show_tables_fixture

    # DESCRIBE: execute on DuckDB, remap columns to Databricks format
    if _is_describe(sql):
        return _handle_describe(conn, sql)

    # All other queries: transpile and execute
    duckdb_sql = transpile_sql(sql)
    try:
        result = conn.execute(duckdb_sql)
    except duckdb.Error as e:
        return f"error\n{e}"
    if result.description:
        rows = result.fetchall()
        return _format_as_csv(result.description, rows)
    return "status\nSUCCEEDED"


def _handle_describe(conn: duckdb.DuckDBPyConnection, sql: str) -> str:
    """Execute DESCRIBE on DuckDB and remap output to Databricks format.

    DuckDB DESCRIBE returns: column_name, column_type, null, key, default, extra
    Databricks returns: col_name, data_type, comment
    """
    # SQLGlot transpiles DESCRIBE TABLE x -> DESCRIBE x
    duckdb_sql = transpile_sql(sql)
    result = conn.execute(duckdb_sql)
    rows = result.fetchall()

    lines = ["col_name,data_type,comment"]
    for row in rows:
        col_name = row[0]
        col_type = _to_databricks_type(str(row[1]))
        lines.append(f"{col_name},{col_type},")
    return "\n".join(lines)
