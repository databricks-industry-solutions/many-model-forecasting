"""Mock Databricks infrastructure for tier 1 skill tests.

Provides a DuckDB-backed execution environment that transpiles Databricks SQL
via SQLGlot, plus mock tool definitions and an agent runner loop.
"""

from skill_test.mock.duckdb_backend import create_test_database, execute_sql, transpile_sql
from skill_test.mock.mock_tools import TOOL_DEFINITIONS, create_tool_handlers
from skill_test.mock.agent_runner import run_skill_agent

__all__ = [
    "create_test_database",
    "execute_sql",
    "transpile_sql",
    "TOOL_DEFINITIONS",
    "create_tool_handlers",
    "run_skill_agent",
]
