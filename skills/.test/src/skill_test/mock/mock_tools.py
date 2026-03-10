"""Mock Databricks tool definitions and handlers for Tier 1 tests.

Provides OpenAI function-calling format tool definitions and Python handlers.
SQL queries are executed against a DuckDB in-memory database via SQLGlot
transpilation. Non-SQL tools return pre-canned fixture data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from skill_test.mock.duckdb_backend import execute_sql
from skill_test.mock.fixtures import FIXTURES

if TYPE_CHECKING:
    import duckdb

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "connect_to_workspace",
            "description": "Connect to a Databricks workspace by name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "workspace": {
                        "type": "string",
                        "description": "Name or URL of the workspace to connect to",
                    }
                },
                "required": ["workspace"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_sql",
            "description": "Execute a SQL statement on Databricks. Returns results as CSV.",
            "parameters": {
                "type": "object",
                "properties": {
                    "statement": {
                        "type": "string",
                        "description": "The SQL statement to execute",
                    },
                    "parameters": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "Optional query parameters",
                    },
                },
                "required": ["statement"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_warehouses",
            "description": "List SQL warehouses available in the Databricks workspace.",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_job",
            "description": "Create a Databricks Workflow job with tasks and cluster definitions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Job name",
                    },
                    "tasks": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of task definitions",
                    },
                    "job_clusters": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": "List of job cluster definitions",
                    },
                },
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_job",
            "description": "Run a Databricks job by job_id. Returns run_id and status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "job_id": {
                        "type": "integer",
                        "description": "The job ID to run",
                    }
                },
                "required": ["job_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "upload_notebook",
            "description": "Upload a notebook to the Databricks workspace. Content is Python source code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Workspace path for the notebook (e.g. /notebooks/run_local)",
                    },
                    "content": {
                        "type": "string",
                        "description": "Python source code content of the notebook",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool handler factory
# ---------------------------------------------------------------------------


def create_tool_handlers(conn: duckdb.DuckDBPyConnection) -> dict[str, callable]:
    """Create tool handlers bound to a DuckDB connection.

    SQL queries are transpiled via SQLGlot and executed on DuckDB.
    Non-SQL tools return pre-canned fixtures.
    """

    def handle_connect_to_workspace(workspace: str = "", **kwargs: object) -> str:
        return FIXTURES["workspace_connected"]

    def handle_execute_sql(statement: str, parameters: list | None = None, **kwargs: object) -> str:
        return execute_sql(conn, statement, FIXTURES["show_tables"])

    def handle_list_warehouses(**kwargs: object) -> str:
        return FIXTURES["warehouses"]

    def handle_create_job(
        name: str = "", tasks: list | None = None, job_clusters: list | None = None, **kwargs: object
    ) -> str:
        return FIXTURES["create_job_success"]

    def handle_run_job(job_id: int = 0, **kwargs: object) -> str:
        return FIXTURES["run_job_success"]

    def handle_upload_notebook(path: str = "", content: str = "", **kwargs: object) -> str:
        return FIXTURES["upload_notebook_success"]

    return {
        "connect_to_workspace": handle_connect_to_workspace,
        "execute_sql": handle_execute_sql,
        "list_warehouses": handle_list_warehouses,
        "create_job": handle_create_job,
        "run_job": handle_run_job,
        "upload_notebook": handle_upload_notebook,
    }
