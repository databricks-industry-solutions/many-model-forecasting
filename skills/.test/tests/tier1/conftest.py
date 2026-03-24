"""Pytest fixtures for Tier 1 skill logic tests."""

import os
from pathlib import Path

import pytest

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from openai import OpenAI

from skill_test.mock.agent_runner import run_skill_agent
from skill_test.mock.duckdb_backend import create_test_database
from skill_test.mock.mock_tools import TOOL_DEFINITIONS, create_tool_handlers

# Mapping from mmf-agent skill names to mmf-dev-kit skill markdown files.
# The mmf-dev-kit stores skills under databricks-skills/many-model-forecasting/.
_SKILL_FILE_MAP = {
    "prep-and-clean-data": "1-prep-and-clean-data.md",
    "profile-and-classify-series": "2-profile-and-classify-series.md",
    "provision-forecasting-resources": "3-provision-forecasting-resources.md",
    "execute-mmf-forecast": "4-execute-mmf-forecast.md",
    "post-process-and-evaluate": "5-post-process-and-evaluate.md",
    # Legacy aliases for backward compatibility in existing tests
    "explore-data": "1-prep-and-clean-data.md",
    "setup-cluster": "3-provision-forecasting-resources.md",
    "run-mmf": "4-execute-mmf-forecast.md",
}

# Root of the mmf-dev-kit repo (two levels up from .test/tests/tier1/)
_REPO_ROOT = Path(__file__).parents[3]
_SKILLS_DIR = _REPO_ROOT / "databricks-skills" / "many-model-forecasting"


@pytest.fixture(scope="session")
def client():
    """OpenAI client pointing at Databricks Foundation Model API."""
    host = os.environ.get(
        "DATABRICKS_HOST",
        "https://fevm-retail-webapp-commerce.cloud.databricks.com",
    )
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        pytest.skip("DATABRICKS_TOKEN not set")
    return OpenAI(api_key=token, base_url=f"{host}/serving-endpoints")


@pytest.fixture(scope="session")
def duckdb_conn():
    """In-memory DuckDB database seeded with test data."""
    conn = create_test_database()
    yield conn
    conn.close()


@pytest.fixture
def skill_prompt():
    """Load a skill markdown file as system prompt.

    Maps mmf-agent skill names (e.g. "explore-data") to the corresponding
    markdown file in databricks-skills/many-model-forecasting/.
    """

    def _load(skill_name: str) -> str:
        filename = _SKILL_FILE_MAP.get(skill_name)
        if filename is None:
            raise ValueError(
                f"Unknown skill '{skill_name}'. "
                f"Known skills: {list(_SKILL_FILE_MAP.keys())}"
            )
        path = _SKILLS_DIR / filename
        return path.read_text()

    return _load


@pytest.fixture
def tools():
    """OpenAI-format tool definitions for mock Databricks tools."""
    return TOOL_DEFINITIONS


@pytest.fixture
def tool_handlers(duckdb_conn):
    """Tool handlers bound to the DuckDB test database."""
    return create_tool_handlers(duckdb_conn)


@pytest.fixture
def run_agent(client, tools, tool_handlers):
    """Convenience fixture: returns a callable that runs the agent loop."""

    def _run(system_prompt: str, user_prompt: str, max_turns: int = 20) -> dict:
        return run_skill_agent(
            client=client,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            tools=tools,
            tool_handlers=tool_handlers,
            max_turns=max_turns,
        )

    return _run
