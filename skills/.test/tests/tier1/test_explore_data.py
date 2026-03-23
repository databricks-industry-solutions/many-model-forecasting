"""Tier 1 tests for the /prep-and-clean-data skill (formerly /explore-data).

Validates that the agent issues correct SQL queries, maps columns properly,
creates the {use_case}_train_data table, and includes cleaning steps.
"""

import pytest


@pytest.mark.tier1
class TestPrepAndCleanTableDiscovery:
    """Test that the agent discovers and profiles tables correctly."""

    def test_lists_tables_and_describes_candidate(self, skill_prompt, run_agent):
        """Agent should SHOW TABLES then DESCRIBE the candidate table."""
        prompt = skill_prompt("prep-and-clean-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. Use case name is 'test'. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        assert any("SHOW TABLES" in s for s in sql_upper), f"Expected SHOW TABLES in SQL calls. Got: {sql_texts}"

        assert any("DESCRIBE" in s and "RAW_TIMESERIES" in s for s in sql_upper), (
            f"Expected DESCRIBE ... raw_timeseries in SQL calls. Got: {sql_texts}"
        )

    def test_maps_columns_correctly(self, skill_prompt, run_agent):
        """Agent should identify unique_id, ds, y from the raw_timeseries schema."""
        prompt = skill_prompt("prep-and-clean-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. Use case name is 'test'. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        response = result["final_response"].lower()

        assert "unique_id" in response, f"Expected 'unique_id' in response. Got: {result['final_response'][:500]}"
        assert "ds" in response, f"Expected 'ds' in response. Got: {result['final_response'][:500]}"
        assert "metadata" not in response or "raw_timeseries" in response, (
            "Agent should focus on raw_timeseries, not metadata"
        )

    def test_creates_train_data(self, skill_prompt, run_agent):
        """Agent should issue CREATE TABLE for {use_case}_train_data."""
        prompt = skill_prompt("prep-and-clean-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. Use case name is 'test'. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        create_stmts = [
            s for s in sql_upper if ("CREATE TABLE" in s or "CREATE OR REPLACE" in s) and "_TRAIN_DATA" in s
        ]
        assert len(create_stmts) > 0, f"Expected CREATE TABLE _train_data. Got SQL calls: {sql_texts}"

        create_sql = create_stmts[0]
        assert "UNIQUE_ID" in create_sql, "CREATE should include unique_id column"
        assert "DS" in create_sql, "CREATE should include ds column"
        assert "Y" in create_sql, "CREATE should include y column"


@pytest.mark.tier1
class TestPrepAndCleanProfiling:
    """Test that the agent profiles table statistics (row count, date range, distinct groups)."""

    def test_runs_profiling_queries(self, skill_prompt, run_agent):
        """Agent should run COUNT(*), MIN/MAX dates, and COUNT(DISTINCT unique_id)."""
        prompt = skill_prompt("prep-and-clean-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. Use case name is 'test'. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        assert any("COUNT(*)" in s or "COUNT (" in s for s in sql_upper), (
            f"Expected COUNT(*) profiling query. Got SQL calls: {sql_texts}"
        )

        assert any(("MIN(" in s or "MIN (" in s) and ("MAX(" in s or "MAX (" in s) for s in sql_upper), (
            f"Expected MIN/MAX date range query. Got SQL calls: {sql_texts}"
        )

        assert any("COUNT" in s and "DISTINCT" in s and "UNIQUE_ID" in s for s in sql_upper), (
            f"Expected COUNT(DISTINCT unique_id) query. Got SQL calls: {sql_texts}"
        )


@pytest.mark.tier1
class TestPrepAndCleanFrequencyAlignment:
    """Test that the agent applies frequency-dependent date alignment in CREATE TABLE."""

    def test_creates_train_data_weekly(self, skill_prompt, run_agent):
        """Agent should use DATE_TRUNC('week', ...) and GROUP BY for weekly data."""
        prompt = skill_prompt("prep-and-clean-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. Use case name is 'test'. "
                "The table to use is raw_weekly_timeseries with freq=W. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        create_stmts = [
            s for s in sql_upper if ("CREATE TABLE" in s or "CREATE OR REPLACE" in s) and "_TRAIN_DATA" in s
        ]
        assert len(create_stmts) > 0, f"Expected CREATE TABLE _train_data. Got SQL calls: {sql_texts}"

        create_sql = create_stmts[0]

        assert "DATE_TRUNC" in create_sql and "WEEK" in create_sql, (
            f"Expected DATE_TRUNC('week', ...) in weekly CREATE TABLE. Got: {create_stmts[0]}"
        )

        assert "GROUP BY" in create_sql, f"Expected GROUP BY in weekly CREATE TABLE. Got: {create_stmts[0]}"

    def test_creates_train_data_monthly(self, skill_prompt, run_agent):
        """Agent should use LAST_DAY() or DATE_TRUNC('month', ...) and GROUP BY for monthly data."""
        prompt = skill_prompt("prep-and-clean-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. Use case name is 'test'. "
                "The table to use is raw_monthly_timeseries with freq=M. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        create_stmts = [
            s for s in sql_upper if ("CREATE TABLE" in s or "CREATE OR REPLACE" in s) and "_TRAIN_DATA" in s
        ]
        assert len(create_stmts) > 0, f"Expected CREATE TABLE _train_data. Got SQL calls: {sql_texts}"

        create_sql = create_stmts[0]

        has_last_day = "LAST_DAY" in create_sql
        has_date_trunc_month = "DATE_TRUNC" in create_sql and "MONTH" in create_sql
        assert has_last_day or has_date_trunc_month, (
            f"Expected LAST_DAY() or DATE_TRUNC('month', ...) in monthly CREATE TABLE. Got: {create_stmts[0]}"
        )

        assert "GROUP BY" in create_sql, f"Expected GROUP BY in monthly CREATE TABLE. Got: {create_stmts[0]}"
