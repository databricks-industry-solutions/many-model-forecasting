"""Tier 1 tests for the /explore-data skill.

Validates that the agent issues correct SQL queries, maps columns properly,
and creates the mmf_train_data table when given mock Databricks responses.
"""

import pytest


@pytest.mark.tier1
class TestExploreDataTableDiscovery:
    """Test that the agent discovers and profiles tables correctly."""

    def test_lists_tables_and_describes_candidate(self, skill_prompt, run_agent):
        """Agent should SHOW TABLES then DESCRIBE the candidate table."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        # Must issue SHOW TABLES
        assert any("SHOW TABLES" in s for s in sql_upper), f"Expected SHOW TABLES in SQL calls. Got: {sql_texts}"

        # Must DESCRIBE the candidate table
        assert any("DESCRIBE" in s and "RAW_TIMESERIES" in s for s in sql_upper), (
            f"Expected DESCRIBE ... raw_timeseries in SQL calls. Got: {sql_texts}"
        )

    def test_maps_columns_correctly(self, skill_prompt, run_agent):
        """Agent should identify unique_id, ds, y from the raw_timeseries schema."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        response = result["final_response"].lower()

        # The agent's response should reference the correct column mappings
        assert "unique_id" in response, f"Expected 'unique_id' in response. Got: {result['final_response'][:500]}"
        assert "ds" in response, f"Expected 'ds' in response. Got: {result['final_response'][:500]}"
        # Check the agent doesn't confuse tables
        assert "metadata" not in response or "raw_timeseries" in response, (
            "Agent should focus on raw_timeseries, not metadata"
        )

    def test_creates_mmf_train_data(self, skill_prompt, run_agent):
        """Agent should issue CREATE TABLE for mmf_train_data."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        # Must issue CREATE TABLE for mmf_train_data
        create_stmts = [
            s for s in sql_upper if ("CREATE TABLE" in s or "CREATE OR REPLACE" in s) and "MMF_TRAIN_DATA" in s
        ]
        assert len(create_stmts) > 0, f"Expected CREATE TABLE mmf_train_data. Got SQL calls: {sql_texts}"

        # The CREATE statement should reference the correct columns
        create_sql = create_stmts[0]
        assert "UNIQUE_ID" in create_sql, "CREATE should include unique_id column"
        assert "DS" in create_sql, "CREATE should include ds column"
        assert "Y" in create_sql, "CREATE should include y column"


@pytest.mark.tier1
class TestExploreDataProfiling:
    """Test that the agent profiles table statistics (row count, date range, distinct groups)."""

    def test_runs_profiling_queries(self, skill_prompt, run_agent):
        """Agent should run COUNT(*), MIN/MAX dates, and COUNT(DISTINCT unique_id)."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        # Should run a row count query
        assert any("COUNT(*)" in s or "COUNT (" in s for s in sql_upper), (
            f"Expected COUNT(*) profiling query. Got SQL calls: {sql_texts}"
        )

        # Should query date range (MIN/MAX on ds)
        assert any(("MIN(" in s or "MIN (" in s) and ("MAX(" in s or "MAX (" in s) for s in sql_upper), (
            f"Expected MIN/MAX date range query. Got SQL calls: {sql_texts}"
        )

        # Should count distinct series
        assert any("COUNT" in s and "DISTINCT" in s and "UNIQUE_ID" in s for s in sql_upper), (
            f"Expected COUNT(DISTINCT unique_id) query. Got SQL calls: {sql_texts}"
        )

    def test_reports_profiling_results(self, skill_prompt, run_agent):
        """Agent should report row count, date range, and number of series in its response."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        response = result["final_response"]

        # Should mention the row count (18,250 rows in raw_timeseries)
        assert "18" in response, f"Expected mention of row count (~18,250). Got: {response[:500]}"

        # Should mention the number of distinct series (50)
        assert "50" in response, f"Expected mention of 50 distinct series. Got: {response[:500]}"

        # Should mention the date range (2022)
        assert "2022" in response, f"Expected mention of date range (2022). Got: {response[:500]}"


@pytest.mark.tier1
class TestExploreDataQuality:
    """Test that the agent runs data quality checks and reports issues."""

    def test_detects_data_quality_issues(self, skill_prompt, run_agent):
        """Agent should run quality checks and report missing/negative values."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Run all data quality checks. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        # Should run a missing values check (NULL detection query)
        assert any("NULL" in s and "UNIQUE_ID" in s for s in sql_upper), (
            f"Expected missing values quality check SQL. Got SQL calls: {sql_texts}"
        )

        # Should check for negative values
        assert any("< 0" in s or "<0" in s or "NEGATIVE" in s for s in sql_upper), (
            f"Expected negative values quality check SQL. Got SQL calls: {sql_texts}"
        )

    def test_reports_specific_problematic_series(self, skill_prompt, run_agent):
        """Agent should identify series_42 (nulls) and/or series_99 (negatives) in its report."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_timeseries. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Run all data quality checks and report which series have issues. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        # Combine response + tool results to check what the agent saw and reported
        response = result["final_response"]
        all_text = response
        for tc in result["tool_calls"]:
            all_text += " " + str(tc.get("arguments", ""))

        all_text_lower = all_text.lower()

        # The DuckDB data has series_42 with 5 nulls and series_17 with 3 nulls
        # Agent should find at least one of these in quality checks
        found_null_series = "series_42" in all_text_lower or "series_17" in all_text_lower
        # The DuckDB data has series_99 with 2 negative values
        found_negative_series = "series_99" in all_text_lower

        assert found_null_series or found_negative_series, (
            f"Expected agent to identify problematic series (series_42, series_17, or series_99). "
            f"Response: {response[:500]}"
        )


@pytest.mark.tier1
class TestExploreDataFrequencyAlignment:
    """Test that the agent applies frequency-dependent date alignment in CREATE TABLE."""

    def test_creates_mmf_train_data_weekly(self, skill_prompt, run_agent):
        """Agent should use DATE_TRUNC('week', ...) and GROUP BY for weekly data."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_weekly_timeseries with freq=W. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        # Must issue CREATE TABLE for mmf_train_data
        create_stmts = [
            s for s in sql_upper if ("CREATE TABLE" in s or "CREATE OR REPLACE" in s) and "MMF_TRAIN_DATA" in s
        ]
        assert len(create_stmts) > 0, f"Expected CREATE TABLE mmf_train_data. Got SQL calls: {sql_texts}"

        create_sql = create_stmts[0]

        # Should use DATE_TRUNC with WEEK for weekly alignment
        assert "DATE_TRUNC" in create_sql and "WEEK" in create_sql, (
            f"Expected DATE_TRUNC('week', ...) in weekly CREATE TABLE. Got: {create_stmts[0]}"
        )

        # Should use GROUP BY for aggregation
        assert "GROUP BY" in create_sql, f"Expected GROUP BY in weekly CREATE TABLE. Got: {create_stmts[0]}"

    def test_creates_mmf_train_data_monthly(self, skill_prompt, run_agent):
        """Agent should use LAST_DAY() or DATE_TRUNC('month', ...) and GROUP BY for monthly data."""
        prompt = skill_prompt("explore-data")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Explore data in test_catalog.default. "
                "The table to use is raw_monthly_timeseries with freq=M. "
                "Columns: unique_id=unique_id, ds=ds, y=y. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        sql_calls = [tc for tc in result["tool_calls"] if tc["name"] == "execute_sql"]
        sql_texts = [tc["arguments"].get("statement", "") for tc in sql_calls]
        sql_upper = [s.upper() for s in sql_texts]

        # Must issue CREATE TABLE for mmf_train_data
        create_stmts = [
            s for s in sql_upper if ("CREATE TABLE" in s or "CREATE OR REPLACE" in s) and "MMF_TRAIN_DATA" in s
        ]
        assert len(create_stmts) > 0, f"Expected CREATE TABLE mmf_train_data. Got SQL calls: {sql_texts}"

        create_sql = create_stmts[0]

        # Should use LAST_DAY or DATE_TRUNC with MONTH for monthly alignment
        has_last_day = "LAST_DAY" in create_sql
        has_date_trunc_month = "DATE_TRUNC" in create_sql and "MONTH" in create_sql
        assert has_last_day or has_date_trunc_month, (
            f"Expected LAST_DAY() or DATE_TRUNC('month', ...) in monthly CREATE TABLE. Got: {create_stmts[0]}"
        )

        # Should use GROUP BY for aggregation
        assert "GROUP BY" in create_sql, f"Expected GROUP BY in monthly CREATE TABLE. Got: {create_stmts[0]}"
