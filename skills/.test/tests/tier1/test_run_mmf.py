"""Tier 1 tests for the /execute-mmf-forecast skill (formerly /run-mmf).

Validates that the agent generates correct notebook parameters
and references the right tables when given mock tool responses.
"""

import pytest


@pytest.mark.tier1
class TestExecuteMMFForecast:
    """Test MMF run parameter generation."""

    def test_generates_correct_parameters(self, skill_prompt, run_agent):
        """Agent should use the specified parameters and reference {use_case}_train_data."""
        prompt = skill_prompt("execute-mmf-forecast")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Run MMF on test_catalog.default. Use case is 'test'. "
                "freq=D, prediction_length=30, metric=smape, "
                "models=StatsForecastAutoArima,StatsForecastAutoETS. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        response = result["final_response"]
        all_text = response
        for tc in result["tool_calls"]:
            all_text += " " + str(tc["arguments"])

        all_text_lower = all_text.lower()

        assert "_train_data" in all_text_lower, f"Expected reference to _train_data. Got: {all_text[:500]}"

        assert "prediction_length" in all_text_lower or "30" in all_text, (
            f"Expected prediction_length=30. Got: {all_text[:500]}"
        )

        assert "freq" in all_text_lower and "d" in all_text_lower, f"Expected freq=D in output. Got: {all_text[:500]}"

    def test_generates_correct_parameters_weekly(self, skill_prompt, run_agent):
        """Agent should reference {use_case}_train_data and freq=W for weekly forecasting."""
        prompt = skill_prompt("execute-mmf-forecast")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Run MMF on test_catalog.default. Use case is 'test'. "
                "freq=W, prediction_length=12, metric=smape, "
                "models=StatsForecastAutoArima,StatsForecastAutoETS. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        response = result["final_response"]
        all_text = response
        for tc in result["tool_calls"]:
            all_text += " " + str(tc["arguments"])

        all_text_lower = all_text.lower()

        assert "_train_data" in all_text_lower, f"Expected reference to _train_data. Got: {all_text[:500]}"

        assert "prediction_length" in all_text_lower or "12" in all_text, (
            f"Expected prediction_length=12. Got: {all_text[:500]}"
        )

        assert "freq" in all_text_lower, f"Expected freq parameter in output. Got: {all_text[:500]}"
        assert "'w'" in all_text_lower or '"w"' in all_text_lower or "=w" in all_text_lower, (
            f"Expected freq=W in output. Got: {all_text[:500]}"
        )

    def test_generates_correct_parameters_monthly(self, skill_prompt, run_agent):
        """Agent should reference {use_case}_train_data and freq=M for monthly forecasting."""
        prompt = skill_prompt("execute-mmf-forecast")
        result = run_agent(
            system_prompt=prompt,
            user_prompt=(
                "Run MMF on test_catalog.default. Use case is 'test'. "
                "freq=M, prediction_length=6, metric=smape, "
                "models=StatsForecastAutoArima,StatsForecastAutoETS. "
                "Confirm and proceed with all steps without asking questions."
            ),
        )

        response = result["final_response"]
        all_text = response
        for tc in result["tool_calls"]:
            all_text += " " + str(tc["arguments"])

        all_text_lower = all_text.lower()

        assert "_train_data" in all_text_lower, f"Expected reference to _train_data. Got: {all_text[:500]}"

        assert "prediction_length" in all_text_lower or "6" in all_text, (
            f"Expected prediction_length=6. Got: {all_text[:500]}"
        )

        assert "freq" in all_text_lower, f"Expected freq parameter in output. Got: {all_text[:500]}"
        assert "'m'" in all_text_lower or '"m"' in all_text_lower or "=m" in all_text_lower, (
            f"Expected freq=M in output. Got: {all_text[:500]}"
        )
