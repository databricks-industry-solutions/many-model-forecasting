"""Tier 1 tests for the /post-process-and-evaluate skill.

Validates that the skill document contains the correct SQL patterns
for best model selection, WAPE calculation, and evaluation summary.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_SKILLS_DIR = _REPO_ROOT / "databricks-skills" / "many-model-forecasting"


@pytest.mark.tier1
class TestPostEvaluateSkillDocument:
    """Test that the post-evaluation skill document is well-formed."""

    def test_skill_document_exists(self):
        """Post-evaluation skill document must exist."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        assert doc_path.exists(), f"Post-evaluation skill document not found at {doc_path}"

    def test_skill_has_best_model_selection_sql(self):
        """Skill must contain SQL for best model selection using RANK()."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "RANK()" in content, "Skill must use RANK() for best model selection"
        assert "_best_models" in content, "Skill must create _best_models table"

    def test_skill_creates_best_models_table(self):
        """Skill must create {use_case}_best_models table."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "CREATE OR REPLACE TABLE" in content, "Skill must CREATE OR REPLACE TABLE"
        assert "_best_models" in content, "Skill must reference _best_models table"

    def test_skill_creates_evaluation_summary_table(self):
        """Skill must create {use_case}_evaluation_summary table."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "_evaluation_summary" in content, "Skill must create _evaluation_summary table"

    def test_skill_has_wape_calculation(self):
        """Skill must contain WAPE calculation using ARRAYS_ZIP/AGGREGATE."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "ARRAYS_ZIP" in content, "Skill must use ARRAYS_ZIP for WAPE calculation"
        assert "AGGREGATE" in content, "Skill must use AGGREGATE for WAPE calculation"
        assert "wape" in content.lower(), "Skill must compute WAPE metric"

    def test_skill_has_model_ranking(self):
        """Skill must contain model ranking (wins count) query."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "wins_count" in content, "Skill must compute wins_count"
        assert "wins_pct" in content, "Skill must compute wins_pct"

    def test_skill_cross_references_profiling(self):
        """Skill must optionally cross-reference with series profile."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "_series_profile" in content, "Skill must reference _series_profile for cross-reference"
        assert "forecastability_class" in content, "Skill must reference forecastability_class"

    def test_skill_verifies_outputs_exist(self):
        """Skill must verify evaluation and scoring output tables exist."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "_evaluation_output" in content, "Skill must verify _evaluation_output exists"
        assert "_scoring_output" in content, "Skill must verify _scoring_output exists"

    def test_skill_uses_use_case_prefix(self):
        """Skill must use {use_case} prefix for all table names."""
        doc_path = _SKILLS_DIR / "5-post-process-and-evaluate.md"
        content = doc_path.read_text()

        assert "{use_case}" in content, "Skill must use {use_case} prefix for table names"
