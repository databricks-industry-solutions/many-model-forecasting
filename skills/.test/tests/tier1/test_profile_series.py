"""Tier 1 tests for the /profile-and-classify-series skill.

Validates that the profiling notebook template has correct placeholders,
defines the profile_series function, and writes to the expected table.
"""

from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parents[3]
_SKILLS_DIR = _REPO_ROOT / "databricks-skills" / "many-model-forecasting"


@pytest.mark.tier1
class TestProfilingNotebookTemplate:
    """Test that the profiling notebook template is well-formed."""

    def test_template_has_required_placeholders(self):
        """Profiling template must contain all required {placeholder} tokens."""
        template_path = _SKILLS_DIR / "mmf_profiling_notebook_template.ipynb"
        assert template_path.exists(), f"Profiling template not found at {template_path}"

        content = template_path.read_text()

        required_placeholders = [
            "{catalog}",
            "{schema}",
            "{use_case}",
            "{train_table}",
            "{freq}",
            "{prediction_length}",
        ]
        for placeholder in required_placeholders:
            assert placeholder in content, (
                f"Profiling template missing required placeholder: {placeholder}"
            )

    def test_template_defines_profile_series_function(self):
        """Template must define the profile_series function."""
        template_path = _SKILLS_DIR / "mmf_profiling_notebook_template.ipynb"
        content = template_path.read_text()

        assert "def profile_series" in content, "Template must define profile_series function"

    def test_template_writes_to_series_profile_table(self):
        """Template must write to {use_case}_series_profile table."""
        template_path = _SKILLS_DIR / "mmf_profiling_notebook_template.ipynb"
        content = template_path.read_text()

        assert "_series_profile" in content, "Template must write to _series_profile table"

    def test_template_computes_required_statistics(self):
        """Template must compute the required statistical properties."""
        template_path = _SKILLS_DIR / "mmf_profiling_notebook_template.ipynb"
        content = template_path.read_text()

        required_stats = [
            "adf_pvalue",
            "seasonality_strength",
            "trend_strength",
            "spectral_entropy",
            "autocorrelation_lag1",
            "snr",
            "sparsity",
            "cv",
        ]
        for stat in required_stats:
            assert stat in content, f"Template must compute {stat}"

    def test_template_classifies_series(self):
        """Template must classify series into high_confidence and low_signal."""
        template_path = _SKILLS_DIR / "mmf_profiling_notebook_template.ipynb"
        content = template_path.read_text()

        assert "high_confidence" in content, "Template must classify series as high_confidence"
        assert "low_signal" in content, "Template must classify series as low_signal"
        assert "forecastability_class" in content, "Template must include forecastability_class column"

    def test_template_recommends_models(self):
        """Template must recommend models based on series characteristics."""
        template_path = _SKILLS_DIR / "mmf_profiling_notebook_template.ipynb"
        content = template_path.read_text()

        assert "recommended_models" in content, "Template must include recommended_models"
        assert "model_types_needed" in content, "Template must include model_types_needed"
        assert "StatsForecastAutoArima" in content, "Template must recommend StatsForecastAutoArima"


@pytest.mark.tier1
class TestProfilingSkillDocument:
    """Test that the profiling skill document is well-formed."""

    def test_skill_document_exists(self):
        """Profiling skill document must exist."""
        doc_path = _SKILLS_DIR / "2-profile-and-classify-series.md"
        assert doc_path.exists(), f"Profiling skill document not found at {doc_path}"

    def test_skill_references_template(self):
        """Skill must reference mmf_profiling_notebook_template.ipynb."""
        doc_path = _SKILLS_DIR / "2-profile-and-classify-series.md"
        content = doc_path.read_text()

        assert "mmf_profiling_notebook_template" in content, (
            "Skill must reference mmf_profiling_notebook_template.ipynb"
        )

    def test_skill_has_classification_logic(self):
        """Skill must document classification thresholds."""
        doc_path = _SKILLS_DIR / "2-profile-and-classify-series.md"
        content = doc_path.read_text()

        assert "high_confidence" in content, "Skill must document high_confidence classification"
        assert "low_signal" in content, "Skill must document low_signal classification"
        assert "spectral_entropy" in content, "Skill must document spectral_entropy threshold"
