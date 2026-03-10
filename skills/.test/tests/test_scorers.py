"""Unit tests for scorers."""
import pytest
from unittest.mock import MagicMock

# Test the routing scorer functions directly (not the decorated versions)
from skill_test.scorers.routing import detect_skills_from_prompt, SKILL_TRIGGERS


class TestDetectSkillsFromPrompt:
    """Tests for skill detection from prompts."""

    def test_detect_mmf(self):
        """Test detection of many-model-forecasting skill."""
        prompt = "Run many model forecasting on my sales data"
        skills = detect_skills_from_prompt(prompt)
        assert "many-model-forecasting" in skills

    def test_detect_mmf_time_series(self):
        """Test detection via time series forecasting keyword."""
        prompt = "Set up a time series forecasting pipeline"
        skills = detect_skills_from_prompt(prompt)
        assert "many-model-forecasting" in skills

    def test_detect_mmf_chronos(self):
        """Test detection via Chronos model keyword."""
        prompt = "Run Chronos foundation models on my data"
        skills = detect_skills_from_prompt(prompt)
        assert "many-model-forecasting" in skills

    def test_detect_no_match(self):
        """Test no skills detected for unrelated prompt."""
        prompt = "What is the weather today?"
        skills = detect_skills_from_prompt(prompt)
        assert len(skills) == 0

    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        prompt = "RUN MMF ON MY DATA"
        skills = detect_skills_from_prompt(prompt)
        assert "many-model-forecasting" in skills


class TestSkillTriggers:
    """Tests for SKILL_TRIGGERS configuration."""

    def test_all_skills_have_triggers(self):
        """Verify all expected skills have trigger keywords."""
        expected_skills = [
            "many-model-forecasting",
        ]
        for skill in expected_skills:
            assert skill in SKILL_TRIGGERS
            assert len(SKILL_TRIGGERS[skill]) > 0

    def test_triggers_are_lowercase(self):
        """Verify all triggers are lowercase for matching."""
        for skill, triggers in SKILL_TRIGGERS.items():
            for trigger in triggers:
                assert trigger == trigger.lower(), f"Trigger '{trigger}' for {skill} should be lowercase"


# Tests for executor module
from skill_test.grp.executor import (
    extract_code_blocks,
    verify_python_syntax,
    verify_sql_structure,
    CodeBlock
)


class TestExtractCodeBlocks:
    """Tests for code block extraction."""

    def test_extract_python_block(self):
        """Test extraction of Python code block."""
        response = '''Here's some code:

```python
def hello():
    print("Hello")
```

That's it.'''
        blocks = extract_code_blocks(response)
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].code

    def test_extract_sql_block(self):
        """Test extraction of SQL code block."""
        response = '''Here's SQL:

```sql
SELECT * FROM table
```'''
        blocks = extract_code_blocks(response)
        assert len(blocks) == 1
        assert blocks[0].language == "sql"

    def test_extract_multiple_blocks(self):
        """Test extraction of multiple code blocks."""
        response = '''
```python
x = 1
```

```sql
SELECT 1
```
'''
        blocks = extract_code_blocks(response)
        assert len(blocks) == 2

    def test_no_code_blocks(self):
        """Test response with no code blocks."""
        response = "Just some text without code."
        blocks = extract_code_blocks(response)
        assert len(blocks) == 0


class TestVerifyPythonSyntax:
    """Tests for Python syntax verification."""

    def test_valid_syntax(self):
        """Test valid Python code."""
        code = "def foo():\n    return 42"
        valid, error = verify_python_syntax(code)
        assert valid is True
        assert error is None

    def test_invalid_syntax(self):
        """Test invalid Python code."""
        code = "def foo(\n    return"
        valid, error = verify_python_syntax(code)
        assert valid is False
        assert error is not None


class TestVerifySqlStructure:
    """Tests for SQL structure verification."""

    def test_valid_select(self):
        """Test valid SELECT statement."""
        code = "SELECT * FROM table WHERE id = 1"
        result = verify_sql_structure(code)
        assert result.success is True

    def test_valid_create(self):
        """Test valid CREATE statement."""
        code = "CREATE TABLE foo (id INT)"
        result = verify_sql_structure(code)
        assert result.success is True

    def test_unbalanced_parens(self):
        """Test unbalanced parentheses."""
        code = "SELECT * FROM foo WHERE (id = 1"
        result = verify_sql_structure(code)
        assert result.success is False
        assert "Unbalanced" in result.error

    def test_no_statement(self):
        """Test code with no SQL statement."""
        code = "just some random text"
        result = verify_sql_structure(code)
        assert result.success is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
