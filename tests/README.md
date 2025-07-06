# Data Quality Checks Tests

This directory contains comprehensive unit tests for the improved `data_quality_checks.py` module.

## Overview

The test suite covers all the enhanced functionality in the data quality checks module, including:

- **New Classes**: `ValidationResult`, `ExternalRegressorValidator`, `DateOffsetUtility`
- **Enhanced Classes**: `DataQualityThresholds`, `DataQualityMetrics`, `DataQualityChecks`
- **Enums**: `SupportedFrequencies`, `ExternalRegressorTypes`
- **Error Handling**: Comprehensive error scenarios and edge cases
- **Integration**: End-to-end pipeline testing
- **Improved Logging**: Generic removal reasons without specific ratio values for cleaner output

## Test Structure

### Unit Tests (`tests/unit/test_data_quality_checks.py`)

The test file is organized into the following test classes:

1. **`TestValidationResult`** - Tests the new `ValidationResult` class
2. **`TestDataQualityThresholds`** - Tests threshold validation and configuration
3. **`TestDataQualityMetrics`** - Tests metrics tracking functionality
4. **`TestSupportedFrequencies`** - Tests the frequency enum
5. **`TestExternalRegressorTypes`** - Tests the external regressor types enum
6. **`TestDateOffsetUtility`** - Tests date offset calculations
7. **`TestExternalRegressorValidator`** - Tests external regressor validation
8. **`TestDataQualityChecks`** - Tests the main data quality checks class
9. **`TestDataQualityChecksIntegration`** - Integration tests for the full pipeline

### Fixtures (`tests/conftest.py`)

Common test fixtures are defined for:
- Sample time series data
- Basic forecasting configurations
- Mock Spark DataFrames
- Problematic data scenarios

## Running the Tests

### Quick Start

```bash
# Run all data quality tests
python tests/run_data_quality_tests.py

# Run with verbose output
python tests/run_data_quality_tests.py --verbose

# Run with coverage reporting
python tests/run_data_quality_tests.py --coverage
```

### Using pytest directly

```bash
# Run all data quality tests
pytest tests/unit/test_data_quality_checks.py

# Run with coverage
pytest tests/unit/test_data_quality_checks.py --cov=mmf_sa.data_quality_checks --cov-report=html

# Run specific test class
pytest tests/unit/test_data_quality_checks.py::TestValidationResult

# Run specific test method
pytest tests/unit/test_data_quality_checks.py::TestDataQualityChecks::test_initialization
```

### Test Categories

You can run different categories of tests:

```bash
# Run only unit tests
python tests/run_data_quality_tests.py --type unit

# Run only integration tests  
python tests/run_data_quality_tests.py --type integration

# Run only fast tests (exclude slow ones)
python tests/run_data_quality_tests.py --type fast
```

## Test Coverage

The tests cover the following scenarios:

### ✅ Success Cases
- Valid configurations and data
- Proper initialization of all classes
- Correct validation results
- Successful pipeline execution

### ❌ Failure Cases
- Invalid threshold values
- Missing required parameters
- Unsupported frequencies
- Data quality violations
- External regressor validation failures

### 🔄 Edge Cases
- Empty datasets
- Data conversion errors
- Null values in external regressors
- Insufficient training data
- High negative data ratios

### 🧪 Integration Tests
- Full pipeline with good data
- Full pipeline with problematic data
- Multiple groups with mixed quality
- External regressor scenarios

## Test Data

The tests use various data scenarios:

1. **Clean Data**: Well-formed time series with no quality issues
2. **Problematic Data**: Data with missing values, negative values, insufficient length
3. **External Regressor Data**: Data with various types of external regressors
4. **Edge Case Data**: Empty datasets, single-point series, extreme values

## Dependencies

The tests require the following packages:

```bash
pip install pytest pytest-cov pytest-mock pytest-xdist
```

You can install these automatically:

```bash
python tests/run_data_quality_tests.py --install-deps
```

## Configuration

Test configuration is managed through:

- `pytest.ini`: Main pytest configuration
- `tests/conftest.py`: Common fixtures and setup
- `tests/unit/fixtures.py`: Existing project fixtures

## Output

Tests generate the following outputs:

- **Console**: Test results and summary
- **Coverage Report**: HTML coverage report (when `--coverage` is used)
- **Log Files**: Detailed test logs (`tests/test.log`)

## Continuous Integration

To run these tests in CI/CD pipelines:

```yaml
# Example GitHub Actions step
- name: Run Data Quality Tests
  run: |
    python tests/run_data_quality_tests.py --coverage
    # Upload coverage reports if needed
```

## Contributing

When adding new functionality to `data_quality_checks.py`:

1. Add corresponding unit tests to `test_data_quality_checks.py`
2. Include both success and failure scenarios
3. Add integration tests for end-to-end functionality
4. Update this README if new test categories are added

## Troubleshooting

### Common Issues

1. **Spark Session Issues**: Ensure you have proper Spark configuration
2. **Missing Dependencies**: Run with `--install-deps` flag
3. **Memory Issues**: Reduce data size in tests or increase available memory
4. **Import Errors**: Ensure the `mmf_sa` module is in your Python path

### Debug Mode

For debugging failing tests:

```bash
# Run with maximum verbosity
pytest tests/unit/test_data_quality_checks.py -vvv --tb=long

# Run with pdb debugger
pytest tests/unit/test_data_quality_checks.py --pdb

# Run specific failing test
pytest tests/unit/test_data_quality_checks.py::TestClass::test_method -vvv
``` 