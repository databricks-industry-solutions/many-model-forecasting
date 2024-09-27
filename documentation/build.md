# Installation

This page covers the install and build process of the MMF Statistical Analysis Package. 

## Installation

1. Clone the repository:
    ```
    git clone https://github.com/databrickslabs/mmf_sa.git
    ```

2. Install the dependencies. This is system specific, but you will need spark, java, and lightgbm.

## Hatchling

MMF uses [Hatchling](https://hatch.pypa.io/latest/) to manage the build process. It is a lightweight build system for Python projects. Hatch allows you to create and manage multiple isolated Python environments for a single project. Key features include:

- Automatic environment creation when running commands
- Support for environment-specific dependencies
- Ability to define multiple environments in pyproject.toml

Common commands:
- `hatch env create`: Create a new environment
- `hatch shell`: Spawn a shell within an environment
- `hatch run <command>`: Execute a command in an environment
- `hatch env remove <name>`: Remove a specific environment
- `hatch env show`: Display all available environments

Hatchling integrates with Databricks Asset Bundles, providing:
- Package building and asset bundle creation
- Dependency management via `pyproject.toml`
- Version control for Databricks assets
- Consistency between local and Databricks environments

## Setting up the development environment

1. Make sure you have the dependencies installed (see install.md)

2. Create a virtual environment in development mode:
    ```
    hatch env create
    ```

## Building a wheel

3. After development, build the project using Hatch:
    ```
    hatch build
    ```
    Note that this command creates a distributable package in the `dist/` directory. This can be used for CI/CD pipelines instead of the repo.

## Integrating with Databricks CI/CD

1. Upload the built package to Databricks:
   ```
   databricks fs cp dist/your_package-version.whl dbfs:/FileStore/packages/
   ```

2. Install the package in a Databricks notebook:
   ```python
   %pip install /dbfs/FileStore/packages/your_package-version.whl
   ```

3. Use the `pyproject.toml` file to create a Databricks library:
   - Go to the Databricks workspace
   - Navigate to "Libraries" in your cluster settings
   - Select "Upload" and choose your `pyproject.toml` file
   - Databricks will install all specified dependencies

4. Version your Databricks assets:
   - Use the version from `pyproject.toml` in your asset names or metadata
   - Example: `my_model_v1.2.3`

5. CI/CD Integration:
   - Use Databricks CLI in your CI/CD pipeline to automate package uploads and asset updates
   - Example GitHub Actions workflow step:
     ```yaml
     - name: Upload to Databricks
       run: |
         databricks fs cp dist/your_package-${{ steps.version.outputs.version }}.whl dbfs:/FileStore/packages/
         databricks jobs update --job-id $JOB_ID --new-settings job_config.json
     ```


2. Build the project using Hatch:
    ```
    hatch build
    ```     