# Copilot Workspace Init

This workspace contains the current KMDS dataset utility package as implemented in the repository today.

## Project overview

- Package name: `kmds-dataset-util`
- Source code directory: `src/`
- Main package: `data_prep`
- Tests: `tests/`
- Current implementation focus: config validation, raw dataset download, prepared data output, and dictionary generation

## Local environment

This project uses a local virtual environment at `.venv/`.

To activate it:

```bash
cd /home/rajiv/programming/kmds-dataset-util
source .venv/bin/activate
```

## Install dependencies

If the environment is not already created or dependencies need to be installed:

```bash
cd /home/rajiv/programming/kmds-dataset-util
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

## Run tests

```bash
cd /home/rajiv/programming/kmds-dataset-util
source .venv/bin/activate
pytest -q
```

Or directly with the venv Python:

```bash
cd /home/rajiv/programming/kmds-dataset-util
.venv/bin/python -m pytest -q
```

## Current verified status

The project currently passes the existing test suite:

```text
10 passed in 0.02s
```

## Key implementation files

- [pyproject.toml](pyproject.toml): project metadata and pytest configuration
- [src/data_prep/config.py](src/data_prep/config.py): config loading, validation, and bootstrap helpers
- [src/data_prep/download.py](src/data_prep/download.py): HTTP/Kaggle download adapters, URL normalization, archive extraction, and download verification
- [src/data_prep/prepare.py](src/data_prep/prepare.py): prepared dataset writing and dictionary export logic
- [src/data_prep/llm.py](src/data_prep/llm.py): dataset dictionary generation using a local LLM interface
- [configs/download_kaggle_sample.yaml](configs/download_kaggle_sample.yaml): example Kaggle download configuration
- [tests/test_config.py](tests/test_config.py): config validation checks
- [tests/test_download.py](tests/test_download.py): download and URL normalization coverage

## What the package does today

- Validates YAML configs for `download`, `prepare`, `inspect`, `transform`, and `write` operations.
- Accepts HTTP or Kaggle download sources.
- Normalizes Kaggle dataset URLs to the canonical `owner/dataset` form.
- Verifies that a downloaded destination exists and contains content.
- Downloads and unpacks archives automatically when needed.
- Creates prepared CSV/Parquet outputs from pandas DataFrames.
- Writes dictionary files alongside staged or prepared data when metadata is available.
- Uses local LLM-backed logic to build a data dictionary from dataset/README context when available.

## Important implementation notes

- The project uses `setuptools` with a `src` layout.
- Pytest is configured with `pythonpath = ["src"]`.
- The package depends on `PyYAML`, the official `kaggle` client, and `pandas`.
- Kaggle credentials are read from `~/.kaggle/kaggle.json` or `~/.kaggle/access_token`.
- `bootstrap_config()` creates a prepare workflow config with explicit `representation`, `task_characterization`, and output path settings.
- `download_from_config()` loads a YAML config and verifies the result before returning the staging path.
- `prepare_dataset()` writes output and can emit dictionary artifacts in either the staging directory or a separate output directory.

## Current status and next likely work

The package is already working as a practical baseline for local raw-data preparation. The next likely step is to expand beyond the current download/prepare baseline into richer provenance metadata, more formal transformation templates, and a fuller KMDS notebook or CLI workflow.
