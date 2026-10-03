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
37 passed, 1 warning in 0.99s
```

The warning is a pandas deprecation warning from an older `30d` frequency alias in a sparse-data regression, not a failing logic issue.

## Key implementation files

- [pyproject.toml](pyproject.toml): project metadata and pytest configuration
- [src/data_prep/config.py](src/data_prep/config.py): config loading, validation, and bootstrap helpers
- [src/data_prep/download.py](src/data_prep/download.py): HTTP/Kaggle download adapters, URL normalization, archive extraction, and download verification
- [src/data_prep/prepare.py](src/data_prep/prepare.py): prepared dataset writing and dictionary export logic
- [src/data_prep/timeseries.py](src/data_prep/timeseries.py): enterprise-aligned count-binning and arrival-intensity time-series utilities
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
- Selects an operational time-window from a fixed enterprise ladder using count-density validation (`5m`, `15m`, `30m`, `1h`, `2h`, `4h`, `8h`, `12h`, `1d`, `2d`, `3d`, `5d`, `1w`, `2w`, `4w`).
- Uses a day-anchored phase-1 resample so the original raw binning process starts from the operational day boundary rather than the first observed arrival timestamp.
- Performs a second-stage integer alignment step after the density search so the chosen grid is anchored to midnight or cycle boundaries and preserves an integer number of bins per day.
- Produces uniform count-binned arrival-intensity series with `bin_id`, `bin_start`, `bin_end`, `window`, `count`, and `intensity` columns.
- Uses window-aware intensity scaling: the intensity is normalized by the selected unit (`count per minute`, `count per hour`, `count per day`, `count per week`) rather than assuming every window is an hour.
- Reports both the original algorithm output and the post-alignment output together via `summarize_integer_alignment(...)` so the user can compare phase 1 vs phase 2 directly.
- Includes notebook workflows for both the hospital dataset and the Seattle 911 2016 dataset that print the selected window, the window unit, the intensity unit, the raw phase-1 result, and the aligned phase-2 result.

## Important implementation notes

- The project uses `setuptools` with a `src` layout.
- Pytest is configured with `pythonpath = ["src"]`.
- The package depends on `PyYAML`, the official `kaggle` client, and `pandas`.
- Kaggle credentials are read from `~/.kaggle/kaggle.json` or `~/.kaggle/access_token`.
- `bootstrap_config()` creates a prepare workflow config with explicit `representation`, `task_characterization`, and output path settings.
- `download_from_config()` loads a YAML config and verifies the result before returning the staging path.
- `prepare_dataset()` writes output and can emit dictionary artifacts in either the staging directory or a separate output directory.

## Time-series interface and usage

The package now includes an enterprise count-binning API in `src/data_prep/timeseries.py` with a two-stage execution model:

1. Stage 1: `find_optimal_time_bucket(...)` searches the fixed enterprise ladder for the valid density window from a day-anchored cadence.
2. Stage 2: `align_to_integer_bin_grid(...)` anchors the grid to the nearest clean phase boundary (midnight or cycle boundary) to preserve integer bin counts per operational day.
3. Summary reporting: `summarize_integer_alignment(...)` returns the raw phase-1 series, the aligned phase-2 series, the algorithm window, the window unit, the intensity unit, and a status/message summarizing the difference.

```python
from data_prep.timeseries import (
    align_to_integer_bin_grid,
    build_arrival_intensity_series,
    find_optimal_time_bucket,
    summarize_integer_alignment,
)

window_selection = find_optimal_time_bucket(df, timestamp_col="timestamp")
arrival_intensity = build_arrival_intensity_series(df, timestamp_col="timestamp")
aligned_bins = align_to_integer_bin_grid(df["timestamp"], window_selection["selected_window"])
summary = summarize_integer_alignment(df[["timestamp"]], timestamp_col="timestamp")
```

The resulting output is a uniform-width, integer-sequenced series with:

- `bin_id`: sequential integer bin index
- `bin_start`: bucket start timestamp aligned to the operational grid
- `bin_end`: bucket end timestamp
- `window`: selected enterprise duration like `12h`
- `count`: raw arrivals in the bucket
- `intensity`: normalized count per selected window unit, rounded to 3 decimal places
- `window_unit`: `minute`, `hour`, `day`, or `week`
- `intensity_unit`: e.g. `count per hour`

The summary metadata therefore makes it explicit whether the intensity is per minute, per hour, per day, or per week.

This is currently exercised in both the healthcare and Seattle 911 EDA notebooks, where the dataset-specific arrival outputs are exported to `arrival_intensity.csv` under the prepared dataset directory and the notebook prints both the raw phase-1 output and the aligned phase-2 output.

## Current status and next likely work

The package is already working as a practical baseline for local raw-data preparation and enterprise time-series binning. The next likely step is to expand beyond the current baseline into richer provenance metadata, more formal transformation templates, and a fuller KMDS notebook or CLI workflow.
