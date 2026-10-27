# Copilot Workspace Init

This workspace contains the current KMDS dataset utility package as implemented in the repository today.

## Project overview

- Package name: `kmds-dataset-util`
- Source code directory: `src/`
- Main package: `data_prep`
- Tests: `tests/`
- Current implementation focus: config validation, raw dataset download, prepared data output, dictionary generation, and Fano-based operating-window screening

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

## Current Fano-factor selection design

`data_prep.fanno_factor_selection.select_poisson_operating_window()` is the current API for screening operational windows under the Fano-factor design. It is separate from the older enterprise count-binning and overlap/Hankel helpers in `data_prep.count_timeseries`.

The API reads an event-arrival CSV and accepts a timestamp column, candidate intervals, a preferred aggregation interval, a minimum sample-window count, a confidence level, and an optional plot path. Defaults are:

- aggregation interval: `1min`
- minimum complete non-overlapping candidate windows: `30`
- confidence level: `0.90` (`confidence`, with `alpha = 1 - confidence`)

For each candidate interval, the implementation:

1. Sorts the candidate intervals from shortest to longest.
2. Counts complete non-overlapping candidate windows in the event-time span. Candidates below `min_aggregation_windows` are reported as insufficient and are not tested.
3. Counts arrivals in the candidate windows, including empty windows, and computes the sample Fano factor as unbiased sample variance divided by sample mean.
4. Computes two-sided chi-squared limits with `n - 1` degrees of freedom and the configured confidence.
5. Marks the candidate acceptable when its Fano factor lies within those limits.

The result contains the full metrics table, an `acceptable_windows` list containing every passing candidate, and a plot with the Fano curve, confidence-bound band, and Poisson reference at `F = 1`. It does not choose one window from that list. The user selects a window before doing the separate downstream TS-EDA/Hankel step.

The hospital EDA notebook combines `Patient Admission Date` and `Patient Admission Time` to construct arrival timestamps, filters to the modal year, writes the event CSV, and calls this API. Its confidence level is exposed as `confidence_level` in the notebook cell, and the following cell displays the Fano-factor-by-window table. The notebook does not perform the downstream TS-EDA selection/export.

Relevant implementation and coverage:

- `src/data_prep/fanno_factor_selection.py`
- `tests/test_fanno_factor_selection.py`
- `notebooks/health_care_anlytics_eda.ipynb`
- `fanno_factor_selection.md`

## Before vs after: midnight-anchored binning

The old behavior effectively anchored to the first observed timestamp instead of the start of the operational day. That produced a drifting phase and an inconsistent number of bins across days.

Example of the broken behavior:

```text
old raw buckets
2024-01-01 01:04:00 -> bucket 2024-01-01 01:00:00
2024-01-01 01:20:00 -> bucket 2024-01-01 01:00:00
2024-01-02 00:15:00 -> bucket 2024-01-02 00:00:00
2024-01-02 12:30:00 -> bucket 2024-01-02 12:00:00
```

This is not a true day-aligned operational grid because the first bucket is offset from midnight and the phase drifts from the first arrival.

The corrected behavior snaps the grid to the daily cycle boundary and preserves the same interval cadence each day:

```text
corrected midnight-anchored buckets
2024-01-01 00:00:00 -> 2024-01-01 00:30:00
2024-01-01 00:30:00 -> 2024-01-01 01:00:00
2024-01-01 01:00:00 -> 2024-01-01 01:30:00
2024-01-01 01:30:00 -> 2024-01-01 02:00:00
...
2024-01-02 00:00:00 -> 2024-01-02 00:30:00
2024-01-02 00:30:00 -> 2024-01-02 01:00:00
```

The corrected series therefore starts at `00:00:00`, advances by the selected interval, and keeps the same number of bins for each operational day.

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
- Retains the older count-binning API, which searches a fixed enterprise ladder; this is distinct from the CSV-based Fano acceptance API described above.
- Uses a day-anchored phase-1 resample so the original raw binning process starts from the operational day boundary rather than the first observed arrival timestamp.
- Performs a second-stage integer alignment step after the density search so the chosen grid is anchored to midnight or cycle boundaries and preserves an integer number of bins per day.
- Produces uniform count-binned arrival-intensity series with `bin_id`, `bin_start`, `bin_end`, `window`, `count`, and `intensity` columns.
- Uses window-aware intensity scaling: the intensity is normalized by the selected unit (`count per minute`, `count per hour`, `count per day`, `count per week`) rather than assuming every window is an hour.
- Reports both the original algorithm output and the post-alignment output together via `summarize_integer_alignment(...)` so the user can compare phase 1 vs phase 2 directly.
- The Seattle 911 notebook uses the older count-binning and alignment workflow. The hospital notebook uses the newer Fano acceptance API and leaves the downstream TS-EDA step to the user.

## Important implementation notes

- The project uses `setuptools` with a `src` layout.
- Pytest is configured with `pythonpath = ["src"]`.
- The package depends on `PyYAML`, the official `kaggle` client, `pandas`, `scipy`, and `matplotlib`.
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

This older interface remains exercised in the Seattle 911 notebook. The hospital notebook uses the independent Fano screening flow described above, displays the Fano-factor-by-window table and plot, and does not perform a TS-EDA export.

## Current status and next likely work

The package is already working as a practical baseline for local raw-data preparation and enterprise time-series binning. The next likely step is to expand beyond the current baseline into richer provenance metadata, more formal transformation templates, and a fuller KMDS notebook or CLI workflow.
