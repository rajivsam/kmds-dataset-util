# KMDS Dataset Utility Design

## Overview
This package implements a practical, config-driven dataset workflow for KMDS work. The current codebase covers the foundational pieces that are actually built now: YAML-based config validation, Kaggle and HTTP downloads, a generic dataset preparation step, and dictionary-generation helpers for downstream analysis.

## Design goals
- Keep the package modular and easy to extend.
- Validate config explicitly at the operation boundary.
- Support a real download workflow for local raw staging.
- Provide a reusable prepare step for dataset output and metadata capture.
- Preserve analyst usability through simple filesystem conventions and clear validation errors.

## Current implementation structure
The actual package layout in the workspace is:

- src/data_prep/config.py: shared config schema validation and bootstrap helpers
- src/data_prep/download.py: source adapters, URL normalization, archive handling, and verification
- src/data_prep/prepare.py: dataset output writing and dictionary export helpers
- src/data_prep/llm.py: local LLM-backed data dictionary generation utilities
- src/data_prep/__init__.py: package exports
- tests/: regression coverage for config, download, and dictionary workflows
- configs/: example YAML configs for download and prepare usage

## Supported operations in the current code

### Download operation
The current implementation validates a download config with:

- version
- dataset_name
- operation: download
- source.type in {http, kaggle}
- source.url for HTTP or source.kaggle_dataset for Kaggle
- destination.path
- destination.format in {csv, parquet}

This is enforced in `validate_operation_config()` in `src/data_prep/config.py`.

### Prepare operation
The prepare workflow supports a more explicit KMDS-style config with:

- version
- dataset_name
- operation: prepare
- representation in {tabular, graph}
- task_characterization in {regression, classification, clustering, survival_analysis, time_series, anomaly_detection}
- staging_dir
- prepared_dataset_dir or output.path
- output.format in {csv, parquet}

The prepare workflow is bootstrapped through `bootstrap_config()` and the actual writer is `prepare_dataset()`.

### Inspect / transform / write operations
The validators accept these operations, but they are currently lightweight schema checks rather than full end-to-end workflows. They require input and output paths consistent with the current package conventions.

## Config schema principles
The code uses a single YAML config model and validates the operation-specific fields before execution.

### Shared fields
- version
- dataset_name
- operation

### Download config requirements
- source.type
- source.url or source.kaggle_dataset
- destination.path
- destination.format

### Prepare config requirements
- staging_dir
- prepared_dataset_dir or output.path
- representation
- task_characterization
- output.format

## Implementation details

### Kaggle source support
The current Kaggle implementation:

- accepts a dataset slug such as `owner/dataset`
- accepts a full Kaggle dataset URL
- normalizes URLs to the canonical `owner/dataset` form
- checks for credentials in `~/.kaggle/kaggle.json` or `~/.kaggle/access_token`
- uses the official Kaggle API via `kaggle.api.kaggle_api_extended.KaggleApi`
- downloads to the target directory and unzips archives automatically
- validates that the destination contains real files after download

The helper `kaggle_url_to_config()` converts a Kaggle URL into a YAML config and can optionally write it to the project `configs/` directory.

### HTTP source support
The current HTTP implementation:

- validates a URL using `urllib.parse`
- creates the destination directory if needed
- downloads the target file or archive to the configured path
- extracts `.zip` and `.tar` archives when needed
- rejects malformed URLs with a clear `ValueError`

### Verification and staging behavior
The package verifies the staging result by checking:

- the requested destination exists
- the destination is a directory or file
- the file is non-empty when a single file is used
- the directory contains at least one file after download

If download content is absent or empty, the workflow raises an explicit error rather than silently treating the result as successful.

### Prepared output behavior
`prepare_dataset()` currently:

- requires a pandas DataFrame
- writes the prepared dataset as CSV or Parquet
- stores output in `prepared_dataset_dir`
- optionally writes dictionary artifacts into the staging directory or a separate dictionary output directory
- normalizes dictionary rows into `attribute`, `description`, and `data_type` columns when possible

## Data flow in the implemented package
1. Load config YAML.
2. Validate the config with `validate_operation_config()`.
3. For download workflows, dispatch to `download_kaggle_dataset()` or `download_http_file()`.
4. Verify the destination path and content with `verify_download_path()`.
5. For prepare workflows, write the prepared dataset artifact and any dictionary files.
6. Return result metadata for downstream inspection and reporting.

## Actual file conventions
The current implementation uses a local, filesystem-first staging model:

```text
project/
├── data/
│   ├── raw/
│   │   └── dataset_name/
│   └── prepared/
│       └── dataset_name/
├── configs/
│   └── dataset_name.yaml
└── notebooks/
```

This matches the practical working pattern used by the current project and is consistent with the sample config in `configs/download_kaggle_sample.yaml`.

## Current scope and maturity
Implemented now:
- HTTP downloads
- Kaggle downloads via the official Kaggle API
- YAML config validation and bootstrap helpers
- disk-based raw staging and prepared output patterns
- CSV and Parquet output writing
- data dictionary export and LLM-assisted dictionary creation

Planned or incomplete relative to the broader KMDS vision:
- a full CLI entry-point layer
- richer provenance manifests
- source-agnostic provider registry beyond Kaggle and HTTP
- end-to-end transform notebooks and canonical graph workflows
- complete metadata versioning and operational orchestration

## Design summary
The current implementation is intentionally lean and grounded in what is already working in the repo: a config-driven pipeline with validation, file-based staging, direct source downloads, and generic prepared-data output. The design remains extensible, but it is now documented around the code that exists rather than the aspirational architecture alone.
