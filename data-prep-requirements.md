# Dataset Download Requirements for KMDS Analysis

## 1. Purpose

This document defines the required behavior for downloading source datasets into a reproducible, analysis-ready staging area for the KMDS dataset preparation workflow. The download process must preserve raw source assets, capture metadata, and support validation before any transformation or modeling is performed.

The download subsystem is intended to support both direct file acquisition and dataset-package acquisition from provider APIs, with the initial implementation focused on Kaggle and HTTP sources.

---

## 2. Scope

The dataset download requirements cover:

- Download configuration and validation
- Source-specific acquisition workflows
- Staging and storage conventions
- Provenance and metadata capture
- File and archive validation
- Error handling and failure reporting
- Extensibility to additional providers in future releases

This workflow is a prerequisite to inspection, transformation, and final dataset preparation.

---

## 3. Core Requirements

### 3.1 Configuration-driven execution

All download operations must be executed through a YAML configuration file. A valid download configuration must include:

- `version`
- `dataset_name`
- `operation` with value `download`
- `source` object
- `destination` object

The source definition must specify either:

- `source.url` for HTTP-based downloads, or
- `source.kaggle_dataset` for Kaggle-based downloads

The destination definition must provide:

- `destination.path`
- `destination.format` with allowed values `csv` or `parquet`

### 3.2 Operational contract

The download workflow shall:

- resolve the dataset source from configuration
- create the destination directory when needed
- perform the source-specific retrieval operation
- verify that the resulting directory contains valid downloaded content
- return the staged target path for downstream processing

### 3.3 Source support

The delivery pipeline must support the following initial sources:

- Kaggle datasets via the official Kaggle API
- HTTP file or archive downloads via standard URL retrieval

Source adapters shall be implemented in a way that keeps the workflow source-agnostic and allows future adapters to be added without modifying the main orchestration layer.

---

## 4. Supported Dataset Sources

### 4.1 Kaggle datasets

The Kaggle download mechanism shall:

- accept either a dataset slug such as `owner/dataset` or a full Kaggle dataset URL
- normalize the dataset identifier into a canonical form
- authenticate using credentials stored in the standard Kaggle configuration location
- download the dataset to the configured staging destination
- unzip archives when required
- verify that the output directory contains files after download

Required credentials:

- `~/.kaggle/kaggle.json`, or
- `~/.kaggle/access_token`

If credentials are missing, the download operation must fail with a clear, actionable error.

### 4.2 HTTP sources

The HTTP download mechanism shall:

- accept a valid URL
- create the destination directory if it does not exist
- download the target file to the configured path
- validate that the download produced a file with content
- support direct file download and archive retrieval patterns

The workflow must reject malformed or missing URLs with a validation error before attempting the network request.

---

## 5. Staging and Storage Requirements

The staging area is the authoritative local workspace for downloaded and intermediate files.

A typical project layout is:

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

### 5.1 Storage rules

- Original downloaded files must not be modified.
- Download targets must be created under the configured staging directory.
- Downloaded assets must remain available for inspection and traceability.
- Source metadata, documentation, and dictionaries must be retained alongside raw source files.

### 5.2 Source documentation retention

The staging area shall preserve:

- downloaded files
- dataset readme or documentation files
- metadata files
- any source dictionaries or schema references
- download logs and provenance records

---

## 6. Validation Requirements

The system shall validate the successful completion of each download before treating the dataset as usable.

### 6.1 Mandatory validation checks

The download workflow must verify:

- the destination path exists
- the destination is a directory
- at least one file is present
- the file or archive is readable
- the size is greater than zero
- archive extraction completed successfully when required
- expected content is present for the dataset source

### 6.2 Validation failure behavior

If validation fails, the system shall:

- raise an explicit error with the path and reason
- stop the workflow before downstream preparation begins
- retain the failed output for diagnosis where possible

### 6.3 Provenance capture

The download subsystem must record enough metadata to explain where the data came from and how it was obtained. At minimum, the metadata should include:

- dataset name
- source type
- source URL or Kaggle identifier
- destination path
- timestamp
- file list or manifest
- status of validation checks

---

## 7. Dataset Dictionary and Metadata Requirements

Downloaded datasets may include documentation or schema artifacts that must be retained for later analysis and transformation.

The system must support acquisition of:

- README files
- data dictionaries
- source metadata files
- schema resources
- analyst-generated notes or mapping documents

The dictionary artifacts must remain on disk in the dataset staging area and be treated as source material for downstream preparation. They should not be embedded directly in configuration files; the configuration file should instead point to the dictionary path in the staging directory when that path is needed for downstream analysis.

---

## 8. Required Configuration Schema

A valid download configuration shall use the following structure:

```yaml
version: 1
dataset_name: healthcare-analytics-patient-flow-data
operation: download
source:
  type: kaggle
  kaggle_dataset: hassanjameelahmed/healthcare-analytics-patient-flow-data
destination:
  path: ./data/raw/healthcare-analytics-patient-flow-data
  format: csv
```

For an HTTP source, the structure is similar:

```yaml
version: 1
dataset_name: example_dataset
operation: download
source:
  type: http
  url: https://example.com/dataset.zip
destination:
  path: ./data/raw/example_dataset
  format: csv
```

---

## 9. Required Functional Behaviors

The dataset download subsystem shall:

1. Accept a supported source configuration.
2. Validate the configuration before network access or file creation.
3. Create the destination directory when necessary.
4. Perform the source-specific download.
5. Preserve the raw source material.
6. Verify the download result.
7. Record metadata and provenance information.
8. Fail fast with clear diagnostics on invalid credentials, malformed URLs, or missing files.

---

## 10. Non-functional Requirements

### 10.1 Reproducibility

The download process must be repeatable and traceable. The same configuration should produce the same staging location and capture the same provenance metadata.

### 10.2 Extensibility

The architecture must support future providers such as:

- S3
- Azure Blob Storage
- Google Cloud Storage
- Hugging Face
- Git repositories
- database exports

without requiring redesign of the main dataset preparation pipeline.

### 10.3 Operational reliability

The implementation must surface actionable errors, validate assumptions before processing, and minimize silent data loss.

### 10.4 Analyst usability

Analysts must be able to inspect downloaded artifacts easily and determine the exact source, destination, and validation status for each dataset.

---

## 11. Acceptance Criteria

The dataset download requirement is satisfied when all of the following are true:

- a valid Kaggle config downloads a dataset to the specified staging path
- a valid HTTP config downloads a file or archive to the specified staging path
- invalid source configuration is rejected before execution
- missing download content causes a clear validation failure
- provenance metadata is retained for each successful download
- source documentation and schema artifacts remain associated with the staged dataset
- the system is extensible to additional data providers without rewriting the core workflow

---

## 12. Summary

The KMDS download workflow must produce a trustworthy, reproducible, and inspectable staging area for analysis. It must support provider-specific acquisition, capture provenance and metadata, validate the result, and preserve raw source materials in a form suitable for downstream transformation and dataset preparation.

---

# Dataset Transformation Requirements

Transformation workflows shall be implemented within preparation notebooks.

Requirements:

- Read data from the staging area.
- Create canonical KMDS datasets.
- Preserve provenance.
- Record transformation metadata.
- Support reusable transformation templates.

---

# Prepared Dataset Requirements

The package shall provide APIs to write prepared datasets.

Example:

```python
write_prepared_dataset(
    data,
    output_directory,
    filename,
    format
)
```

Supported formats:

- CSV
- Parquet

Requirements:

- Configurable directory
- Configurable filename
- Configurable format

---

# Prepared Data Dictionary Requirements

The package shall provide APIs to write prepared data dictionaries.

Example:

```python
write_data_dictionary(
    dictionary,
    output_directory,
    filename
)
```

Default behavior:

- Dictionary location defaults to prepared dataset location.

---

# Representation-Specific Output Requirements

## Tabular Representation

Outputs:

```text
dataset.parquet
dataset_dictionary.csv
```

or

```text
dataset.csv
dataset_dictionary.csv
```

---

## Graph Representation

Outputs:

```text
nodes.parquet
edges.parquet

nodes_dictionary.csv
edges_dictionary.csv
```

or

```text
nodes.csv
edges.csv

nodes_dictionary.csv
edges_dictionary.csv
```

---
