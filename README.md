# KMDS Dataset Utility

A config-driven dataset utility for downloading, inspecting, transforming, and writing canonical KMDS datasets. The goal is to prime the model development workflow for a KMDS pipeline.

A kmds-dataset-util project is organized as a pair of notebooks:

1. Download notebook: Given an HTTPS endpoint or a connection code snippet for an enterprise data source, this notebook downloads the tabular data resources into a staging area. The user can use an LLM coding agent to generate the download code, then use code-generation tools to build the data dictionary. The workflow varies by source. For example, for a healthcare dataset, the Kaggle data card often contains attribute names and descriptions that can be copied into a chat window and used to generate a CSV file with attribute and description columns. For Olist, the coding agent can consolidate attributes and descriptions into a master data dictionary. Once the data dictionary and download code are ready, the package provides a utility to bootstrap a config file for future downloads. On a subsequent run, you can point the utility at that config file and ask it to complete the download.

2. EDA notebook: You can use the config file from the previous step to read both the data dictionary and the raw data for exploratory data analysis. EDA is performed in the context of a task, and the tasks currently supported are classification, clustering, and time series. You can complete the EDA manually with coding-agent assistance and then ask the package to generate a prepare_dataset() method that applies the required transformations and cleaning steps, writes the final subset of attributes, produces the final data dictionary, and saves the config file capturing the task details.

## Quick start

```python
from data_prep.config import bootstrap_config

config_path = bootstrap_config(
    config_dir="./configs",
    config_name="kmds-prep.yaml",
    dataset_name="example",
    representation="graph",
    task_characterization="clustering",
    staging_dir="./data/raw/example",
    prepared_dataset_dir="./data/prepared/example",
    write=True,
)
print(config_path)
```

## Example config

```yaml
version: 1
dataset_name: example
operation: prepare
representation: graph
task_characterization: clustering
staging_dir: ./data/raw/example
prepared_dataset_dir: ./data/prepared/example
output:
  path: ./data/prepared/example
  format: csv
```

## Supported operations

- `download`: fetch from HTTP or Kaggle sources
- `prepare`: configure staging and prepared output for KMDS pipeline work
- `inspect`: validate source data before transformation
- `transform`: convert to canonical data forms
- `write`: persist the prepared dataset

## Current status

This implementation now matches the revised KMDS requirements by supporting the prepare workflow, representation selection, task characterization, and explicit bootstrap config output paths.
