# KMDS Dataset Utility

A config-driven dataset utility for downloading, inspecting, transforming, and writing canonical KMDS datasets.

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
