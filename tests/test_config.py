from pathlib import Path

import pytest
import yaml

from data_prep.config import (
    bootstrap_config,
    get_learning_options,
    get_representation_options,
    load_config,
    validate_operation_config,
)


def test_download_requires_url_or_kaggle_source():
    config = {
        "version": 1,
        "dataset_name": "example",
        "operation": "download",
        "source": {"type": "http"},
        "destination": {"path": "/tmp/data"},
    }

    with pytest.raises(ValueError, match="source.url|source.kaggle_dataset"):
        validate_operation_config(config)


def test_write_requires_output_path_and_format():
    config = {
        "version": 1,
        "dataset_name": "example",
        "operation": "write",
        "input_path": "/tmp/input.csv",
        "output": {"format": "csv"},
    }

    with pytest.raises(ValueError, match="output.path"):
        validate_operation_config(config)


def test_load_config_parses_yaml_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: 1
dataset_name: demo
action: download
source:
  type: http
  url: https://example.com/data.csv
destination:
  path: /tmp/demo
""".strip()
    )

    config = load_config(config_path)
    assert config["operation"] == "download"
    assert config["source"]["url"] == "https://example.com/data.csv"


def test_download_with_url_is_valid():
    config = {
        "version": 1,
        "dataset_name": "example",
        "operation": "download",
        "source": {"type": "http", "url": "https://example.com/data.csv"},
        "destination": {"path": "/tmp/data"},
    }

    validated = validate_operation_config(config)
    assert validated["operation"] == "download"
    assert validated["source"]["url"] == "https://example.com/data.csv"


def test_download_requires_source_type():
    config = {
        "version": 1,
        "dataset_name": "example",
        "operation": "download",
        "source": {"url": "https://example.com/data.csv"},
        "destination": {"path": "/tmp/data"},
    }

    with pytest.raises(ValueError, match="source.type"):
        validate_operation_config(config)


def test_download_rejects_unsupported_source_type():
    config = {
        "version": 1,
        "dataset_name": "example",
        "operation": "download",
        "source": {"type": "ftp", "url": "https://example.com/data.csv"},
        "destination": {"path": "/tmp/data"},
    }

    with pytest.raises(ValueError, match="source.type"):
        validate_operation_config(config)


def test_prepare_config_requires_representation_and_task_characterization():
    config = {
        "version": 1,
        "dataset_name": "example",
        "operation": "prepare",
        "staging_dir": "./data/raw/example",
        "prepared_dataset_dir": "./data/prepared/example",
    }

    with pytest.raises(ValueError, match="representation|task_characterization"):
        validate_operation_config(config)


def test_bootstrap_config_writes_yaml_with_explicit_path_and_name(tmp_path):
    config_path = bootstrap_config(
        config_dir=tmp_path,
        config_name="kmds-prep.yaml",
        dataset_name="example",
        representation="graph",
        task_characterization="clustering",
        staging_dir="./data/raw/example",
        prepared_dataset_dir="./data/prepared/example",
        write=True,
    )

    assert config_path == tmp_path / "kmds-prep.yaml"
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert data["operation"] == "prepare"
    assert data["representation"] == "graph"
    assert data["task_characterization"] == "clustering"
    assert data["staging_dir"] == "./data/raw/example"


def test_menu_options_cover_required_task_and_representation_values():
    assert "tabular" in get_representation_options()
    assert "graph" in get_representation_options()
    assert "classification" in get_learning_options()
    assert "survival_analysis" in get_learning_options()
