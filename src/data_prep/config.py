from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


def get_representation_options() -> list[str]:
    """Return supported dataset representations."""
    return ["tabular", "graph"]


def get_learning_options() -> list[str]:
    """Return supported task-characterization choices for the modeler."""
    return [
        "regression",
        "classification",
        "clustering",
        "survival_analysis",
        "time_series",
        "anomaly_detection",
    ]


def _normalize_operation(config: dict[str, Any]) -> dict[str, Any]:
    operation = config.get("operation") or config.get("action")
    if not operation:
        raise ValueError(
            "Config requires an operation: download, prepare, inspect, transform, or write"
        )
    config["operation"] = str(operation).lower()
    return config


def validate_operation_config(config: dict[str, Any]) -> dict[str, Any]:
    """Validate a data prep config based on the selected operation."""
    if not isinstance(config, dict):
        raise ValueError("Config must be a dictionary")

    config = _normalize_operation(config)
    if config.get("version") is None:
        raise ValueError("Config requires a version")

    operation = config["operation"]

    if operation == "download":
        source = config.get("source") or {}
        if not isinstance(source, dict):
            raise ValueError("download operation requires a source object")

        source_type = str(source.get("type", "")).lower()
        if not source_type:
            raise ValueError("download operation requires source.type")
        if source_type not in {"http", "kaggle"}:
            raise ValueError("download operation requires source.type to be either 'http' or 'kaggle'")
        source["type"] = source_type

        if source_type == "http" and not source.get("url"):
            raise ValueError("download operation requires source.url for http sources")
        if source_type == "kaggle" and not source.get("kaggle_dataset"):
            raise ValueError("download operation requires source.kaggle_dataset for kaggle sources")

        destination = config.get("destination") or {}
        if not isinstance(destination, dict):
            raise ValueError("download operation requires a destination object")
        if not destination.get("path"):
            raise ValueError("download operation requires destination.path")
        destination_format = str(destination.get("format", "csv")).lower()
        if destination_format not in {"csv", "parquet"}:
            raise ValueError("download operation destination.format must be csv or parquet")
        destination["format"] = destination_format

    elif operation == "prepare":
        if not config.get("staging_dir"):
            raise ValueError("prepare operation requires staging_dir")
        if not config.get("prepared_dataset_dir") and not config.get("output"):
            raise ValueError("prepare operation requires prepared_dataset_dir or output.path")

        representation = str(config.get("representation", "")).lower()
        if representation not in get_representation_options():
            raise ValueError(
                "prepare operation requires a valid representation: "
                + " or ".join(get_representation_options())
            )
        config["representation"] = representation

        task_characterization = str(config.get("task_characterization", "")).lower()
        if not task_characterization:
            raise ValueError("prepare operation requires task_characterization")
        if task_characterization not in get_learning_options():
            raise ValueError(
                "prepare operation requires a valid task_characterization: "
                + " or ".join(get_learning_options())
            )
        config["task_characterization"] = task_characterization

        output = config.get("output") or {}
        if isinstance(output, dict):
            if output.get("path"):
                output_path = str(output["path"])
                if not output_path:
                    raise ValueError("prepare operation requires output.path")
                format_value = str(output.get("format", "csv")).lower()
                if format_value not in {"csv", "parquet"}:
                    raise ValueError("prepare operation output.format must be csv or parquet")
                output["format"] = format_value
                config["output"] = output

    elif operation == "inspect":
        if not config.get("input_path"):
            raise ValueError("inspect operation requires input_path")

    elif operation == "transform":
        if not config.get("input_path"):
            raise ValueError("transform operation requires input_path")
        output = config.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("transform operation requires an output object")
        if not output.get("path"):
            raise ValueError("transform operation requires output.path")
        if not output.get("format"):
            raise ValueError("transform operation requires output.format")

    elif operation == "write":
        if not config.get("input_path"):
            raise ValueError("write operation requires input_path")
        output = config.get("output") or {}
        if not isinstance(output, dict):
            raise ValueError("write operation requires an output object")
        if not output.get("path"):
            raise ValueError("write operation requires output.path")
        if not output.get("format"):
            raise ValueError("write operation requires output.format")

    else:
        raise ValueError(
            "Unsupported operation. Use download, prepare, inspect, transform, or write"
        )

    return config


def bootstrap_config(
    *,
    config_dir: str | Path | None = None,
    config_name: str = "kmds-prep.yaml",
    dataset_name: str | None = None,
    representation: str = "tabular",
    task_characterization: str = "classification",
    staging_dir: str | None = None,
    prepared_dataset_dir: str | None = None,
    output_format: str = "csv",
    write: bool = False,
) -> dict[str, Any] | Path:
    """Create a YAML config for the prepare workflow with explicit config file path support."""
    if not dataset_name:
        raise ValueError("dataset_name is required for bootstrap_config")

    normalized_representation = str(representation).lower()
    if normalized_representation not in get_representation_options():
        raise ValueError(
            "representation must be one of: " + " or ".join(get_representation_options())
        )

    normalized_task = str(task_characterization).lower()
    if normalized_task not in get_learning_options():
        raise ValueError(
            "task_characterization must be one of: " + " or ".join(get_learning_options())
        )

    resolved_staging_dir = staging_dir or f"./data/raw/{dataset_name}"
    resolved_prepared_dir = prepared_dataset_dir or f"./data/prepared/{dataset_name}"

    config = {
        "version": 1,
        "dataset_name": dataset_name,
        "operation": "prepare",
        "representation": normalized_representation,
        "task_characterization": normalized_task,
        "staging_dir": resolved_staging_dir,
        "prepared_dataset_dir": resolved_prepared_dir,
        "output": {
            "path": resolved_prepared_dir,
            "format": str(output_format).lower(),
        },
    }

    if str(config["output"]["format"]).lower() not in {"csv", "parquet"}:
        raise ValueError("output_format must be csv or parquet")

    validated = validate_operation_config(config)

    if not yaml:
        raise RuntimeError("PyYAML is required to create config files")

    target_dir = Path(config_dir).expanduser() if config_dir is not None else Path.cwd() / "configs"
    target_dir.mkdir(parents=True, exist_ok=True)
    file_name = config_name if config_name.endswith(".yaml") or config_name.endswith(".yml") else f"{config_name}.yaml"
    config_path = target_dir / file_name

    if write:
        config_path.write_text(yaml.safe_dump(validated, sort_keys=False), encoding="utf-8")
        return config_path

    return validated


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files")

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    if not isinstance(data, dict):
        raise ValueError("The config file must contain a YAML mapping at the top level")

    return validate_operation_config(data)
