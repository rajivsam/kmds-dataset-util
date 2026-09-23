from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from data_prep.config import get_learning_options, get_representation_options


def _normalize_dictionary_frame(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()

    rename_map: dict[str, str] = {}

    if "attribute" not in df.columns:
        for candidate, target in {
            "column_name": "attribute",
            "field_name": "attribute",
            "Field Name": "attribute",
            "field": "attribute",
        }.items():
            if candidate in df.columns:
                rename_map[candidate] = target
                break

    if "description" not in df.columns:
        for candidate, target in {
            "desc": "description",
            "definition": "description",
            "Definition": "description",
            "description_text": "description",
        }.items():
            if candidate in df.columns:
                rename_map[candidate] = target
                break

    if "data_type" not in df.columns:
        for candidate, target in {
            "python_type": "data_type",
            "dtype": "data_type",
            "type": "data_type",
        }.items():
            if candidate in df.columns:
                rename_map[candidate] = target
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    if "attribute" not in df.columns:
        raise ValueError("Dictionary rows must contain an 'attribute' column")
    if "description" not in df.columns:
        df["description"] = ""
    if "data_type" not in df.columns:
        df["data_type"] = "unknown"

    df = df[["attribute", "description", "data_type"]].copy()
    df["attribute"] = df["attribute"].astype(str).str.strip()
    return df


def _write_dictionary_files(metadata: dict[str, Any] | None, target_dir: Path) -> dict[str, str]:
    if not metadata:
        return {}

    file_map = {
        "data_dictionary": "data_dictionary_processed.csv",
        "raw_data_dictionary": "data_dictionary_raw.csv",
        "analysis_data_dictionary": "data_dictionary_analysis.csv",
        "prepared_data_dictionary": "data_dictionary_prepared.csv",
    }
    written: dict[str, str] = {}

    for key, file_name in file_map.items():
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, pd.DataFrame):
            rows = value
        elif isinstance(value, list):
            rows = pd.DataFrame(value)
        elif isinstance(value, dict):
            rows = pd.DataFrame(value.get("records", value.get("rows", [])))
        else:
            continue

        if rows.empty:
            continue

        try:
            dictionary_df = _normalize_dictionary_frame(rows)
        except ValueError:
            continue

        output_path = target_dir / file_name
        dictionary_df.to_csv(output_path, index=False)
        written[key] = str(output_path)

    return written


def prepare_dataset(
    *,
    dataset_name: str,
    representation: str,
    task_characterization: str,
    staging_dir: str | Path,
    prepared_dataset_dir: str | Path,
    data: pd.DataFrame | None = None,
    output_format: str = "csv",
    filename: str | None = None,
    metadata: dict[str, Any] | None = None,
    dictionary_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a generic dataset preparation workflow and write the prepared CSV/Parquet artifact."""
    representation = str(representation).lower()
    task_characterization = str(task_characterization).lower()

    if representation not in get_representation_options():
        raise ValueError(f"Unsupported representation: {representation}")
    if task_characterization not in get_learning_options():
        raise ValueError(f"Unsupported task_characterization: {task_characterization}")

    if output_format.lower() not in {"csv", "parquet"}:
        raise ValueError("output_format must be csv or parquet")

    if data is None:
        raise ValueError("A pandas DataFrame is required for prepare_dataset")

    target_dir = Path(prepared_dataset_dir).expanduser()
    target_dir.mkdir(parents=True, exist_ok=True)

    dictionary_dir = (
        Path(dictionary_output_dir).expanduser()
        if dictionary_output_dir is not None
        else Path(staging_dir).expanduser()
    )
    dictionary_dir.mkdir(parents=True, exist_ok=True)

    file_name = filename or f"{dataset_name}_{representation}"
    output_path = target_dir / f"{file_name}.{output_format.lower()}"

    if output_format.lower() == "csv":
        data.to_csv(output_path, index=False)
    else:
        data.to_parquet(output_path, index=False)

    dictionary_outputs = _write_dictionary_files(metadata, dictionary_dir)

    result = {
        "dataset_name": dataset_name,
        "representation": representation,
        "task_characterization": task_characterization,
        "staging_dir": str(Path(staging_dir).expanduser()),
        "prepared_dataset_dir": str(target_dir),
        "output_path": str(output_path),
        "output_format": output_format.lower(),
        "metadata": metadata or {},
    }
    if dictionary_outputs:
        result["dictionary_files"] = dictionary_outputs
        if "data_dictionary" in dictionary_outputs:
            result["data_dictionary_path"] = dictionary_outputs["data_dictionary"]
        elif "prepared_data_dictionary" in dictionary_outputs:
            result["data_dictionary_path"] = dictionary_outputs["prepared_data_dictionary"]
    return result
