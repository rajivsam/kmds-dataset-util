from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import yaml

from data_prep.config import load_config


def _normalize_kaggle_dataset_id(dataset_id: str) -> str:
    """Accept a Kaggle dataset slug or full URL and return the canonical owner/slug form."""
    value = str(dataset_id).strip().strip("/")
    if not value:
        raise ValueError("Kaggle dataset identifier is required")

    if "://" in value:
        parsed = urlparse(value)
        path = parsed.path.strip("/")
        if "datasets/" not in path:
            raise ValueError(f"Unsupported Kaggle URL: {dataset_id}")
        value = path.split("datasets/", 1)[1].strip("/")

    if "/" not in value:
        raise ValueError(
            "Kaggle dataset identifier must be in the form 'owner/dataset' or a Kaggle dataset URL"
        )

    return value


def _has_kaggle_credentials() -> bool:
    """Return true when Kaggle credentials are present in either JSON or access-token format."""
    config_dir = Path.home() / ".kaggle"
    if not config_dir.exists():
        return False

    return any(path.exists() for path in (config_dir / "kaggle.json", config_dir / "access_token"))


def verify_download_path(destination: str | Path) -> Path:
    """Validate that a download path exists and contains usable content."""
    path = Path(destination).expanduser()

    if not path.exists():
        raise FileNotFoundError(
            f"Download path does not exist: {path}. Create the folder before verifying the download."
        )

    if path.is_file():
        if path.stat().st_size <= 0:
            raise FileNotFoundError(
                f"Download file exists but is empty: {path}. The dataset may not have downloaded correctly."
            )
        return path

    if not path.is_dir():
        raise NotADirectoryError(f"Download destination is not a directory or file: {path}")

    entries = list(path.iterdir())
    if not entries:
        raise FileNotFoundError(
            f"Download path exists but contains no files: {path}. The dataset may not have downloaded correctly."
        )

    return path


def _extract_archive(archive_path: Path, dest_path: Path) -> Path:
    """Extract supported archive types into a destination directory and return it."""
    archive_path = archive_path.expanduser()
    if not archive_path.exists() or archive_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Archive is missing or empty: {archive_path}")

    destination_dir = dest_path / archive_path.stem.replace(".tar", "").replace(".gz", "").replace(".bz2", "")
    destination_dir.mkdir(parents=True, exist_ok=True)

    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(destination_dir)
        return destination_dir

    if tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(destination_dir)
        return destination_dir

    raise ValueError(f"Unsupported archive format: {archive_path}")


def _is_archive_download(path: Path) -> bool:
    """Return true only for archive formats that should be unpacked after download."""
    name = path.name.lower()
    if name.endswith((".xlsx", ".xls", ".xlsm", ".xlsb")):
        return False
    if name.endswith((".zip", ".tar", ".gz", ".tgz", ".bz2", ".tbz", ".xz", ".txz")):
        return True
    return False


def generate_data_dictionary_for_download(dataset_dir: str | Path) -> Path:
    """Create a raw data dictionary alongside the downloaded raw dataset files."""
    data_source = Path(dataset_dir).expanduser()
    if data_source.is_file():
        files_to_scan = [data_source]
        target_dir = data_source.parent
    else:
        target_dir = data_source
        target_dir.mkdir(parents=True, exist_ok=True)
        files_to_scan = sorted(
            p for p in target_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() == ".csv"
            and p.name not in {"data_dictionary.csv", "data_dictionary_raw.csv"}
        )

    if data_source.is_file() and data_source.suffix.lower() != ".csv":
        output_path = target_dir / "data_dictionary_raw.csv"
        pd.DataFrame(columns=["attribute", "description", "data_type"]).to_csv(output_path, index=False)
        return output_path

    rows: list[dict[str, str]] = []
    seen: set[str] = set()

    for csv_path in files_to_scan:
        if csv_path.is_dir():
            continue
        try:
            sample = pd.read_csv(csv_path, nrows=5)
        except Exception:
            continue
        for column in sample.columns:
            column_name = str(column)
            if column_name in seen:
                continue
            seen.add(column_name)
            rows.append(
                {
                    "attribute": column_name,
                    "description": f"{csv_path.stem} field: {column_name}",
                    "data_type": str(sample[column].dtype),
                }
            )

    output_path = target_dir / "data_dictionary_raw.csv"
    dictionary_df = pd.DataFrame(rows, columns=["attribute", "description", "data_type"])
    if dictionary_df.empty:
        dictionary_df = pd.DataFrame(columns=["attribute", "description", "data_type"])
    dictionary_df.to_csv(output_path, index=False)
    return output_path


def download_http_file(url: str, destination: str | Path) -> Path:
    """Download a file or archive from a URL to a target directory."""
    import urllib.request

    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    parsed = urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid HTTP URL: {url}")

    filename = Path(parsed.path).name or "downloaded_file"
    output_path = dest_path / filename

    urllib.request.urlretrieve(url, output_path)

    if _is_archive_download(output_path):
        extracted = _extract_archive(output_path, dest_path)
        generate_data_dictionary_for_download(extracted)
        return extracted

    generate_data_dictionary_for_download(output_path)
    return output_path


def download_kaggle_dataset(dataset_id: str, destination: str | Path) -> Path:
    """Download a Kaggle dataset using the official Kaggle API.

    The Kaggle dataset can be supplied as either a slug like 'owner/name' or as a full
    dataset URL. This uses the Kaggle client directly instead of the older opendatasets
    wrapper.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("kaggle is required for Kaggle dataset downloads") from exc

    if not _has_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Add ~/.kaggle/kaggle.json or ~/.kaggle/access_token "
            "for your Kaggle account."
        )

    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    dataset_slug = _normalize_kaggle_dataset_id(dataset_id)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(dataset_slug, path=str(dest_path), unzip=True, quiet=False)
    verify_download_path(dest_path)
    generate_data_dictionary_for_download(dest_path)
    return dest_path


def download_dataset(source: dict, destination: str | Path) -> Path:
    """Dispatch to the appropriate download implementation."""
    if source.get("type") == "http" and source.get("url"):
        return download_http_file(source["url"], destination)

    if source.get("type") == "kaggle" and source.get("kaggle_dataset"):
        return download_kaggle_dataset(source["kaggle_dataset"], destination)

    raise ValueError("Unsupported source type or missing source configuration")


def download_from_config(config_path: str | Path) -> Path:
    """Load a YAML config and execute the configured download, verifying the result path."""
    config = load_config(config_path)
    if config.get("operation") != "download":
        raise ValueError(f"Config file is not a download configuration: {config_path}")

    source = config.get("source") or {}
    destination = config.get("destination") or {}
    destination_path = destination.get("path")
    if not destination_path:
        raise ValueError(f"Config file is missing destination.path: {config_path}")

    resolved_destination = Path(destination_path)
    if not resolved_destination.is_absolute():
        resolved_destination = (Path(config_path).resolve().parent / resolved_destination).resolve()

    if source.get("type") == "kaggle" and source.get("kaggle_dataset"):
        result = download_kaggle_dataset(source["kaggle_dataset"], resolved_destination)
    elif source.get("type") == "http" and source.get("url"):
        result = download_http_file(source["url"], resolved_destination)
    else:
        raise ValueError(
            f"Unsupported source configuration in {config_path}: expected source.type and source.url or source.kaggle_dataset"
        )

    verify_download_path(result)
    generate_data_dictionary_for_download(result)
    return result


def _default_config_dir() -> Path:
    """Default to a config folder in the current working directory, not inside the package source tree."""
    return Path.cwd() / "configs"


def kaggle_url_to_config(
    kaggle_url: str,
    *,
    write: bool = False,
    output_dir: str | Path | None = None,
    config_dir: str | Path | None = None,
) -> dict | Path:
    """Turn a Kaggle dataset URL into a config dict, optionally writing it to a runtime config directory.

    When write is False, the YAML config is printed to stdout and returned as a dict.
    When write is True, the config is written to the current working directory's config folder
    unless a different config_dir is provided. The config is never created under the package's
    source tree.

    output_dir overrides the default destination path, which remains ./data/raw/<dataset_name>
    when not provided.
    """
    dataset_id = _normalize_kaggle_dataset_id(kaggle_url)
    dataset_name = dataset_id.split("/", 1)[1]
    destination_path = str(output_dir) if output_dir is not None else f"./data/raw/{dataset_name}"

    config = {
        "version": 1,
        "dataset_name": dataset_name,
        "operation": "download",
        "source": {
            "type": "kaggle",
            "kaggle_dataset": dataset_id,
        },
        "destination": {
            "path": destination_path,
            "format": "csv",
        },
    }

    yaml_text = yaml.safe_dump(config, sort_keys=False, default_flow_style=False)

    if not write:
        print(yaml_text, end="")
        return config

    target_config_dir = Path(config_dir).expanduser() if config_dir is not None else _default_config_dir()
    try:
        target_config_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise PermissionError(
            f"Cannot create config directory at {target_config_dir}. "
            "Choose a writable location or pass config_dir=... to this function."
        ) from exc

    file_path = target_config_dir / f"{dataset_name}.yaml"
    try:
        file_path.write_text(yaml_text, encoding="utf-8")
    except OSError as exc:
        raise PermissionError(
            f"Cannot write config file to {file_path}. Ensure the config directory is writable."
        ) from exc
    return file_path
