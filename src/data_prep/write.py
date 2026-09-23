from __future__ import annotations

from pathlib import Path


def write_output_file(input_path: str | Path, output_path: str | Path, output_format: str) -> Path:
    """Write a prepared dataset to the configured output location.

    For the initial version, this supports copying an existing file to the target output path.
    """
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if output_format.lower() not in {"csv", "parquet"}:
        raise ValueError("Unsupported output format. Use csv or parquet")

    if src.suffix.lower().lstrip(".") != output_format.lower():
        target = target.with_suffix(f".{output_format.lower()}")

    target.write_bytes(src.read_bytes())
    return target
