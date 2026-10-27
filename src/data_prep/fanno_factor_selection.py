"""File-based Poisson operating-window selection using Fano factors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_AGGREGATION_WINDOW = "1min"
DEFAULT_MIN_AGGREGATION_WINDOWS = 30
DEFAULT_CONFIDENCE = 0.90

__all__ = [
    "DEFAULT_AGGREGATION_WINDOW",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MIN_AGGREGATION_WINDOWS",
    "select_poisson_operating_window",
]


def _as_timedelta(value: str | pd.Timedelta, *, name: str) -> pd.Timedelta:
    try:
        interval = pd.to_timedelta(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a valid fixed-duration interval: {value!r}") from exc
    if pd.isna(interval) or interval <= pd.Timedelta(0):
        raise ValueError(f"{name} must be a positive fixed-duration interval")
    return interval


def _aligned_aggregation_interval(
    window: pd.Timedelta,
    preferred_interval: pd.Timedelta,
) -> tuple[pd.Timedelta, int]:
    window_ns = window.value
    preferred_ns = preferred_interval.value
    intervals_per_window = (window_ns + preferred_ns - 1) // preferred_ns
    if window_ns % preferred_ns == 0:
        return preferred_interval, intervals_per_window

    adjusted_ns, remainder = divmod(window_ns, intervals_per_window)
    if remainder:
        raise ValueError(
            "The candidate and aggregation intervals cannot be aligned to whole nanoseconds"
        )
    return pd.Timedelta(adjusted_ns, unit="ns"), intervals_per_window


def _window_metrics(
    timestamps: pd.Series,
    *,
    window_label: str,
    window: pd.Timedelta,
    aggregation_window: pd.Timedelta,
    min_aggregation_windows: int,
    confidence: float,
) -> dict[str, Any]:
    total_span = timestamps.iloc[-1] - timestamps.iloc[0]
    num_windows = int(total_span // window)
    aligned_interval, intervals_per_window = _aligned_aggregation_interval(
        window, aggregation_window
    )
    row: dict[str, Any] = {
        "window": window_label,
        "window_minutes": window.total_seconds() / 60,
        "aggregation_interval": aligned_interval,
        "intervals_per_window": intervals_per_window,
        "aggregation_window_count": num_windows,
        "fano_factor": float("nan"),
        "fano_lower": float("nan"),
        "fano_upper": float("nan"),
        "fano_deviation": float("nan"),
        "poisson_assumption_satisfied": False,
        "status": "insufficient_data",
    }
    if num_windows < min_aggregation_windows:
        return row

    if num_windows < 2:
        return row

    bucket_ids = ((timestamps - timestamps.iloc[0]) // window).to_numpy(dtype=np.int64)
    included = bucket_ids < num_windows
    counts = np.bincount(bucket_ids[included], minlength=num_windows).astype(float)
    mean_count = float(counts.mean())
    if mean_count == 0:
        row["status"] = "no_events"
        return row

    fano_factor = float(counts.var(ddof=1) / mean_count)
    degrees_of_freedom = num_windows - 1
    alpha = 1.0 - confidence
    lower = float(stats.chi2.ppf(alpha / 2, degrees_of_freedom) / degrees_of_freedom)
    upper = float(stats.chi2.ppf(1.0 - alpha / 2, degrees_of_freedom) / degrees_of_freedom)
    accepted = lower <= fano_factor <= upper
    row.update(
        {
            "fano_factor": fano_factor,
            "fano_lower": lower,
            "fano_upper": upper,
            "fano_deviation": abs(fano_factor - 1.0),
            "poisson_assumption_satisfied": accepted,
            "status": "poisson_compatible" if accepted else "not_poisson_compatible",
        }
    )
    return row


def _plot_metrics(metrics: pd.DataFrame) -> plt.Axes:
    figure, axes = plt.subplots()
    x_values = np.arange(len(metrics))
    tested = metrics["fano_factor"].notna()
    accepted = metrics["poisson_assumption_satisfied"]

    if tested.any():
        tested_x = x_values[tested.to_numpy()]
        values = metrics.loc[tested, "fano_factor"].to_numpy(dtype=float)
        lower = metrics.loc[tested, "fano_lower"].to_numpy(dtype=float)
        upper = metrics.loc[tested, "fano_upper"].to_numpy(dtype=float)
        axes.fill_between(
            tested_x, lower, upper, alpha=0.25, color="C2", label="Acceptable range (chi-squared limits)"
        )
        axes.plot(tested_x, lower, color="C2", linewidth=1)
        axes.plot(tested_x, upper, color="C2", linewidth=1)
        axes.plot(tested_x, values, marker="o", label="Fano factor")

    if accepted.any():
        accepted_x = x_values[accepted.to_numpy()]
        accepted_values = metrics.loc[accepted, "fano_factor"].to_numpy(dtype=float)
        axes.scatter(accepted_x, accepted_values, marker="o", s=80, label="Poisson-compatible")

    axes.axhline(1.0, linestyle="--", linewidth=1, color="gray", label="Poisson reference (F = 1)")
    axes.set_xticks(x_values, metrics["window"].tolist(), rotation=45)
    axes.set_title("Operating window vs Fano factor")
    axes.set_xlabel("Candidate operating window")
    axes.set_yscale("log")
    axes.set_ylabel("Fano factor (log scale)")
    axes.grid(alpha=0.3)
    axes.legend()
    figure.tight_layout()
    return axes


def select_poisson_operating_window(
    event_file: str | Path,
    windows: list[str | pd.Timedelta] | tuple[str | pd.Timedelta, ...],
    *,
    timestamp_col: str = "timestamp",
    aggregation_window: str | pd.Timedelta = DEFAULT_AGGREGATION_WINDOW,
    min_aggregation_windows: int = DEFAULT_MIN_AGGREGATION_WINDOWS,
    confidence: float = DEFAULT_CONFIDENCE,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate candidate windows in a CSV event log and return metrics and a Fano plot.

    Candidate intervals are tested in ascending duration order. The reported aggregation-window
    count is the number of complete, non-overlapping candidate windows in the timestamp span.
    """
    if not windows:
        raise ValueError("At least one candidate window must be supplied")
    if (
        isinstance(min_aggregation_windows, bool)
        or not isinstance(min_aggregation_windows, (int, np.integer))
        or min_aggregation_windows < 2
    ):
        raise ValueError("min_aggregation_windows must be an integer of at least 2")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")
    if not timestamp_col:
        raise ValueError("timestamp_col must not be empty")

    preferred_interval = _as_timedelta(aggregation_window, name="aggregation_window")
    candidates = [
        (str(value), _as_timedelta(value, name="candidate window"))
        for value in windows
    ]
    candidates.sort(key=lambda candidate: candidate[1])
    if len({window.value for _, window in candidates}) != len(candidates):
        raise ValueError("Candidate windows must not contain duplicate intervals")

    frame = pd.read_csv(event_file)
    if timestamp_col not in frame.columns:
        raise KeyError(f"Column {timestamp_col!r} was not found in {event_file}")
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if timestamps.empty:
        raise ValueError(f"No valid event-arrival timestamps were found in {event_file}")

    metrics = pd.DataFrame(
        [
            _window_metrics(
                timestamps,
                window_label=label,
                window=window,
                aggregation_window=preferred_interval,
                min_aggregation_windows=int(min_aggregation_windows),
                confidence=float(confidence),
            )
            for label, window in candidates
        ]
    )
    acceptable_windows = metrics.loc[
        metrics["poisson_assumption_satisfied"], "window"
    ].tolist()

    axes = _plot_metrics(metrics)
    output_plot_path = None
    if plot_path is not None:
        output_plot_path = Path(plot_path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)
        axes.figure.savefig(output_plot_path, bbox_inches="tight")

    return {
        "metrics": metrics,
        "acceptable_windows": acceptable_windows,
        "selection_status": (
            "acceptable_windows_found" if acceptable_windows else "no_acceptable_windows"
        ),
        "aggregation_window": preferred_interval,
        "min_aggregation_windows": int(min_aggregation_windows),
        "confidence": float(confidence),
        "plot": axes.figure,
        "axes": axes,
        "plot_path": output_plot_path,
    }
