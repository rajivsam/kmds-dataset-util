"""Lean arrival-intensity and Fano-based window selection utilities."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - handled by the public API when SciPy is absent.
    scipy_stats = None

ENTERPRISE_WINDOWS = ["5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "2d", "3d", "5d", "1w", "2w", "4w"]
WINDOWS = ENTERPRISE_WINDOWS
DEFAULT_MIN_EVENTS = 5
DEFAULT_TARGET_PASS_RATE = 0.80
DEFAULT_HOME_BASE = "5m"
DEFAULT_TAU_BLOCK = "4h"

__all__ = [
    "ENTERPRISE_WINDOWS",
    "WINDOWS",
    "DEFAULT_HOME_BASE",
    "DEFAULT_TAU_BLOCK",
    "acceptable_over_dispersed_window_size",
    "align_to_integer_bin_grid",
    "build_arrival_intensity_series",
    "find_optimal_time_bucket",
    "optimize_arrival_binning",
    "plot_window_fano_factors",
    "select_count_binning_window",
    "summarize_integer_alignment",
    "tseda_dataset_export",
]


def _window_to_timedelta(window: str) -> pd.Timedelta:
    mapping = {
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1h": pd.Timedelta(hours=1),
        "2h": pd.Timedelta(hours=2),
        "4h": pd.Timedelta(hours=4),
        "8h": pd.Timedelta(hours=8),
        "12h": pd.Timedelta(hours=12),
        "1d": pd.Timedelta(days=1),
        "2d": pd.Timedelta(days=2),
        "3d": pd.Timedelta(days=3),
        "5d": pd.Timedelta(days=5),
        "1w": pd.Timedelta(weeks=1),
        "2w": pd.Timedelta(weeks=2),
        "4w": pd.Timedelta(weeks=4),
    }
    if window not in mapping:
        raise ValueError(f"Unsupported time window: {window!r}. Allowed values: {ENTERPRISE_WINDOWS}")
    return mapping[window]


def _window_to_freq(window: str) -> str:
    mapping = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "h",
        "2h": "2h",
        "4h": "4h",
        "8h": "8h",
        "12h": "12h",
        "1d": "D",
        "2d": "2D",
        "3d": "3D",
        "5d": "5D",
        "1w": "W",
        "2w": "2W",
        "4w": "4W",
    }
    if window not in mapping:
        raise ValueError(f"Unsupported time window: {window!r}. Allowed values: {ENTERPRISE_WINDOWS}")
    return mapping[window]


def _normalize_windows(windows: list[str] | tuple[str, ...] | None) -> list[str]:
    if windows is None:
        return list(ENTERPRISE_WINDOWS)
    values = [str(window) for window in windows]
    if not values:
        raise ValueError("At least one time window must be supplied")
    return sorted(values, key=lambda value: _window_to_timedelta(value))


def _coerce_tau_block(tau_block: str | pd.Timedelta | None) -> pd.Timedelta:
    if tau_block is None:
        tau_block = DEFAULT_TAU_BLOCK
    if isinstance(tau_block, str):
        stripped = tau_block.strip().lower()
        if stripped.endswith("m"):
            return pd.Timedelta(minutes=float(stripped[:-1]))
        if stripped.endswith("h"):
            return pd.Timedelta(hours=float(stripped[:-1]))
        if stripped.endswith("d"):
            return pd.Timedelta(days=float(stripped[:-1]))
        if stripped.endswith("w"):
            return pd.Timedelta(weeks=float(stripped[:-1]))
        raise ValueError(f"Unsupported tau_block: {tau_block!r}")
    return pd.to_timedelta(tau_block)


def _window_unit_value(window: str) -> float:
    return float(window[:-1])


def _window_unit_label(window: str) -> str:
    suffix = window[-1]
    if suffix == "m":
        return "minute"
    if suffix == "h":
        return "hour"
    if suffix == "d":
        return "day"
    if suffix == "w":
        return "week"
    raise ValueError(f"Unsupported window suffix: {window!r}")


def _empty_arrival_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["bin_id", "bin_start", "bin_end", "window", "count", "intensity"])


def _count_buckets_for_window(ts: pd.Series | pd.DatetimeIndex, window: str) -> pd.Series:
    values = pd.to_datetime(ts, errors="coerce").dropna().sort_values()
    if values.empty:
        return pd.Series(dtype="int64", name="count")

    freq = _window_to_freq(window)
    counts = pd.Series(1, index=values).resample(freq, label="left", closed="left").count().rename("count")
    if counts.empty:
        return counts

    last_allowed_end = values.max()
    while len(counts) and counts.index[-1] + _window_to_timedelta(window) > last_allowed_end:
        counts = counts.iloc[:-1]
    return counts


def _local_fano_factors(counts: pd.Series, window: str, tau_block: str | pd.Timedelta) -> list[float]:
    values = counts.astype(float).tolist()
    if not values:
        return []

    block_td = _coerce_tau_block(tau_block)
    block_size = max(1, int(block_td / _window_to_timedelta(window)))
    fanos: list[float] = []
    for start in range(0, len(values), block_size):
        block = values[start : start + block_size]
        if not block:
            continue
        mean = float(sum(block) / len(block))
        if mean <= 0:
            continue
        variance = float(sum((value - mean) ** 2 for value in block) / len(block))
        fanos.append(variance / mean)
    return fanos


def expected_poisson_zeros(lambda_value: float, num_observations: int | float, *, print_output: bool = False) -> float:
    """Return N * exp(-lambda), the expected number of zero events under a Poisson model."""
    lam = float(lambda_value)
    n = float(num_observations)
    if lam < 0:
        raise ValueError("lambda_value must be non-negative")
    if n < 0:
        raise ValueError("num_observations must be non-negative")
    expected = n * math.exp(-lam)
    if print_output:
        print(f"Poisson expected zeros: lambda(mean count per selected bucket)={lam:.6f}, observations={int(n)} -> expected_zeros={expected:.6f}")
    return expected


def fano_acceptance_interval(
    counts: pd.Series | list[float] | list[int],
    *,
    window: str,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Return a dataset-specific 95% Fano acceptance interval under a homogeneous Poisson null model."""
    if scipy_stats is None:
        raise ImportError("scipy is required to compute the chi-squared Fano acceptance interval")
    if not 0.0 < float(confidence) < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1")

    values = pd.Series(counts, dtype="float64")
    local_fanos = _local_fano_factors(values, window, tau_block)
    if not local_fanos:
        return {
            "window": window,
            "confidence": float(confidence),
            "n_blocks": 0,
            "df": 0,
            "fano_mean": float("nan"),
            "fano_lower": float("nan"),
            "fano_upper": float("nan"),
            "within_acceptance_range": False,
            "acceptance_interval": (float("nan"), float("nan")),
        }

    avg_fano = float(sum(local_fanos) / len(local_fanos))
    df = max(1, len(local_fanos) - 1)
    alpha = 1.0 - confidence
    lower = float(scipy_stats.chi2.ppf(alpha / 2.0, df) / df)
    upper = float(scipy_stats.chi2.ppf(1.0 - alpha / 2.0, df) / df)
    within = bool(lower <= avg_fano <= upper)
    return {
        "window": window,
        "confidence": float(confidence),
        "n_blocks": len(local_fanos),
        "df": df,
        "fano_mean": avg_fano,
        "fano_lower": lower,
        "fano_upper": upper,
        "within_acceptance_range": within,
        "acceptance_interval": (lower, upper),
    }


def acceptable_over_dispersed_window_size(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int | None = None,
    target_pass_rate: float | None = None,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Return a dataset-specific Fano-by-window table for each candidate window.

    The min-events and pass-rate arguments are retained for backward compatibility only and are
    intentionally ignored because the Fano selection is governed by the reference-window acceptance
    region rather than a count-based eligibility rule.
    """
    candidate_windows = _normalize_windows(windows)

    if isinstance(df, pd.Series):
        timestamps = pd.to_datetime(df, errors="coerce").dropna().sort_values()
    else:
        if timestamp_col not in df.columns:
            raise KeyError(f"Column {timestamp_col!r} was not found in the input DataFrame")
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()

    if timestamps.empty:
        raise ValueError("No valid timestamps were available for Fano-window selection.")

    rows: list[dict[str, Any]] = []
    reference_window = None
    reference_fano = None
    reference_interval = None

    for window in candidate_windows:
        counts = _count_buckets_for_window(timestamps, window)
        if counts.empty:
            row = {
                "window": window,
                "window_minutes": float(_window_to_timedelta(window).total_seconds() / 60.0),
                "average_local_fano": float("nan"),
                "fano_deviation": float("nan"),
                "fano_lower": float("nan"),
                "fano_upper": float("nan"),
                "is_admissible": False,
            }
            rows.append(row)
            continue

        fanos = _local_fano_factors(counts, window, tau_block)
        avg_fano = float(sum(fanos) / len(fanos)) if fanos else 1.0
        interval = fano_acceptance_interval(counts, window=window, tau_block=tau_block, confidence=confidence)
        row = {
            "window": window,
            "window_minutes": float(_window_to_timedelta(window).total_seconds() / 60.0),
            "average_local_fano": avg_fano,
            "fano_deviation": abs(avg_fano - 1.0),
            "fano_lower": float(interval["fano_lower"]),
            "fano_upper": float(interval["fano_upper"]),
            "is_admissible": True,
        }
        rows.append(row)

        current_reference_distance = abs(reference_fano - 1.0) if reference_fano is not None else float("inf")
        if reference_window is None or row["fano_deviation"] < current_reference_distance:
            reference_window = window
            reference_fano = avg_fano
            reference_interval = interval

    table = pd.DataFrame(rows).sort_values("window_minutes", kind="mergesort").reset_index(drop=True)
    if table.empty:
        raise ValueError("No valid Fano-window rows were produced for the supplied timestamps.")

    reference_candidates = table[table["average_local_fano"].notna()].copy()
    if reference_candidates.empty:
        raise ValueError("No candidate windows had a usable average local Fano value.")

    if reference_window is None or reference_fano is None or reference_interval is None:
        reference_row = reference_candidates.sort_values("fano_deviation", kind="mergesort").iloc[0]
        reference_window = reference_row["window"]
        reference_fano = float(reference_row["average_local_fano"]) if pd.notna(reference_row["average_local_fano"]) else 1.0
        reference_interval = {
            "fano_lower": float(reference_row["fano_lower"]),
            "fano_upper": float(reference_row["fano_upper"]),
        }

    table.loc[table["window"] == reference_window, "average_local_fano"] = reference_fano
    table.loc[table["window"] == reference_window, "fano_lower"] = float(reference_interval["fano_lower"])
    table.loc[table["window"] == reference_window, "fano_upper"] = float(reference_interval["fano_upper"])

    dataset_lower_bound = float(reference_interval["fano_lower"])
    dataset_upper_bound = float(reference_interval["fano_upper"])
    table["reference_window"] = reference_window
    table["reference_window_average_local_fano"] = reference_fano
    table["dataset_acceptance_lower_bound"] = dataset_lower_bound
    table["dataset_acceptance_upper_bound"] = dataset_upper_bound
    table["acceptable_for_dataset"] = table["average_local_fano"].apply(
        lambda value: pd.notna(value) and dataset_lower_bound <= float(value) <= dataset_upper_bound
    )
    table["recommendation_reason"] = table.apply(
        lambda row: (
            "within_reference_acceptance_region"
            if bool(row["acceptable_for_dataset"])
            else "outside_reference_acceptance_region"
        ),
        axis=1,
    )
    return table


def align_to_integer_bin_grid(ts: pd.Series | pd.DatetimeIndex, window: str) -> pd.Series:
    values = pd.to_datetime(ts, errors="coerce").dropna()
    if values.empty:
        return pd.Series(dtype="datetime64[ns]")

    series = pd.Series(values)
    if window.endswith("m"):
        interval_minutes = int(window[:-1])
        base = series.dt.normalize()
        minutes_since_start = (series.dt.hour * 60 + series.dt.minute + series.dt.second / 60.0)
        aligned_minutes = (minutes_since_start // interval_minutes) * interval_minutes
        return (base + pd.to_timedelta(aligned_minutes, unit="min")).astype("datetime64[ns]")

    if window.endswith("h"):
        interval_hours = int(window[:-1])
        base = series.dt.normalize()
        hours_since_start = (series.dt.hour + series.dt.minute / 60.0 + series.dt.second / 3600.0)
        aligned_hours = (hours_since_start // interval_hours) * interval_hours
        return (base + pd.to_timedelta(aligned_hours, unit="h")).astype("datetime64[ns]")

    if window.endswith("d"):
        days = int(window[:-1])
        if days == 1:
            return series.dt.normalize().astype("datetime64[ns]")
        baseline = series.dt.normalize().min()
        day_offsets = (series.dt.normalize() - baseline).dt.days
        aligned_days = (day_offsets // days) * days
        return (baseline + pd.to_timedelta(aligned_days, unit="D")).astype("datetime64[ns]")

    if window.endswith("w"):
        weeks = int(window[:-1])
        baseline = series.dt.normalize().min()
        week_offsets = (series.dt.normalize() - baseline).dt.days // 7
        aligned_weeks = (week_offsets // weeks) * weeks
        return (baseline + pd.to_timedelta(aligned_weeks * 7, unit="D")).astype("datetime64[ns]")

    return series.astype("datetime64[ns]")


def optimize_arrival_binning(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
    min_events: int | None = None,
    target_pass_rate: float | None = None,
    home_base: str = DEFAULT_HOME_BASE,
    verbose: bool = True,
) -> dict[str, Any]:
    """Return the candidate window with the Fano factor closest to 1.

    The legacy min-events/pass-rate arguments are retained for compatibility but ignored; the
    selection is now driven solely by the Fano factor pattern.
    """
    candidate_windows = _normalize_windows(windows)
    if isinstance(df, pd.Series):
        timestamps = pd.to_datetime(df, errors="coerce").dropna()
    else:
        if timestamp_col not in df.columns:
            raise KeyError(f"Column {timestamp_col!r} was not found in the input DataFrame")
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce").dropna()

    if timestamps.empty:
        return {
            "windows": candidate_windows,
            "window_metrics": pd.DataFrame(columns=["window", "average_local_fano", "fano_deviation", "selected", "valid", "adjustment"]),
            "optimal_window": None,
            "optimal_counts": pd.Series(dtype="int64"),
            "optimal_intensity": pd.Series(dtype="float64"),
            "tau_block": _coerce_tau_block(tau_block),
        }

    rows: list[dict[str, Any]] = []
    for index, window in enumerate(candidate_windows):
        counts = _count_buckets_for_window(timestamps, window)
        fanos = _local_fano_factors(counts, window, tau_block)
        avg_fano = float(sum(fanos) / len(fanos)) if fanos else 1.0
        interval = fano_acceptance_interval(counts, window=window, tau_block=tau_block, confidence=0.95)

        row = {
            "index": index,
            "window": window,
            "average_local_fano": avg_fano,
            "fano_deviation": abs(avg_fano - 1.0),
            "fano_acceptance_lower": interval["fano_lower"],
            "fano_acceptance_upper": interval["fano_upper"],
            "within_95pct_fano_acceptance_range": interval["within_acceptance_range"],
            "selected": False,
            "valid": not counts.empty,
            "adjustment": "keep" if window == home_base else "coarsen",
        }
        if not row["valid"]:
            row["adjustment"] = "skip"
        rows.append(row)

    metrics = pd.DataFrame(rows)
    valid_metrics = metrics[metrics["valid"]].copy()
    if valid_metrics.empty:
        raise ValueError("No non-empty candidate windows were available for Fano selection.")

    winner_index = valid_metrics["fano_deviation"].idxmin()
    metrics["selected"] = False
    metrics.loc[winner_index, "selected"] = True
    metrics.loc[metrics["selected"], "adjustment"] = "closest_to_poisson_fano"
    metrics.loc[~metrics["valid"], "adjustment"] = "skip"
    metrics.loc[metrics["valid"] & ~metrics["selected"], "adjustment"] = "skip"

    metrics = metrics[["index", "window", "average_local_fano", "fano_deviation", "fano_acceptance_lower", "fano_acceptance_upper", "within_95pct_fano_acceptance_range", "selected", "valid", "adjustment"]]
    selected_window = str(metrics.loc[metrics["selected"], "window"].iloc[0])
    selected_counts = _count_buckets_for_window(timestamps, selected_window)
    selected_intensity = (selected_counts.astype(float) / _window_unit_value(selected_window)).round(3)
    selected_lambda = float(selected_counts.mean()) if not selected_counts.empty else 0.0
    expected_zero_count = expected_poisson_zeros(selected_lambda, len(timestamps))
    selected_acceptance = fano_acceptance_interval(selected_counts, window=selected_window, tau_block=tau_block, confidence=0.95)

    return {
        "windows": candidate_windows,
        "window_metrics": metrics,
        "optimal_window": selected_window,
        "optimal_counts": selected_counts,
        "optimal_intensity": selected_intensity,
        "selected_window": selected_window,
        "lambda_value": selected_lambda,
        "expected_zero_count": expected_zero_count,
        "fano_acceptance_interval": selected_acceptance,
        "fano_acceptance_lower": selected_acceptance["fano_lower"],
        "fano_acceptance_upper": selected_acceptance["fano_upper"],
        "within_95pct_fano_acceptance_range": selected_acceptance["within_acceptance_range"],
        "average_local_fano": float(metrics.loc[metrics["selected"], "average_local_fano"].iloc[0]),
        "tau_block": _coerce_tau_block(tau_block),
    }


def find_optimal_time_bucket(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int | None = None,
    target_pass_rate: float | None = None,
    home_base: str = DEFAULT_HOME_BASE,
    count_target_tolerance: float | None = None,
    verbose: bool = True,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
) -> dict[str, Any]:
    """Backward-compatible alias for the optimized Fano-based selector."""
    selection = optimize_arrival_binning(
        df,
        timestamp_col=timestamp_col,
        windows=windows,
        count_col=count_col,
        tau_block=tau_block,
        min_events=min_events,
        target_pass_rate=target_pass_rate,
        home_base=home_base,
        verbose=verbose,
    )
    selection_log = selection["window_metrics"].copy()
    if "adjustment" not in selection_log.columns:
        selection_log["adjustment"] = "coarsen"
    winner = selection_log.loc[selection_log["selected"]].iloc[0]
    return {
        "selected_window": selection["optimal_window"],
        "selection_log": selection_log,
        "average_local_fano": float(winner["average_local_fano"]),
        "fano_deviation": float(winner["fano_deviation"]),
    }


def select_count_binning_window(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int | None = None,
    target_pass_rate: float | None = None,
    home_base: str = DEFAULT_HOME_BASE,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
) -> dict[str, Any]:
    return find_optimal_time_bucket(
        df,
        timestamp_col=timestamp_col,
        windows=windows,
        count_col=count_col,
        min_events=min_events,
        target_pass_rate=target_pass_rate,
        home_base=home_base,
        tau_block=tau_block,
    )


def bin_time_series_for_window(
    df: pd.DataFrame | pd.Series,
    window: str,
    timestamp_col: str = "timestamp",
    *,
    count_col: str | None = None,
) -> pd.DataFrame:
    """Bin a timestamp series into fixed-width arrival counts for a single provided window."""
    if isinstance(df, pd.Series):
        timestamps = pd.to_datetime(df, errors="coerce").dropna().sort_values()
    else:
        if timestamp_col not in df.columns:
            raise KeyError(f"Column {timestamp_col!r} was not found in the input DataFrame")
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()

    if timestamps.empty:
        return _empty_arrival_frame()

    counts = _count_buckets_for_window(timestamps, window)
    if counts.empty:
        return _empty_arrival_frame()

    data = counts.rename_axis("bin_start").reset_index()
    data.columns = ["bin_start", "count"]
    data["bin_end"] = data["bin_start"] + _window_to_timedelta(window)
    data["window"] = window
    data["bin_id"] = range(1, len(data) + 1)
    data["intensity"] = (data["count"] / _window_unit_value(window)).round(3)
    return data[["bin_id", "bin_start", "bin_end", "window", "count", "intensity"]].reset_index(drop=True)


def build_arrival_intensity_series(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int | None = None,
    target_pass_rate: float | None = None,
    home_base: str = DEFAULT_HOME_BASE,
    verbose: bool = True,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
) -> pd.DataFrame:
    """Construct a clean arrival-intensity table for the chosen enterprise window."""
    if isinstance(df, pd.Series):
        timestamps = pd.to_datetime(df, errors="coerce").dropna().sort_values()
    else:
        if timestamp_col not in df.columns:
            raise KeyError(f"Column {timestamp_col!r} was not found in the input DataFrame")
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()

    if timestamps.empty:
        return _empty_arrival_frame()

    try:
        selection = optimize_arrival_binning(
            df,
            timestamp_col=timestamp_col,
            windows=windows,
            count_col=count_col,
            tau_block=tau_block,
            min_events=min_events,
            target_pass_rate=target_pass_rate,
            home_base=home_base,
            verbose=verbose,
        )
    except ValueError:
        return _empty_arrival_frame()

    window = selection["optimal_window"]
    return bin_time_series_for_window(df, window, timestamp_col=timestamp_col)


def select_largest_window_in_fano_acceptance_interval(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int | None = None,
    target_pass_rate: float | None = None,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Select the largest admissible window whose Fano value falls within the reference acceptance region.

    This method supports two modes:
    1. table mode: receives the Fano/window table produced by acceptable_over_dispersed_window_size()
    2. raw-data mode: recomputes the table from timestamps for backward compatibility

    In both cases, the reference window is the admissible window whose average local Fano factor is
    closest to 1. The chosen window is the largest admissible window whose Fano value remains within
    the reference window's acceptance region. If none do, the reference window is retained.
    """
    table_mode = isinstance(df, pd.DataFrame) and {"window", "average_local_fano"}.issubset(df.columns)

    if table_mode:
        table = df.copy()
        if "window_minutes" not in table.columns:
            table["window_minutes"] = table["window"].map(lambda value: _window_to_timedelta(str(value)).total_seconds() / 60.0)
        if "fano_deviation" not in table.columns:
            table["fano_deviation"] = (table["average_local_fano"] - 1.0).abs()

        if table.empty:
            raise ValueError("No windows were available in the Fano table for selection.")

        reference_row = table[table["average_local_fano"].notna()].sort_values("fano_deviation", kind="mergesort").iloc[0]
        reference_window = str(reference_row["window"])
        reference_fano = float(reference_row["average_local_fano"])
        lower = float(reference_row["fano_lower"]) if "fano_lower" in reference_row and pd.notna(reference_row["fano_lower"]) else reference_fano
        upper = float(reference_row["fano_upper"]) if "fano_upper" in reference_row and pd.notna(reference_row["fano_upper"]) else reference_fano

        compatible = table[table["average_local_fano"].notna()].copy()
        compatible = compatible[
            (compatible["average_local_fano"] >= lower) & (compatible["average_local_fano"] <= upper)
        ].copy()

        if compatible.empty:
            chosen_row = reference_row
            selection_status = "reference_window_retained"
        else:
            chosen_row = compatible.sort_values("window_minutes", ascending=False, kind="mergesort").iloc[0]
            selection_status = "largest_window_in_reference_acceptance_region"

        chosen_window = str(chosen_row["window"])
        return {
            "reference_window": reference_window,
            "reference_fano": reference_fano,
            "selected_window": chosen_window,
            "average_local_fano": float(chosen_row["average_local_fano"]),
            "fano_lower": lower,
            "fano_upper": upper,
            "window_metrics": table,
            "selected_row": {
                "window": chosen_window,
                "average_local_fano": float(chosen_row["average_local_fano"]),
                "fano_deviation": abs(float(chosen_row["average_local_fano"]) - 1.0),
            },
            "selection_status": selection_status,
            "confidence": confidence,
        }

    candidate_windows = _normalize_windows(windows)
    rows: list[dict[str, Any]] = []

    if isinstance(df, pd.Series):
        timestamps = pd.to_datetime(df, errors="coerce").dropna().sort_values()
    else:
        if timestamp_col not in df.columns:
            raise KeyError(f"Column {timestamp_col!r} was not found in the input DataFrame")
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()

    if timestamps.empty:
        raise ValueError("No valid timestamps were available for Fano-window selection.")

    all_rows: list[dict[str, Any]] = []

    for window in candidate_windows:
        counts = _count_buckets_for_window(timestamps, window)
        if counts.empty:
            continue
        fanos = _local_fano_factors(counts, window, tau_block)
        avg_fano = float(sum(fanos) / len(fanos)) if fanos else 1.0
        row = {
            "window": window,
            "average_local_fano": avg_fano,
            "fano_deviation": abs(avg_fano - 1.0),
        }
        all_rows.append(row)

    if not all_rows:
        raise ValueError("No time windows were available for Fano-based selection.")

    base_row = min(all_rows, key=lambda row: row["fano_deviation"])
    base_window = base_row["window"]
    base_counts = _count_buckets_for_window(timestamps, base_window)
    base_interval = fano_acceptance_interval(base_counts, window=base_window, tau_block=tau_block, confidence=confidence)
    lower = base_interval["fano_lower"]
    upper = base_interval["fano_upper"]

    compatible = []
    for row in all_rows:
        fano = float(row["average_local_fano"])
        if lower <= fano <= upper:
            compatible.append(row)

    if compatible:
        chosen = max(compatible, key=lambda row: _window_to_timedelta(row["window"]).total_seconds())
        selection_status = "largest_window_in_reference_acceptance_region"
    else:
        chosen = base_row
        selection_status = "reference_window_retained"

    return {
        "reference_window": base_window,
        "reference_fano": base_row["average_local_fano"],
        "selected_window": chosen["window"],
        "average_local_fano": chosen["average_local_fano"],
        "fano_lower": lower,
        "fano_upper": upper,
        "window_metrics": pd.DataFrame(all_rows),
        "selected_row": {
            "window": chosen["window"],
            "average_local_fano": chosen["average_local_fano"],
            "fano_deviation": abs(chosen["average_local_fano"] - 1.0),
        },
        "selection_status": selection_status,
        "confidence": confidence,
    }


def summarize_integer_alignment(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int = DEFAULT_MIN_EVENTS,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
    home_base: str = DEFAULT_HOME_BASE,
    verbose: bool = True,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
) -> dict[str, Any]:
    """Return both the raw and aligned summary outputs for the selected enterprise window."""
    if isinstance(df, pd.Series):
        frame = pd.DataFrame({timestamp_col: df})
    else:
        frame = df.copy()

    if timestamp_col not in frame.columns:
        raise KeyError(f"Column {timestamp_col!r} was not found in the input data")

    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce").dropna()
    if timestamps.empty:
        return {
            "algorithm_window": None,
            "algorithm_result": None,
            "alignment_window": None,
            "alignment_result": None,
            "alignment_applied": False,
            "original_series": _empty_arrival_frame(),
            "aligned_series": _empty_arrival_frame(),
            "status": "no_data",
            "message": "No valid timestamps were available for the time-binning workflow.",
        }

    try:
        selection = optimize_arrival_binning(
            frame,
            timestamp_col=timestamp_col,
            windows=windows,
            count_col=count_col,
            min_events=min_events,
            target_pass_rate=target_pass_rate,
            home_base=home_base,
            tau_block=tau_block,
            verbose=verbose,
        )
    except ValueError:
        return {
            "algorithm_window": None,
            "algorithm_result": None,
            "alignment_window": None,
            "alignment_result": None,
            "alignment_applied": False,
            "original_series": _empty_arrival_frame(),
            "aligned_series": _empty_arrival_frame(),
            "status": "sparse_data",
            "message": "No non-empty enterprise windows were available for Fano-based selection.",
        }

    window = selection["optimal_window"]
    original_series = build_arrival_intensity_series(
        frame,
        timestamp_col=timestamp_col,
        windows=[window],
        min_events=min_events,
        target_pass_rate=target_pass_rate,
        home_base=home_base,
        verbose=False,
        tau_block=tau_block,
    )
    aligned_series = original_series.copy()
    if not aligned_series.empty:
        aligned_series["bin_start"] = align_to_integer_bin_grid(aligned_series["bin_start"], window)
        aligned_series["bin_end"] = aligned_series["bin_start"] + _window_to_timedelta(window)

    alignment_applied = not original_series.equals(aligned_series)
    status = "alignment_applied" if alignment_applied else "no_alignment_needed"
    message = (
        f"Algorithm selected {window}; window unit={_window_unit_label(window)}; intensity unit=count per {_window_unit_label(window)}; integer alignment {'applied' if alignment_applied else 'did not change the grid'}."
    )

    result = {
        "algorithm_window": window,
        "window_unit": _window_unit_label(window),
        "intensity_unit": f"count per {_window_unit_label(window)}",
        "algorithm_result": selection,
        "alignment_window": window,
        "alignment_result": {"selected_window": window, "alignment_applied": alignment_applied},
        "alignment_applied": alignment_applied,
        "original_series": original_series,
        "aligned_series": aligned_series,
        "status": status,
        "message": message,
    }
    if verbose:
        result["selection_log"] = selection["window_metrics"]
    return result


def plot_window_fano_factors(
    fano_df: pd.DataFrame,
    *,
    x_col: str = "window",
    y_col: str = "average_local_fano",
    ax=None,
    title: str = "Window size vs Fano factor",
):
    """Plot the average local Fano factor by candidate window size."""
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required to plot the Fano-factor comparison.") from exc

    if ax is None:
        _, ax = plt.subplots()

    ordered = fano_df.copy()
    if x_col in ordered.columns and y_col in ordered.columns:
        ordered = ordered.sort_values(x_col, key=lambda values: values.map(_window_to_timedelta)).reset_index(drop=True)
        ax.plot(ordered[x_col], ordered[y_col], marker="o", label="average_local_fano")
        ax.axhline(1.0, linestyle="--", linewidth=1, color="gray")
        ax.set_title(title)
        ax.set_xlabel("Window size")
        ax.set_ylabel("Average local Fano factor")
        ax.grid(alpha=0.3)

    return ax


def tseda_dataset_export(
    dataset_name: str | Path | None = None,
    *,
    dataset_dir: str | Path | None = None,
    output_name: str | None = None,
    source_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    """Export only the bin_end and intensity fields for downstream TS-EDA consumers."""
    if source_path is None and dataset_name is None:
        raise ValueError("A dataset name or a source path must be provided")

    if source_path is not None:
        input_path = Path(source_path)
    elif isinstance(dataset_name, Path):
        input_path = dataset_name if dataset_name.suffix.lower() == ".csv" else None
    elif str(dataset_name).lower().endswith(".csv"):
        input_path = Path(dataset_name)
    else:
        input_path = None

    if input_path is None:
        base_dir = Path(dataset_dir) if dataset_dir is not None else Path.cwd()
        candidates = [
            base_dir / "arrival_intensity.csv",
            base_dir / f"{dataset_name}_arrival_intensity.csv",
            base_dir / f"{dataset_name}.csv",
        ]
        for candidate in candidates:
            if candidate.exists():
                input_path = candidate
                break
        if input_path is None:
            csv_candidates = sorted(base_dir.glob("*.csv"))
            if not csv_candidates:
                raise FileNotFoundError(f"No CSV source file was found in {base_dir}")
            input_path = csv_candidates[0]

    input_path = Path(input_path)
    frame = pd.read_csv(input_path)
    missing = sorted({"bin_end", "intensity"}.difference(frame.columns))
    if missing:
        raise KeyError(f"Dataset {input_path} is missing required columns: {missing}")

    export_frame = frame[["bin_end", "intensity"]].copy()
    if output_path is not None:
        destination = Path(output_path)
    elif output_name is not None:
        destination = input_path.with_name(output_name)
    else:
        destination = input_path.with_name(f"{input_path.stem}_tseda.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_frame.to_csv(destination, index=False)
    return destination
