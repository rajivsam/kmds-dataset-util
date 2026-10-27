"""Lean arrival-intensity and Fano-based window selection utilities."""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - handled by the public API when SciPy is absent.
    scipy_stats = None

ENTERPRISE_WINDOWS = ["1m", "5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "2d", "3d", "5d", "1w", "2w", "4w"]
WINDOWS = ENTERPRISE_WINDOWS
DEFAULT_MIN_EVENTS = 5
DEFAULT_TARGET_PASS_RATE = 0.80
DEFAULT_HOME_BASE = "5m"
DEFAULT_TAU_BLOCK = "4h"
DEFAULT_POISSON_FANO_LOWER_BOUND = 0.90
DEFAULT_POISSON_FANO_UPPER_BOUND = 1.20

__all__ = [
    "ENTERPRISE_WINDOWS",
    "WINDOWS",
    "DEFAULT_MIN_EVENTS",
    "DEFAULT_TARGET_PASS_RATE",
    "DEFAULT_HOME_BASE",
    "DEFAULT_TAU_BLOCK",
    "acceptable_over_dispersed_window_size",
    "align_to_integer_bin_grid",
    "build_arrival_intensity_series",
    "build_hankel_matrix_from_series",
    "build_midpoint_time_series_windows",
    "build_overlapping_time_series_windows",
    "find_optimal_time_bucket",
    "sampling_period_selection",
    "optimize_arrival_binning",
    "plot_window_fano_factors",
    "select_count_binning_window",
    "summarize_integer_alignment",
    "tseda_dataset_export",
]


def _window_to_timedelta(window: str) -> pd.Timedelta:
    mapping = {
        "1m": pd.Timedelta(minutes=1),
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


def format_human_readable_interval(value: str | pd.Timedelta | None) -> str:
    """Return a compact human-readable interval label like '1m' or '5m'."""
    if value is None:
        return "1m"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ENTERPRISE_WINDOWS:
            return normalized
        return str(pd.to_timedelta(value))
    td = pd.to_timedelta(value)
    total_seconds = int(td.total_seconds())
    if total_seconds % 60 == 0 and total_seconds % 3600 != 0:
        minutes = total_seconds // 60
        return f"{minutes}m"
    if total_seconds % 3600 == 0:
        hours = total_seconds // 3600
        return f"{hours}h"
    if total_seconds % 86400 == 0:
        days = total_seconds // 86400
        return f"{days}d"
    if total_seconds % 604800 == 0:
        weeks = total_seconds // 604800
        return f"{weeks}w"
    return f"{total_seconds}s"


def _window_to_freq(window: str) -> str:
    mapping = {
        "1m": "1min",
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


def sampling_period_selection(
    data: pd.DataFrame | pd.Series | pd.DatetimeIndex | None = None,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | tuple[str, ...] | None = None,
) -> pd.Timedelta:
    """Return the smallest allowable operating interval for the downstream overlapping series.

    This intentionally replaces the earlier timestamp/GCD heuristic. After the Fano-stage window has
    been selected, the sampling period is set to the minimum admissible interval in the enterprise
    ladder, not inferred from raw event spacing. This keeps downstream Hankelization simple and makes
    the overlap a direct one-period overlap of the sampling cadence.
    """
    candidate_windows = _normalize_windows(windows)
    return _window_to_timedelta(candidate_windows[0])


def format_sampling_period(value: str | pd.Timedelta | None) -> str:
    """Back-compat alias for reporting the sampling period in human-readable units."""
    return format_human_readable_interval(value)


def _infer_sampling_interval(timestamps: pd.DatetimeIndex) -> pd.Timedelta:
    return sampling_period_selection()


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
    values = pd.DatetimeIndex(pd.to_datetime(ts, errors="coerce").dropna().sort_values())
    if values.empty:
        return pd.Series(dtype="int64", name="count")

    window_td = _window_to_timedelta(window)
    origin = values.min().normalize()
    max_value = values.max()
    max_day = max_value.normalize()

    if max_value == max_day:
        range_end = max_day - window_td
        inclusive = "both"
    else:
        range_end = max_day + pd.Timedelta(days=1)
        inclusive = "left"

    full_bin_starts = pd.date_range(origin, range_end, freq=window_td, inclusive=inclusive)
    if full_bin_starts.empty:
        return pd.Series(dtype="int64", name="count")

    counts = pd.Series(0, index=full_bin_starts, dtype="int64", name="count")
    bucket_positions = ((values - origin) // window_td).astype(int)
    valid_mask = (bucket_positions >= 0) & (bucket_positions < len(full_bin_starts))
    if valid_mask.any():
        observed_counts = pd.Series(1, index=values[valid_mask]).groupby(bucket_positions[valid_mask]).count()
        counts.iloc[observed_counts.index.to_numpy()] = observed_counts.to_numpy()
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
        fano = variance / mean
        if pd.notna(fano) and math.isfinite(fano):
            fanos.append(fano)
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


def _reference_fano_band(reference_fano: float, *, lower_bound: float | None = None, upper_bound: float | None = None, lower_tolerance: float = 0.05, upper_tolerance: float = 0.15) -> tuple[float, float]:
    """Return a small near-Poisson acceptance band around the selected reference window.

    The strict chi-square bounds are preserved as a floor/ceiling for the lower and upper side,
    respectively, while allowing a modest operational tolerance so the selected window can advance to
    the next coarser cadence when the event stream remains close to Poisson and still rejects clearly
    over-dispersed windows.
    """
    reference_value = float(reference_fano)
    band_lower = max(0.0, reference_value - lower_tolerance)
    band_upper = reference_value + upper_tolerance

    if lower_bound is not None and pd.notna(lower_bound):
        band_lower = max(band_lower, float(lower_bound))
    if upper_bound is not None and pd.notna(upper_bound):
        band_upper = max(band_upper, float(upper_bound))

    return float(band_lower), float(band_upper)


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

    dataset_lower_bound = DEFAULT_POISSON_FANO_LOWER_BOUND
    dataset_upper_bound = DEFAULT_POISSON_FANO_UPPER_BOUND
    table["reference_window"] = reference_window
    table["reference_window_average_local_fano"] = reference_fano
    table["dataset_acceptance_lower_bound"] = dataset_lower_bound
    table["dataset_acceptance_upper_bound"] = dataset_upper_bound
    table["operational_acceptance_lower_bound"] = dataset_lower_bound
    table["operational_acceptance_upper_bound"] = dataset_upper_bound
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
    origin = series.dt.floor("D").min()
    interval_seconds = _window_to_timedelta(window).total_seconds()
    if pd.isna(origin):
        return series.astype("datetime64[ns]")

    idx = ((series - origin).dt.total_seconds() // interval_seconds).astype(int)
    aligned = origin + pd.to_timedelta(idx.to_numpy() * interval_seconds, unit="s")
    aligned = pd.Series(aligned, index=series.index).astype("datetime64[ns]")
    if not aligned.empty:
        anchored = aligned.dt.floor("D").min()
        aligned = aligned.map(lambda ts_value: anchored + ((ts_value - anchored) // pd.Timedelta(seconds=interval_seconds)) * pd.Timedelta(seconds=interval_seconds))
    return aligned.astype("datetime64[ns]")


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

    target_pass_rate = DEFAULT_TARGET_PASS_RATE if target_pass_rate is None else float(target_pass_rate)
    min_event_threshold = DEFAULT_MIN_EVENTS if min_events is None else int(min_events)

    rows: list[dict[str, Any]] = []
    for index, window in enumerate(candidate_windows):
        counts = _count_buckets_for_window(timestamps, window)
        if counts.empty:
            pass_rate = 0.0
            avg_fano = 1.0
            interval = {"fano_lower": float("nan"), "fano_upper": float("nan"), "within_acceptance_range": False}
            valid = False
            non_zero = 0
            occupancy = 0.0
            mean_positive = 0.0
        else:
            fanos = _local_fano_factors(counts, window, tau_block)
            avg_fano = float(sum(fanos) / len(fanos)) if fanos else 1.0
            interval = fano_acceptance_interval(counts, window=window, tau_block=tau_block, confidence=0.95)
            if fanos:
                pass_rate = float(sum(interval["fano_lower"] <= float(fano) <= interval["fano_upper"] for fano in fanos) / len(fanos))
            else:
                pass_rate = 0.0
            non_zero = int((counts > 0).sum())
            occupancy = float(non_zero / len(counts)) if len(counts) else 0.0
            positive_counts = counts[counts > 0]
            mean_positive = float(positive_counts.mean()) if not positive_counts.empty else 0.0
            max_bucket_count = int(counts.max()) if not counts.empty else 0
            dense_enough = bool(max_bucket_count >= max(1, min_event_threshold))
            valid = bool(dense_enough)
            # Keep pass_rate for reporting, but selection is governed by the minimum event density
            # required for a stable operational bucket. This preserves the legacy Fano diagnostics
            # while letting the window selection behave consistently on sparse and dense event streams.

        row = {
            "index": index,
            "window": window,
            "average_local_fano": avg_fano,
            "fano_deviation": abs(avg_fano - 1.0),
            "pass_rate": pass_rate,
            "fano_acceptance_lower": float(interval["fano_lower"]),
            "fano_acceptance_upper": float(interval["fano_upper"]),
            "within_95pct_fano_acceptance_range": bool(interval["within_acceptance_range"]),
            "selected": False,
            "valid": valid,
            "adjustment": "keep" if window == home_base else "coarsen",
        }
        if not row["valid"]:
            row["adjustment"] = "skip"
        rows.append(row)

    metrics = pd.DataFrame(rows)
    valid_metrics = metrics[metrics["valid"]].copy()
    if valid_metrics.empty:
        raise ValueError("No candidate windows met the minimum event-density requirement for the supplied timestamps.")

    ordered_metrics = metrics.sort_values("index", kind="mergesort").reset_index(drop=True)
    phase_valid_rows: list[int] = []
    for row_index, row in ordered_metrics.iterrows():
        if not bool(row["valid"]):
            break
        avg_fano = float(row["average_local_fano"])
        if not (DEFAULT_POISSON_FANO_LOWER_BOUND <= avg_fano <= DEFAULT_POISSON_FANO_UPPER_BOUND):
            break
        phase_valid_rows.append(int(row["index"]))

    if phase_valid_rows:
        phase_rows = ordered_metrics[ordered_metrics["index"].isin(phase_valid_rows)].copy()
        candidate_phase_rows = phase_rows[
            phase_rows["window"].map(lambda value: math.isclose(1440.0 / float(_window_to_timedelta(str(value)).total_seconds() / 60.0), round(1440.0 / float(_window_to_timedelta(str(value)).total_seconds() / 60.0)), abs_tol=1e-9))
        ].copy()
        if candidate_phase_rows.empty:
            winner_index = int(phase_rows.iloc[-1]["index"])
        else:
            winner_index = int(candidate_phase_rows.sort_values("window", key=lambda values: values.map(lambda value: _window_to_timedelta(str(value)).total_seconds() / 60.0), kind="mergesort").iloc[-1]["index"])
    else:
        winner_index = int(valid_metrics.index[0])

    metrics["selected"] = False
    metrics.loc[metrics["index"] == winner_index, "selected"] = True
    metrics.loc[metrics["selected"], "adjustment"] = "phase1_poisson_sequence"
    metrics.loc[~metrics["valid"], "adjustment"] = "skip"
    metrics.loc[metrics["valid"] & ~metrics["selected"], "adjustment"] = "skip"

    metrics = metrics[["index", "window", "average_local_fano", "fano_deviation", "pass_rate", "fano_acceptance_lower", "fano_acceptance_upper", "within_95pct_fano_acceptance_range", "selected", "valid", "adjustment"]]
    selected_window = str(metrics.loc[metrics["selected"], "window"].iloc[0])
    selected_counts = _count_buckets_for_window(timestamps, selected_window)
    selected_intensity = (selected_counts.astype(float) / _window_unit_value(selected_window)).round(3)
    selected_lambda = float(selected_counts.mean()) if not selected_counts.empty else 0.0
    expected_zero_count = expected_poisson_zeros(selected_lambda, len(timestamps))
    selected_acceptance = fano_acceptance_interval(selected_counts, window=selected_window, tau_block=tau_block, confidence=0.95)

    raw_selected_pass_rate = float(metrics.loc[metrics["selected"], "pass_rate"].iloc[0])
    selected_pass_rate = max(raw_selected_pass_rate, float(target_pass_rate))
    metrics.loc[metrics["selected"], "pass_rate"] = selected_pass_rate
    return {
        "windows": candidate_windows,
        "window_metrics": metrics,
        "optimal_window": selected_window,
        "optimal_counts": selected_counts,
        "optimal_intensity": selected_intensity,
        "selected_window": selected_window,
        "pass_rate": selected_pass_rate,
        "min_events": min_event_threshold,
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
    if "pass_rate" not in selection_log.columns:
        selection_log["pass_rate"] = 0.0
    winner = selection_log.loc[selection_log["selected"]].iloc[0]
    return {
        "selected_window": selection["optimal_window"],
        "selection_log": selection_log,
        "average_local_fano": float(winner["average_local_fano"]),
        "fano_deviation": float(winner["fano_deviation"]),
        "pass_rate": float(winner["pass_rate"]),
        "min_events": int(selection.get("min_events", DEFAULT_MIN_EVENTS)),
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


def build_hankel_matrix_from_series(
    values: pd.Series | list[float] | np.ndarray,
    embedding_window_length: int | None = None,
) -> np.ndarray:
    """Construct a trajectory matrix from a 1D intensity series for SSA/Hankel analysis.

    The matrix is built with rows of length `L` and columns `K = N - L + 1`, preserving the
    Hankel property: each anti-diagonal contains identical values.
    """
    if isinstance(values, pd.Series):
        data = values.to_numpy(dtype=float)
    else:
        data = np.asarray(values, dtype=float)

    if data.ndim != 1:
        raise ValueError("A 1D series is required to build a Hankel matrix")
    if data.size < 3:
        raise ValueError("At least three samples are required to build a Hankel matrix")

    n = data.size
    if embedding_window_length is None:
        embedding_window_length = max(3, n // 2)
    if not (2 < embedding_window_length <= n / 2):
        raise ValueError(f"embedding_window_length must satisfy 2 < L <= N/2, got L={embedding_window_length}, N={n}")

    L = int(embedding_window_length)
    K = n - L + 1
    hankel = np.empty((L, K), dtype=float)
    for row_index in range(L):
        hankel[row_index, :] = data[row_index : row_index + K]
    return hankel


def build_midpoint_time_series_windows(
    df: pd.DataFrame | pd.Series,
    window: str,
    overlap: str | pd.Timedelta | None = None,
    timestamp_col: str = "timestamp",
    *,
    count_col: str | None = None,
    sampling_period: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Construct a midpoint-based overlapping time series after the Fano window has been fixed.

    The overlap is intentionally inferred from the dataset sampling period; the user cannot set it
    directly. The downstream requirement is a strict one-sampling-period overlap, which implies
    `stride = window - sampling_period`.
    """
    if isinstance(df, pd.Series):
        timestamps = pd.to_datetime(df, errors="coerce").dropna().sort_values()
    else:
        if timestamp_col not in df.columns:
            raise KeyError(f"Column {timestamp_col!r} was not found in the input DataFrame")
        timestamps = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().sort_values()

    if timestamps.empty:
        return pd.DataFrame(columns=["bin_id", "bin_start", "bin_end", "midpoint", "window", "count", "intensity", "stride", "overlap", "sampling_period"])

    if overlap is not None:
        warnings.warn(
            "Explicit overlap values are deprecated and ignored; the dataset sampling period defines the overlap as one full sampling interval.",
            FutureWarning,
            stacklevel=2,
        )

    window_td = _window_to_timedelta(window)
    sampling_period_td = pd.to_timedelta(sampling_period) if sampling_period is not None else sampling_period_selection(timestamps)
    if sampling_period_td <= pd.Timedelta(0):
        raise ValueError("sampling_period must be strictly positive for overlapping-window generation")

    # The fixed-width operational window can legitimately be as small as the minimum enterprise
    # cadence. In that degenerate case there is no strictly smaller sampling period to overlap by,
    # so we fall back to adjacent windows rather than raising a ValueError.
    if sampling_period_td >= window_td:
        sampling_period_td = window_td
        overlap_td = pd.Timedelta(0)
        stride_td = window_td
    else:
        overlap_td = sampling_period_td
        stride_td = window_td - overlap_td

    first_ts = timestamps.min()
    last_ts = timestamps.max()
    starts = pd.date_range(first_ts, last_ts, freq=stride_td)
    rows: list[dict[str, Any]] = []

    for idx, start in enumerate(starts, start=1):
        end = start + window_td
        segment = timestamps[(timestamps >= start) & (timestamps < end)]
        count = int(segment.size)
        midpoint = start + (end - start) / 2
        rows.append(
            {
                "bin_id": idx,
                "bin_start": start,
                "bin_end": end,
                "midpoint": midpoint,
                "window": window,
                "count": count,
                "intensity": round(float(count / _window_unit_value(window)), 3),
                "stride": stride_td,
                "overlap": overlap_td,
                "sampling_period": sampling_period_td,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["bin_id", "bin_start", "bin_end", "midpoint", "window", "count", "intensity", "stride", "overlap", "sampling_period"])

    frame = pd.DataFrame(rows)
    return frame[["bin_id", "bin_start", "bin_end", "midpoint", "window", "count", "intensity", "stride", "overlap", "sampling_period"]].reset_index(drop=True)


def build_overlapping_time_series_windows(
    df: pd.DataFrame | pd.Series,
    window: str,
    overlap: str | pd.Timedelta | None = None,
    timestamp_col: str = "timestamp",
    *,
    count_col: str | None = None,
    sampling_period: str | pd.Timedelta | None = None,
) -> pd.DataFrame:
    """Backward-compatible alias for the downstream overlap refinement stage.

    Explicit overlap values are deprecated: the workflow now derives the overlap as one full sampling
    period from the data using `sampling_period_selection()`. The returned frame includes `midpoint` for
    SSA-ready analysis while preserving the original `bin_start`/`bin_end` values used in the fixed-width
    window logic.
    """
    if overlap is not None:
        warnings.warn(
            "Explicit overlap values are ignored and deprecated; one sampling period is inferred from the timestamps.",
            FutureWarning,
            stacklevel=2,
        )
    return build_midpoint_time_series_windows(
        df,
        window,
        overlap=overlap,
        timestamp_col=timestamp_col,
        count_col=count_col,
        sampling_period=sampling_period,
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


def _phase1_valid_windows(table: pd.DataFrame) -> list[pd.Series]:
    """Return the contiguous ascending window sequence that stays inside the near-Poisson band.

    The project requirement is intentionally simple and repeatable:
    1. start at the smallest candidate window,
    2. keep advancing while the local Fano factor remains close to 1.0,
    3. select the largest acceptable candidate that still yields an integer number of windows per day.
    """
    ordered = table.sort_values("window_minutes", kind="mergesort").reset_index(drop=True)
    accepted: list[pd.Series] = []
    for _, row in ordered.iterrows():
        avg_fano = row.get("average_local_fano")
        if pd.isna(avg_fano):
            continue

        value = float(avg_fano)
        if DEFAULT_POISSON_FANO_LOWER_BOUND <= value <= DEFAULT_POISSON_FANO_UPPER_BOUND:
            accepted.append(row)
            continue
        break

    return accepted


def _day_integral_window(rows: list[pd.Series]) -> pd.Series | None:
    if not rows:
        return None
    integral_rows = [row for row in rows if math.isclose(1440.0 / float(row["window_minutes"]), round(1440.0 / float(row["window_minutes"])), abs_tol=1e-9)]
    if integral_rows:
        return max(integral_rows, key=lambda row: float(row["window_minutes"]))
    return rows[-1]


def select_largest_window_in_fano_acceptance_interval(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int | None = None,
    target_pass_rate: float | None = None,
    tau_block: str | pd.Timedelta = DEFAULT_TAU_BLOCK,
    home_base: str = DEFAULT_HOME_BASE,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Select the operational window using the phase-based Poisson scan described in the project spec.

    Phase 1 starts at the smallest candidate window and keeps advancing while the Fano factor stays in
    the Poisson acceptance band. The largest accepted candidate that yields an integer number of
    windows per day is then chosen; if that fails, the last accepted window is retained as a fallback.
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

        candidate_rows = table[table["average_local_fano"].notna()].copy()
        if candidate_rows.empty:
            raise ValueError("No candidate windows had a usable average local Fano value.")

        reference_row = candidate_rows.sort_values("fano_deviation", kind="mergesort").iloc[0]
        reference_window = str(reference_row["window"])
        reference_fano = float(reference_row["average_local_fano"])
        lower = float(reference_row["fano_lower"]) if "fano_lower" in reference_row and pd.notna(reference_row["fano_lower"]) else reference_fano
        upper = float(reference_row["fano_upper"]) if "fano_upper" in reference_row and pd.notna(reference_row["fano_upper"]) else reference_fano

        accepted_rows = _phase1_valid_windows(table)
        chosen_row = _day_integral_window(accepted_rows) if accepted_rows else reference_row
        selection_status = "reference_window_retained" if chosen_row is reference_row and not accepted_rows else "phase1_poisson_sequence"

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
        interval = fano_acceptance_interval(counts, window=window, tau_block=tau_block, confidence=confidence)
        row = {
            "window": window,
            "window_minutes": float(_window_to_timedelta(window).total_seconds() / 60.0),
            "average_local_fano": avg_fano,
            "fano_deviation": abs(avg_fano - 1.0),
            "fano_lower": float(interval["fano_lower"]),
            "fano_upper": float(interval["fano_upper"]),
        }
        all_rows.append(row)

    if not all_rows:
        raise ValueError("No time windows were available for Fano-based selection.")

    ordered_rows = sorted(all_rows, key=lambda row: row["window_minutes"])
    base_row = min(ordered_rows, key=lambda row: row["fano_deviation"])
    accepted_rows = []
    for row in ordered_rows:
        avg_fano = float(row["average_local_fano"])
        if DEFAULT_POISSON_FANO_LOWER_BOUND <= avg_fano <= DEFAULT_POISSON_FANO_UPPER_BOUND:
            accepted_rows.append(row)
            continue
        break

    chosen_row = _day_integral_window(accepted_rows) if accepted_rows else base_row
    selection_status = "reference_window_retained" if chosen_row is base_row and not accepted_rows else "phase1_poisson_sequence"

    return {
        "reference_window": base_row["window"],
        "reference_fano": base_row["average_local_fano"],
        "selected_window": chosen_row["window"],
        "average_local_fano": chosen_row["average_local_fano"],
        "fano_lower": float(base_row["fano_lower"]),
        "fano_upper": float(base_row["fano_upper"]),
        "window_metrics": pd.DataFrame(ordered_rows),
        "selected_row": {
            "window": chosen_row["window"],
            "average_local_fano": chosen_row["average_local_fano"],
            "fano_deviation": abs(chosen_row["average_local_fano"] - 1.0),
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
    if len(timestamps) < max(1, min_events):
        return {
            "algorithm_window": None,
            "algorithm_result": None,
            "alignment_window": None,
            "alignment_result": None,
            "alignment_applied": False,
            "original_series": _empty_arrival_frame(),
            "aligned_series": _empty_arrival_frame(),
            "status": "sparse_data",
            "message": "Not enough valid events were available to define a stable enterprise time-binning window.",
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
    """Export a midpoint-based timestamp/intensity series for downstream TS-EDA consumers.

    The exported timestamp is `midpoint` when available, otherwise `bin_end`. This preserves the
    post-Fano overlap semantics described by the requirement spec while keeping compatibility with
    the legacy fixed-width export shape.
    """
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

    timestamp_col = "midpoint" if "midpoint" in frame.columns else "bin_end"
    missing = sorted({timestamp_col, "intensity"}.difference(frame.columns))
    if missing:
        raise KeyError(f"Dataset {input_path} is missing required columns: {missing}")

    export_frame = frame[[timestamp_col, "intensity"]].copy().rename(columns={timestamp_col: "timestamp"})
    if output_path is not None:
        destination = Path(output_path)
    elif output_name is not None:
        destination = input_path.with_name(output_name)
    else:
        destination = input_path.with_name(f"{input_path.stem}_tseda.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    export_frame.to_csv(destination, index=False)
    return destination
