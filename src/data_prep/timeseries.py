"""Enterprise time-window selection for count-based time series data."""

from __future__ import annotations

from typing import Any

import pandas as pd

ENTERPRISE_WINDOWS = ["5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "2d", "3d", "5d", "1w", "2w", "4w"]
WINDOWS = ENTERPRISE_WINDOWS
DEFAULT_MIN_EVENTS = 5
DEFAULT_TARGET_PASS_RATE = 0.80
DEFAULT_HOME_BASE = "1d"


def _window_to_pandas_freq(window: str) -> str:
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


def align_to_integer_bin_grid(ts: pd.Series, window: str) -> pd.Series:
    """Post-search phase-alignment step for integer-anchored, enterprise-size bins.

    This second-stage adjustment preserves the selected window but snaps each timestamp to an
    integer boundary relative to the start of the operational day (or the whole cycle for larger
    windows), so the final grid has a clean integer number of bins per day.
    """
    if ts.empty:
        return ts

    normalized = pd.to_datetime(ts, errors="coerce").dropna()
    if normalized.empty:
        return pd.Series(dtype="datetime64[ns]")

    window_td = _window_to_timedelta(window)
    if window_td < pd.Timedelta(days=1):
        day_starts = normalized.dt.normalize()
        return day_starts + ((normalized - day_starts) // window_td) * window_td

    cycle_start = normalized.min().normalize()
    day_offsets = (normalized.dt.normalize() - cycle_start).dt.days
    cycle_index = (day_offsets // max(1, int(window_td.days)))
    return cycle_start + cycle_index * window_td


def evaluate_count_density(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    window: str,
    count_col: str | None = None,
    min_events: int = DEFAULT_MIN_EVENTS,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
) -> dict[str, Any]:
    """Return pass/fail metrics for count-based bucket density checking.

    The check evaluates the fraction of continuous buckets for a given time window whose
    event count is at least `min_events`. Buckets are generated continuously between the
    minimum and maximum valid timestamps so that empty buckets are counted as failures.
    """
    if min_events <= 0:
        raise ValueError("min_events must be greater than zero")
    if not 0.0 <= target_pass_rate <= 1.0:
        raise ValueError("target_pass_rate must be between 0 and 1")

    if not isinstance(window, str):
        raise TypeError("window must be a string using the enterprise bucket format")

    if isinstance(df, pd.Series):
        ts = pd.to_datetime(df, errors="coerce")
        event_counts = pd.Series(1, index=ts.index, dtype="float64")
    else:
        if timestamp_col not in df.columns:
            raise KeyError(f"Column {timestamp_col!r} was not found in the input DataFrame")
        ts = pd.to_datetime(df[timestamp_col], errors="coerce")
        if count_col is None:
            event_counts = pd.Series(1, index=df.index, dtype="float64")
        else:
            if count_col not in df.columns:
                raise KeyError(f"Column {count_col!r} was not found in the input DataFrame")
            event_counts = pd.to_numeric(df[count_col], errors="coerce")

    valid = pd.DataFrame({"timestamp": ts, "event_count": event_counts})
    valid = valid.dropna(subset=["timestamp", "event_count"]).sort_values("timestamp")
    if valid.empty:
        return {
            "window": window,
            "passed": False,
            "pass_rate": 0.0,
            "min_events": min_events,
            "target_pass_rate": target_pass_rate,
            "bucket_count": 0,
            "dense_bucket_count": 0,
            "total_event_count": 0,
        }

    freq = _window_to_pandas_freq(window)
    bucket_counts = (
        valid.set_index("timestamp")["event_count"]
        .resample(freq, label="left", closed="left", origin="start_day")
        .sum()
        .fillna(0.0)
    )

    dense_bucket_count = int((bucket_counts >= min_events).sum())
    bucket_count = int(len(bucket_counts))
    pass_rate = dense_bucket_count / bucket_count if bucket_count else 0.0

    return {
        "window": window,
        "pandas_freq": freq,
        "passed": bool(pass_rate >= target_pass_rate),
        "pass_rate": pass_rate,
        "min_events": min_events,
        "target_pass_rate": target_pass_rate,
        "bucket_count": bucket_count,
        "dense_bucket_count": dense_bucket_count,
        "total_event_count": int(valid["event_count"].sum()),
    }


def find_optimal_time_bucket(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int = DEFAULT_MIN_EVENTS,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
    home_base: str = DEFAULT_HOME_BASE,
) -> dict[str, Any]:
    """Find the finest valid enterprise bucket duration for count-based time-series data.

    The search begins at the baseline window (default: 1 day) and then traverses the fixed
    enterprise window ladder toward finer or coarser aggregation depending on whether the
    baseline passes the density constraint. Fractional windows are not allowed.
    """
    window_list = list(windows or ENTERPRISE_WINDOWS)
    if not window_list:
        raise ValueError("At least one enterprise window must be supplied")
    effective_home_base = home_base if home_base in window_list else window_list[0]

    home_index = window_list.index(effective_home_base)
    history: list[dict[str, Any]] = []

    def evaluate_at(index: int) -> dict[str, Any]:
        window = window_list[index]
        result = evaluate_count_density(
            df,
            timestamp_col=timestamp_col,
            count_col=count_col,
            window=window,
            min_events=min_events,
            target_pass_rate=target_pass_rate,
        )
        result["index"] = index
        history.append(result)
        return result

    home_result = evaluate_at(home_index)
    if home_result["passed"]:
        current_index = home_index
        while current_index > 0:
            candidate_index = current_index - 1
            candidate = evaluate_at(candidate_index)
            if not candidate["passed"]:
                selected = evaluate_at(current_index)
                return {
                    "selected_window": window_list[current_index],
                    "index": current_index,
                    "direction": "finer",
                    "home_base": effective_home_base,
                    "min_events": min_events,
                    "target_pass_rate": target_pass_rate,
                    "pass_rate": selected["pass_rate"],
                    "bucket_count": selected["bucket_count"],
                    "dense_bucket_count": selected["dense_bucket_count"],
                    "history": history,
                }
            current_index = candidate_index

        selected = evaluate_at(0)
        return {
            "selected_window": window_list[0],
            "index": 0,
            "direction": "finer",
            "home_base": effective_home_base,
            "min_events": min_events,
            "target_pass_rate": target_pass_rate,
            "pass_rate": selected["pass_rate"],
            "bucket_count": selected["bucket_count"],
            "dense_bucket_count": selected["dense_bucket_count"],
            "history": history,
        }

    current_index = home_index
    while current_index < len(window_list) - 1:
        candidate_index = current_index + 1
        candidate = evaluate_at(candidate_index)
        if candidate["passed"]:
            return {
                "selected_window": window_list[candidate_index],
                "index": candidate_index,
                "direction": "coarser",
                "home_base": effective_home_base,
                "min_events": min_events,
                "target_pass_rate": target_pass_rate,
                "pass_rate": candidate["pass_rate"],
                "bucket_count": candidate["bucket_count"],
                "dense_bucket_count": candidate["dense_bucket_count"],
                "history": history,
            }
        current_index = candidate_index

    raise ValueError(
        "Sparse Data Alert: no valid enterprise window satisfied the density constraint "
        f"at or above the maximum operational aggregation level ({window_list[-1]})."
    )


def select_count_binning_window(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int = DEFAULT_MIN_EVENTS,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
    home_base: str = DEFAULT_HOME_BASE,
) -> dict[str, Any]:
    """Alias for the count-binning time-window selection API."""
    return find_optimal_time_bucket(
        df,
        timestamp_col=timestamp_col,
        windows=windows,
        count_col=count_col,
        min_events=min_events,
        target_pass_rate=target_pass_rate,
        home_base=home_base,
    )


def _empty_arrival_intensity_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["bin_id", "bin_start", "bin_end", "window", "count", "intensity"])


def _window_to_unit_duration(window: str) -> tuple[str, float]:
    if window.endswith("m"):
        return "minute", float(window[:-1])
    if window.endswith("h"):
        return "hour", float(window[:-1])
    if window.endswith("d"):
        return "day", float(window[:-1])
    if window.endswith("w"):
        return "week", float(window[:-1])
    raise ValueError(f"Unsupported window unit for intensity scaling: {window!r}")


def _window_to_unit_label(window: str) -> str:
    if window.endswith("m"):
        return "minute"
    if window.endswith("h"):
        return "hour"
    if window.endswith("d"):
        return "day"
    if window.endswith("w"):
        return "week"
    raise ValueError(f"Unsupported window unit for intensity scaling: {window!r}")


def _window_to_intensity_unit(window: str) -> str:
    unit = _window_to_unit_label(window)
    return f"count per {unit}"


def _build_window_count_frame(
    df: pd.DataFrame,
    timestamp_col: str,
    window: str,
    *,
    align_bins: bool,
) -> pd.DataFrame:
    window_td = _window_to_timedelta(window)
    freq = _window_to_pandas_freq(window)
    _, unit_value = _window_to_unit_duration(window)
    series = pd.to_datetime(df[timestamp_col], errors="coerce").dropna().copy()

    if align_bins:
        series = align_to_integer_bin_grid(series, window)
        counts = pd.DataFrame({"bin_start": series}).groupby("bin_start").size().rename("count").reset_index()
    else:
        counts = (
            pd.Series(1, index=series)
            .resample(freq, label="left", closed="left", origin="start_day")
            .sum()
            .rename("count")
            .reset_index()
        )
        counts.columns = ["bin_start", "count"]

    if counts.empty:
        return _empty_arrival_intensity_frame()

    start = counts["bin_start"].min()
    end = counts["bin_start"].max()
    full_range = pd.date_range(start=start, end=end, freq=freq, inclusive="both")
    counts = (
        counts.set_index("bin_start")["count"]
        .reindex(full_range, fill_value=0)
        .rename("count")
        .reset_index()
        .rename(columns={"index": "bin_start"})
    )

    counts["bin_end"] = counts["bin_start"] + window_td
    counts["bin_id"] = range(1, len(counts) + 1)
    counts["window"] = window
    counts["count"] = counts["count"].astype(int)
    counts["intensity"] = (counts["count"] / unit_value).round(3)
    counts = counts[["bin_id", "bin_start", "bin_end", "window", "count", "intensity"]]
    return counts.sort_values("bin_start").reset_index(drop=True)


def summarize_integer_alignment(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int = DEFAULT_MIN_EVENTS,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
    home_base: str = DEFAULT_HOME_BASE,
) -> dict[str, Any]:
    """Summarize both the original search result and the post-alignment result.

    Returns the selected algorithm window, the original unaligned bin series, the aligned series,
    whether any adjustment was applied, and a human-readable summary message.
    """
    if isinstance(df, pd.Series):
        frame = pd.DataFrame({timestamp_col: df})
    else:
        frame = df.copy()

    if timestamp_col not in frame.columns:
        raise KeyError(f"Column {timestamp_col!r} was not found in the input data")

    ts = pd.to_datetime(frame[timestamp_col], errors="coerce")
    work = frame.assign(**{timestamp_col: ts}).dropna(subset=[timestamp_col]).copy()
    if work.empty:
        return {
            "algorithm_window": None,
            "algorithm_result": None,
            "alignment_window": None,
            "alignment_result": None,
            "alignment_applied": False,
            "original_series": _empty_arrival_intensity_frame(),
            "aligned_series": _empty_arrival_intensity_frame(),
            "status": "no_data",
            "message": "No valid timestamps were available for the time-binning workflow.",
        }

    try:
        selection = find_optimal_time_bucket(
            work,
            timestamp_col=timestamp_col,
            windows=windows,
            count_col=count_col,
            min_events=min_events,
            target_pass_rate=target_pass_rate,
            home_base=home_base,
        )
    except ValueError as exc:
        message = str(exc)
        return {
            "algorithm_window": None,
            "algorithm_result": None,
            "alignment_window": None,
            "alignment_result": None,
            "alignment_applied": False,
            "original_series": _empty_arrival_intensity_frame(),
            "aligned_series": _empty_arrival_intensity_frame(),
            "status": "sparse_data",
            "message": message,
        }

    algorithm_window = selection["selected_window"]
    window_unit = _window_to_unit_label(algorithm_window)
    intensity_unit = _window_to_intensity_unit(algorithm_window)
    original_series = _build_window_count_frame(work, timestamp_col, algorithm_window, align_bins=False)
    aligned_series = _build_window_count_frame(work, timestamp_col, algorithm_window, align_bins=True)
    alignment_applied = not aligned_series.equals(original_series)
    status = "alignment_applied" if alignment_applied else "no_alignment_needed"
    message = (
        f"Algorithm selected {algorithm_window}; window unit={window_unit}; intensity unit={intensity_unit}; integer alignment applied a grid-phase adjustment."
        if alignment_applied
        else f"Algorithm selected {algorithm_window}; window unit={window_unit}; intensity unit={intensity_unit}; integer alignment did not change the grid."
    )

    return {
        "algorithm_window": algorithm_window,
        "window_unit": window_unit,
        "intensity_unit": intensity_unit,
        "algorithm_result": selection,
        "alignment_window": algorithm_window,
        "alignment_result": {
            "selected_window": algorithm_window,
            "window_unit": window_unit,
            "intensity_unit": intensity_unit,
            "alignment_applied": alignment_applied,
            "bins": len(aligned_series),
            "first_bin_start": aligned_series["bin_start"].iloc[0] if not aligned_series.empty else None,
            "last_bin_end": aligned_series["bin_end"].iloc[-1] if not aligned_series.empty else None,
        },
        "alignment_applied": alignment_applied,
        "original_series": original_series,
        "aligned_series": aligned_series,
        "status": status,
        "message": message,
    }


def build_arrival_intensity_series(
    df: pd.DataFrame | pd.Series,
    timestamp_col: str = "timestamp",
    *,
    windows: list[str] | None = None,
    count_col: str | None = None,
    min_events: int = DEFAULT_MIN_EVENTS,
    target_pass_rate: float = DEFAULT_TARGET_PASS_RATE,
    home_base: str = DEFAULT_HOME_BASE,
) -> pd.DataFrame:
    """Create a uniform-width arrival intensity series aligned to the selected enterprise window.

    Returns rows with integer bin IDs and explicit bin start/end timestamps. The intensity is
    the count of events in each fixed-width bucket, which is the empirical arrival intensity for
    that bin under the selected enterprise interval.
    """
    if isinstance(df, pd.Series):
        frame = pd.DataFrame({timestamp_col: df})
    else:
        frame = df.copy()

    if timestamp_col not in frame.columns:
        raise KeyError(f"Column {timestamp_col!r} was not found in the input data")

    ts = pd.to_datetime(frame[timestamp_col], errors="coerce")
    work = frame.assign(**{timestamp_col: ts}).dropna(subset=[timestamp_col]).copy()
    if work.empty:
        return _empty_arrival_intensity_frame()

    try:
        selection = find_optimal_time_bucket(
            work,
            timestamp_col=timestamp_col,
            windows=windows,
            count_col=count_col,
            min_events=min_events,
            target_pass_rate=target_pass_rate,
            home_base=home_base,
        )
    except ValueError:
        return _empty_arrival_intensity_frame()

    window = selection["selected_window"]
    return _build_window_count_frame(work, timestamp_col, window, align_bins=True)
