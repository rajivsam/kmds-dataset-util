import matplotlib
import pandas as pd
import pytest

matplotlib.use("Agg")


def test_select_poisson_operating_window_reports_metrics_and_plot(tmp_path):
    from data_prep import select_poisson_operating_window

    arrivals = [
        timestamp
        for minute in range(0, 600, 2)
        for timestamp in (
            pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute),
            pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute, seconds=10),
        )
    ]
    event_file = tmp_path / "arrivals.csv"
    pd.DataFrame({"arrival_time": arrivals}).to_csv(event_file, index=False)
    plot_path = tmp_path / "report" / "fano.png"

    result = select_poisson_operating_window(
        event_file,
        ["20m", "2m", "1m"],
        timestamp_col="arrival_time",
        plot_path=plot_path,
    )

    metrics = result["metrics"].set_index("window")
    assert metrics.index.tolist() == ["1m", "2m", "20m"]
    assert metrics.loc["1m", "aggregation_window_count"] == 598
    assert metrics.loc["1m", "poisson_assumption_satisfied"]
    assert not metrics.loc["2m", "poisson_assumption_satisfied"]
    assert metrics.loc["20m", "aggregation_window_count"] == 29
    assert metrics.loc["20m", "status"] == "insufficient_data"
    assert result["acceptable_windows"] == ["1m"]
    assert result["selection_status"] == "acceptable_windows_found"
    assert result["axes"].get_ylabel().startswith("Fano factor")
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0

    result["plot"].clf()


def test_select_poisson_operating_window_returns_every_acceptable_window(tmp_path):
    from data_prep import select_poisson_operating_window

    minute_counts = [0, 0, 0, 2, 2, 0, 2, 2]
    arrivals = [
        pd.Timestamp("2024-01-01")
        + pd.Timedelta(minutes=minute, seconds=10 * event_index)
        for minute in range(8 * 2000)
        for event_index in range(minute_counts[minute % len(minute_counts)])
    ]
    event_file = tmp_path / "arrivals.csv"
    pd.DataFrame({"timestamp": arrivals}).to_csv(event_file, index=False)

    result = select_poisson_operating_window(event_file, ["2m", "1m"])

    assert result["acceptable_windows"] == ["1m", "2m"]
    assert result["metrics"]["poisson_assumption_satisfied"].all()
    result["plot"].clf()


def test_confidence_parameter_controls_chi_squared_bounds(tmp_path):
    from data_prep.fanno_factor_selection import (
        DEFAULT_CONFIDENCE,
        select_poisson_operating_window,
    )

    assert DEFAULT_CONFIDENCE == 0.90
    arrivals = [
        timestamp
        for minute in range(0, 1200, 2)
        for timestamp in (
            pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute),
            pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=minute, seconds=10),
        )
    ]
    event_file = tmp_path / "arrivals.csv"
    pd.DataFrame({"timestamp": arrivals}).to_csv(event_file, index=False)

    default_result = select_poisson_operating_window(event_file, ["1m"])
    stricter_result = select_poisson_operating_window(event_file, ["1m"], confidence=0.95)

    default_row = default_result["metrics"].iloc[0]
    stricter_row = stricter_result["metrics"].iloc[0]
    assert default_result["confidence"] == 0.90
    assert stricter_result["confidence"] == 0.95
    assert default_row["fano_lower"] > stricter_row["fano_lower"]
    assert default_row["fano_upper"] < stricter_row["fano_upper"]
    default_result["plot"].clf()
    stricter_result["plot"].clf()


def test_select_poisson_operating_window_adjusts_aggregation_interval(tmp_path):
    from data_prep import select_poisson_operating_window

    arrivals = pd.date_range("2024-01-01", periods=2701, freq="1min")
    event_file = tmp_path / "arrivals.csv"
    pd.DataFrame({"timestamp": arrivals}).to_csv(event_file, index=False)

    result = select_poisson_operating_window(
        event_file,
        ["90min"],
        aggregation_window="1h",
    )

    row = result["metrics"].iloc[0]
    assert row["aggregation_window_count"] == 30
    assert row["aggregation_interval"] == pd.Timedelta("45min")
    assert row["intervals_per_window"] == 2
    assert result["aggregation_window"] == pd.Timedelta("1h")
    result["plot"].clf()


def test_select_poisson_operating_window_rejects_invalid_inputs(tmp_path):
    from data_prep import select_poisson_operating_window

    event_file = tmp_path / "arrivals.csv"
    pd.DataFrame({"timestamp": ["not-a-timestamp"]}).to_csv(event_file, index=False)

    with pytest.raises(ValueError, match="No valid event-arrival timestamps"):
        select_poisson_operating_window(event_file, ["1m"])

    pd.DataFrame({"timestamp": ["2024-01-01"]}).to_csv(event_file, index=False)
    with pytest.raises(ValueError, match="at least 2"):
        select_poisson_operating_window(event_file, ["1m"], min_aggregation_windows=1)
