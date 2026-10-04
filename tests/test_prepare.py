from pathlib import Path

import pandas as pd

import data_prep.prepare as prepare_module
from data_prep.prepare import prepare_dataset


def test_clean_count_timeseries_api_has_only_the_lean_public_surface():
    import data_prep.count_timeseries as count_timeseries

    expected = {
        "ENTERPRISE_WINDOWS",
        "WINDOWS",
        "DEFAULT_MIN_EVENTS",
        "DEFAULT_TARGET_PASS_RATE",
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
    }

    assert set(count_timeseries.__all__) == expected


def test_prepare_dataset_writes_csv_and_returns_metadata(tmp_path):
    df = pd.DataFrame({"value": [1, 2, 3]})
    result = prepare_dataset(
        dataset_name="demo",
        representation="tabular",
        task_characterization="classification",
        staging_dir="./data/raw/demo",
        prepared_dataset_dir=str(tmp_path / "prepared"),
        data=df,
        output_format="csv",
    )

    assert result["dataset_name"] == "demo"
    assert result["representation"] == "tabular"
    assert result["output_path"].endswith("demo_tabular.csv")
    assert Path(result["output_path"]).exists()


def test_prepare_dataset_writes_data_dictionary_metadata(tmp_path):
    df = pd.DataFrame({"value": [1, 2, 3]})
    metadata = {
        "data_dictionary": [
            {"attribute": "value", "description": "Numeric value", "data_type": "int"},
        ]
    }

    result = prepare_dataset(
        dataset_name="demo",
        representation="tabular",
        task_characterization="classification",
        staging_dir="./data/raw/demo",
        prepared_dataset_dir=str(tmp_path / "prepared"),
        data=df,
        output_format="csv",
        metadata=metadata,
    )

    assert "data_dictionary_path" in result
    dictionary_path = Path(result["data_dictionary_path"])
    assert dictionary_path.exists()
    dictionary = pd.read_csv(dictionary_path)
    assert list(dictionary.columns) == ["attribute", "description", "data_type"]


def test_prepare_dataset_accepts_sba_style_dictionary_columns(tmp_path):
    df = pd.DataFrame({"loan_amount": [1000.0, 2000.0], "approval_date": ["2020-01-01", "2020-02-01"]})
    metadata = {
        "data_dictionary": [
            {"Field Name": "loan_amount", "Definition": "Loan amount"},
            {"Field Name": "approval_date", "Definition": "Approval date"},
        ]
    }

    result = prepare_dataset(
        dataset_name="sba",
        representation="tabular",
        task_characterization="classification",
        staging_dir="./data/raw/sba",
        prepared_dataset_dir=str(tmp_path / "prepared"),
        data=df,
        output_format="csv",
        metadata=metadata,
    )

    assert "dictionary_files" in result
    dictionary_path = Path(result["dictionary_files"]["data_dictionary"])
    assert dictionary_path.exists()
    dictionary = pd.read_csv(dictionary_path)
    assert list(dictionary.columns) == ["attribute", "description", "data_type"]
    assert set(dictionary["attribute"]) >= {"loan_amount", "approval_date"}


def test_package_does_not_define_custom_dataset_preparation_helpers():
    assert not hasattr(prepare_module, "prepare_healthcare_dataset")
    assert not hasattr(prepare_module, "prepare_seattle_911_dataset")


def test_optimize_arrival_binning_uses_fano_factor_for_window_selection():
    from data_prep.timeseries import optimize_arrival_binning

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=180, freq="1min")
    result = optimize_arrival_binning(
        pd.DataFrame({"timestamp": timestamps}),
        windows=["5m", "15m", "30m", "1h"],
        tau_block="4h",
    )

    assert result["optimal_window"] == "5m"
    assert "window_metrics" in result
    assert {"window", "pass_rate", "average_local_fano", "fano_deviation"}.issubset(result["window_metrics"].columns)
    assert result["window_metrics"]["pass_rate"].between(0.0, 1.0).all()
    assert result["window_metrics"]["window"].nunique() >= 1


def test_expected_poisson_zeros_matches_closed_form():
    from data_prep.count_timeseries import expected_poisson_zeros

    assert expected_poisson_zeros(0.0, 100) == 100.0
    assert abs(expected_poisson_zeros(1.0, 10) - 10 * 2.718281828459045 ** -1) < 1e-12


def test_fano_acceptance_interval_matches_chi_squared_reference():
    from data_prep.count_timeseries import fano_acceptance_interval

    counts = pd.Series([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    interval = fano_acceptance_interval(counts, window="5m", tau_block="1h", confidence=0.95)

    assert interval["confidence"] == 0.95
    assert interval["n_blocks"] >= 1
    assert interval["fano_lower"] < interval["fano_upper"]
    assert interval["fano_mean"] >= 0.0


def test_acceptable_over_dispersed_window_size_returns_window_fano_table():
    from data_prep.count_timeseries import acceptable_over_dispersed_window_size

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=180, freq="1min")
    table = acceptable_over_dispersed_window_size(
        pd.DataFrame({"timestamp": timestamps}),
        windows=["5m", "15m", "30m", "1h"],
        tau_block="4h",
        confidence=0.95,
    )

    assert isinstance(table, pd.DataFrame)
    assert {"window", "average_local_fano", "fano_upper", "acceptable_for_dataset"}.issubset(table.columns)
    assert len(table) >= 1
    assert "window" in table.columns
    assert table["average_local_fano"].notna().all()
    assert table["acceptable_for_dataset"].isin([True, False]).all()


def test_select_largest_window_in_fano_acceptance_interval_uses_precomputed_table():
    from data_prep.count_timeseries import acceptable_over_dispersed_window_size, select_largest_window_in_fano_acceptance_interval

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=180, freq="1min")
    table = acceptable_over_dispersed_window_size(
        pd.DataFrame({"timestamp": timestamps}),
        windows=["5m", "15m", "30m", "1h"],
        tau_block="4h",
        confidence=0.95,
    )

    selection = select_largest_window_in_fano_acceptance_interval(table)

    assert isinstance(selection, dict)
    assert selection["reference_window"] in table["window"].tolist()
    assert selection["selected_window"] in table["window"].tolist()
    assert selection["selected_window"] == selection["reference_window"] or selection["selected_window"] in {"15m", "30m", "1h"}


def test_hospital_dataset_allows_larger_window_inside_acceptance_band():
    from pathlib import Path
    from data_prep.count_timeseries import acceptable_over_dispersed_window_size, select_largest_window_in_fano_acceptance_interval

    project_root = Path(__file__).resolve().parents[1]
    raw_csv = project_root / "data" / "raw" / "healthcare-analytics-patient-flow-data" / "healthcare_analytics_patient_flow_data.csv"
    if not raw_csv.exists():
        pytest.skip(f"Hospital dataset fixture not present: {raw_csv}")

    df = pd.read_csv(raw_csv)
    arrival_df = df[["Patient Admission Date", "Patient Admission Time"]].copy()
    arrival_df["timestamp"] = pd.to_datetime(
        arrival_df["Patient Admission Date"].astype(str) + " " + arrival_df["Patient Admission Time"].astype(str),
        errors="coerce",
    )
    arrival_df = arrival_df.dropna(subset=["timestamp"]).reset_index(drop=True)
    target_year = arrival_df["timestamp"].dt.year.mode().iloc[0]
    arrival_df = arrival_df[arrival_df["timestamp"].dt.year == target_year].reset_index(drop=True)

    table = acceptable_over_dispersed_window_size(
        arrival_df[["timestamp"]],
        timestamp_col="timestamp",
        windows=["5m", "15m", "30m", "1h", "2h", "4h", "8h", "12h", "1d", "2d", "3d", "5d", "1w", "2w", "4w"],
        min_events=5,
        target_pass_rate=0.75,
        tau_block="4h",
        confidence=0.95,
    )

    selection = select_largest_window_in_fano_acceptance_interval(table)

    assert selection["reference_window"] == "5m"
    assert table.loc[table["window"] == "15m", "average_local_fano"].notna().all()
    assert table.loc[table["window"] == "15m", "average_local_fano"].iloc[0] > 0.895846
    assert selection["selected_window"] == "15m"


def test_build_arrival_intensity_series_returns_uniform_bin_dataframe():
    from data_prep.timeseries import build_arrival_intensity_series

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=180, freq="1min")
    series = build_arrival_intensity_series(pd.DataFrame({"timestamp": timestamps}))
    window = series["window"].iloc[0]
    window_value = float(window[:-1])

    assert list(series.columns) == ["bin_id", "bin_start", "bin_end", "window", "count", "intensity"]
    assert series["bin_id"].tolist() == list(range(1, len(series) + 1))
    assert series["window"].nunique() == 1
    assert (series["bin_end"] - series["bin_start"]).eq(pd.to_timedelta(series["window"].iloc[0])).all()
    assert (series["intensity"] == (series["count"] / window_value).round(3)).all()

def test_align_to_integer_bin_grid_anchors_bins_to_midnight_grid():
    from data_prep.timeseries import align_to_integer_bin_grid

    timestamps = pd.to_datetime([
        "2024-01-01 03:14:00",
        "2024-01-01 03:20:00",
        "2024-01-01 04:03:00",
        "2024-01-01 04:25:00",
        "2024-01-01 08:12:00",
        "2024-01-02 02:10:00",
    ])
    aligned = align_to_integer_bin_grid(pd.Series(timestamps), "4h")

    assert aligned.dt.time.map(str).str.startswith("00:00:00").any()
    assert (aligned.dt.hour % 4 == 0).all()
    assert len(aligned) == len(timestamps)

def test_summarize_integer_alignment_reports_original_and_aligned_outputs():
    from data_prep.timeseries import summarize_integer_alignment

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=180, freq="1min")
    summary = summarize_integer_alignment(pd.DataFrame({"timestamp": timestamps}), "timestamp")

    assert summary["algorithm_window"] == "5m"
    assert summary["alignment_window"] == "5m"
    assert summary["status"] in {"alignment_applied", "no_alignment_needed"}
    assert isinstance(summary["original_series"], pd.DataFrame)
    assert isinstance(summary["aligned_series"], pd.DataFrame)


def test_summarize_integer_alignment_starts_raw_phase_on_day_boundary():
    from data_prep.timeseries import summarize_integer_alignment

    timestamps = pd.to_datetime([
        "2024-01-01 00:05:00",
        "2024-01-01 00:12:00",
        "2024-01-01 00:18:00",
        "2024-01-01 00:24:00",
        "2024-01-01 00:30:00",
        "2024-01-01 00:36:00",
        "2024-01-01 12:05:00",
        "2024-01-01 12:12:00",
        "2024-01-01 12:18:00",
        "2024-01-01 12:24:00",
        "2024-01-01 12:30:00",
        "2024-01-01 12:36:00",
    ])
    summary = summarize_integer_alignment(pd.DataFrame({"timestamp": timestamps}), "timestamp", windows=["12h"])

    assert summary["algorithm_window"] == "12h"
    assert summary["original_series"].empty is False
    assert summary["original_series"]["bin_start"].iloc[0] == pd.Timestamp("2024-01-01 00:00:00")


def test_summarize_integer_alignment_handles_sparse_data_gracefully():
    from data_prep.timeseries import summarize_integer_alignment

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=2, freq="30d")
    summary = summarize_integer_alignment(pd.DataFrame({"timestamp": timestamps}), "timestamp")

    assert summary["status"] == "sparse_data"
    assert summary["algorithm_window"] is None
    assert summary["aligned_series"].empty


def test_select_time_bucket_prefers_finer_valid_window_for_dense_count_data():
    from data_prep.timeseries import find_optimal_time_bucket

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=180, freq="1min")
    result = find_optimal_time_bucket(pd.DataFrame({"timestamp": timestamps}), "timestamp")

    assert result["selected_window"] == "5m"
    assert result["pass_rate"] >= 0.8
    assert result["min_events"] == 5


def test_select_time_bucket_aggregates_to_coarser_window_for_sparse_count_data():
    from data_prep.timeseries import find_optimal_time_bucket

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=30, freq="12h")
    result = find_optimal_time_bucket(pd.DataFrame({"timestamp": timestamps}), "timestamp")

    assert result["selected_window"] == "3d"
    assert result["pass_rate"] >= 0.8


def test_select_time_bucket_stops_at_first_valid_window_in_ascending_order():
    from data_prep.timeseries import find_optimal_time_bucket

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=72 * 5, freq="12min")
    result = find_optimal_time_bucket(pd.DataFrame({"timestamp": timestamps}), "timestamp")

    assert result["selected_window"] == "1h"
    assert result["pass_rate"] >= 0.8
    assert result["min_events"] == 5


def test_build_arrival_intensity_series_drops_partial_final_window_for_multi_day_bins():
    from data_prep.timeseries import build_arrival_intensity_series

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=10, freq="1d")
    series = build_arrival_intensity_series(
        pd.DataFrame({"timestamp": timestamps}),
        windows=["2d"],
        min_events=1,
        target_pass_rate=0.5,
    )

    assert len(series) == 4
    assert series["bin_end"].max() == pd.Timestamp("2024-01-09 00:00:00")


def test_find_optimal_time_bucket_returns_verbose_selection_log():
    from data_prep.timeseries import find_optimal_time_bucket

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=72 * 5, freq="12min")
    result = find_optimal_time_bucket(pd.DataFrame({"timestamp": timestamps}), "timestamp", verbose=True)

    assert isinstance(result["selection_log"], pd.DataFrame)
    assert {"window", "pass_rate", "selected", "adjustment"}.issubset(result["selection_log"].columns)
    assert result["selected_window"] == result["selection_log"].loc[result["selection_log"]["selected"], "window"].iloc[0]


def test_count_timeseries_exports_only_bin_end_and_intensity_columns(tmp_path):
    from data_prep.count_timeseries import tseda_dataset_export

    source_dir = tmp_path / "prepared"
    source_dir.mkdir()
    source_df = pd.DataFrame(
        {
            "bin_id": [1, 2],
            "bin_start": [pd.Timestamp("2024-01-01 00:00:00"), pd.Timestamp("2024-01-01 00:05:00")],
            "bin_end": [pd.Timestamp("2024-01-01 00:05:00"), pd.Timestamp("2024-01-01 00:10:00")],
            "window": ["5m", "5m"],
            "count": [3, 5],
            "intensity": [0.6, 1.0],
        }
    )
    source_path = source_dir / "hospital_arrival_intensity.csv"
    source_df.to_csv(source_path, index=False)

    exported_path = source_dir / "hospital_arrival_intensity_tseda.csv"
    result_path = tseda_dataset_export(
        source_path=source_path,
        output_path=exported_path,
    )

    exported = pd.read_csv(result_path)
    assert list(exported.columns) == ["bin_end", "intensity"]
    assert result_path.name == "hospital_arrival_intensity_tseda.csv"
    assert result_path.exists()
    assert exported["bin_end"].tolist() == ["2024-01-01 00:05:00", "2024-01-01 00:10:00"]
