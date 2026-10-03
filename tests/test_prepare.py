from pathlib import Path

import pandas as pd

import data_prep.prepare as prepare_module
from data_prep.prepare import prepare_dataset


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
