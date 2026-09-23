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
