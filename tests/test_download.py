from pathlib import Path

import pandas as pd
import pytest

from data_prep.config import load_config
from data_prep.download import (
    _normalize_kaggle_dataset_id,
    download_from_config,
    download_http_file,
    kaggle_url_to_config,
    verify_download_path,
)
from data_prep.llm import DatasetDictionaryBuilder, OllamaLLM


def test_normalize_kaggle_dataset_id_accepts_slug():
    assert _normalize_kaggle_dataset_id("paultimothymooney/chess") == "paultimothymooney/chess"


def test_normalize_kaggle_dataset_id_accepts_url():
    url = "https://www.kaggle.com/datasets/hassanjameelahmed/healthcare-analytics-patient-flow-data"
    assert _normalize_kaggle_dataset_id(url) == "hassanjameelahmed/healthcare-analytics-patient-flow-data"


def test_download_from_config_reads_yaml_file(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
version: 1
dataset_name: demo
dataset_id: demo/dataset
operation: download
source:
  type: kaggle
  kaggle_dataset: owner/sample
destination:
  path: ./data/raw/sample
""".strip()
    )

    config = load_config(config_path)
    assert config["source"]["kaggle_dataset"] == "owner/sample"
    assert config["destination"]["path"] == "./data/raw/sample"


def test_verify_download_path_raises_clear_error_for_empty_directory(tmp_path):
    target = tmp_path / "downloaded"
    target.mkdir()

    try:
        verify_download_path(target)
        assert False, "verify_download_path should raise for an empty directory"
    except FileNotFoundError as exc:
        assert "contains no files" in str(exc)
        assert str(target) in str(exc)


def test_download_http_file_preserves_excel_workbook(monkeypatch, tmp_path):
    import io
    import zipfile

    workbook = io.BytesIO()
    with zipfile.ZipFile(workbook, "w") as zf:
        zf.writestr("[Content_Types].xml", "<Types></Types>")
        zf.writestr("_rels/.rels", "<Relationships></Relationships>")
        zf.writestr("docProps/core.xml", "<cp:coreProperties/>")
        zf.writestr("docProps/app.xml", "<Properties/>")
        zf.writestr("xl/workbook.xml", "<workbook/>")
        zf.writestr("xl/_rels/workbook.xml.rels", "<Relationships/>")
        zf.writestr("xl/worksheets/sheet1.xml", "<worksheet/>")

    def fake_urlretrieve(url, output_path):
        output_path.write_bytes(workbook.getvalue())

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    result = download_http_file(
        "https://example.com/7a_504_foia_data_dictionary.xlsx",
        tmp_path / "downloaded",
    )

    assert result == tmp_path / "downloaded" / "7a_504_foia_data_dictionary.xlsx"
    assert result.is_file()
    assert not (tmp_path / "downloaded" / "7a_504_foia_data_dictionary").exists()


def test_download_http_file_creates_raw_data_dictionary(monkeypatch, tmp_path):
    csv_path = tmp_path / "downloaded"
    csv_path.mkdir()

    def fake_urlretrieve(url, output_path):
        output_path.write_text("customer_id,order_total\n1,42\n2,53\n", encoding="utf-8")

    monkeypatch.setattr("urllib.request.urlretrieve", fake_urlretrieve)

    result = download_http_file("https://example.com/data.csv", csv_path)

    assert result == csv_path / "data.csv"
    dictionary_path = csv_path / "data_dictionary_raw.csv"
    assert dictionary_path.exists()
    dictionary = pd.read_csv(dictionary_path)
    assert set(dictionary["attribute"]) == {"customer_id", "order_total"}


def test_kaggle_url_to_config_prints_yaml_when_write_false(monkeypatch, capsys):
    import data_prep.download as download_module

    fake_root = Path("/tmp/fake-project")
    monkeypatch.setattr(download_module, "__file__", str(fake_root / "src/data_prep/download.py"))

    config = kaggle_url_to_config(
        "https://www.kaggle.com/datasets/hassanjameelahmed/healthcare-analytics-patient-flow-data",
        write=False,
    )

    assert config["dataset_name"] == "healthcare-analytics-patient-flow-data"
    captured = capsys.readouterr()
    assert "operation: download" in captured.out
    assert "kaggle_dataset: hassanjameelahmed/healthcare-analytics-patient-flow-data" in captured.out


def test_kaggle_url_to_config_writes_file_when_write_true(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    path = kaggle_url_to_config(
        "https://www.kaggle.com/datasets/hassanjameelahmed/healthcare-analytics-patient-flow-data",
        write=True,
    )

    assert path.exists()
    assert path.parent == tmp_path / "configs"
    assert path.parent.parent == tmp_path
    assert "healthcare-analytics-patient-flow-data" in path.name


def test_kaggle_url_to_config_accepts_custom_output_dir(monkeypatch, capsys):
    import data_prep.download as download_module

    fake_root = Path("/tmp/fake-project")
    monkeypatch.setattr(download_module, "__file__", str(fake_root / "src/data_prep/download.py"))

    config = kaggle_url_to_config(
        "https://www.kaggle.com/datasets/hassanjameelahmed/healthcare-analytics-patient-flow-data",
        write=False,
        output_dir="./custom/output",
    )

    assert config["destination"]["path"] == "./custom/output"
    assert config["destination"]["format"] == "csv"
    captured = capsys.readouterr()
    assert "./custom/output" in captured.out
    assert "format: csv" in captured.out


def test_kaggle_url_to_config_defaults_format_to_csv(monkeypatch, capsys):
    import data_prep.download as download_module

    fake_root = Path("/tmp/fake-project")
    monkeypatch.setattr(download_module, "__file__", str(fake_root / "src/data_prep/download.py"))

    config = kaggle_url_to_config(
        "https://www.kaggle.com/datasets/hassanjameelahmed/healthcare-analytics-patient-flow-data",
        write=False,
    )

    assert config["destination"]["format"] == "csv"
    captured = capsys.readouterr()
    assert "format: csv" in captured.out


def test_ollama_llm_calls_local_api(monkeypatch):
    captured = {}

    class FakeResponse:
        def __init__(self):
            self._json = {"response": "ok"}

        def json(self):
            return self._json

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("requests.post", fake_post)

    llm = OllamaLLM(model="llama3.2")
    assert llm.generate("hello") == "ok"
    assert captured["url"] == "http://localhost:11434/api/generate"
    assert captured["json"]["model"] == "llama3.2"


def test_build_data_dictionary_extracts_descriptions_from_docs(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("first_name,last_name\nAda,Lovelace\n", encoding="utf-8")
    (tmp_path / "README.md").write_text(
        "first_name: given name of a person. last_name: family name of a person.",
        encoding="utf-8",
    )

    class FakeLLM:
        def generate(self, prompt, **kwargs):
            return (
                '{"status":"has_attribute_descriptions","attribute_descriptions":['
                '{"column_name":"first_name","description":"Given name of a person","data_type":"string"},'
                '{"column_name":"last_name","description":"Family name of a person","data_type":"string"}'
                '],"source":"README.md"}'
            )

    builder = DatasetDictionaryBuilder(llm=FakeLLM(), output_path=tmp_path / "data_dictionary.csv")
    result = builder.create_from_directory(tmp_path)

    assert result["status"] == "has_attribute_descriptions"
    assert (tmp_path / "data_dictionary.csv").exists()
    dictionary = pd.read_csv(tmp_path / "data_dictionary.csv")
    assert set(dictionary["column_name"]) == {"first_name", "last_name"}


def test_build_data_dictionary_reports_graceful_failure_when_no_descriptions_exist(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("first_name,last_name\nAda,Lovelace\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("This dataset contains people names", encoding="utf-8")

    class FakeLLM:
        def generate(self, prompt, **kwargs):
            return '{"status":"no_attribute_descriptions","attribute_descriptions":[],"source":null}'

    builder = DatasetDictionaryBuilder(llm=FakeLLM(), output_path=tmp_path / "data_dictionary.csv")
    result = builder.create_from_directory(tmp_path, use_llm=True)

    assert result["status"] == "no_attribute_descriptions"
    assert "message" in result
    assert "No attribute descriptions" in result["message"]
    assert not (tmp_path / "data_dictionary.csv").exists()


def test_build_data_dictionary_accepts_explicit_dictionary_file(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("first_name,last_name\nAda,Lovelace\n", encoding="utf-8")
    dictionary_path = tmp_path / "data_dictionary.csv"
    dictionary_path.write_text(
        "attribute,description\nfirst_name,Given name\nlast_name,Family name\n",
        encoding="utf-8",
    )

    builder = DatasetDictionaryBuilder(output_path=tmp_path / "output_dictionary.csv")
    result = builder.create_from_directory(tmp_path, dictionary_path=dictionary_path)

    assert result["status"] == "has_attribute_descriptions"
    assert result["source"] == str(dictionary_path)
    assert (tmp_path / "output_dictionary.csv").exists()


def test_build_data_dictionary_reports_llm_failure_when_generation_fails(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("first_name,last_name\nAda,Lovelace\n", encoding="utf-8")

    class FailingLLM:
        def generate(self, prompt, **kwargs):
            raise RuntimeError("Local model unavailable")

    builder = DatasetDictionaryBuilder(llm=FailingLLM(), output_path=tmp_path / "data_dictionary.csv")
    result = builder.create_from_directory(tmp_path, use_llm=True)

    assert result["status"] == "no_attribute_descriptions"
    assert "Could not create the data dictionary using the LLM" in result["message"]
    assert not (tmp_path / "data_dictionary.csv").exists()


def test_update_data_dictionary_merges_attribute_metadata(tmp_path):
    dictionary_path = tmp_path / "data_dictionary.csv"
    dictionary_path.write_text(
        "attribute,description,data_type\nfirst_name,Old first name,unknown\nlast_name,Family name,string\n",
        encoding="utf-8",
    )

    updates = {
        "first_name": {"description": "Given name of a person", "data_type": "string"},
        "last_name": "Surname of a person",
    }

    builder = DatasetDictionaryBuilder()
    result = builder.update_data_dictionary(dictionary_path, updates)

    assert result["status"] == "updated"
    updated = pd.read_csv(dictionary_path)
    assert updated.loc[updated["attribute"] == "first_name", "description"].iat[0] == "Given name of a person"
    assert updated.loc[updated["attribute"] == "first_name", "data_type"].iat[0] == "string"
    assert updated.loc[updated["attribute"] == "last_name", "description"].iat[0] == "Surname of a person"
    assert updated.loc[updated["attribute"] == "last_name", "data_type"].iat[0] == "string"


def test_download_http_file_extracts_zip_archive(tmp_path):
    import http.server
    import socketserver
    import threading
    import zipfile

    archive_path = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("nested/sample.csv", "id,name\n1,alpha\n")

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(tmp_path), **kwargs)

        def log_message(self, format, *args):
            return

    with socketserver.TCPServer(("127.0.0.1", 0), QuietHandler) as httpd:
        port = httpd.server_address[1]
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        try:
            destination = tmp_path / "downloaded"
            result = Path(__import__("data_prep.download", fromlist=["download_http_file"]).download_http_file(
                f"http://127.0.0.1:{port}/dataset.zip",
                destination,
            ))
            assert result.is_dir()
            assert (result / "nested" / "sample.csv").exists()
            assert (result / "nested" / "sample.csv").read_text(encoding="utf-8") == "id,name\n1,alpha\n"
        finally:
            httpd.shutdown()
            thread.join(timeout=5)
