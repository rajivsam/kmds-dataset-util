from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import pandas as pd
import requests


class BaseLLM(ABC):
    """Common contract for model providers used in metadata extraction."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Return a raw text response from the model."""


class OllamaLLM(BaseLLM):
    """Adapter for the local Ollama model server."""

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        timeout: int = 120,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        if hasattr(response, "raise_for_status"):
            response.raise_for_status()
        data = response.json() if hasattr(response, "json") else {}
        return str(data.get("response", ""))


class DatasetDictionaryBuilder:
    """Build a data dictionary from a dataset using either a supplied dictionary file or an LLM."""

    def __init__(self, llm: BaseLLM | None = None, output_path: str | Path | None = None) -> None:
        self.llm = llm or OllamaLLM()
        self.output_path = Path(output_path) if output_path else None

    def _load_dictionary_csv(self, dictionary_path: str | Path) -> dict[str, Any]:
        path = Path(dictionary_path)
        if not path.exists():
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": str(path),
                "message": f"The specified data dictionary file was not found: {path}",
            }

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive path
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": str(path),
                "message": f"The specified data dictionary file could not be read: {exc}",
            }

        required_columns = {"attribute", "description"}
        if not required_columns.issubset(df.columns):
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": str(path),
                "message": "The supplied dictionary must contain 'attribute' and 'description' columns.",
            }

        records = [
            {"column_name": row["attribute"], "description": row["description"], "data_type": "unknown"}
            for _, row in df.iterrows()
        ]
        return {
            "status": "has_attribute_descriptions",
            "attribute_descriptions": records,
            "source": str(path),
        }

    def detect_attribute_descriptions(self, dataset_dir: str | Path) -> dict[str, Any]:
        directory = Path(dataset_dir)
        if not directory.exists():
            raise FileNotFoundError(f"Dataset directory not found: {directory}")

        files = sorted(directory.iterdir())
        candidate_files = [
            path
            for path in files
            if path.is_file() and path.suffix.lower() in {".csv", ".tsv", ".json", ".md", ".txt", ".yaml", ".yml"}
        ]

        if not candidate_files:
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": None,
                "message": "No dataset files were found to inspect for attribute descriptions.",
            }

        csv_files = [path for path in candidate_files if path.suffix.lower() in {".csv", ".tsv"}]
        doc_files = [path for path in candidate_files if path.suffix.lower() in {".md", ".txt", ".json", ".yaml", ".yml"}]

        if not csv_files:
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": None,
                "message": "No tabular dataset file was found for attribute description extraction.",
            }

        dataset_file = csv_files[0]
        columns = pd.read_csv(dataset_file).columns.tolist()
        descriptions_text = "\n".join(
            f"{path.name}:\n{path.read_text(encoding='utf-8', errors='replace')}"
            for path in doc_files[:5]
        )

        if not descriptions_text.strip():
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": None,
                "message": "No attribute descriptions were found in available documentation for this dataset.",
            }

        prompt = (
            "You are extracting a dataset data dictionary from available source documentation.\n"
            "Analyze the following dataset schema and documentation.\n\n"
            f"Columns: {columns}\n\n"
            "Documentation:\n"
            f"{descriptions_text}\n\n"
            "Return valid JSON with exactly this structure:\n"
            '{"status":"has_attribute_descriptions","attribute_descriptions":[{"column_name":"...","description":"...","data_type":"..."}],"source":"..."}'
            "If no attribute descriptions are present, return:"
            '{"status":"no_attribute_descriptions","attribute_descriptions":[],"source":null}'
            "Do not add extra commentary."
        )

        try:
            raw = self.llm.generate(prompt)
        except Exception as exc:
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": None,
                "message": f"Could not create the data dictionary using the LLM: {exc}",
            }

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": None,
                "message": "The LLM response was not valid JSON, so no attribute descriptions could be extracted.",
            }

        status = str(payload.get("status", "no_attribute_descriptions"))
        attribute_descriptions = payload.get("attribute_descriptions") or []
        source = payload.get("source")
        if status != "has_attribute_descriptions" or not isinstance(attribute_descriptions, list):
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": source,
                "message": "No attribute descriptions were available for extraction from the dataset documentation.",
            }

        return {
            "status": status,
            "attribute_descriptions": attribute_descriptions,
            "source": source,
        }

    def update_data_dictionary(
        self,
        dictionary_path: str | Path,
        updates: dict[str, Any],
        *,
        output_path: str | Path | None = None,
    ) -> dict[str, Any]:
        """Update a CSV data dictionary keyed by attribute name.

        `updates` can take either of these forms:
        - {"column_name": "new description"}
        - {"column_name": {"description": "...", "data_type": "..."}}
        - {"column_name": {"description": "...", "type": "..."}}
        """
        path = Path(dictionary_path)
        if not path.exists():
            return {
                "status": "not_found",
                "attribute_descriptions": [],
                "source": str(path),
                "message": f"The data dictionary file was not found: {path}",
            }

        try:
            df = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive path
            return {
                "status": "invalid",
                "attribute_descriptions": [],
                "source": str(path),
                "message": f"The data dictionary file could not be read: {exc}",
            }

        if "attribute" not in df.columns:
            return {
                "status": "invalid",
                "attribute_descriptions": [],
                "source": str(path),
                "message": "The dictionary must contain an 'attribute' column.",
            }

        if "description" not in df.columns:
            df["description"] = ""
        if "data_type" not in df.columns:
            df["data_type"] = "unknown"

        updated_attributes: list[str] = []
        for attribute, value in (updates or {}).items():
            if not isinstance(attribute, str):
                attribute = str(attribute)

            if isinstance(value, dict):
                description = value.get("description")
                data_type = value.get("data_type", value.get("type"))
            else:
                description = value
                data_type = None

            mask = df["attribute"].astype(str) == attribute
            if not mask.any():
                record = {"attribute": attribute, "description": description or "", "data_type": data_type or "unknown"}
                df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
                updated_attributes.append(attribute)
                continue

            if description is not None:
                df.loc[mask, "description"] = str(description)
            if data_type is not None:
                df.loc[mask, "data_type"] = str(data_type)
            updated_attributes.append(attribute)

        output_file = Path(output_path) if output_path else path
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)

        return {
            "status": "updated",
            "attribute_descriptions": [
                {"column_name": str(row["attribute"]), "description": row["description"], "data_type": row["data_type"]}
                for _, row in df.iterrows()
                if str(row["attribute"]) in {str(attr) for attr in updated_attributes}
            ],
            "source": str(output_file),
            "updated_attributes": updated_attributes,
        }

    def create_from_directory(
        self,
        dataset_dir: str | Path,
        *,
        output_path: str | Path | None = None,
        dictionary_path: str | Path | None = None,
        use_llm: bool = True,
    ) -> dict[str, Any]:
        if dictionary_path is not None:
            result = self._load_dictionary_csv(dictionary_path)
            if result["status"] != "has_attribute_descriptions":
                return result

            output_file = Path(output_path) if output_path else (self.output_path or Path(dataset_dir) / "data_dictionary.csv")
            output_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(result["attribute_descriptions"])[["column_name", "description"]].to_csv(output_file, index=False)
            result["output_path"] = str(output_file)
            return result

        if not use_llm:
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": None,
                "message": "No data dictionary source was provided. Choose a supplied dictionary file or enable LLM extraction.",
            }

        result = self.detect_attribute_descriptions(dataset_dir)
        if result["status"] != "has_attribute_descriptions":
            base_message = result.get("message") or "Could not create the data dictionary using the LLM."
            result["message"] = (
                "Could not create the data dictionary using the LLM. "
                f"{base_message}"
                if "Could not create the data dictionary using the LLM" not in base_message
                else base_message
            )
            return result

        df = pd.DataFrame(result["attribute_descriptions"])
        if "column_name" not in df.columns:
            return {
                "status": "no_attribute_descriptions",
                "attribute_descriptions": [],
                "source": result.get("source"),
                "message": "No attribute descriptions could be mapped to dataset columns.",
            }

        output_file = Path(output_path) if output_path else (self.output_path or Path(dataset_dir) / "data_dictionary.csv")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df[["column_name", "description"]].to_csv(output_file, index=False)
        result["output_path"] = str(output_file)
        return result
