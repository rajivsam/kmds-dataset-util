"""Dataset preparation utilities for KMDS."""

from data_prep.llm import BaseLLM, DatasetDictionaryBuilder, OllamaLLM
from data_prep.timeseries import (
    ENTERPRISE_WINDOWS,
    DEFAULT_HOME_BASE,
    DEFAULT_MIN_EVENTS,
    DEFAULT_TARGET_PASS_RATE,
    WINDOWS,
    build_arrival_intensity_series,
    evaluate_count_density,
    find_optimal_time_bucket,
    select_count_binning_window,
)

__all__ = [
    "__version__",
    "BaseLLM",
    "OllamaLLM",
    "DatasetDictionaryBuilder",
    "ENTERPRISE_WINDOWS",
    "WINDOWS",
    "DEFAULT_MIN_EVENTS",
    "DEFAULT_TARGET_PASS_RATE",
    "DEFAULT_HOME_BASE",
    "evaluate_count_density",
    "find_optimal_time_bucket",
    "select_count_binning_window",
    "build_arrival_intensity_series",
]

__version__ = "0.1.0"
