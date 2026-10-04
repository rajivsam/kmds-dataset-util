"""Public API for the KMDS dataset utilities."""

from data_prep.count_timeseries import (
    DEFAULT_HOME_BASE,
    DEFAULT_TAU_BLOCK,
    ENTERPRISE_WINDOWS,
    WINDOWS,
    acceptable_over_dispersed_window_size,
    align_to_integer_bin_grid,
    bin_time_series_for_window,
    build_arrival_intensity_series,
    expected_poisson_zeros,
    fano_acceptance_interval,
    find_optimal_time_bucket,
    optimize_arrival_binning,
    plot_window_fano_factors,
    select_count_binning_window,
    select_largest_window_in_fano_acceptance_interval,
    summarize_integer_alignment,
    tseda_dataset_export,
)

__all__ = [
    "ENTERPRISE_WINDOWS",
    "WINDOWS",
    "DEFAULT_HOME_BASE",
    "DEFAULT_TAU_BLOCK",
    "acceptable_over_dispersed_window_size",
    "align_to_integer_bin_grid",
    "bin_time_series_for_window",
    "build_arrival_intensity_series",
    "expected_poisson_zeros",
    "fano_acceptance_interval",
    "find_optimal_time_bucket",
    "optimize_arrival_binning",
    "plot_window_fano_factors",
    "select_count_binning_window",
    "select_largest_window_in_fano_acceptance_interval",
    "summarize_integer_alignment",
    "tseda_dataset_export",
]

__version__ = "0.1.0"
