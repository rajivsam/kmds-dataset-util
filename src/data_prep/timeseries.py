"""Compatibility wrapper for the count-based time-series module.

The canonical API is now exposed via ``data_prep.count_timeseries``. This module remains as a
backward-compatible import path for older callers.
"""

from data_prep.count_timeseries import *  # noqa: F401,F403
