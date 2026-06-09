"""Data loading utilities for the Streamlit app.

All functions return pandas DataFrames / xarray Datasets.
Results are cached with @st.cache_data for performance.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent.parent

TIMESERIES_CSV = (
    ROOT
    / "data/processed/Caravan-nc/timeseries/csv/itajai/itajai_83500000.csv"
)

COMPARISON_CSV = ROOT / "reports/comparison_table.csv"

# Zarr paths for each experiment
ZARR_PATHS = {
    "Baseline (MSE)": ROOT / "models/experiments/mef_lstm_itajai_baseline_0906_083244/test/model_epoch015",
    "+ lag features": ROOT / "models/experiments/exp02_lag_features/exp02_lag_features_0906_111809/test/model_epoch030",
    "+ NSELoss ★":    ROOT / "models/experiments/exp03_nse_loss/exp03_nse_loss_0906_112336/test/model_epoch030",
    "+ seq 30d":      ROOT / "models/experiments/exp04_seq_30d/exp04_seq_30d_0906_112529/test/model_epoch019",
    "+ ERA5 statics": ROOT / "models/experiments/exp05_era5_statics/exp05_era5_statics_0906_114408/test/model_epoch028",
}

ALERT_LEVEL_M3S = 1200.0  # Blumenau civil defence pre-alert threshold


@st.cache_data
def load_timeseries() -> pd.DataFrame:
    """Load daily streamflow + precipitation CSV."""
    df = pd.read_csv(TIMESERIES_CSV, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


@st.cache_data
def load_comparison_table() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_CSV, index_col=0)


@st.cache_data
def load_zarr_simulation(label: str, time_step: int = 0) -> pd.Series:
    """Load simulated streamflow from zarr at a given time_step.

    time_step=0 → nowcast (t+0); time_step=7 → 7-day lead.
    """
    import xarray as xr

    zarr_path = ZARR_PATHS.get(label)
    if zarr_path is None:
        raise ValueError(f"Unknown experiment: {label}")
    ds = xr.open_zarr(str(zarr_path / "test_results.zarr"), consolidated=False)
    sim = ds["streamflow_sim"].sel(basin="itajai_83500000", time_step=time_step)
    return sim.to_series().dropna().reset_index(level="freq", drop=True)


@st.cache_data
def load_observed_streamflow() -> pd.Series:
    """Load observed streamflow from zarr (any run — observations are the same)."""
    import xarray as xr

    zarr_path = list(ZARR_PATHS.values())[1]  # use exp02 (has test period)
    ds = xr.open_zarr(str(zarr_path / "test_results.zarr"), consolidated=False)
    obs = ds["streamflow_obs"].sel(basin="itajai_83500000", time_step=0)
    return obs.to_series().dropna().reset_index(level="freq", drop=True)


def top_flood_events(n: int = 3) -> pd.DataFrame:
    """Return top-n flood peaks from the full timeseries."""
    df = load_timeseries()
    if "streamflow" not in df.columns:
        return pd.DataFrame()
    peaks = (
        df["streamflow"]
        .resample("YE")
        .max()
        .nlargest(n)
        .reset_index()
    )
    peaks.columns = ["year", "peak_m3s"]
    peaks["year"] = peaks["year"].dt.year
    return peaks


def event_slice(
    start: str, end: str, sim_label: str = "+ NSELoss ★", time_step: int = 0
) -> pd.DataFrame:
    """Return DataFrame with obs + sim for a given event window."""
    obs = load_observed_streamflow()
    sim = load_zarr_simulation(sim_label, time_step=time_step)

    baseline_sim = load_zarr_simulation("Baseline (MSE)", time_step=time_step)

    df = pd.DataFrame(
        {
            "obs": obs,
            "best_sim": sim,
            "baseline_sim": baseline_sim,
        }
    )
    return df.loc[start:end]
