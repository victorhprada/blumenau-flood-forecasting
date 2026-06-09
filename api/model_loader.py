"""Singleton model loader for MEF-LSTM inference.

Loads config, scaler, and checkpoint once and caches them for reuse.
Thread-safe via a module-level lock.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import xarray as xr
import yaml

# Root is one level above api/
ROOT = Path(__file__).parent.parent

BEST_RUN_DIR = (
    ROOT / "models/experiments/exp03_nse_loss/exp03_nse_loss_0906_112336"
)
CONFIG_PATH = ROOT / "configs/training/experiments/03_nse_loss.yml"
CHECKPOINT = BEST_RUN_DIR / "model_epoch030.pt"
SCALER_PATH = BEST_RUN_DIR / "scaler.nc"


@dataclass
class ScalerParams:
    # Normalization: norm = (x - center) / scale
    # Denormalization: x = norm * scale + center
    precip_center: float
    precip_scale: float
    flow_center: float
    flow_scale: float
    lag1_center: float
    lag1_scale: float
    lag7_center: float
    lag7_scale: float


@dataclass
class ModelBundle:
    model: torch.nn.Module
    scaler: ScalerParams
    run_dir: Path
    config_path: Path
    epoch: int = 30


_bundle: ModelBundle | None = None
_lock = threading.Lock()


def get_model_bundle() -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle
    with _lock:
        if _bundle is None:
            _bundle = _load_bundle()
    return _bundle


def _load_bundle() -> ModelBundle:
    import sys
    import yaml

    sys.path.insert(0, str(ROOT / "vendor/flood-forecasting"))

    from googlehydrology.modelzoo import get_model
    from googlehydrology.utils.config import Config

    # Override run_dir to the timestamped subdirectory where scaler.nc lives.
    # The YAML has the base dir; the framework creates a timestamped subdir at training time.
    with open(CONFIG_PATH) as f:
        raw = yaml.safe_load(f)
    raw["run_dir"] = str(BEST_RUN_DIR)

    cfg = Config(raw)
    model = get_model(cfg)
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    scaler = _load_scaler(SCALER_PATH)

    return ModelBundle(
        model=model,
        scaler=scaler,
        run_dir=BEST_RUN_DIR,
        config_path=CONFIG_PATH,
    )


def _load_scaler(path: Path) -> ScalerParams:
    ds = xr.open_dataset(path)
    # params dim: [center, scale, mean, std] (indices 0, 1, 2, 3)

    def _get(name: str) -> tuple[float, float]:
        v = ds[name].values.astype(float)
        return float(v[0]), float(v[1])  # center, scale

    precip_center, precip_scale   = _get("total_precipitation")
    flow_center, flow_scale       = _get("streamflow")
    lag1_center, lag1_scale       = _get("streamflow_lag1")
    lag7_center, lag7_scale       = _get("streamflow_lag7")

    return ScalerParams(
        precip_center=precip_center,
        precip_scale=precip_scale,
        flow_center=flow_center,
        flow_scale=flow_scale,
        lag1_center=lag1_center,
        lag1_scale=lag1_scale,
        lag7_center=lag7_center,
        lag7_scale=lag7_scale,
    )
