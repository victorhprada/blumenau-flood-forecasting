"""Inference pipeline for MEF-LSTM forecast.

Tensor conventions (all arrays sorted oldest→newest):
  precip_history    — 20 values covering [D-19 … D] inclusive (mm/day)
  streamflow_history — 21 values covering [D-20 … D] inclusive (m³/s)

Where D is the issue date (today).

Hindcast window (seq_length=14) covers dates D-13 … D (inclusive):
  - total_precipitation comes from the *forecast* dataset at lead_time=0,
    shifted one day backwards: dates D-14 … D-1 (14 values)
  - streamflow_lag1[k] = streamflow[D-13+k - 1] = q[D-14+k]
  - streamflow_lag7[k] = streamflow[D-13+k - 7] = q[D-20+k]

Forecast overlap tensor (21 values): [p_{D-19} … p_D, p_D] (last element repeated).
"""

from __future__ import annotations

import numpy as np
import torch

from .model_loader import ModelBundle, ScalerParams

SEQ_LENGTH      = 14
FORECAST_OVERLAP = 20
LEAD_TIME       = 7
PREDICT_LAST_N  = 8   # time_step 0 (nowcast) to time_step 7 (+7 days)
N_STATIC        = 6   # all ≈ 0 after mean-centering for single-basin training


def run_forecast(
    bundle: ModelBundle,
    precip_history: list[float],
    streamflow_history: list[float],
) -> dict:
    """Return 8-step discharge forecast [t+0 … t+7] in m³/s.

    Parameters
    ----------
    precip_history :
        ≥14 precipitation values [mm/day], oldest first, ending on issue date D.
        Padded to 20 values if fewer supplied.
    streamflow_history :
        ≥21 streamflow values [m³/s], oldest first, ending on issue date D.
        Padded to 21 values if fewer supplied.
    """
    sc = bundle.scaler

    precip     = _ensure_length(precip_history,     20)  # [D-19 … D]
    streamflow = _ensure_length(streamflow_history,  21)  # [D-20 … D]

    p_raw = np.array(precip,     dtype=np.float32)   # shape (20,)
    q_raw = np.array(streamflow, dtype=np.float32)   # shape (21,)

    p_norm = (p_raw - sc.precip_center) / sc.precip_scale

    # ── Hindcast tensors ──────────────────────────────────────────────────────
    # Precip hindcast: dates D-14 … D-1  (= p_raw[5:19])
    # p_raw[k] = p_{D-19+k}  →  p_{D-14} = p_raw[5], p_{D-1} = p_raw[18]
    hind_precip = p_norm[5:19]                                          # (14,)

    # Lag1: q[D-14] … q[D-1]  (= q_raw[6:20])
    # q_raw[k] = q_{D-20+k}  →  q_{D-14} = q_raw[6], q_{D-1} = q_raw[19]
    hind_lag1 = (q_raw[6:20] - sc.lag1_center) / sc.lag1_scale         # (14,)

    # Lag7: q[D-20] … q[D-7]  (= q_raw[0:14])
    # q_{D-20} = q_raw[0], q_{D-7} = q_raw[13]
    hind_lag7 = (q_raw[0:14] - sc.lag7_center) / sc.lag7_scale         # (14,)

    # ── Forecast overlap tensor: overlap [D-19 … D] + lead_time-0 at D ───────
    # = p_norm[:] (20 values) + [p_norm[-1]] repeated = 21 values
    forecast_precip = np.append(p_norm, p_norm[-1])                     # (21,)

    # ── Build batch (B=1) ────────────────────────────────────────────────────
    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.tensor(arr[:, np.newaxis][np.newaxis], dtype=torch.float32)  # (1,T,1)

    data = {
        "x_s": torch.zeros(1, N_STATIC, dtype=torch.float32),
        "x_d_hindcast": {
            "total_precipitation": _t(hind_precip),
            "streamflow_lag1":     _t(hind_lag1),
            "streamflow_lag7":     _t(hind_lag7),
            "hindcast_counter":    torch.zeros(1, SEQ_LENGTH, 1, dtype=torch.float32),
        },
        "x_d_forecast": {
            "total_precipitation": _t(forecast_precip),
            "forecast_counter":    torch.zeros(1, FORECAST_OVERLAP + LEAD_TIME + 1, 1, dtype=torch.float32),
        },
        "basin_index": np.array(0),
    }

    # ── Forward pass ─────────────────────────────────────────────────────────
    with torch.no_grad():
        out = bundle.model(data)

    y_hat = out["y_hat"] if isinstance(out, dict) else out
    preds_norm = y_hat[0, -PREDICT_LAST_N:, 0].cpu().numpy()            # (8,)

    # Denormalize and clip
    preds_m3s = np.maximum(preds_norm * sc.flow_scale + sc.flow_center, 0.0)

    return {
        "lead_times":    list(range(PREDICT_LAST_N)),
        "discharge_m3s": preds_m3s.tolist(),
    }


def _ensure_length(values: list[float], min_len: int) -> list[float]:
    """Return last min_len values; pad front with first value if shorter."""
    if len(values) >= min_len:
        return list(values[-min_len:])
    pad = [values[0]] * (min_len - len(values))
    return pad + list(values)
