"""Computa RMSE por lead-time (t+0 … t+7) do exp03_nse_loss sobre o período de teste.

Resultado salvo em reports/forecast_rmse.json — versionado no repo e carregado
pela página Forecast para exibir bandas de incerteza ±RMSE.

Uso:
    python scripts/compute_forecast_rmse.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

ZARR_PATH = (
    ROOT
    / "models/experiments/exp03_nse_loss"
    / "exp03_nse_loss_0906_112336"
    / "test/model_epoch030"
    / "test_results.zarr"
)
OUT_PATH = ROOT / "reports" / "forecast_rmse.json"
BASIN = "itajai_83500000"


def main() -> None:
    import xarray as xr

    if not ZARR_PATH.exists():
        print(f"Zarr não encontrado: {ZARR_PATH}")
        sys.exit(1)

    ds = xr.open_zarr(str(ZARR_PATH), consolidated=False)

    rmse_per_lead: list[float] = []
    for ts in range(8):
        obs = (
            ds["streamflow_obs"]
            .sel(basin=BASIN, time_step=ts)
            .to_series()
            .dropna()
            .reset_index(level="freq", drop=True)
        )
        sim = (
            ds["streamflow_sim"]
            .sel(basin=BASIN, time_step=ts)
            .to_series()
            .dropna()
            .reset_index(level="freq", drop=True)
        )
        aligned = obs.align(sim, join="inner")
        rmse = float(np.sqrt(np.mean((aligned[0].values - aligned[1].values) ** 2)))
        rmse_per_lead.append(round(rmse, 2))
        print(f"  t+{ts}: RMSE = {rmse:.2f} m³/s  (n={len(aligned[0])})")

    result = {
        "model": "exp03_nse_loss",
        "metric": "RMSE (m³/s)",
        "period": "test 2008–2024",
        "lead_days": list(range(8)),
        "rmse_m3s": rmse_per_lead,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2))
    print(f"\nSalvo: {OUT_PATH}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
