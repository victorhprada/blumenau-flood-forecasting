"""FastAPI endpoint for 7-day flood forecast — Itajaí-Açu / Blumenau.

Routes:
  POST /forecast  — returns 8-step discharge forecast (t+0 … t+7)
  GET  /health    — liveness check + model version info
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from .model_loader import get_model_bundle
from .inference import run_forecast

_startup_time = time.time()


@asynccontextmanager
async def _lifespan(app: FastAPI):
    # Load model on startup so the first /forecast request isn't slow
    get_model_bundle()
    yield


app = FastAPI(
    title="Itajaí-Açu Flood Forecast API",
    version="1.0.0",
    description=(
        "MEF-LSTM model for 7-day ahead daily discharge forecasts "
        "at the Blumenau gauge (ANA 83500000)."
    ),
    lifespan=_lifespan,
)


# ── Request / Response schemas ────────────────────────────────────────────────

class ForecastRequest(BaseModel):
    precip_history: list[float] = Field(
        ...,
        min_length=14,
        description=(
            "Daily precipitation [mm/day], oldest first. "
            "Minimum 14 values; 20+ recommended to fill the forecast overlap window."
        ),
    )
    streamflow_history: list[float] = Field(
        ...,
        min_length=21,
        description=(
            "Daily streamflow observations [m³/s], oldest first. "
            "Must cover at least 21 days for the lag-7 autoregressive features."
        ),
    )
    precip_forecast_7d: Optional[list[float]] = Field(
        default=None,
        description=(
            "Optional 7-day NWP precipitation forecast [mm/day]. "
            "Stored in the response for display; the current model uses "
            "historical precipitation only."
        ),
    )
    issue_date: Optional[str] = Field(
        default=None,
        description="ISO date string (YYYY-MM-DD) for labelling the forecast.",
    )

    @field_validator("precip_history", "streamflow_history", mode="before")
    @classmethod
    def _no_negatives(cls, v: list[float]) -> list[float]:
        if any(x < 0 for x in v):
            raise ValueError("Hydrological values must be non-negative.")
        return v


class ForecastResponse(BaseModel):
    lead_days: list[int]          = Field(description="Lead time in days: 0 (nowcast) … 7")
    discharge_m3s: list[float]    = Field(description="Predicted discharge [m³/s]")
    issue_date: Optional[str]     = Field(default=None)
    precip_forecast_7d: Optional[list[float]] = Field(default=None)
    alert_level_m3s: float        = Field(default=1200.0, description="Alert threshold for Blumenau")


class HealthResponse(BaseModel):
    status: str
    model: str
    experiment: str
    epoch: int
    uptime_seconds: float


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    bundle = get_model_bundle()
    return HealthResponse(
        status="ok",
        model="MEF-LSTM (mean_embedding_forecast_lstm)",
        experiment="exp03_nse_loss — NSE t+7=0.474",
        epoch=bundle.epoch,
        uptime_seconds=round(time.time() - _startup_time, 1),
    )


@app.post("/forecast", response_model=ForecastResponse, tags=["forecast"])
def forecast(req: ForecastRequest):
    try:
        bundle = get_model_bundle()
        result = run_forecast(
            bundle=bundle,
            precip_history=req.precip_history,
            streamflow_history=req.streamflow_history,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ForecastResponse(
        lead_days=result["lead_times"],
        discharge_m3s=result["discharge_m3s"],
        issue_date=req.issue_date,
        precip_forecast_7d=req.precip_forecast_7d,
    )
