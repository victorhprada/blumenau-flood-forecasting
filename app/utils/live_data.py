"""Busca de dados em tempo real — ANA (streamflow) + INMET (precipitação).

Nota de viés de distribuição:
    O modelo MEF-LSTM foi treinado com precipitação CHIRPS (grade 0.05°, média areal
    da bacia). O INMET é pontual e tende a sub ou superestimar eventos convectivos
    intensos em relação ao CHIRPS. O UI exibe aviso explícito sobre essa limitação.

Decisão de design:
    Sem escrita em disco — todos os dados ficam em memória e cacheados via
    @st.cache_data(ttl=3600) para não sobrecarregar as APIs externas.
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# INMET stations inside the Itajaí-Açu basin (operante, validated 2026-06)
# Ordered by proximity to Blumenau gauge 83500000 (~-26.92°, -49.07°)
INMET_STATIONS = [
    {"id": "A868", "name": "Itajaí", "lat": -26.95, "lon": -48.76},
    {"id": "A861", "name": "Rio do Campo", "lat": -26.94, "lon": -50.15},
    {"id": "A863", "name": "Ituporanga", "lat": -27.42, "lon": -49.65},
]

ANA_STATION = "83500000"


def _fetch_inmet_daily(station_id: str, start: str, end: str) -> pd.Series:
    """Busca precipitação diária de uma estação INMET em memória (sem disco)."""
    import requests

    base = "https://apitempo.inmet.gov.br"
    # API INMET: GET /estacao/dados/{cod}/{ini}/{fim}  (formato YYYY-MM-DD)
    url = f"{base}/estacao/dados/{station_id}/{start}/{end}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()
    if not data:
        return pd.Series(dtype=float)

    df = pd.DataFrame(data)
    precip_col = next((c for c in df.columns if "CHUVA" in c.upper()), None)
    dt_col = next((c for c in df.columns if "DT_MEDICAO" in c.upper() or "DATETIME" in c.upper()), None)
    if not precip_col or not dt_col:
        return pd.Series(dtype=float)

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")
    df = df.set_index(dt_col).sort_index()

    # Aggregate hourly → daily (hydrological day 07:00–07:00)
    daily = (
        df[precip_col]
        .resample("D", offset="7h")
        .sum(min_count=12)
    )
    daily.index = daily.index.normalize()
    return daily.rename("precip_mm")


def _fetch_ana_streamflow_last_n(days: int = 30) -> pd.Series:
    """Busca vazão diária recente da ANA (apenas últimos N dias)."""
    from src.data.ana_downloader import download_series

    today = date.today()
    # Fetch 2 years to ensure the last N days are available (ANA may lag ~1 week)
    start_year = today.year - 1
    df = download_series(
        station=ANA_STATION,
        variable="vazao",
        start_year=start_year,
        end_year=today.year,
    )
    return df["value"].dropna()


def fetch_last_21_days(
    issue_date: str | date | None = None,
    n_days: int = 21,
) -> dict:
    """Busca os últimos 21 dias de precipitação (INMET) e vazão (ANA).

    Retorna dict com:
        precip      : list[float]  — 21 valores diários, mais recente por último
        streamflow  : list[float]  — 21 valores diários
        precip_source : str        — estações INMET usadas
        fetched_at  : str          — timestamp da busca (DD/MM/YYYY HH:MM)
        streamflow_last_date : str — data do dado de vazão mais recente
        warning     : str          — aviso sobre viés CHIRPS/INMET
        error       : str | None   — mensagem de erro se falha parcial
    """
    import streamlit as st

    @st.cache_data(ttl=3600)
    def _cached(issue_date_str: str) -> dict:
        from datetime import datetime

        end_dt = pd.Timestamp(issue_date_str)
        start_dt = end_dt - pd.Timedelta(days=n_days + 14)  # extra buffer

        errors: list[str] = []

        # ── Precipitação INMET ─────────────────────────────────────────────────
        precip_series_list: list[pd.Series] = []
        used_stations: list[str] = []

        for station in INMET_STATIONS:
            try:
                s = _fetch_inmet_daily(
                    station["id"],
                    start_dt.strftime("%Y-%m-%d"),
                    end_dt.strftime("%Y-%m-%d"),
                )
                if not s.empty and s.notna().sum() >= 7:
                    precip_series_list.append(s)
                    used_stations.append(f"{station['id']} ({station['name']})")
            except Exception as exc:
                errors.append(f"INMET {station['id']}: {exc}")

        if precip_series_list:
            precip_df = pd.concat(precip_series_list, axis=1).mean(axis=1)
            precip_df = precip_df.loc[:end_dt]
            precip_raw = precip_df.reindex(
                pd.date_range(end_dt - pd.Timedelta(days=n_days - 1), end_dt, freq="D")
            ).fillna(0.0)
            precip_list = [round(float(v), 2) for v in precip_raw.values]
            precip_source = "INMET: " + ", ".join(used_stations)
        else:
            precip_list = [0.0] * n_days
            precip_source = "N/A — todas as estações INMET falharam"
            errors.append("Nenhuma estação INMET respondeu — usando zeros para precipitação.")

        # ── Vazão ANA ─────────────────────────────────────────────────────────
        try:
            q_series = _fetch_ana_streamflow_last_n(days=60)
            q_series = q_series.loc[:end_dt]
            q_reindexed = q_series.reindex(
                pd.date_range(end_dt - pd.Timedelta(days=n_days - 1), end_dt, freq="D")
            )
            # Forward-fill gaps up to 3 days (ANA consistency lag)
            q_reindexed = q_reindexed.ffill(limit=3)
            # Remaining NaN → fill with series mean
            fallback_q = float(q_series.tail(30).mean()) if not q_series.empty else 200.0
            q_reindexed = q_reindexed.fillna(fallback_q)
            streamflow_list = [round(float(v), 1) for v in q_reindexed.values]
            streamflow_last_date = (
                q_series.index[-1].strftime("%d/%m/%Y") if not q_series.empty else "N/A"
            )
        except Exception as exc:
            streamflow_list = [200.0] * n_days
            streamflow_last_date = "N/A"
            errors.append(f"ANA: {exc}")

        warning = (
            "⚠️ **Atenção — desvio de distribuição**: o modelo foi treinado com precipitação "
            "CHIRPS (grade espacial, média areal da bacia). O INMET usa estações pontuais "
            "e pode sub ou superestimar eventos convectivos intensos em relação ao CHIRPS. "
            "Para uso operacional, recomenda-se recalibração do modelo com INMET como input."
        )

        return {
            "precip": precip_list,
            "streamflow": streamflow_list,
            "precip_source": precip_source,
            "streamflow_last_date": streamflow_last_date,
            "fetched_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
            "warning": warning,
            "error": "\n".join(errors) if errors else None,
        }

    issue_str = (
        str(issue_date) if issue_date else date.today().isoformat()
    )
    return _cached(issue_str)
