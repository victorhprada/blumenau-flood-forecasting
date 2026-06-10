"""Página 5 — Previsão ao Vivo.

Sliders para precipitação histórica + botão "Gerar Previsão" → consome FastAPI.

UX rationale: esta é a última página — o usuário já entendeu o contexto (pág. 1),
os dados históricos (pág. 2), os experimentos (pág. 3) e as limitações (pág. 4).
Agora pode brincar com a ferramenta com expectativas calibradas.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.data_loader import load_timeseries, ALERT_LEVEL_M3S
from app.utils.charts import forecast_chart
from app.utils.live_data import fetch_last_21_days

# ── Load per-lead RMSE (computed once from test set, versionado em reports/) ──
_RMSE_PATH = Path(__file__).parent.parent.parent / "reports" / "forecast_rmse.json"
try:
    _rmse_data = json.loads(_RMSE_PATH.read_text())
    FORECAST_RMSE: list[float] | None = _rmse_data["rmse_m3s"]
except Exception:
    FORECAST_RMSE = None

st.set_page_config(page_title="Previsão ao Vivo", page_icon="📡", layout="wide")

st.title("📡 Previsão ao Vivo — 7 dias")

import os as _os
_default_api_url = _os.environ.get(
    "FORECAST_API_URL",
    "http://localhost:8000",  # override via env var FORECAST_API_URL for Railway deploy
)

API_URL = st.sidebar.text_input(
    "URL da API",
    value=_default_api_url,
    help="URL base da FastAPI (sem trailing slash). Configure FORECAST_API_URL para apontar para a instância Railway.",
)

st.sidebar.markdown(
    """
**Como usar:**
1. Ajuste a data de emissão
2. Configure a precipitação dos últimos 21 dias
3. Configure o histórico de vazão (21 dias)
4. Clique em **Gerar Previsão**

A API retorna previsões de t+0 a t+7 em m³/s.
"""
)


# ── Helper function (must be defined before the button call) ─────────────────

def _run_forecast(
    api_url: str,
    precip_history: list[float],
    streamflow_history: list[float],
    issue_date: str,
):
    import requests

    payload = {
        "precip_history": precip_history,
        "streamflow_history": streamflow_history,
        "issue_date": issue_date,
    }

    try:
        with st.spinner("Consultando a API..."):
            resp = requests.post(f"{api_url}/forecast", json=payload, timeout=30)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.ConnectionError:
        st.error(
            f"Não foi possível conectar à API em `{api_url}`. "
            "Verifique se o servidor FastAPI está rodando:\n\n"
            "```bash\ncd /path/to/project\n"
            "source .venv/bin/activate\n"
            "PYTHONPATH=vendor/flood-forecasting:. uvicorn api.main:app --reload\n```"
        )
        return
    except Exception as exc:
        st.error(f"Erro na API: {exc}")
        return

    lead_days = result["lead_days"]
    discharge = result["discharge_m3s"]

    # ── Alert check ───────────────────────────────────────────────────────────
    max_q = max(discharge)
    alert_days = [d for d, q in zip(lead_days, discharge) if q >= ALERT_LEVEL_M3S]
    if alert_days:
        st.warning(
            f"⚠️ **ATENÇÃO**: Previsão acima do pré-alerta ({ALERT_LEVEL_M3S:.0f} m³/s) "
            f"em t+{alert_days[0]} (pico previsto: {max_q:.0f} m³/s)."
        )
    else:
        st.success(f"✅ Sem alertas previstos — pico máximo: {max_q:.0f} m³/s.")

    # ── Forecast chart ────────────────────────────────────────────────────────
    ts_full = load_timeseries()
    ts_obs   = ts_full.get("streamflow", pd.Series(dtype=float))
    ts_precip = ts_full.get("total_precipitation", pd.Series(dtype=float))
    fig = forecast_chart(
        lead_days=lead_days,
        discharge_m3s=discharge,
        issue_date=issue_date,
        alert_level=ALERT_LEVEL_M3S,
        historical_obs=ts_obs,
        historical_precip=ts_precip,
        uncertainty_per_lead=FORECAST_RMSE,
        height=450,
    )
    st.plotly_chart(fig, width="stretch", theme=None)

    # ── Table ─────────────────────────────────────────────────────────────────
    df_result = pd.DataFrame(
        {
            "Horizonte": [f"t+{d}" for d in lead_days],
            "Data prevista": [
                (pd.Timestamp(issue_date) + pd.Timedelta(days=d)).strftime("%d/%m/%Y")
                for d in lead_days
            ],
            "Vazão prevista (m³/s)": [round(q, 1) for q in discharge],
            "Alerta": ["⚠️" if q >= ALERT_LEVEL_M3S else "✅" for q in discharge],
        }
    )
    st.dataframe(df_result, hide_index=True, use_container_width=True)
    st.caption("✅ Abaixo do pré-alerta · ⚠️ Acima do pré-alerta (≥ 1 200 m³/s — Defesa Civil Blumenau)")

    rmse_note = (
        f" · Banda ±RMSE: {FORECAST_RMSE[0]:.0f}–{FORECAST_RMSE[-1]:.0f} m³/s (t+0 a t+7)"
        if FORECAST_RMSE else ""
    )
    st.caption(
        f"Modelo: MEF-LSTM exp03_nse_loss · NSE(t+7)=0.474{rmse_note} · "
        "Limite de alerta: Defesa Civil Blumenau"
    )


# ── Pre-populate from CSV ─────────────────────────────────────────────────────

ts = load_timeseries()

if len(ts) > 0:
    last_date = ts.index[-1]
    _csv_precip = ts["total_precipitation"].dropna().tail(21).tolist()
    _csv_flow = ts["streamflow"].dropna().tail(21).tolist()
    while len(_csv_precip) < 21:
        _csv_precip.insert(0, float(np.nanmean(_csv_precip) if _csv_precip else 4.5))
    while len(_csv_flow) < 21:
        _csv_flow.insert(0, float(np.nanmean(_csv_flow) if _csv_flow else 200.0))
else:
    last_date = pd.Timestamp.today() - pd.Timedelta(days=1)
    _csv_precip = [4.5] * 21
    _csv_flow = [200.0] * 21

issue_date = st.date_input("Data de emissão da previsão", value=last_date.date())

# ── Auto-fetch button ─────────────────────────────────────────────────────────
st.subheader("Dados de entrada")

col_btn, col_status = st.columns([1, 3])
with col_btn:
    fetch_clicked = st.button(
        "🔄 Buscar dados reais (ANA + INMET)",
        use_container_width=True,
        help="Busca os últimos 21 dias de vazão (ANA HidroWeb) e precipitação (INMET). Cache de 1h.",
    )

if fetch_clicked:
    with st.spinner("Buscando dados da ANA e INMET..."):
        live = fetch_last_21_days(issue_date=str(issue_date))

    if live["error"]:
        with col_status:
            st.error(f"Erros durante a busca:\n{live['error']}")
    else:
        with col_status:
            st.success(
                f"✅ Dados carregados — Vazão ANA até {live['streamflow_last_date']} · "
                f"Precipitação: {live['precip_source']} · Atualizado às {live['fetched_at']}"
            )
    st.warning(live["warning"])

    # Write fetched values into session state so sliders pick them up on rerun
    for i in range(21):
        st.session_state[f"p_{i}"] = float(round(live["precip"][i], 1))
        st.session_state[f"q_{i}"] = float(round(live["streamflow"][i], 0))
    st.rerun()

# Determine default values for slider initialisation
# (session state keys already set → Streamlit ignores `value` param automatically)
default_precip = _csv_precip
default_flow = _csv_flow

# ── Precipitation sliders ─────────────────────────────────────────────────────
st.subheader("Precipitação histórica (últimos 21 dias)")
st.caption("Dia 1 = 20 dias atrás; Dia 21 = hoje (data de emissão).")

day_labels = [
    (pd.Timestamp(issue_date) - pd.Timedelta(days=20 - i)).strftime("%d/%m")
    for i in range(21)
]

precip_values = list(default_precip)
with st.expander("Ajustar precipitação dia a dia", expanded=False):
    slider_cols = st.columns(7)
    for i in range(21):
        col = slider_cols[i % 7]
        with col:
            precip_values[i] = st.slider(
                day_labels[i],
                min_value=0.0,
                max_value=150.0,
                value=float(round(default_precip[i], 1)),
                step=0.5,
                key=f"p_{i}",
            )

# ── Streamflow sliders ────────────────────────────────────────────────────────
st.subheader("Histórico de vazão (últimos 21 dias)")
st.caption("Necessário para as features autorregressivas (lag1, lag7) do modelo.")

flow_values = list(default_flow)
with st.expander("Ajustar vazão dia a dia", expanded=False):
    flow_cols = st.columns(7)
    for i in range(21):
        col = flow_cols[i % 7]
        with col:
            flow_values[i] = st.slider(
                day_labels[i],
                min_value=0.0,
                max_value=3000.0,
                value=float(round(default_flow[i], 0)),
                step=10.0,
                key=f"q_{i}",
            )

# ── Current conditions summary ────────────────────────────────────────────────
st.subheader("Condições atuais")
c1, c2, c3 = st.columns(3)
c1.metric("Precipitação média 14d", f"{np.mean(precip_values[-14:]):.1f} mm/dia")
c2.metric("Precipitação acumulada 7d", f"{np.sum(precip_values[-7:]):.1f} mm")
c3.metric("Vazão atual (t=0)", f"{flow_values[-1]:.0f} m³/s")

# ── Forecast button ───────────────────────────────────────────────────────────
st.divider()

if st.button("🚀 Gerar Previsão", type="primary", use_container_width=True):
    _run_forecast(
        api_url=API_URL,
        precip_history=precip_values,
        streamflow_history=flow_values,
        issue_date=str(issue_date),
    )
