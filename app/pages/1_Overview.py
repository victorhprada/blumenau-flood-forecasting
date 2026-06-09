"""Página 1 — Visão Geral.

Contexto histórico, mapa interativo da bacia, top-3 enchentes e stack tecnológico.

UX rationale: esta é a "porta de entrada" — deve responder "onde estamos, por quê
importa, como funciona" antes de mostrar qualquer dado técnico. A ordem de páginas
segue a pirâmide invertida do jornalismo: contexto → dados históricos → metodologia
→ eventos extremos → ferramenta interativa.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.data_loader import load_timeseries, top_flood_events

st.set_page_config(
    page_title="Visão Geral — Blumenau",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 Previsão de Cheias — Bacia do Itajaí-Açu")
st.caption("Estação Blumenau · ANA 83500000 · Horizonte 7 dias")

# ── Intro ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
Blumenau sofre enchentes catastróficas desde o século XIX.
A bacia do **Itajaí-Açu** (~15.000 km²) drena o Vale do Itajaí e desemboca no oceano
próximo a Itajaí — com tempo de resposta de 2 a 5 dias. Eventos como **novembro de
2008** e **setembro de 2011** causaram dezenas de mortes e danos bilionários.

Este projeto treina um modelo de aprendizado profundo — **MEF-LSTM** (*Mean-Embedding
Forecast LSTM*) do Google Hydrology — para prever a vazão diária com 7 dias de antecedência,
integrando dados de precipitação CHIRPS e atributos ERA5.
"""
)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Área da bacia", "~15.000 km²")
with col2:
    st.metric("Período de dados", "1996 – 2024")
with col3:
    st.metric("Melhor NSE (t+7)", "0.474 ★")

st.divider()

# ── Map ───────────────────────────────────────────────────────────────────────
st.subheader("Localização da bacia")

try:
    import folium
    from streamlit_folium import st_folium

    m = folium.Map(location=[-27.05, -49.40], zoom_start=8, tiles="CartoDB positron")
    folium.Marker(
        location=[-26.9189, -49.0661],
        tooltip="Estação Blumenau (83500000)",
        popup="ANA 83500000 — Blumenau",
        icon=folium.Icon(color="blue", icon="tint", prefix="fa"),
    ).add_to(m)
    folium.Circle(
        location=[-27.05, -49.40],
        radius=55_000,
        color="#1f77b4",
        fill=True,
        fill_opacity=0.1,
        tooltip="Centróide aproximado da bacia",
    ).add_to(m)
    st_folium(m, height=380, use_container_width=True)

except ImportError:
    st.info(
        "Instale `folium` e `streamlit-folium` para ver o mapa interativo.\n"
        "Coordenadas: Blumenau −26.92 S, −49.07 W"
    )

st.divider()

# ── Top 3 floods ──────────────────────────────────────────────────────────────
st.subheader("Maiores picos históricos")

col_flood, col_events = st.columns([1, 1])

with col_flood:
    df = load_timeseries()
    if "streamflow" in df.columns:
        top = (
            df["streamflow"]
            .nlargest(3)
            .reset_index()
        )
        top.columns = ["Data", "Pico (m³/s)"]
        top["Data"] = top["Data"].dt.strftime("%d/%m/%Y")
        top["Pico (m³/s)"] = top["Pico (m³/s)"].round(0).astype(int)
        st.dataframe(top, hide_index=True, use_container_width=True)

with col_events:
    st.markdown(
        """
| Evento | Pico observado | Característica |
|--------|---------------|----------------|
| Set/2011 | **2.534 m³/s** | Maior registro histórico digital |
| Nov/2008 | ~1.800 m³/s | Deslizamentos em massa |
| Jul/1983 | ~3.800 m³/s* | Maior do século, sem dados digitais |

*estimativa por régua limnimétrica manual
"""
    )

st.divider()

# ── Stack ─────────────────────────────────────────────────────────────────────
st.subheader("Stack tecnológico")

cols = st.columns(4)
stack = [
    ("🔬 Modelo", "MEF-LSTM\nGoogle Hydrology"),
    ("📡 Dados", "CHIRPS precip\nANA streamflow\nNASA POWER PET"),
    ("🛠️ Framework", "PyTorch · FastAPI\nStreamlit · MLflow"),
    ("📊 Métricas", "NSE · KGE · PBIAS\nt+0 e t+7 separados"),
]
for col, (icon_title, text) in zip(cols, stack):
    with col:
        st.info(f"**{icon_title}**\n\n{text}")
