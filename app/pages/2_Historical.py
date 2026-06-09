"""Página 2 — Série Histórica.

Série temporal interativa 1996–2024 com slider de período e destaque de extremos.

UX rationale: ANTES de mostrar métricas do modelo, o usuário precisa ter uma intuição
do dado que estamos modelando. A série histórica completa dá contexto para "o que é
um NSE de 0.47" — mostrando a variabilidade que o modelo precisa capturar.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.data_loader import load_timeseries

st.set_page_config(page_title="Série Histórica", page_icon="📈", layout="wide")

st.title("📈 Série Histórica — Blumenau (1996–2024)")

df = load_timeseries()
if "streamflow" not in df.columns:
    st.error("Coluna `streamflow` não encontrada no CSV.")
    st.stop()

# ── Period slider ─────────────────────────────────────────────────────────────
years = list(range(df.index.year.min(), df.index.year.max() + 1))
col1, col2 = st.columns(2)
with col1:
    year_start = st.selectbox("Ano inicial", years, index=0)
with col2:
    year_end = st.selectbox(
        "Ano final", years, index=len(years) - 1
    )

if year_end < year_start:
    st.warning("Ano final deve ser maior ou igual ao ano inicial.")
    st.stop()

mask = (df.index.year >= year_start) & (df.index.year <= year_end)
df_view = df.loc[mask]

# ── Highlight extreme events ──────────────────────────────────────────────────
EVENTS = {
    "Nov/2008": ("2008-11-20", "2008-12-05"),
    "Set/2011": ("2011-09-05", "2011-09-20"),
    "Mar/2022": ("2022-03-25", "2022-04-10"),
}

fig = go.Figure()

# Precipitation as background bars (if available)
if "total_precipitation" in df_view.columns:
    fig.add_trace(
        go.Bar(
            x=df_view.index,
            y=df_view["total_precipitation"],
            name="Precipitação (mm/d)",
            marker_color="#9ecae1",
            opacity=0.5,
            yaxis="y2",
        )
    )

# Main streamflow line
fig.add_trace(
    go.Scatter(
        x=df_view.index,
        y=df_view["streamflow"],
        name="Vazão observada (m³/s)",
        line=dict(color="#1f77b4", width=1.2),
        mode="lines",
    )
)

# Alert threshold
fig.add_hline(
    y=1200,
    line=dict(color="#ff7f0e", width=1.5, dash="dash"),
    annotation_text="Pré-alerta 1200 m³/s",
    annotation_position="top left",
)

# Highlight flood events that fall in the visible range
for label, (start, end) in EVENTS.items():
    s = max(df_view.index.min(), df_view.index[df_view.index >= start][0] if any(df_view.index >= start) else df_view.index.min())
    e_idx = df_view.index[df_view.index <= end]
    if len(e_idx) == 0:
        continue
    e = e_idx[-1]
    fig.add_vrect(
        x0=start, x1=end,
        fillcolor="#d62728", opacity=0.1, line_width=0,
        annotation_text=label, annotation_position="top left",
    )

fig.update_layout(
    yaxis=dict(title="Vazão (m³/s)", showgrid=True, gridcolor="#333333", color="#aaaaaa"),
    yaxis2=dict(title="Precip. (mm/d)", overlaying="y", side="right", autorange="reversed", showgrid=False, color="#6baed6", tickfont=dict(color="#6baed6"), title_font=dict(color="#6baed6")),
    xaxis=dict(showgrid=True, gridcolor="#333333", color="#aaaaaa"),
    hovermode="x unified",
    height=500,
    legend=dict(orientation="h", y=-0.12, font=dict(color="#aaaaaa")),
    margin=dict(l=60, r=60, t=30, b=70),
    plot_bgcolor="#111111",
    paper_bgcolor="#111111",
    font=dict(color="#aaaaaa"),
)

st.plotly_chart(fig, use_container_width=True, theme=None)

# ── Summary stats ─────────────────────────────────────────────────────────────
st.subheader("Estatísticas do período selecionado")
q = df_view["streamflow"].dropna()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Média (m³/s)", f"{q.mean():.1f}")
c2.metric("Mediana (m³/s)", f"{q.median():.1f}")
c3.metric("Pico máximo (m³/s)", f"{q.max():.0f}")
c4.metric("Dias > pré-alerta", f"{(q > 1200).sum()}")

if "total_precipitation" in df_view.columns:
    p = df_view["total_precipitation"].dropna()
    st.caption(
        f"Precipitação média no período: {p.mean():.2f} mm/dia "
        f"· Máxima diária: {p.max():.1f} mm/dia"
    )
