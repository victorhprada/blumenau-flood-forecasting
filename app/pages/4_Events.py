"""Página 4 — Eventos Históricos.

Hidrogramas dos eventos de 2008 e 2011: baseline vs melhor modelo.
Análise quantitativa da subestimação do pico de 2011.

UX rationale: depois de entender "as métricas gerais melhoram" (pág. 3), é hora de
mostrar "o que isso significa na prática". Hidrogramas de eventos extremos são
a prova mais honesta de um modelo de cheia.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.utils.charts import hydrograph
from app.utils.data_loader import event_slice, load_timeseries

st.set_page_config(page_title="Eventos Históricos", page_icon="🌧️", layout="wide")

st.title("🌧️ Eventos Históricos — 2008 e 2011")

# ── Event selector ────────────────────────────────────────────────────────────
event = st.radio(
    "Selecione o evento",
    ["Novembro 2008", "Setembro 2011"],
    horizontal=True,
)

lead = st.select_slider(
    "Horizonte de previsão",
    options=[0, 1, 2, 3, 4, 5, 6, 7],
    value=0,
    format_func=lambda x: f"t+{x} ({x} dias à frente)",
)

if event == "Novembro 2008":
    start, end = "2008-11-01", "2008-12-15"
    title = "Cheia de Novembro/2008 — Blumenau"
else:
    start, end = "2011-09-01", "2011-09-30"
    title = "Cheia de Setembro/2011 — Blumenau (pico histórico)"

# ── Load data ─────────────────────────────────────────────────────────────────
try:
    df = event_slice(start, end, sim_label="+ NSELoss ★", time_step=lead)
except Exception as e:
    st.error(f"Erro ao carregar dados: {e}")
    st.info(
        "Certifique-se de que os arquivos zarr de teste foram gerados "
        "(execute `scripts/run_experiment.py` para o exp03)."
    )
    st.stop()

# Load precipitation for the subplot
ts = load_timeseries()
precip = ts.get("total_precipitation", pd.Series(dtype=float)).loc[start:end]

fig = hydrograph(
    obs=df["obs"] if "obs" in df.columns else None,
    best_sim=df["best_sim"] if "best_sim" in df.columns else None,
    baseline_sim=df["baseline_sim"] if "baseline_sim" in df.columns else None,
    precip=precip if len(precip) > 0 else None,
    title=f"{title} — horizonte t+{lead}",
    height=500,
)
st.plotly_chart(fig, use_container_width=True, theme=None)

# ── Metrics for the event ─────────────────────────────────────────────────────
if "obs" in df.columns and "best_sim" in df.columns:
    obs_peak = df["obs"].max()
    sim_peak = df["best_sim"].max()
    base_peak = df["baseline_sim"].max() if "baseline_sim" in df.columns else None

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pico observado", f"{obs_peak:.0f} m³/s")
    c2.metric(
        "Pico simulado (★)",
        f"{sim_peak:.0f} m³/s",
        delta=f"{sim_peak - obs_peak:.0f} m³/s",
        delta_color="inverse",
    )
    if base_peak:
        c3.metric("Pico simulado (Baseline)", f"{base_peak:.0f} m³/s")
    c4.metric(
        "Subestimação (%)",
        f"{100*(obs_peak - sim_peak)/obs_peak:.1f}%",
        delta_color="inverse",
    )

# ── 2011 diagnostic ───────────────────────────────────────────────────────────
if "2011" in event:
    st.divider()
    st.subheader("Por que o modelo subestima o pico de setembro/2011?")

    st.markdown(
        """
O pico observado em 11/09/2011 foi **2.534 m³/s** — o maior registro digital da estação.
O melhor modelo simulou apenas **~737 m³/s** (subestimação de ~71%). As principais causas:
"""
    )

    with st.expander("1. Extrapolação fora do domínio de treinamento"):
        st.markdown(
            """
O conjunto de treino (1996–2005) não contém nenhum evento com pico acima de ~600 m³/s.
A LSTM aprende a **interpolar** bem dentro dos padrões vistos, mas não a **extrapolar**
para magnitudes sem precedente. Sem amostras de eventos >1000 m³/s no treino, o modelo
não tem gradientes que o "ensinem" a prever esses valores.
"""
        )

    with st.expander("2. Regressão à média em redes recorrentes"):
        st.markdown(
            """
LSTMs tendem a regredir à média para eventos raros. O mecanismo de forget gate da LSTM
"esquece" estados extremos gradualmente durante o backpropagation. Isso é deliberado para
generalização, mas tem o efeito colateral de suavizar os picos.
"""
        )

    with st.expander("3. CHIRPS subestima precipitação convectiva intensa"):
        st.markdown(
            """
A precipitação CHIRPS é baseada em krigagem de estações meteorológicas e satélite.
Em eventos convectivos de alta intensidade como setembro/2011, o CHIRPS pode
subestimar a chuva real em 20–50% em subcélulas menores que a resolução (~5 km).
Sem o sinal de entrada correto, o modelo não pode gerar a resposta hidrológica correta.
"""
        )

    with st.expander("4. Saturação do solo — efeito de memória de 30+ dias"):
        st.markdown(
            """
O evento de 2011 ocorreu após semanas de chuvas acima da média. Com `seq_length=14`,
o modelo vê apenas os últimos 14 dias — insuficiente para capturar a saturação
prévia do solo que amplifica a resposta da bacia. Com `seq_length=30+` e dados de
umidade do solo (ERA5-Land), esse efeito poderia ser parcialmente capturado.
"""
        )

    st.info(
        "**Implicação prática**: para alertas de cheia extrema, o modelo atual "
        "sub-estima o risco. Uma abordagem complementar seria usar FDC (curva de "
        "permanência) para calibrar o output em percentis extremos, ou treinar um "
        "modelo CMAL com distribuição assimétrica (heavy tail)."
    )
