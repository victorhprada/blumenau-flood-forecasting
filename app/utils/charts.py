"""Reusable Plotly chart builders for the Streamlit app."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "obs":      "#1f77b4",
    "best":     "#d62728",
    "baseline": "#aec7e8",
    "precip":   "#9ecae1",
    "alert":    "#ff7f0e",
    "forecast": "#2ca02c",
}

ALERT_LEVEL = 1200.0  # m³/s


def hydrograph(
    obs: pd.Series | None = None,
    best_sim: pd.Series | None = None,
    baseline_sim: pd.Series | None = None,
    precip: pd.Series | None = None,
    title: str = "",
    alert_level: float = ALERT_LEVEL,
    height: int = 440,
) -> go.Figure:
    """Dual-axis hydrograph: streamflow on primary axis, precip bars on secondary."""
    rows = 2 if precip is not None else 1
    row_heights = [0.72, 0.28] if rows == 2 else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=row_heights,
        subplot_titles=["", "Precipitação (mm/dia)"] if rows == 2 else [""],
    )

    # ── Observations ──────────────────────────────────────────────────────────
    if obs is not None:
        fig.add_trace(
            go.Scatter(
                x=obs.index, y=obs.values,
                name="Observado",
                line=dict(color=COLORS["obs"], width=1.8),
                mode="lines",
            ),
            row=1, col=1,
        )

    # ── Baseline simulation ───────────────────────────────────────────────────
    if baseline_sim is not None:
        fig.add_trace(
            go.Scatter(
                x=baseline_sim.index, y=baseline_sim.values,
                name="Baseline (MSE)",
                line=dict(color=COLORS["baseline"], width=1.2, dash="dot"),
                mode="lines",
            ),
            row=1, col=1,
        )

    # ── Best model simulation ─────────────────────────────────────────────────
    if best_sim is not None:
        fig.add_trace(
            go.Scatter(
                x=best_sim.index, y=best_sim.values,
                name="+ NSELoss ★",
                line=dict(color=COLORS["best"], width=1.8),
                mode="lines",
            ),
            row=1, col=1,
        )

    # ── Alert level ───────────────────────────────────────────────────────────
    if obs is not None or best_sim is not None:
        ref = obs if obs is not None else best_sim
        fig.add_hline(
            y=alert_level,
            line=dict(color=COLORS["alert"], width=1.5, dash="dash"),
            annotation_text=f"Pré-alerta {alert_level:.0f} m³/s",
            annotation_position="top right",
            row=1, col=1,
        )

    # ── Precipitation bars ────────────────────────────────────────────────────
    if precip is not None and rows == 2:
        fig.add_trace(
            go.Bar(
                x=precip.index, y=precip.values,
                name="Precipitação",
                marker_color=COLORS["precip"],
                opacity=0.75,
            ),
            row=2, col=1,
        )
        fig.update_yaxes(
            title_text="mm/dia", autorange="reversed", row=2, col=1
        )

    fig.update_yaxes(title_text="Vazão (m³/s)", row=1, col=1, gridcolor="#333333", color="#aaaaaa")
    fig.update_xaxes(showgrid=True, gridcolor="#333333", color="#aaaaaa", row=1, col=1)
    fig.update_layout(
        title=title,
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.12, font=dict(color="#aaaaaa")),
        margin=dict(l=60, r=20, t=40, b=60),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#aaaaaa"),
    )
    return fig


def metrics_bar(df: pd.DataFrame, metrics: list[str], title: str = "") -> go.Figure:
    """Grouped bar chart of metrics across experiments."""
    fig = go.Figure()
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    valid = [m for m in metrics if m in df.columns]
    n = len(valid)
    experiments = list(df.index)

    for i, metric in enumerate(valid):
        fig.add_trace(
            go.Bar(
                name=metric,
                x=experiments,
                y=df[metric].tolist(),
                marker_color=palette[i % len(palette)],
            )
        )

    # Add value annotations manually — Plotly.js never auto-hides these
    # For grouped bars: bar groups are at integer positions 0..n_exp-1,
    # each bar within a group is offset by (i - (n-1)/2) * barwidth
    barwidth = 0.8 / n  # Plotly default: bargroupgap=0, barwidth splits evenly
    for i, metric in enumerate(valid):
        offset = (i - (n - 1) / 2) * barwidth
        for j, exp in enumerate(experiments):
            val = df.loc[exp, metric]
            sign = 1 if val >= 0 else -1
            fig.add_annotation(
                x=j + offset,
                y=val,
                text=f"<b>{val:.3f}</b>",
                showarrow=False,
                xanchor="center",
                yanchor="bottom" if val >= 0 else "top",
                yshift=sign * 4,
                font=dict(color="white", size=10, family="monospace"),
            )

    vals = df[valid].values.flatten()
    lo = float(vals.min())
    hi = float(vals.max())
    span = hi - lo or 1.0
    yrange = [lo - 0.20 * span, hi + 0.30 * span]

    fig.update_layout(
        template="plotly_dark",
        title=title,
        barmode="group",
        yaxis_title="Métrica",
        height=420,
        margin=dict(l=60, r=20, t=70, b=150),
        xaxis=dict(tickangle=-30, automargin=True),
        yaxis=dict(range=yrange),
    )
    return fig


def forecast_chart(
    lead_days: list[int],
    discharge_m3s: list[float],
    issue_date: str = "",
    alert_level: float = ALERT_LEVEL,
    historical_obs: pd.Series | None = None,
    historical_precip: pd.Series | None = None,
    uncertainty_per_lead: list[float] | None = None,
    height: int = 440,
) -> go.Figure:
    """Operational hydrograph — NOAA/ECMWF style.

    Layout: 21-day window (14d history + 7d forecast), dual Y-axis
    (streamflow left, precipitation right inverted), alert annotation,
    vertical separator at issue_date.
    """
    try:
        base = pd.Timestamp(issue_date) if issue_date else pd.Timestamp.today()
    except Exception:
        base = pd.Timestamp.today()

    window_start = base - pd.Timedelta(days=14)
    window_end   = base + pd.Timedelta(days=7)
    forecast_dates = [base + pd.Timedelta(days=d) for d in lead_days]

    # ── Anchor to last observed point for visual continuity ───────────────────
    fc_x: list = list(forecast_dates)
    fc_y: list = list(discharge_m3s)
    has_anchor = False
    if historical_obs is not None:
        hist_q_tail = historical_obs.loc[window_start:base].dropna()
        if not hist_q_tail.empty:
            fc_x = [hist_q_tail.index[-1]] + fc_x
            fc_y = [float(hist_q_tail.iloc[-1])] + fc_y
            has_anchor = True

    # ── Y-range: always include the alert line ────────────────────────────────
    hist_max = (
        float(historical_obs.loc[window_start:base].max())
        if historical_obs is not None and not historical_obs.empty
        else 0.0
    )
    fc_max = max(discharge_m3s) if discharge_m3s else 0.0
    rmse_max = max(uncertainty_per_lead) if uncertainty_per_lead else 0.0
    y_max = max(fc_max + rmse_max, hist_max, alert_level) * 1.15

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # ── Historical streamflow — gray, primary Y ───────────────────────────────
    if historical_obs is not None:
        hist_q = historical_obs.loc[window_start:base].dropna()
        fig.add_trace(
            go.Scatter(
                x=hist_q.index,
                y=hist_q.values,
                name="Observado (14d)",
                line=dict(color="#888888", width=2),
                mode="lines+markers",
                marker=dict(size=4),
            ),
            secondary_y=False,
        )

    # ── Uncertainty band ±RMSE — add BEFORE forecast line so line renders on top
    if uncertainty_per_lead is not None:
        # anchor gets RMSE=0 (known value), then grows per lead step
        rmse_band = ([0.0] if has_anchor else []) + list(uncertainty_per_lead)
        upper_y = [y + r for y, r in zip(fc_y, rmse_band)]
        lower_y = [max(0.0, y - r) for y, r in zip(fc_y, rmse_band)]

        # invisible lower boundary — reference for fill='tonexty'
        fig.add_trace(
            go.Scatter(
                x=fc_x, y=lower_y,
                mode="lines",
                line=dict(color="rgba(214,39,40,0)", width=0),
                showlegend=False,
                hoverinfo="skip",
            ),
            secondary_y=False,
        )
        # upper boundary fills down to the lower trace
        fig.add_trace(
            go.Scatter(
                x=fc_x, y=upper_y,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(214,39,40,0.18)",
                line=dict(color="rgba(214,39,40,0)", width=0),
                name="Banda ±RMSE",
                hovertemplate="±RMSE: %{customdata:.0f} m³/s<extra></extra>",
                customdata=rmse_band,
            ),
            secondary_y=False,
        )

    # ── Forecast streamflow — red line, primary Y ─────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=fc_x,
            y=fc_y,
            name="Previsão 7 dias",
            line=dict(color="#d62728", width=2.5),
            mode="lines+markers",
            marker=dict(size=7, symbol="circle"),
        ),
        secondary_y=False,
    )

    # ── Precipitation bars — light blue, secondary Y (inverted = top) ─────────
    if historical_precip is not None:
        hist_p = historical_precip.loc[window_start:base].dropna()
        fig.add_trace(
            go.Bar(
                x=hist_p.index,
                y=hist_p.values,
                name="Precipitação (mm/dia)",
                marker_color="rgba(158, 202, 225, 0.75)",
                marker_line_color="rgba(70, 130, 180, 0.5)",
                marker_line_width=0.5,
            ),
            secondary_y=True,
        )

    # ── Alert level line + text annotation ───────────────────────────────────
    fig.add_hline(
        y=alert_level,
        line=dict(color=COLORS["alert"], width=2, dash="dash"),
        annotation_text=f"⚠ Pré-alerta {alert_level:.0f} m³/s",
        annotation_position="top left",
        annotation=dict(
            font=dict(color=COLORS["alert"], size=12, family="Arial Black"),
            bgcolor="rgba(30,30,30,0.85)",
            bordercolor=COLORS["alert"],
            borderwidth=1,
        ),
    )

    # ── Vertical dashed separator: "hoje" ─────────────────────────────────────
    fig.add_vline(
        x=str(base.date()),
        line=dict(color="#444444", width=1.5, dash="dot"),
        annotation_text=" Hoje",
        annotation_position="top",
        annotation=dict(
            font=dict(color="#444444", size=11),
            bgcolor="rgba(30,30,30,0.8)",
        ),
    )

    # ── Axis styling ──────────────────────────────────────────────────────────
    fig.update_yaxes(
        title_text="Vazão (m³/s)",
        range=[0, y_max],
        secondary_y=False,
        showgrid=True,
        gridcolor="#333333",
        zeroline=True,
        zerolinecolor="#444444",
        color="#aaaaaa",
    )
    fig.update_yaxes(
        title_text="Precipitação (mm/dia)",
        autorange="reversed",
        secondary_y=True,
        showgrid=False,
        tickfont=dict(color="#6baed6"),
        title_font=dict(color="#6baed6"),
    )
    fig.update_xaxes(
        range=[window_start, window_end],
        showgrid=True,
        gridcolor="#333333",
        tickformat="%d/%m",
        color="#aaaaaa",
    )

    fig.update_layout(
        title=dict(
            text=f"Previsão emitida em {base.strftime('%d/%m/%Y')} — Janela 21 dias",
            font=dict(size=14, color="#dddddd"),
        ),
        height=height,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.15, x=0, font=dict(color="#aaaaaa")),
        margin=dict(l=70, r=80, t=55, b=75),
        plot_bgcolor="#111111",
        paper_bgcolor="#111111",
        font=dict(color="#aaaaaa"),
        bargap=0.15,
    )
    return fig
