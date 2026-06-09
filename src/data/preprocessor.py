"""
Pré-processamento da série de vazão/cota do Itajaí-Açu.

Pipeline:
  1. Carrega série bruta (Parquet)
  2. Remove duplicatas e outliers físicos
  3. Detecta e classifica gaps
  4. Interpolação curta (≤ 3 dias)
  5. Log-transform + padronização por bacia
  6. Salva série processada e estatísticas de normalização
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Limites físicos da bacia do Itajaí-Açu ────────────────────────────────────
#
# Estes limites não são arbitrários — derivam do conhecimento da bacia:
#   Q_MAX: a maior vazão registrada em Blumenau foi ~11.000 m³/s (nov/1983).
#   Usamos 15.000 com margem. Qualquer valor acima é erro de digitação ou
#   sensor defeituoso.
#   COTA_MAX: cota máxima histórica ~17 m. 20 m tem margem suficiente.
#   Valores negativos são fisicamente impossíveis em ambos os casos.
#
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "vazao": (0.0, 15_000.0),   # m³/s
    "cota": (0.0, 20.0),        # m
    "chuva": (0.0, 500.0),      # mm/dia (máximo plausível em eventos extremos)
}

# Máximo de dias consecutivos que interpolamos linearmente
MAX_GAP_INTERPOLATE = 3


# ── 1. Carga ──────────────────────────────────────────────────────────────────


def load_raw(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ── 2. Outliers físicos ────────────────────────────────────────────────────────


def remove_physical_outliers(
    df: pd.DataFrame,
    variable: str,
    col: str = "value",
) -> pd.DataFrame:
    """
    Seta como NaN valores fora dos limites físicos da bacia.

    Por que este passo antes de qualquer estatística?
    ──────────────────────────────────────────────────
    Outliers físicos (ex: -999 como código de missing, ou valores absurdos
    por falha de sensor) distorcem a média, desvio-padrão e percentis que
    serão usados na normalização posterior. Se normalizarmos antes de limpar,
    os extremos "puxam" a escala e comprimem os valores reais.
    Nunca calcule estatísticas de normalização sobre dados não filtrados.
    """
    lo, hi = PHYSICAL_BOUNDS.get(variable, (-np.inf, np.inf))
    mask = (df[col] < lo) | (df[col] > hi)
    n_bad = mask.sum()
    if n_bad:
        logger.warning("Outliers físicos removidos: %d valores em [%.1f, %.1f]", n_bad, lo, hi)
    df = df.copy()
    df.loc[mask, col] = np.nan
    return df


# ── 3. Análise de gaps ────────────────────────────────────────────────────────


def classify_gaps(series: pd.Series) -> pd.DataFrame:
    """
    Retorna DataFrame descrevendo cada sequência de NaN: início, fim, duração.

    Por que classificar gaps antes de preencher?
    ─────────────────────────────────────────────
    Nem todos os gaps são iguais:
    - 1-3 dias: provavelmente falha de sensor ou transmissão.
      Interpolação linear é defensável porque a vazão não muda
      abruptamente em escala diária (exceto em cheias rápidas —
      o que o step seguinte trata).
    - 4-30 dias: zona cinza. Interpolação introduziria dados
      sintéticos que o modelo poderia "aprender" como real.
      Marcamos e deixamos para decisão do usuário.
    - > 30 dias: buracos estruturais. O modelo simplesmente não
      treinará nesses períodos (máscaras de NaN).
    Ter essa tabela permite auditar a qualidade da série e tomar
    decisões informadas — não só preencher cegamente.
    """
    is_nan = series.isna()
    gaps = []
    in_gap = False
    start = None

    for date, val in series.items():
        if val != val and not in_gap:  # NaN
            in_gap = True
            start = date
        elif val == val and in_gap:
            gaps.append({"start": start, "end": date, "days": (date - start).days})
            in_gap = False

    if in_gap:
        gaps.append({"start": start, "end": series.index[-1], "days": (series.index[-1] - start).days + 1})

    return pd.DataFrame(gaps)


# ── 4. Interpolação curta ─────────────────────────────────────────────────────


def interpolate_short_gaps(series: pd.Series, max_gap: int = MAX_GAP_INTERPOLATE) -> pd.Series:
    """
    Preenche gaps de até `max_gap` dias com interpolação linear.

    Por que linear e não spline cúbica?
    ─────────────────────────────────────
    Splines cúbicas podem oscilar (fenômeno de Runge) entre dois valores
    muito diferentes — ex: de 100 m³/s para 5000 m³/s em 3 dias.
    A interpolação linear é mais conservadora: assume uma transição
    monótona, que é errada mas menos perigosa para o treinamento.
    Importante: o campo `interpolated` marca quais dias foram sintéticos,
    permitindo excluí-los do cálculo de métricas de avaliação.

    Por que limit_area="inside"?
    ─────────────────────────────
    Sem esse parâmetro, pandas extrapola além do início e fim da série.
    Como não temos dados antes de 1940 e após hoje, extrapolação geraria
    valores negativos ou absurdos nas bordas.
    """
    filled = series.interpolate(method="linear", limit=max_gap, limit_area="inside")
    return filled


# ── 5. Log-transform ──────────────────────────────────────────────────────────


def log_transform(series: pd.Series, eps: float = 1.0) -> pd.Series:
    """
    Aplica log1p(Q + eps - 1) ≈ log(Q) para Q >> 1.

    Por que transformar a vazão em log?
    ─────────────────────────────────────
    A distribuição de vazão é fortemente assimétrica à direita
    (cauda pesada — log-normal). O LSTM otimiza o erro quadrático
    médio (MSE) no espaço de features. Se treinarmos no espaço linear,
    um erro de 100 m³/s na baixa vazão (100→200 m³/s) tem o mesmo peso
    que um erro de 100 m³/s durante uma cheia (5000→5100 m³/s).
    No espaço log, erros são relativos — o modelo aprende a prever bem
    tanto o regime baixo quanto as cheias.

    Atenção: eps=1.0 evita log(0) em dias de vazão zero (possível em
    tributários pequenos; raro em Blumenau mas presente em sub-bacias).
    """
    return np.log1p(series + eps - 1)


def inverse_log_transform(series: pd.Series, eps: float = 1.0) -> pd.Series:
    """Inversa de log_transform — use para converter predições de volta."""
    return np.expm1(series) - eps + 1


# ── 6. Normalização ───────────────────────────────────────────────────────────


def compute_normalization_stats(
    series: pd.Series,
    train_end: str = "2010-12-31",
) -> dict[str, float]:
    """
    Calcula média e desvio-padrão apenas no período de treino.

    Por que só no período de treino?
    ──────────────────────────────────
    Se calcularmos a normalização sobre toda a série, o modelo "vê"
    estatísticas do período de validação e teste durante o treinamento.
    Isso é data leakage — o modelo fica artificialmente bem calibrado
    para valores que "não deveria conhecer".
    As mesmas estatísticas devem ser salvas em JSON e reutilizadas
    na inferência operacional (nunca recalcular no momento de previsão).
    """
    train_slice = series[:train_end].dropna()
    return {
        "mean": float(train_slice.mean()),
        "std": float(train_slice.std(ddof=1)),
        "min": float(train_slice.min()),
        "max": float(train_slice.max()),
        "n_train": int(len(train_slice)),
        "train_end": train_end,
    }


def standardize(series: pd.Series, stats: dict[str, float]) -> pd.Series:
    """Z-score: (x - mean) / std"""
    return (series - stats["mean"]) / stats["std"]


def inverse_standardize(series: pd.Series, stats: dict[str, float]) -> pd.Series:
    return series * stats["std"] + stats["mean"]


# ── Pipeline completo ─────────────────────────────────────────────────────────


def preprocess(
    raw_path: str | Path,
    variable: Literal["vazao", "cota", "chuva"],
    out_dir: str | Path = "data/processed",
    train_end: str = "2010-12-31",
    apply_log: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Executa o pipeline completo de pré-processamento.

    Retorna
    -------
    (df_processed, norm_stats)
        df_processed : DataFrame com colunas value, value_log, value_norm,
                       consistency, interpolated
        norm_stats   : dict com mean/std usados — salvar para inferência!
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Carga
    df = load_raw(raw_path)
    logger.info("Carregado: %d dias, %.1f%% válidos", len(df), df["value"].notna().mean() * 100)

    # 2. Outliers físicos
    df = remove_physical_outliers(df, variable)

    # 3. Análise de gaps (só para logging/auditoria)
    gaps = classify_gaps(df["value"])
    if not gaps.empty:
        logger.info(
            "Gaps detectados: %d total | curtos(≤3d): %d | longos(>30d): %d",
            len(gaps),
            (gaps["days"] <= MAX_GAP_INTERPOLATE).sum(),
            (gaps["days"] > 30).sum(),
        )
        long_gaps = gaps[gaps["days"] > 30]
        if not long_gaps.empty:
            logger.warning("Gaps longos:\n%s", long_gaps.to_string())

    # 4. Interpolação curta
    filled = interpolate_short_gaps(df["value"])
    df["interpolated"] = df["value"].isna() & filled.notna()
    df["value"] = filled

    # 5. Log-transform (apenas para vazão e cota)
    if apply_log and variable in ("vazao", "cota"):
        df["value_log"] = log_transform(df["value"])
    else:
        df["value_log"] = df["value"]

    # 6. Normalização (usando apenas período de treino)
    norm_stats = compute_normalization_stats(df["value_log"], train_end=train_end)
    df["value_norm"] = standardize(df["value_log"], norm_stats)

    # ── Salvar ────────────────────────────────────────────────────────────────
    station_id = Path(raw_path).stem.split("_")[0]
    out_parquet = out_dir / f"{station_id}_{variable}_processed.parquet"
    df.to_parquet(out_parquet)

    out_stats = out_dir / f"{station_id}_{variable}_norm_stats.json"
    with open(out_stats, "w") as f:
        json.dump(norm_stats, f, indent=2)

    logger.info("Processado salvo: %s", out_parquet)
    logger.info("Estatísticas de normalização: %s", norm_stats)
    return df, norm_stats


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", help="Caminho para o .parquet bruto")
    parser.add_argument("--variable", choices=["vazao", "cota", "chuva"], required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--train-end", default="2010-12-31")
    args = parser.parse_args()

    df, stats = preprocess(args.raw_path, args.variable, args.out_dir, args.train_end)
    print(df[["value", "value_log", "value_norm", "interpolated"]].describe())
