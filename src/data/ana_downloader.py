"""
Downloader para a API REST do HidroWeb (ANA).

Endpoint base: http://telemetriaws1.ana.gov.br/ServiceANA.asmx
Série histórica: /HidroSerieHistorica
  - codEstacao   : código de 8 dígitos da estação
  - dataInicio   : YYYY-MM-DD (opcional)
  - dataFim      : YYYY-MM-DD (opcional)
  - tipoDados    : 1 = cota (nível), 2 = chuva, 3 = vazão
  - nivelConsistencia : 1 = bruto, 2 = consolidado (vazio = ambos)

A resposta é XML com registros mensais. Cada mês tem campos
Vazao01...Vazao31 (ou Cota01...) representando dias.
"""

from __future__ import annotations

import calendar
import datetime
import logging
import time
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Constantes ────────────────────────────────────────────────────────────────

BASE_URL = "http://telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica"

TIPO_DADOS = {
    "cota": 1,
    "chuva": 2,
    "vazao": 3,
}

VAR_PREFIX = {
    1: "Cota",
    2: "Chuva",
    3: "Vazao",
}

STATION_BLUMENAU = "83500000"

# ── Funções de baixo nível ────────────────────────────────────────────────────


def _fetch_month_chunk(
    station: str,
    start: str,
    end: str,
    tipo: int,
    nivel: str = "",
    timeout: int = 60,
    retries: int = 3,
) -> bytes:
    """
    Faz uma requisição GET e retorna o conteúdo XML bruto.

    Por que retries com backoff exponencial?
    O servidor da ANA é instável — timeouts esporádicos são comuns,
    especialmente para janelas longas. Três tentativas com espera dobrada
    (2 s, 4 s, 8 s) resolve a maioria dos problemas transitórios sem
    travar o script por muito tempo.
    """
    params = {
        "codEstacao": station,
        "dataInicio": start,
        "dataFim": end,
        "tipoDados": tipo,
        "nivelConsistencia": nivel,
    }
    for attempt in range(retries):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.content
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("Tentativa %d falhou (%s). Aguardando %ds...", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")  # mypy


def _parse_xml_to_daily(xml_bytes: bytes, tipo: int) -> pd.DataFrame:
    """
    Converte o XML mensal da ANA em série diária.

    Por que não usar pd.read_xml direto?
    O XML da ANA tem uma estrutura aninhada não trivial (DocumentElement >
    DadosHidrometereologicos > SerieHistorica). pd.read_xml consegue lidar
    com isso via xpath, mas a lógica de montar as datas diárias (cada linha
    do XML = 1 mês, com Vazao01..Vazao31) requer pós-processamento manual
    de qualquer forma. Optei por ET para controle total e clareza.
    """
    import xml.etree.ElementTree as ET

    root = ET.fromstring(xml_bytes)
    prefix = VAR_PREFIX[tipo]
    records: list[dict] = []

    for node in root.iter("SerieHistorica"):
        raw_date = node.findtext("DataHora", "")
        if not raw_date:
            continue
        month_start = datetime.datetime.strptime(raw_date[:10], "%Y-%m-%d")
        n_days = calendar.monthrange(month_start.year, month_start.month)[1]
        consistencia = int(node.findtext("NivelConsistencia", "0"))

        for d in range(1, n_days + 1):
            tag = f"{prefix}{d:02d}"
            raw_val = node.findtext(tag)
            try:
                value = float(raw_val) if raw_val is not None else float("nan")
            except (ValueError, TypeError):
                value = float("nan")

            records.append(
                {
                    "date": month_start + datetime.timedelta(days=d - 1),
                    "value": value,
                    "consistency": consistencia,
                }
            )

    if not records:
        return pd.DataFrame(columns=["date", "value", "consistency"])

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


# ── Função principal ───────────────────────────────────────────────────────────


def download_series(
    station: str = STATION_BLUMENAU,
    variable: Literal["vazao", "cota", "chuva"] = "vazao",
    start_year: int = 1940,
    end_year: int | None = None,
    nivel_consistencia: str = "",
    chunk_years: int = 5,
) -> pd.DataFrame:
    """
    Baixa a série histórica completa da estação e retorna DataFrame diário.

    Parâmetros
    ----------
    station : str
        Código de 8 dígitos da estação ANA.
    variable : {"vazao", "cota", "chuva"}
        Tipo de dado a baixar.
    start_year : int
        Ano de início da série.
    end_year : int, opcional
        Ano de fim (padrão: ano corrente).
    nivel_consistencia : str
        "" = ambos; "1" = bruto; "2" = consolidado.
    chunk_years : int
        Tamanho do bloco em anos por requisição.

        Por que chunking?
        ─────────────────
        A API não tem limite de datas documentado, mas requisições
        abrangendo décadas inteiras frequentemente travam ou retornam
        XML incompleto. Blocos de 5 anos são um equilíbrio entre
        número de requisições e confiabilidade. Ajuste para 1-2 se
        a conexão for instável.

    Retorna
    -------
    pd.DataFrame com colunas: date (index), value, consistency
    """
    end_year = end_year or datetime.date.today().year
    tipo = TIPO_DADOS[variable]
    all_chunks: list[pd.DataFrame] = []

    years = range(start_year, end_year + 1, chunk_years)
    for yr_start in years:
        yr_end = min(yr_start + chunk_years - 1, end_year)
        start_str = f"{yr_start}-01-01"
        end_str = f"{yr_end}-12-31"
        logger.info("Baixando %s: %s → %s", variable, start_str, end_str)

        try:
            xml_bytes = _fetch_month_chunk(station, start_str, end_str, tipo, nivel_consistencia)
            chunk_df = _parse_xml_to_daily(xml_bytes, tipo)
            if not chunk_df.empty:
                all_chunks.append(chunk_df)
        except Exception as exc:
            logger.error("Erro no bloco %s–%s: %s", start_str, end_str, exc)

    if not all_chunks:
        return pd.DataFrame(columns=["date", "value", "consistency"])

    df = pd.concat(all_chunks, ignore_index=True)

    # ── Deduplicação por data + preferência de nível de consistência ──────────
    #
    # Por que priorizar consistency == 2?
    # ────────────────────────────────────
    # A ANA publica dois "layers" de dado:
    #   Nível 1 (bruto): inserido em tempo quase real, sem revisão.
    #   Nível 2 (consolidado): revisado manualmente, valores corrigidos.
    # Para datas antigas, o nível 2 já existe e é mais confiável.
    # Para datas recentes (últimos 1-2 anos), só o nível 1 existe.
    # Ao ordenar por consistency desc e fazer drop_duplicates(keep='first'),
    # sempre usamos o melhor dado disponível para cada dia.
    #
    df = (
        df.sort_values(["date", "consistency"], ascending=[True, False])
        .drop_duplicates(subset="date", keep="first")
        .set_index("date")
        .sort_index()
    )

    logger.info(
        "Série completa: %s a %s (%d dias, %.1f%% válidos)",
        df.index.min().date(),
        df.index.max().date(),
        len(df),
        df["value"].notna().mean() * 100,
    )
    return df


# ── Persistência ──────────────────────────────────────────────────────────────


def save_raw(
    df: pd.DataFrame,
    station: str,
    variable: str,
    out_dir: str | Path = "data/raw/streamflow",
) -> Path:
    """
    Salva a série bruta em Parquet.

    Por que Parquet e não CSV?
    ───────────────────────────
    Parquet preserva os tipos (datetime64, float32) sem ambiguidade,
    comprime bem séries de floats (~10x menor que CSV) e é lido pela
    maioria das ferramentas Python (pandas, polars, dask, xarray via
    pyarrow). CSV é legível por humanos, mas para séries de décadas
    com metadados de consistência, Parquet é superior em todos os
    critérios práticos.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{station}_{variable}_raw.parquet"
    df.to_parquet(path)
    logger.info("Salvo: %s", path)
    return path


# ── CLI simples ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Baixa dados históricos do HidroWeb (ANA)")
    parser.add_argument("--station", default=STATION_BLUMENAU)
    parser.add_argument("--variable", choices=["vazao", "cota", "chuva"], default="vazao")
    parser.add_argument("--start-year", type=int, default=1940)
    parser.add_argument("--end-year", type=int, default=None)
    parser.add_argument("--out-dir", default="data/raw/streamflow")
    args = parser.parse_args()

    df = download_series(
        station=args.station,
        variable=args.variable,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    save_raw(df, args.station, args.variable, args.out_dir)
    print(df.tail(10))
