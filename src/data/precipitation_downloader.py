"""
Download de precipitação para a bacia do Itajaí-Açu.

Duas fontes complementares — cada uma cobre uma lacuna da outra:

┌─────────────────────────────────────────────────────────────────────────┐
│  INMET (Automáticas)          │  CHIRPS (Gridded)                       │
│  Tipo: pontual                │  Tipo: espacialmente contínuo           │
│  Resolução temporal: horária  │  Resolução temporal: diária             │
│  Cobertura espacial: ~15 est. │  Cobertura espacial: grade 0.05°        │
│  Período: 2000–presente       │  Período: 1981–presente                 │
│  Qualidade: alta (mas gaps)   │  Qualidade: média (reanalysis)          │
└─────────────────────────────────────────────────────────────────────────┘

Quando usar cada uma:
  - INMET: treinamento com dados observados reais; correlação chuva-vazão;
    validação do CHIRPS.
  - CHIRPS: cobertura espacial completa para médias areais da bacia;
    retropolação para períodos sem estação automática (antes de 2000);
    substituto de previsão numérica em testes de sensibilidade.

API INMET base: https://apitempo.inmet.gov.br
CHIRPS download: https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25/
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

logger = logging.getLogger(__name__)

# ── Bounding box da bacia do Itajaí-Açu ──────────────────────────────────────
#
# Derivado do polígono da bacia (ANA / SNIRH):
#   Itajaí do Sul: nasce em Ituporanga (~-27.4°S)
#   Foz em Itajaí: ~-26.9°S, -48.7°W
#   Cabeceira Oeste: Campos Novos/Curitibanos ~-51.0°W
#
BASIN_BBOX = {
    "lat_min": -27.6,
    "lat_max": -26.5,
    "lon_min": -50.2,
    "lon_max": -48.6,
}

INMET_BASE = "https://apitempo.inmet.gov.br"

# ── INMET ─────────────────────────────────────────────────────────────────────


def _inmet_get(endpoint: str, timeout: int = 30, retries: int = 3) -> list | dict:
    """GET com retry exponencial para a API do INMET."""
    url = f"{INMET_BASE}/{endpoint.lstrip('/')}"
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            logger.warning("INMET attempt %d failed (%s). Retrying in %ds...", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError("Unreachable")


def list_inmet_stations_in_basin(
    bbox: dict = BASIN_BBOX,
) -> pd.DataFrame:
    """
    Retorna estações automáticas INMET dentro da bounding box da bacia.

    Por que só automáticas (tipo 'T')?
    ────────────────────────────────────
    As estações convencionais (tipo 'M') medem precipitação acumulada
    diária por observador humano — horário inconsistente e dependente
    de disponibilidade. As automáticas (tipo 'T') medem a cada hora,
    integradas e transmitidas automaticamente. Para análise de lag
    chuva-vazão com resolução sub-diária, só as automáticas servem.
    Desvantagem: a rede automática é mais recente (~2000 em diante)
    e menos densa que a convencional.
    """
    data = _inmet_get("estacoes/T")
    df = pd.DataFrame(data)

    # Normalizar nomes de colunas (API retorna PT-BR)
    rename = {
        "CD_ESTACAO": "station_id",
        "DC_NOME": "name",
        "VL_LATITUDE": "lat",
        "VL_LONGITUDE": "lon",
        "VL_ALTITUDE": "altitude_m",
        "DT_INICIO_OPERACAO": "start_date",
        "SG_ESTADO": "state",
        "CD_SITUACAO": "status",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Converter lat/lon para float (API pode retornar string)
    for col in ("lat", "lon"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Filtrar por bounding box
    mask = (
        df["lat"].between(bbox["lat_min"], bbox["lat_max"])
        & df["lon"].between(bbox["lon_min"], bbox["lon_max"])
    )
    in_basin = df[mask].copy()
    logger.info("Estações INMET na bacia: %d encontradas", len(in_basin))
    return in_basin


def download_inmet_station(
    station_id: str,
    start_date: str,
    end_date: str,
    out_dir: str | Path = "data/raw/precipitation",
) -> pd.DataFrame:
    """
    Baixa dados horários de uma estação INMET e retorna DataFrame diário.

    Endpoint: GET /estacao/dados/{codigo}/{data-ini}/{data-fim}
    Datas no formato YYYY-MM-DD.

    Decisão de agregação horária → diária:
    ────────────────────────────────────────
    Mesmo que treinar com resolução horária seja ideal para bacias rápidas
    como o Itajaí-Açu (tempo de concentração ~24h), o OpenHydroNet base
    opera em resolução diária. Agregamos para diário aqui, mas preservamos
    o CSV horário bruto para uso futuro (modelagem intra-diária).

    Definição de "dia hidrológico":
    A ANA define o dia hidrológico como 07:00–07:00 (fuso Brasília UTC-3).
    Usar meia-noite UTC geraria discrepâncias com os dados de vazão da ANA.
    Usamos resample('D', offset='7h') para alinhar os acumulados.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Chunk por mês para evitar timeout
    date_range = pd.date_range(start_date, end_date, freq="MS")
    chunks: list[pd.DataFrame] = []

    for month_start in date_range:
        month_end = month_start + pd.offsets.MonthEnd(0)
        s = month_start.strftime("%Y-%m-%d")
        e = min(month_end, pd.Timestamp(end_date)).strftime("%Y-%m-%d")
        endpoint = f"estacao/dados/{station_id}/{s}/{e}"

        try:
            data = _inmet_get(endpoint)
            if not data:
                continue
            df_chunk = pd.DataFrame(data)
            chunks.append(df_chunk)
        except Exception as exc:
            logger.warning("Erro estação %s mês %s: %s", station_id, s, exc)

    if not chunks:
        logger.warning("Nenhum dado encontrado para estação %s", station_id)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Coluna de precipitação pode aparecer como 'CHUVA' ou 'PRE_INS'
    precip_col = next((c for c in df.columns if "CHUVA" in c.upper()), None)
    dt_col = next((c for c in df.columns if "DT_MEDICAO" in c.upper() or "DATETIME" in c.upper()), None)

    if not precip_col or not dt_col:
        logger.warning("Colunas de precipitação/data não encontradas para %s. Colunas: %s", station_id, list(df.columns))
        return df

    df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
    df[precip_col] = pd.to_numeric(df[precip_col], errors="coerce")
    df = df.set_index(dt_col).sort_index()

    # Salvar horário bruto
    raw_path = out_dir / f"inmet_{station_id}_hourly_raw.parquet"
    df[[precip_col]].to_parquet(raw_path)
    logger.info("Horário bruto salvo: %s", raw_path)

    # Agregar para diário (dia hidrológico 07:00–07:00)
    daily = (
        df[precip_col]
        .resample("D", offset="7h")
        .sum(min_count=18)  # exige ao menos 18 horas válidas no dia
        .rename("precip_mm")
        .to_frame()
    )
    # Remover offset do índice para manter datas limpas
    daily.index = daily.index.normalize()

    daily_path = out_dir / f"inmet_{station_id}_daily.parquet"
    daily.to_parquet(daily_path)
    logger.info("Diário salvo: %s (%d dias, %.0f%% válidos)", daily_path, len(daily), daily["precip_mm"].notna().mean() * 100)
    return daily


def download_all_basin_stations(
    start_date: str = "2000-01-01",
    end_date: str | None = None,
    out_dir: str | Path = "data/raw/precipitation",
) -> dict[str, pd.DataFrame]:
    """Baixa todas as estações INMET dentro da bacia e retorna dict station_id → DataFrame."""
    end_date = end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    stations = list_inmet_stations_in_basin()

    results: dict[str, pd.DataFrame] = {}
    for _, row in stations.iterrows():
        sid = row["station_id"]
        logger.info("Baixando estação %s (%s)...", sid, row.get("name", ""))
        try:
            df = download_inmet_station(sid, start_date, end_date, out_dir)
            if not df.empty:
                results[sid] = df
        except Exception as exc:
            logger.error("Falhou estação %s: %s", sid, exc)

    return results


# ── CHIRPS ────────────────────────────────────────────────────────────────────

CHIRPS_BASE = "https://data.chc.ucsb.edu/products/CHIRPS-2.0/global_daily/netcdf/p25"


def _chirps_url(year: int) -> str:
    return f"{CHIRPS_BASE}/chirps-v2.0.{year}.days_p25.nc"


def download_chirps_year(
    year: int,
    bbox: dict = BASIN_BBOX,
    out_dir: str | Path = "data/raw/precipitation",
    timeout: int = 300,
) -> Path:
    """
    Baixa e recorta o arquivo NetCDF anual do CHIRPS para a bounding box.

    Por que baixar o arquivo global e depois recortar?
    ──────────────────────────────────────────────────
    O CHIRPS não oferece API de recorte espacial público. O arquivo anual
    global em 0.25° tem ~50 MB — manejável. A resolução 0.05° (~5.5 km)
    chega a ~500 MB/ano; usamos 0.25° (~27 km) para o pipeline de treino.
    Para análise espacial fina ou geração de features gridded de alta
    resolução, trocar para p05 no URL.

    O recorte (sel com slice) é feito em memória pelo xarray antes de
    salvar — nenhum dado fora da bacia é persistido em disco.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"chirps_{year}_itajai.nc"

    if out_path.exists():
        logger.info("CHIRPS %d já existe: %s", year, out_path)
        return out_path

    url = _chirps_url(year)
    logger.info("Baixando CHIRPS %d de %s...", year, url)

    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        tmp_path = out_dir / f"chirps_{year}_global_tmp.nc"
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                f.write(chunk)

    # Recortar para a bacia
    ds = xr.open_dataset(tmp_path, engine="netcdf4")

    # CHIRPS usa longitude 0–360; converter se necessário
    if ds["longitude"].min() > 0:
        ds = ds.assign_coords(longitude=(ds["longitude"] + 180) % 360 - 180)
        ds = ds.sortby("longitude")

    ds_clipped = ds.sel(
        latitude=slice(bbox["lat_min"], bbox["lat_max"]),  # CHIRPS lat é ascendente
        longitude=slice(bbox["lon_min"], bbox["lon_max"]),
    )
    ds_clipped.to_netcdf(out_path)
    tmp_path.unlink()  # apagar arquivo global
    logger.info("CHIRPS %d recortado: %s", year, out_path)
    return out_path


def download_chirps_range(
    start_year: int = 1981,
    end_year: int | None = None,
    bbox: dict = BASIN_BBOX,
    out_dir: str | Path = "data/raw/precipitation",
) -> list[Path]:
    """Baixa série CHIRPS para todos os anos no intervalo."""
    end_year = end_year or pd.Timestamp.today().year - 1  # CHIRPS tem ~6 meses de lag
    paths = []
    for year in range(start_year, end_year + 1):
        try:
            p = download_chirps_year(year, bbox=bbox, out_dir=out_dir)
            paths.append(p)
        except Exception as exc:
            logger.error("CHIRPS %d falhou: %s", year, exc)
    return paths


def chirps_basin_average(
    chirps_dir: str | Path,
    basin_mask: "np.ndarray | None" = None,
) -> pd.DataFrame:
    """
    Calcula precipitação média areal da bacia a partir dos arquivos CHIRPS anuais.

    Por que média areal e não um único pixel?
    ──────────────────────────────────────────
    A bacia do Itajaí-Açu tem ~15.000 km². Cada pixel de 0.25° (~700 km²)
    cobre cerca de 5% da bacia. Uma única estação pontual ou pixel pode estar
    em uma zona de sombra orográfica (lado seco da Serra), enquanto outra
    está no barlavento úmido. A média areal do CHIRPS é a melhor aproximação
    da "precipitação que entra na bacia" e é o que o modelo hidrológico precisa.

    O parâmetro `basin_mask` aceita um array 2D bool/float com os pesos
    de cada pixel na média (ex: fração da área do pixel dentro do polígono
    da bacia). Se None, usa média simples de todos os pixels.

    Para gerar um basin_mask mais preciso:
        1. Baixar o shapefile da bacia do SNIRH/ANA
        2. Rasterizar sobre a grade do CHIRPS com rasterio/exactextract
        3. Normalizar para sum == 1 e passar como parâmetro
    """
    chirps_dir = Path(chirps_dir)
    nc_files = sorted(chirps_dir.glob("chirps_*_itajai.nc"))
    if not nc_files:
        raise FileNotFoundError(f"Nenhum arquivo CHIRPS encontrado em {chirps_dir}")

    datasets = [xr.open_dataset(f, engine="netcdf4") for f in nc_files]
    ds = xr.concat(datasets, dim="time")

    precip_var = next((v for v in ds.data_vars if "precip" in v.lower()), list(ds.data_vars)[0])

    if basin_mask is not None:
        weights = xr.DataArray(basin_mask, dims=["latitude", "longitude"])
        basin_mean = (ds[precip_var] * weights).sum(dim=["latitude", "longitude"])
    else:
        basin_mean = ds[precip_var].mean(dim=["latitude", "longitude"])

    df = basin_mean.to_dataframe(name="precip_mm_chirps")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Baixa dados de precipitação da bacia do Itajaí-Açu")
    sub = parser.add_subparsers(dest="source")

    p_inmet = sub.add_parser("inmet", help="Baixar estações INMET da bacia")
    p_inmet.add_argument("--start", default="2000-01-01")
    p_inmet.add_argument("--end", default=None)
    p_inmet.add_argument("--out-dir", default="data/raw/precipitation")

    p_chirps = sub.add_parser("chirps", help="Baixar CHIRPS gridded")
    p_chirps.add_argument("--start-year", type=int, default=1981)
    p_chirps.add_argument("--end-year", type=int, default=None)
    p_chirps.add_argument("--out-dir", default="data/raw/precipitation")

    args = parser.parse_args()

    if args.source == "inmet":
        download_all_basin_stations(args.start, args.end, args.out_dir)
    elif args.source == "chirps":
        download_chirps_range(args.start_year, args.end_year, out_dir=args.out_dir)
    else:
        parser.print_help()
