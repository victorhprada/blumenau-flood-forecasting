"""
Converte dados locais (ANA + INMET/CHIRPS) para o formato Caravan NetCDF.

O OpenHydroNet espera esta estrutura de diretórios exata (load_as_csv: true):

    Caravan-nc/
    ├── timeseries/
    │   ├── csv/
    │   │   └── {subdataset}/
    │   │       └── {subdataset}_{gauge_id}.csv   ← hindcast + targets (CSV)
    │   └── netcdf/
    │       └── {subdataset}/
    │           └── {subdataset}_{gauge_id}.nc    ← mantido como referência
    └── attributes/
        └── {subdataset}/
            ├── attributes_caravan_{subdataset}.csv   ← atributos climáticos
            ├── attributes_hydroatlas_{subdataset}.csv ← morfologia/hidrologia
            └── attributes_other_{subdataset}.csv     ← metadados da estação

Com load_as_csv: true, o Multimet dataset usa:
  _load_hindcast_as_csv → timeseries/csv/{subdataset}/{basin}.csv
  _load_forecast_as_csv → fallback para timeseries/csv/{subdataset}/{basin}.csv
  _load_target_features → timeseries/csv/{subdataset}/{basin}.csv
  load_caravan_attributes → attributes/{subdataset}/*.csv

Nomenclatura de basin IDs:
    O framework usa "{subdataset}_{gauge_id}" — ex: "itajai_83500000".
    O prefixo do subdataset precisa bater exatamente com os nomes de diretório.
    Escolhemos "itajai" como subdataset name.

────────────────────────────────────────────────────────────────────────────
Timeseries vs. Static Attributes — por que essa separação?
────────────────────────────────────────────────────────────────────────────
O LSTM processa dois tipos de informação de forma muito diferente:

  TIMESERIES (dinâmico):
  - Varia a cada passo de tempo (diariamente)
  - Entra na camada de hindcast/forecast do LSTM como sequência
  - Exemplos: precipitação, vazão, temperatura, umidade do solo

  STATIC ATTRIBUTES (estático):
  - Não varia no tempo (ou varia muito lentamente — décadas)
  - Codifica "o que torna esta bacia única"
  - Passa por uma rede de embedding separada (antes do LSTM)
  - O embedding transforma ~60 atributos → vetor de 20 dimensões
  - Isso permite ao modelo generalizar para bacias não vistas (zero-shot)
  - Exemplos: área de drenagem, declividade média, tipo de solo, cobertura

A separação não é arbitrária — é fundamental para a arquitetura:
sem atributos estáticos, o modelo não teria como saber que Blumenau
tem resposta rápida por conta da topografia. Ele aprenderia isso
implicitamente dos dados históricos, mas generalização para eventos
fora do período de treino seria muito pior.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

SUBDATASET = "itajai"
GAUGE_ID = "83500000"
BASIN_ID = f"{SUBDATASET}_{GAUGE_ID}"


# ── Estrutura de diretórios ───────────────────────────────────────────────────


def caravan_dirs(root: str | Path) -> dict[str, Path]:
    """Retorna (e cria) os diretórios do Caravan."""
    root = Path(root)
    dirs = {
        "timeseries_nc": root / "timeseries" / "netcdf" / SUBDATASET,
        "timeseries_csv": root / "timeseries" / "csv" / SUBDATASET,
        "attributes": root / "attributes" / SUBDATASET,
        # compat alias para código legado que usava "timeseries"
        "timeseries": root / "timeseries" / "netcdf" / SUBDATASET,
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ── 1. Timeseries NetCDF ──────────────────────────────────────────────────────


def build_timeseries_nc(
    streamflow: pd.Series,
    precipitation: pd.DataFrame | None = None,
    basin_id: str = BASIN_ID,
    out_dir: str | Path = "data/processed/Caravan-nc",
) -> Path:
    """
    Cria o arquivo NetCDF de série temporal para uma bacia.

    Variáveis incluídas:
      streamflow          : m³/s, alvo principal do modelo
      total_precipitation : mm/dia, da fonte disponível (CHIRPS ou INMET)

    Por que NetCDF e não CSV?
    ──────────────────────────
    O caravan.py do framework tem dois backends: CSV e NetCDF.
    O NetCDF é preferido porque:
    1. Embute metadados (unidades, long_name) que o framework usa
    2. Permite chunking eficiente para grandes conjuntos de bacias
    3. Suporta leitura lazy via xarray/dask — importante quando você
       escalar de 1 bacia para centenas

    Estrutura esperada do NC:
      Dimensão: date (freq diária, completa, sem gaps — NaN para missing)
      Variáveis: float32 (o caravan.py faz cast explícito para float32)
      Atributos globais: gauge_id, subdataset_name, units por variável
    """
    dirs = caravan_dirs(out_dir)

    # ── Montar índice de datas completo ───────────────────────────────────────
    #
    # Por que reindex para índice diário completo?
    # O LSTM precisa de sequências sem gaps. O framework trata NaN como
    # dado faltante (mask), mas o índice de datas deve ser contínuo —
    # caso contrário o carregador de dados calcula offsets errados entre
    # passos de tempo consecutivos.
    #
    all_dates = streamflow.dropna().index
    if precipitation is not None and not precipitation.empty:
        all_dates = all_dates.union(precipitation.dropna(how="all").index)

    date_range = pd.date_range(all_dates.min(), all_dates.max(), freq="D", name="date")

    # ── Construir Dataset ─────────────────────────────────────────────────────
    data_vars: dict[str, Any] = {}

    q = streamflow.reindex(date_range).astype("float32")
    data_vars["streamflow"] = xr.DataArray(
        q.values,
        dims=["date"],
        attrs={"units": "m3/s", "long_name": "Streamflow"},
    )

    if precipitation is not None and not precipitation.empty:
        precip_col = precipitation.columns[0]
        p = precipitation[precip_col].reindex(date_range).astype("float32")
        data_vars["total_precipitation"] = xr.DataArray(
            p.values,
            dims=["date"],
            attrs={"units": "mm/day", "long_name": "Total Precipitation"},
        )

    ds = xr.Dataset(
        data_vars,
        coords={"date": date_range},
        attrs={
            "gauge_id": basin_id,
            "subdataset_name": SUBDATASET,
            "created_by": "blumenau-flood-forecasting/caravan_formatter.py",
        },
    )

    out_path = dirs["timeseries_nc"] / f"{basin_id}.nc"
    ds.to_netcdf(out_path, encoding={v: {"zlib": True, "complevel": 4} for v in data_vars})
    logger.info("Timeseries NetCDF criado: %s (%d dias)", out_path, len(date_range))
    return out_path


def build_timeseries_csv(
    streamflow: pd.Series,
    precipitation: pd.DataFrame | None = None,
    basin_id: str = BASIN_ID,
    out_dir: str | Path = "data/processed/Caravan-nc",
    lag_days: list[int] | None = None,
) -> Path:
    """
    Cria o CSV de série temporal para uso com load_as_csv: true.

    O Multimet dataset com load_as_csv: true carrega hindcast e targets deste
    arquivo via load_caravan_timeseries_together(..., csv=True).
    O dask.read_csv espera:
      - Coluna 'date' (parse_dates=True) — não como índice
      - Colunas de features como float32

    Parâmetros
    ----------
    lag_days : list[int] | None
        Dias de lag autoregressivo de vazão a incluir.
        Ex: [1, 7] adiciona colunas 'streamflow_lag1' e 'streamflow_lag7'.

        Os prefixos '_lag' evitam conflito dimensional com 'streamflow' em
        target_variables: o framework infere o grupo pelo prefixo antes do
        primeiro '_', então 'streamflow_lag1' e 'streamflow_lag7' formam o
        grupo 'streamflow' nos hindcast_inputs (separado do target).
    """
    dirs = caravan_dirs(out_dir)

    all_dates = streamflow.dropna().index
    if precipitation is not None and not precipitation.empty:
        all_dates = all_dates.union(precipitation.dropna(how="all").index)

    date_range = pd.date_range(all_dates.min(), all_dates.max(), freq="D", name="date")

    df = pd.DataFrame(index=date_range)
    df.index.name = "date"

    q = streamflow.reindex(date_range).astype("float32")
    df["streamflow"] = q

    if lag_days:
        for lag in lag_days:
            df[f"streamflow_lag{lag}"] = q.shift(lag).astype("float32")

    if precipitation is not None and not precipitation.empty:
        precip_col = precipitation.columns[0]
        p = precipitation[precip_col].reindex(date_range).astype("float32")
        df["total_precipitation"] = p
    else:
        df["total_precipitation"] = float("nan")

    # Ordenação: precipitation → streamflow_lags → streamflow (target por último)
    lag_cols = [f"streamflow_lag{d}" for d in (lag_days or [])]
    ordered = ["total_precipitation"] + lag_cols + ["streamflow"]
    df = df[[c for c in ordered if c in df.columns]]

    df = df.reset_index()

    out_path = dirs["timeseries_csv"] / f"{basin_id}.csv"
    df.to_csv(out_path, index=False, float_format="%.6g")
    logger.info("Timeseries CSV criado: %s (%d dias, lags=%s)", out_path, len(date_range), lag_days)
    return out_path


# ── 2. Attributes CSVs ────────────────────────────────────────────────────────
#
# O framework espera três arquivos CSV de atributos por subdataset.
# O nome do arquivo determina qual é qual:
#   attributes_caravan_*   → atributos climáticos (p_mean, aridity, etc.)
#   attributes_hydroatlas_* → morfologia (de HydroAtlas) — pode estar vazio
#   attributes_other_*     → metadados da estação (área, lat/lon, etc.)
#
# O índice de todos os CSVs deve ser "gauge_id" com os mesmos IDs usados
# nos arquivos NetCDF de timeseries.


def build_attributes_caravan(
    basin_id: str = BASIN_ID,
    out_dir: str | Path = "data/processed/Caravan-nc",
    # Atributos climáticos da bacia do Itajaí-Açu
    # Fonte: ERA5-Land climatology (1981–2010) — preencher após EDA
    p_mean: float = float("nan"),           # mm/dia
    pet_mean: float = float("nan"),         # mm/dia
    aridity: float = float("nan"),          # PET/P
    frac_snow: float = 0.0,                 # fração da precipitação como neve
    moisture_index: float = float("nan"),   # (P - PET) / P
    seasonality: float = float("nan"),      # índice de sazonalidade de Walsh & Lawler
    high_prec_freq: float = float("nan"),   # dias/ano com P > 5x média diária
    high_prec_dur: float = float("nan"),    # duração média eventos intensos (dias)
    low_prec_freq: float = float("nan"),    # dias/ano com P < 1 mm
    low_prec_dur: float = float("nan"),     # duração média períodos secos (dias)
) -> Path:
    """
    Cria attributes_caravan_itajai.csv.

    Estes atributos descrevem o REGIME CLIMÁTICO da bacia, derivados de
    climatologias de longo prazo (normalmente ERA5-Land 1981–2010).
    O modelo usa esses valores para entender "que tipo de bacia" é esta
    e ajustar seu comportamento antes de ver qualquer dado de vazão.

    Por que atributos climáticos e não apenas topográficos?
    ─────────────────────────────────────────────────────────
    Um LSTM treinado em bacias temperadas da Europa pode não saber que
    o Itajaí-Açu tem alta sazonalidade tropical, precipitação intensa
    concentrada em poucas horas e pouca neve (frac_snow ≈ 0 abaixo de
    500m). Os atributos de aridity, moisture_index e seasonality codificam
    essas diferenças — essenciais para o embedding generalizar corretamente.

    ATENÇÃO: Os valores padrão são NaN — você deve calculá-los da
    climatologia ERA5-Land ou CHIRPS após o download dos dados.
    """
    dirs = caravan_dirs(out_dir)

    df = pd.DataFrame(
        {
            "gauge_id": [basin_id],
            "p_mean": [p_mean],
            "pet_mean_ERA5_LAND": [pet_mean],
            "aridity_ERA5_LAND": [aridity],
            "frac_snow": [frac_snow],
            "moisture_index_ERA5_LAND": [moisture_index],
            "seasonality_ERA5_LAND": [seasonality],
            "high_prec_freq": [high_prec_freq],
            "high_prec_dur": [high_prec_dur],
            "low_prec_freq": [low_prec_freq],
            "low_prec_dur": [low_prec_dur],
        }
    ).set_index("gauge_id")

    out_path = dirs["attributes"] / f"attributes_caravan_{SUBDATASET}.csv"
    df.to_csv(out_path)
    logger.info("Caravan attributes: %s", out_path)
    return out_path


def build_attributes_hydroatlas(
    basin_id: str = BASIN_ID,
    out_dir: str | Path = "data/processed/Caravan-nc",
    hydroatlas_values: dict[str, float] | None = None,
) -> Path:
    """
    Cria attributes_hydroatlas_itajai.csv.

    HydroAtlas fornece ~60 atributos de morfologia e geologia derivados
    de datasets globais (MERIT Hydro, SoilGrids, GLC, etc.).
    Para obter os valores da bacia 83500000:
      1. Baixar HydroAtlas v1.0: https://www.hydrosheds.org/hydroatlas
      2. Fazer spatial join com o polígono da bacia
      3. Extrair os atributos listados no config de treinamento

    Enquanto esses valores não estão disponíveis, o arquivo existe com
    NaN para que o framework não quebre ao carregar os atributos.
    O modelo simplesmente não usará features que sejam todos NaN.
    """
    dirs = caravan_dirs(out_dir)

    # Atributos HydroAtlas esperados pelo config do OpenHydroNet
    hydroatlas_cols = [
        "aet_mm_syr", "ari_ix_sav", "crp_pc_sse", "ele_mt_sav", "ero_kh_sav",
        "for_pc_sse", "gla_pc_sse",
        "glc_pc_s01", "glc_pc_s02", "glc_pc_s03", "glc_pc_s04", "glc_pc_s05",
        "glc_pc_s06", "glc_pc_s07", "glc_pc_s08", "glc_pc_s09", "glc_pc_s10",
        "glc_pc_s11", "glc_pc_s12", "glc_pc_s13", "glc_pc_s14", "glc_pc_s15",
        "glc_pc_s16", "glc_pc_s17", "glc_pc_s18", "glc_pc_s19", "glc_pc_s20",
        "glc_pc_s21", "glc_pc_s22",
        "inu_pc_slt", "inu_pc_smn", "inu_pc_smx",
        "kar_pc_sse", "lka_pc_sse", "nli_ix_sav", "pac_pc_sse",
        "pet_mm_syr", "ppd_pk_sav", "pre_mm_syr", "prm_pc_sse", "rdd_mk_sav",
        "snw_pc_syr", "swc_pc_syr", "tmp_dc_syr", "urb_pc_sse",
        "wet_pc_s01", "wet_pc_s02", "wet_pc_s03", "wet_pc_s04", "wet_pc_s05",
        "wet_pc_s06", "wet_pc_s07", "wet_pc_s08", "wet_pc_s09", "wet_pc_sg1", "wet_pc_sg2",
    ]

    vals = hydroatlas_values or {}
    row = {col: vals.get(col, float("nan")) for col in hydroatlas_cols}
    row["gauge_id"] = basin_id

    df = pd.DataFrame([row]).set_index("gauge_id")
    out_path = dirs["attributes"] / f"attributes_hydroatlas_{SUBDATASET}.csv"
    df.to_csv(out_path)
    logger.info("HydroAtlas attributes: %s", out_path)
    return out_path


def build_attributes_other(
    basin_id: str = BASIN_ID,
    out_dir: str | Path = "data/processed/Caravan-nc",
    area_km2: float = 15_000.0,
    lat: float = -26.915,
    lon: float = -49.065,
    alt_m: float = 14.0,
    country: str = "Brazil",
    river_name: str = "Itajai-Acu",
    gauge_name: str = "Blumenau",
    record_start: str = "1939-01-01",
    record_end: str = "",
) -> Path:
    """
    Cria attributes_other_itajai.csv com metadados da estação.

    Estes atributos são usados pelo framework para:
    1. Plotar mapas (lat/lon)
    2. Filtrar bacias por área para análise de escala
    3. Normalizar vazão específica (m³/s/km²) se necessário
    4. Informação de proveniência dos dados

    Valores padrão para Blumenau / ANA 83500000:
      - Área: ~15.000 km² (bacia total acima de Blumenau)
      - Coordenadas: estação ANA em Blumenau
      - Altitude: ~14 m (altitude do leito do rio na estação)
    """
    dirs = caravan_dirs(out_dir)

    df = pd.DataFrame(
        {
            "gauge_id": [basin_id],
            "gauge_name": [gauge_name],
            "country": [country],
            "river_name": [river_name],
            "area_km2": [area_km2],
            "gauge_lat": [lat],
            "gauge_lon": [lon],
            "gauge_alt_m": [alt_m],
            "record_period_start": [record_start],
            "record_period_end": [record_end or pd.Timestamp.today().strftime("%Y-%m-%d")],
        }
    ).set_index("gauge_id")

    out_path = dirs["attributes"] / f"attributes_other_{SUBDATASET}.csv"
    df.to_csv(out_path)
    logger.info("Other attributes: %s", out_path)
    return out_path


# ── 3. Pipeline completo ──────────────────────────────────────────────────────


def format_caravan(
    streamflow_parquet: str | Path,
    precipitation_parquet: str | Path | None = None,
    caravan_root: str | Path = "data/processed/Caravan-nc",
    hydroatlas_json: str | Path | None = None,
    climate_stats_json: str | Path | None = None,
) -> dict[str, Path]:
    """
    Executa o pipeline completo de formatação Caravan para a bacia do Itajaí-Açu.

    Parâmetros
    ----------
    streamflow_parquet : str | Path
        Série de vazão processada (saída de preprocessor.py), com coluna 'value'.
    precipitation_parquet : str | Path | None
        Série de precipitação diária (INMET ou CHIRPS), com coluna 'precip_mm'
        ou 'precip_mm_chirps'. Opcional — pode ser adicionada depois.
    caravan_root : str | Path
        Raiz do diretório Caravan. Vai apontar neste campo do YAML de treino:
            targets_data_dir: {caravan_root}
            statics_data_dir: {caravan_root}
    hydroatlas_json : str | Path | None
        JSON com atributos HydroAtlas (gauge_id → {attr: value}).
        Gerado separadamente via extração espacial do HydroAtlas.
    climate_stats_json : str | Path | None
        JSON com estatísticas climáticas da bacia (p_mean, aridity, etc.).
        Gerado via compute_climate_stats() após download do CHIRPS/ERA5.

    Retorna
    -------
    dict com caminhos dos arquivos criados.
    """
    # Carregar vazão processada
    df_q = pd.read_parquet(streamflow_parquet)
    streamflow = df_q["value"].rename("streamflow")

    # Carregar precipitação se disponível
    precip_df = None
    if precipitation_parquet:
        precip_df = pd.read_parquet(precipitation_parquet)
        # Normalizar nome da coluna
        col = next((c for c in precip_df.columns if "precip" in c.lower()), None)
        if col and col != "precip_mm":
            precip_df = precip_df.rename(columns={col: "precip_mm"})

    # Carregar atributos externos se fornecidos
    hydroatlas_vals = {}
    if hydroatlas_json:
        with open(hydroatlas_json) as f:
            hydroatlas_vals = json.load(f).get(BASIN_ID, {})

    climate_vals: dict[str, float] = {}
    if climate_stats_json:
        with open(climate_stats_json) as f:
            climate_vals = json.load(f)

    # Criar arquivos
    paths = {}
    paths["timeseries_nc"] = build_timeseries_nc(streamflow, precip_df, out_dir=caravan_root)
    paths["timeseries_csv"] = build_timeseries_csv(streamflow, precip_df, out_dir=caravan_root)
    paths["attr_caravan"] = build_attributes_caravan(out_dir=caravan_root, **climate_vals)
    paths["attr_hydroatlas"] = build_attributes_hydroatlas(
        out_dir=caravan_root, hydroatlas_values=hydroatlas_vals
    )
    paths["attr_other"] = build_attributes_other(out_dir=caravan_root)

    logger.info("Caravan formatado com sucesso. Arquivos criados:")
    for key, path in paths.items():
        logger.info("  %s: %s", key, path)

    return paths


def generate_basins_list(
    caravan_root: str | Path,
    out_path: str | Path = "configs/training/itajai_basins.txt",
) -> Path:
    """
    Gera o arquivo de lista de bacias (.txt) necessário pelo config YAML.

    O YAML espera:
        train_basin_file: configs/training/itajai_basins.txt

    Cada linha é um basin ID no formato "{subdataset}_{gauge_id}".
    Prioriza o diretório CSV (usado com load_as_csv: true), com fallback
    para NetCDF.
    """
    caravan_root = Path(caravan_root)
    csv_dir = caravan_root / "timeseries" / "csv" / SUBDATASET
    nc_dir = caravan_root / "timeseries" / "netcdf" / SUBDATASET

    if csv_dir.is_dir() and any(csv_dir.glob("*.csv")):
        basin_ids = [f.stem for f in csv_dir.glob("*.csv")]
    else:
        basin_ids = [f.stem for f in nc_dir.glob("*.nc")]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sorted(basin_ids)) + "\n")
    logger.info("Lista de bacias gerada: %s (%d bacias)", out_path, len(basin_ids))
    return out_path


# ── Utilitários de diagnóstico ────────────────────────────────────────────────


def validate_caravan_structure(caravan_root: str | Path) -> dict[str, bool]:
    """
    Verifica se a estrutura Caravan está completa e retorna dict de checks.

    Útil para rodar antes de iniciar o treinamento — evita erros crípticos
    de FileNotFoundError no meio do primeiro epoch.
    Verifica tanto CSV (usado com load_as_csv: true) quanto NetCDF.
    """
    root = Path(caravan_root)
    checks: dict[str, bool] = {}

    # CSV timeseries (necessário para load_as_csv: true)
    csv_dir = root / "timeseries" / "csv" / SUBDATASET
    checks["timeseries_csv_dir"] = csv_dir.is_dir()
    csv_file = csv_dir / f"{BASIN_ID}.csv"
    checks["timeseries_csv_file"] = csv_file.is_file()

    if csv_file.is_file():
        try:
            df = pd.read_csv(csv_file, nrows=5)
            checks["csv_has_date"] = "date" in df.columns
            checks["csv_has_streamflow"] = "streamflow" in df.columns
            checks["csv_has_precipitation"] = "total_precipitation" in df.columns
        except Exception:
            checks["csv_has_date"] = False
            checks["csv_has_streamflow"] = False
            checks["csv_has_precipitation"] = False

    # NetCDF timeseries (opcional, mantido como referência)
    nc_dir = root / "timeseries" / "netcdf" / SUBDATASET
    checks["timeseries_nc_dir"] = nc_dir.is_dir()
    checks["timeseries_nc"] = any(nc_dir.glob("*.nc")) if nc_dir.is_dir() else False

    # Atributos estáticos
    attr_dir = root / "attributes" / SUBDATASET
    checks["attributes_dir"] = attr_dir.is_dir()
    for kind in ("caravan", "hydroatlas", "other"):
        fname = f"attributes_{kind}_{SUBDATASET}.csv"
        checks[f"attr_{kind}"] = (attr_dir / fname).is_file() if attr_dir.is_dir() else False

    # Verificar conteúdo do NetCDF (se existir)
    nc_files = list(nc_dir.glob("*.nc")) if nc_dir.is_dir() else []
    if nc_files:
        try:
            ds = xr.open_dataset(nc_files[0])
            checks["nc_has_streamflow"] = "streamflow" in ds.data_vars
            checks["nc_date_complete"] = not bool(
                pd.date_range(
                    ds["date"].values[0], ds["date"].values[-1], freq="D"
                )
                .difference(pd.DatetimeIndex(ds["date"].values))
                .size
            )
            ds.close()
        except Exception:
            checks["nc_has_streamflow"] = False
            checks["nc_date_complete"] = False

    ok = all(checks.values())
    logger.info("Validação Caravan: %s", "OK" if ok else "PROBLEMAS ENCONTRADOS")
    for k, v in checks.items():
        if not v:
            logger.warning("  FALHOU: %s", k)
    return checks


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Formata dados locais para Caravan NetCDF")
    parser.add_argument("streamflow_parquet", help="Parquet processado de vazão")
    parser.add_argument("--precip", default=None, help="Parquet de precipitação diária")
    parser.add_argument("--caravan-root", default="data/processed/Caravan-nc")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    if args.validate:
        checks = validate_caravan_structure(args.caravan_root)
        print(json.dumps(checks, indent=2))
    else:
        paths = format_caravan(args.streamflow_parquet, args.precip, args.caravan_root)
        print("Arquivos criados:")
        for k, v in paths.items():
            print(f"  {k}: {v}")
