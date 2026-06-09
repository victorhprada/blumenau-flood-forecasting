"""
Cálculo de PET via Hargreaves-Samani usando temperatura da NASA POWER API.

A NASA POWER fornece dados de reanálise derivados de ERA5 + MERRA-2,
acessíveis gratuitamente via HTTP sem API key.

Método de Hargreaves-Samani (1985):
    PET = 0.0023 × (T_mean + 17.8) × ΔT^0.5 × Ra

Onde:
    T_mean = (T_max + T_min) / 2  [°C]
    ΔT     = T_max - T_min        [°C — proxy da demanda evaporativa]
    Ra     = radiação extraterrestre [MJ/m²/dia]

Hargreaves-Samani é o método recomendado pela FAO quando não há dados
de umidade, vento e radiação solar observados. Válido para bacias
subtropicais úmidas como o Itajaí-Açu (erros típicos < 10%).

Atributos derivados (convenção Caravan / CAMELS):
    pet_mean         — média diária de PET [mm/dia]
    aridity          — PET_anual / P_anual  (> 1 = árido; < 1 = úmido)
    moisture_index   — (P - PET) / P        (> 0 = excesso hídrico)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Centróide aproximado da bacia do Itajaí-Açu ──────────────────────────────
# Derivado do bounding box CHIRPS: lat[-27.6, -26.5], lon[-50.2, -48.6]
BASIN_LAT = -27.05
BASIN_LON = -49.40

POWER_BASE = "https://power.larc.nasa.gov/api/temporal/daily/point"


# ── NASA POWER download ───────────────────────────────────────────────────────

def download_temperature(
    lat: float = BASIN_LAT,
    lon: float = BASIN_LON,
    start: str = "19950101",
    end: str = "20241231",
    cache_path: str | Path | None = None,
) -> pd.DataFrame:
    """
    Baixa T2M_MAX e T2M_MIN do NASA POWER API para um ponto.

    Retorna DataFrame com colunas [T_max, T_min] indexado por data.
    """
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            logger.info("Temperatura carregada do cache: %s", cache_path)
            return pd.read_parquet(cache_path)

    url = POWER_BASE
    params = {
        "parameters": "T2M_MAX,T2M_MIN",
        "community":  "RE",
        "longitude":  lon,
        "latitude":   lat,
        "start":      start,
        "end":        end,
        "format":     "JSON",
    }

    logger.info("Baixando temperatura do NASA POWER (%s → %s)...", start, end)
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=120)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            if attempt == 2:
                raise
            logger.warning("Tentativa %d falhou: %s. Aguardando 5 s...", attempt + 1, e)
            time.sleep(5)

    data = resp.json()["properties"]["parameter"]
    df = pd.DataFrame({
        "T_max": pd.Series(data["T2M_MAX"]),
        "T_min": pd.Series(data["T2M_MIN"]),
    })
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df.index.name = "date"

    # NASA POWER usa -999 para dados faltantes
    df.replace(-999.0, np.nan, inplace=True)
    df = df.astype("float32")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        logger.info("Cache salvo: %s", cache_path)

    logger.info("Temperatura baixada: %d dias, T_mean médio %.1f°C",
                len(df), ((df["T_max"] + df["T_min"]) / 2).mean())
    return df


# ── Radiação extraterrestre (Ra) ──────────────────────────────────────────────

def calc_extraterrestrial_radiation(doy: np.ndarray, lat_deg: float) -> np.ndarray:
    """
    Calcula Ra [MJ/m²/dia] para cada dia do ano.

    Fórmula FAO-56 (Allen et al., 1998), Eq. 21.

    Parâmetros físicos:
        dr    = fator de correção da distância Terra-Sol (eccentricidade)
        delta = declinação solar (ângulo entre o plano equatorial e o Sol)
        ws    = ângulo horário ao pôr-do-sol
    """
    lat_rad = np.deg2rad(lat_deg)
    dr    = 1 + 0.033 * np.cos(2 * np.pi / 365 * doy)
    delta = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    ws    = np.arccos(-np.tan(lat_rad) * np.tan(delta))
    Ra    = (
        (24 * 60 / np.pi)
        * 0.0820
        * dr
        * (
            ws * np.sin(lat_rad) * np.sin(delta)
            + np.cos(lat_rad) * np.cos(delta) * np.sin(ws)
        )
    )
    return Ra.astype("float32")


# ── Hargreaves-Samani PET ────────────────────────────────────────────────────

def calc_pet_hargreaves(temp_df: pd.DataFrame, lat_deg: float = BASIN_LAT) -> pd.Series:
    """
    Calcula PET diária pelo método de Hargreaves-Samani [mm/dia].

    PET = 0.0023 × (T_mean + 17.8) × ΔT^0.5 × Ra

    Validação para o Itajaí-Açu:
    - PET média anual esperada: ~850–1000 mm/ano (~2.3–2.7 mm/dia)
    - P média anual: ~1650 mm/ano → aridity ≈ 0.55–0.65 (úmido)
    - Para comparação: Blumenau tem Cfa (Köppen) — subtropical sem estação seca
    """
    t_max = temp_df["T_max"].values
    t_min = temp_df["T_min"].values
    t_mean = (t_max + t_min) / 2

    doy = temp_df.index.day_of_year.values
    Ra  = calc_extraterrestrial_radiation(doy, lat_deg)

    delta_t = np.maximum(t_max - t_min, 0)  # ΔT nunca negativo
    # 0.408 converte Ra de MJ/m²/dia → mm/dia (λ=2.45 MJ/kg, ρ=1000 kg/m³)
    pet     = 0.0023 * (t_mean + 17.8) * np.sqrt(delta_t) * Ra * 0.408
    pet     = np.maximum(pet, 0)            # PET ≥ 0

    return pd.Series(pet.astype("float32"), index=temp_df.index, name="pet_mm_day")


# ── Atributos estáticos derivados ────────────────────────────────────────────

def calc_static_attributes(
    pet: pd.Series,
    precip: pd.Series,
    start_year: int = 1995,
    end_year: int = 2024,
) -> dict[str, float]:
    """
    Deriva atributos estáticos de clima (convenção Caravan / CAMELS).

    pet_mean_ERA5_LAND:
        Média diária de PET [mm/dia] sobre o período de referência.
        Representa a demanda evaporativa do clima. Para o Itajaí-Açu,
        valores esperados 2.3–2.7 mm/dia (subtropical úmido).

    aridity_ERA5_LAND (Budyko 1974):
        φ = PET_anual / P_anual  (adimensional)
        φ < 1 → energia é o fator limitante (bacia úmida) → evaporação real < PET
        φ > 1 → água é o fator limitante (bacia árida)
        Itajaí-Açu: φ ≈ 0.55–0.65 → clima úmido com escoamento robusto.
        Físicamente: quanto de chuva "sobra" após satisfazer a evaporação.

    moisture_index_ERA5_LAND:
        MI = (P - PET) / P  (adimensional, entre -∞ e 1)
        MI > 0 → excesso hídrico → alimenta escoamento e recarga de aquíferos
        MI < 0 → déficit hídrico → bacia seca. Para Itajaí: MI ≈ 0.35–0.45.
        É o complemento do aridity index — mede a "margem de segurança hídrica".
    """
    # Filtrar pelo período de referência
    mask = (pet.index.year >= start_year) & (pet.index.year <= end_year)
    pet_ref    = pet.loc[mask].dropna()
    precip_ref = precip.loc[mask].dropna()

    # Alinhar índices
    common = pet_ref.index.intersection(precip_ref.index)
    pet_ref    = pet_ref.loc[common]
    precip_ref = precip_ref.loc[common]

    pet_mean_day  = float(pet_ref.mean())           # mm/dia
    pet_annual    = float(pet_ref.mean() * 365.25)  # mm/ano
    p_annual      = float(precip_ref.mean() * 365.25)

    aridity       = pet_annual / p_annual if p_annual > 0 else float("nan")
    moisture_idx  = (p_annual - pet_annual) / p_annual if p_annual > 0 else float("nan")

    attrs = {
        "pet_mean_ERA5_LAND":      round(pet_mean_day, 4),
        "aridity_ERA5_LAND":       round(aridity, 4),
        "moisture_index_ERA5_LAND": round(moisture_idx, 4),
    }

    logger.info("Atributos calculados:")
    for k, v in attrs.items():
        logger.info("  %s = %.4f", k, v)

    return attrs


# ── Pipeline principal ────────────────────────────────────────────────────────

def compute_era5_attributes(
    precip_parquet: str | Path,
    cache_dir: str | Path = "data/raw/era5",
    lat: float = BASIN_LAT,
    lon: float = BASIN_LON,
    start: str = "19950101",
    end: str = "20241231",
) -> dict[str, float]:
    """
    Pipeline completo: download → PET → atributos estáticos.

    Parâmetros
    ----------
    precip_parquet : path para o parquet CHIRPS (precipitação diária média da bacia)
    cache_dir      : diretório para cache dos dados de temperatura NASA POWER
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / "nasa_power_temperature.parquet"

    # 1. Download temperatura
    temp_df = download_temperature(lat=lat, lon=lon, start=start, end=end,
                                   cache_path=cache_path)

    # 2. PET Hargreaves-Samani
    pet = calc_pet_hargreaves(temp_df, lat_deg=lat)
    logger.info("PET calculado: %.2f mm/dia (média), %.0f mm/ano (estimativa)",
                pet.mean(), pet.mean() * 365.25)

    # 3. Carregar precipitação
    p_df = pd.read_parquet(precip_parquet)
    p_df.index = pd.to_datetime(p_df.index)
    precip = p_df.iloc[:, 0]

    # 4. Atributos estáticos
    attrs = calc_static_attributes(pet, precip)

    # 5. Salvar PET como parquet para referência futura
    pet_path = cache_dir / "pet_hargreaves_daily.parquet"
    pet.to_frame().to_parquet(pet_path)
    logger.info("PET diário salvo: %s", pet_path)

    return attrs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    attrs = compute_era5_attributes(
        precip_parquet="data/raw/chirps/chirps_itajai_mean_all.parquet",
    )
    print("\n=== ATRIBUTOS ERA5/POWER ===")
    for k, v in attrs.items():
        print(f"  {k}: {v:.4f}")
