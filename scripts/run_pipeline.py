#!/usr/bin/env python3
"""
Pipeline completo: CHIRPS → Caravan → Validação → Treinamento.
Roda após o download CHIRPS estar concluído.
"""
import sys, json, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "flood-forecasting"))

import pandas as pd

CHIRPS_DIR = ROOT / "data/raw/chirps"
ANA_PROCESSED = ROOT / "data/processed/ana/83500000_vazao_processed.parquet"
CARAVAN_ROOT  = ROOT / "data/processed/Caravan-nc"
CONFIG_FILE   = ROOT / "configs/training/mef_lstm_itajai.yml"
BASINS_TXT    = ROOT / "configs/training/itajai_basins.txt"


def step1_concat_chirps() -> pd.DataFrame:
    """Concatena parquets CHIRPS anuais em série diária única."""
    log.info("=== Passo 1: Concatenando CHIRPS ===")
    parquets = sorted(CHIRPS_DIR.glob("chirps_*_itajai_mean.parquet"))
    if not parquets:
        raise FileNotFoundError(f"Nenhum parquet CHIRPS em {CHIRPS_DIR}")
    log.info("Parquets encontrados: %d", len(parquets))

    dfs = []
    for p in parquets:
        df = pd.read_parquet(p)
        df["date"] = pd.to_datetime(df["date"])
        dfs.append(df)

    chirps = pd.concat(dfs).sort_values("date").reset_index(drop=True)
    chirps = chirps.set_index("date")
    chirps.index = pd.DatetimeIndex(chirps.index).normalize()
    chirps = chirps[~chirps.index.duplicated(keep="first")]

    out = CHIRPS_DIR / "chirps_itajai_mean_all.parquet"
    chirps.to_parquet(out)
    log.info(
        "CHIRPS concatenado: %d dias | %s a %s | média=%.2f mm/dia",
        len(chirps),
        chirps.index.min().date(),
        chirps.index.max().date(),
        chirps["precip_mm_chirps"].mean(),
    )
    return chirps


def step2_format_caravan(chirps_df: pd.DataFrame) -> dict:
    """Cria estrutura Caravan (NetCDF + CSV + atributos)."""
    log.info("=== Passo 2: Formatando Caravan ===")
    from src.data.caravan_formatter import format_caravan, generate_basins_list

    if not ANA_PROCESSED.exists():
        raise FileNotFoundError(f"Parquet de vazão processada não encontrado: {ANA_PROCESSED}")

    paths = format_caravan(
        streamflow_parquet=ANA_PROCESSED,
        precipitation_parquet=CHIRPS_DIR / "chirps_itajai_mean_all.parquet",
        caravan_root=CARAVAN_ROOT,
    )
    log.info("Arquivos Caravan criados:")
    for k, v in paths.items():
        log.info("  %s: %s", k, v)

    generate_basins_list(CARAVAN_ROOT, out_path=BASINS_TXT)
    log.info("Basins list: %s", BASINS_TXT)
    return paths


def step3_validate() -> bool:
    """Valida estrutura Caravan antes de treinar."""
    log.info("=== Passo 3: Validando estrutura Caravan ===")
    from src.data.caravan_formatter import validate_caravan_structure

    checks = validate_caravan_structure(CARAVAN_ROOT)
    all_ok = all(checks.values())
    for k, v in checks.items():
        status = "OK" if v else "FALHOU"
        log.info("  [%s] %s", status, k)

    if not all_ok:
        log.error("Validação falhou. Corrija os problemas antes de treinar.")
    return all_ok


def step4_train():
    """Executa o treinamento via CLI do googlehydrology."""
    import subprocess
    log.info("=== Passo 4: Treinamento ===")
    log.info("Config: %s", CONFIG_FILE)

    cmd = ["run", "train", str(CONFIG_FILE)]
    log.info("Comando: %s", " ".join(cmd))

    result = subprocess.run(cmd, cwd=ROOT, capture_output=False)
    if result.returncode != 0:
        log.error("Treinamento falhou com código %d", result.returncode)
        return False
    log.info("Treinamento concluído.")
    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true",
                        help="Pular etapa de treinamento (só prepara dados)")
    args = parser.parse_args()

    try:
        chirps = step1_concat_chirps()
        paths  = step2_format_caravan(chirps)
        ok     = step3_validate()
        if not ok:
            sys.exit(1)
        if not args.skip_train:
            step4_train()
        else:
            log.info("--skip-train: pulando treinamento. Execute manualmente:")
            log.info("  source .venv/bin/activate && run train %s", CONFIG_FILE)
    except Exception as e:
        log.exception("Erro no pipeline: %s", e)
        sys.exit(1)
