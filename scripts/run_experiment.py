#!/usr/bin/env python3
"""
Executor de experimentos com MLflow tracking.

Uso:
    python scripts/run_experiment.py configs/training/experiments/02_lag_features.yml
    python scripts/run_experiment.py configs/training/experiments/03_nse_loss.yml --epoch-select best_val_nse

O script:
  1. Lê o config YAML e registra todos os parâmetros no MLflow
  2. Executa 'run train' e captura métricas por epoch do log
  3. Seleciona o melhor epoch (por val NSE)
  4. Executa 'run infer --period test' no melhor epoch
  5. Calcula e registra métricas de teste (global + eventos de cheia)
  6. Plota hidrogramas e os salva como artefatos MLflow
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import mlflow
import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "flood-forecasting"))

MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlruns.db'}"
MLFLOW_EXPERIMENT   = "itajai-mef-lstm"

VENV_RUN = str(ROOT / ".venv" / "bin" / "run")


# ── Utilidades ────────────────────────────────────────────────────────────────

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) == 0:
        return float("nan")
    return float(1 - np.sum((o - s) ** 2) / np.sum((o - o.mean()) ** 2))


def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if len(o) < 2:
        return float("nan")
    r = float(np.corrcoef(o, s)[0, 1])
    alpha = float(s.std() / o.std()) if o.std() > 0 else float("nan")
    beta  = float(s.mean() / o.mean()) if o.mean() != 0 else float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def pbias(obs: np.ndarray, sim: np.ndarray) -> float:
    mask = ~(np.isnan(obs) | np.isnan(sim))
    o, s = obs[mask], sim[mask]
    if o.sum() == 0:
        return float("nan")
    return float(100 * (s.sum() - o.sum()) / o.sum())


# ── Parsing do log ────────────────────────────────────────────────────────────

_EPOCH_RE = re.compile(
    r"Epoch (\d+) average loss: .*?avg_loss: ([\d.]+)"
)
_VAL_RE = re.compile(
    r"Epoch (\d+) average validation loss: ([\d.]+) -- Median validation metrics: "
    r".*?NSE: ([-\d.]+).*?KGE: ([-\d.]+)"
)


def parse_training_log(log_path: Path) -> pd.DataFrame:
    rows = []
    text = log_path.read_text()
    train_losses = {int(m.group(1)): float(m.group(2)) for m in _EPOCH_RE.finditer(text)}
    for m in _VAL_RE.finditer(text):
        ep  = int(m.group(1))
        rows.append({
            "epoch":       ep,
            "train_loss":  train_losses.get(ep, float("nan")),
            "val_loss":    float(m.group(2)),
            "val_nse":     float(m.group(3)),
            "val_kge":     float(m.group(4)),
        })
    return pd.DataFrame(rows)


# ── Treinamento ───────────────────────────────────────────────────────────────

def run_training(config_path: Path) -> Path:
    """Executa 'run train' e retorna o diretório do experimento."""
    cmd = [VENV_RUN, "train", "--config-file", str(config_path)]
    print(f"\n{'='*60}")
    print(f"TREINAMENTO: {config_path.name}")
    print(f"Comando: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Treinamento falhou (código {result.returncode})")

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(cfg["run_dir"])
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir

    # Framework cria subdiretório com timestamp: {experiment_name}_{timestamp}
    exp_name = cfg["experiment_name"]
    candidates = sorted(run_dir.glob(f"{exp_name}_*"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"Nenhum diretório de experimento encontrado em {run_dir}")
    return candidates[-1]


# ── Inferência ────────────────────────────────────────────────────────────────

def run_inference(exp_dir: Path, epoch: int) -> Path:
    """Executa 'run infer' no epoch especificado e retorna o diretório de resultados."""
    cmd = [VENV_RUN, "infer",
           "--run-dir", str(exp_dir),
           "--epoch", str(epoch),
           "--period", "test"]
    print(f"\nINFERÊNCIA: epoch={epoch}")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise RuntimeError(f"Inferência falhou (código {result.returncode})")

    results_dir = exp_dir / "test" / f"model_epoch{epoch:03d}"
    if not results_dir.exists():
        raise FileNotFoundError(f"Diretório de resultados não encontrado: {results_dir}")
    return results_dir


# ── Métricas de teste ─────────────────────────────────────────────────────────

def compute_test_metrics(results_dir: Path) -> dict[str, float]:
    zarr_path = results_dir / "test_results.zarr"
    if not zarr_path.exists():
        raise FileNotFoundError(f"test_results.zarr não encontrado em {results_dir}")

    ds = xr.open_zarr(str(zarr_path), consolidated=False).compute()
    basin = str(ds.basin.values[0])
    # time_step=0 no zarr = lead_time=7 (7 dias à frente)
    # time_step=-1 no zarr = lead_time=0 (hindcast)
    obs   = ds["streamflow_obs"].sel(basin=basin, freq="1D").isel(time_step=0).values
    sim7  = ds["streamflow_sim"].sel(basin=basin, freq="1D").isel(time_step=0).values
    sim0  = ds["streamflow_sim"].sel(basin=basin, freq="1D").isel(time_step=-1).values
    dates = pd.DatetimeIndex(ds.date.values)

    metrics = {
        "test/nse_t0":   nse(obs, sim0),
        "test/nse_t7":   nse(obs, sim7),
        "test/kge_t7":   kge(obs, sim7),
        "test/pbias_t7": pbias(obs, sim7),
    }

    # Eventos de cheia
    events = {
        "nov2008": ("2008-10-01", "2008-12-15"),
        "sep2011": ("2011-08-01", "2011-10-15"),
    }
    for ev, (t0, t1) in events.items():
        mask = (dates >= t0) & (dates <= t1)
        if mask.sum() == 0:
            continue
        o, s = obs[mask], sim7[mask]
        metrics[f"test/nse_{ev}"] = nse(o, s)
        metrics[f"test/kge_{ev}"] = kge(o, s)
        metrics[f"test/obs_peak_{ev}"] = float(np.nanmax(o))
        metrics[f"test/sim_peak_{ev}"] = float(np.nanmax(s))

    return metrics, ds


def plot_hydrographs(ds: xr.Dataset, exp_name: str, out_dir: Path) -> Path:
    basin = str(ds.basin.values[0])
    obs_all  = ds["streamflow_obs"].sel(basin=basin, freq="1D").isel(time_step=0).to_pandas()
    sim_lt7  = ds["streamflow_sim"].sel(basin=basin, freq="1D").isel(time_step=0).to_pandas()
    sim_lt0  = ds["streamflow_sim"].sel(basin=basin, freq="1D").isel(time_step=-1).to_pandas()

    events = [
        ("Cheia de Novembro 2008", "2008-10-01", "2008-12-15"),
        ("Cheia de Setembro 2011", "2011-08-01", "2011-10-15"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9))
    fig.suptitle(
        f"MEF-LSTM · {exp_name}\nBacia do Itajaí-Açu (Blumenau 83500000)",
        fontsize=12, fontweight="bold",
    )
    for ax, (title, t0, t1) in zip(axes, events):
        o  = obs_all.loc[t0:t1]
        s0 = sim_lt0.loc[t0:t1]
        s7 = sim_lt7.loc[t0:t1]
        n0 = nse(o.values, s0.values)
        n7 = nse(o.values, s7.values)

        ax.fill_between(o.index, 0, o.values, alpha=0.20, color="steelblue")
        ax.plot(o.index, o.values, color="steelblue", lw=2.0,
                label=f"Observado (pico {o.max():.0f} m³/s)")
        ax.plot(s0.index, s0.values, color="tomato", lw=1.8,
                label=f"t+0  NSE={n0:.3f}")
        ax.plot(s7.index, s7.values, color="darkorange", lw=1.8, linestyle="--",
                label=f"t+7  NSE={n7:.3f}")
        ax.axhline(obs_all.mean(), color="gray", lw=0.8, linestyle=":", alpha=0.6,
                   label=f"Média ({obs_all.mean():.0f} m³/s)")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("Vazão (m³/s)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%b/%Y"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = out_dir / f"{exp_name}_hydrographs.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Hidrogram salvo: {out_path}")
    return out_path


# ── Pipeline principal ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Executa experimento MEF-LSTM com MLflow")
    parser.add_argument("config", type=Path, help="Path para o config YAML do experimento")
    parser.add_argument("--epoch-select", choices=["best_val_nse", "last"],
                        default="best_val_nse")
    parser.add_argument("--skip-train", action="store_true",
                        help="Pula treinamento; usa último run_dir existente")
    args = parser.parse_args()

    config_path = args.config if args.config.is_absolute() else ROOT / args.config
    if not config_path.exists():
        sys.exit(f"Config não encontrado: {config_path}")

    with open(config_path) as f:
        cfg_yaml = yaml.safe_load(f)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=cfg_yaml["experiment_name"]):
        # ── Log de parâmetros ─────────────────────────────────────────────────
        flat_params = {
            "experiment_name": cfg_yaml.get("experiment_name"),
            "loss":            cfg_yaml.get("loss"),
            "seq_length":      cfg_yaml.get("seq_length"),
            "lead_time":       cfg_yaml.get("lead_time"),
            "hidden_size":     cfg_yaml.get("hidden_size"),
            "epochs":          cfg_yaml.get("epochs"),
            "batch_size":      cfg_yaml.get("batch_size"),
            "learning_rate":   cfg_yaml.get("initial_learning_rate"),
            "hindcast_inputs": str(cfg_yaml.get("hindcast_inputs")),
            "static_attributes": str(cfg_yaml.get("static_attributes")),
        }
        mlflow.log_params(flat_params)
        mlflow.log_artifact(str(config_path), artifact_path="config")

        # ── Treinamento ───────────────────────────────────────────────────────
        if args.skip_train:
            run_dir_base = Path(cfg_yaml["run_dir"])
            if not run_dir_base.is_absolute():
                run_dir_base = ROOT / run_dir_base
            candidates = sorted(run_dir_base.glob(f"{cfg_yaml['experiment_name']}_*"),
                                 key=lambda p: p.stat().st_mtime)
            if not candidates:
                sys.exit(f"Nenhum run encontrado em {run_dir_base}")
            exp_dir = candidates[-1]
            print(f"Usando run existente: {exp_dir}")
        else:
            exp_dir = run_training(config_path)
            print(f"\nExperimento salvo em: {exp_dir}")

        # ── Parse e log de métricas por epoch ────────────────────────────────
        log_path = exp_dir / "output.log"
        epoch_df = parse_training_log(log_path)
        mlflow.log_artifact(str(log_path), artifact_path="logs")

        for _, row in epoch_df.iterrows():
            ep = int(row["epoch"])
            mlflow.log_metric("train_loss", row["train_loss"], step=ep)
            mlflow.log_metric("val_loss",   row["val_loss"],   step=ep)
            mlflow.log_metric("val_nse",    row["val_nse"],    step=ep)
            mlflow.log_metric("val_kge",    row["val_kge"],    step=ep)

        # ── Selecionar melhor epoch ───────────────────────────────────────────
        if args.epoch_select == "best_val_nse" and not epoch_df.empty:
            best_row = epoch_df.loc[epoch_df["val_nse"].idxmax()]
            best_epoch = int(best_row["epoch"])
        else:
            best_epoch = int(epoch_df["epoch"].max()) if not epoch_df.empty else cfg_yaml.get("epochs", 30)

        print(f"\nMelhor epoch: {best_epoch}  (val_NSE={epoch_df.loc[epoch_df['epoch']==best_epoch, 'val_nse'].values[0]:.4f})")
        mlflow.log_param("best_epoch", best_epoch)

        # ── Inferência de teste ───────────────────────────────────────────────
        results_dir = run_inference(exp_dir, best_epoch)

        # ── Métricas de teste ─────────────────────────────────────────────────
        test_metrics, ds = compute_test_metrics(results_dir)
        mlflow.log_metrics(test_metrics)

        print("\n=== MÉTRICAS DE TESTE ===")
        for k, v in test_metrics.items():
            if not isinstance(v, float) or not (k.endswith("peak_nov2008") or k.endswith("peak_sep2011")):
                print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

        # ── Hidrogramas ───────────────────────────────────────────────────────
        fig_dir = ROOT / "reports" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig_path = plot_hydrographs(ds, cfg_yaml["experiment_name"], fig_dir)
        mlflow.log_artifact(str(fig_path), artifact_path="figures")

        print(f"\nMLflow run id: {mlflow.active_run().info.run_id}")
        print(f"Para visualizar: mlflow ui --backend-store-uri '{MLFLOW_TRACKING_URI}'")


if __name__ == "__main__":
    main()
