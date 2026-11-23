"""Training-run summaries for the training notebooks: status, GPU time, curves.

Reads only files written by the wrappers (``results.csv``, ``weights/``, the GPU time log), so it
works for finished, interrupted and capped runs alike.

The validation metric differs per framework: YOLO's curve is Ultralytics' own validator
(``metrics/mAP50-95(M)``), the torch models use pycocotools mask AP (``val_segm_AP``). The two are
not directly comparable; the comparable numbers come from ``iiss.predict`` + ``iiss.evaluate``.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from iiss.models.common import apply_smoke, gpu_time_log, run_dir
from iiss.utils.config import deep_merge, load_run_config
from iiss.utils.paths import repo_root

# framework column -> normalised name, per model
YOLO_COLUMNS = {
    "metrics/mAP50-95(M)": "val_mask_ap",
    "metrics/mAP50(M)": "val_mask_ap50",
    "lr/pg0": "lr",
}
YOLO_LOSSES = ("train/box_loss", "train/seg_loss", "train/cls_loss", "train/dfl_loss")
TORCH_COLUMNS = {
    "val_segm_AP": "val_mask_ap",
    "val_segm_AP50": "val_mask_ap50",
    "loss": "train_loss",
}


def notebook_config(
    model: str,
    size: int,
    seed: int = 42,
    smoke: bool = False,
    runs_root: str | None = None,
) -> dict[str, Any]:
    """Resolved config of ``configs/runs/<model>_s<size>_seed<seed>.yaml`` for a notebook.

    ``runs_root`` moves the run directory and GPU time log (e.g. to Google Drive on Colab);
    ``smoke`` applies the smoke-test limits (S=320, ``runs/smoke/``).
    """
    cfg = load_run_config(repo_root() / "configs" / "runs" / f"{model}_s{size}_seed{seed}.yaml")
    if runs_root is not None:
        paths = {"runs_root": runs_root, "gpu_time_log": f"{runs_root}/gpu_time_log.csv"}
        cfg = deep_merge(cfg, {"paths": paths})
    return apply_smoke(cfg) if smoke else cfg


def config_table(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Key settings of a run (one row per setting) for display."""
    rows = {
        "run_id": cfg["run_id"],
        "input_size": cfg["input_size"],
        "time_cap_hours": cfg.get("time_cap_hours"),
        "run_dir": str(run_dir(cfg)),
        **{f"train.{k}": v for k, v in cfg["train"].items()},
    }
    return pd.Series(rows, dtype=object).to_frame("value")


def val_metric_source(model: str) -> str:
    return "Ultralytics val (mask mAP50-95)" if model == "yolo" else "pycocotools (segm AP)"


def load_history(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Per-epoch history with normalised columns: epoch, train_loss, val_mask_ap, ..., lr.

    ``train_loss`` for YOLO is the sum of its four training losses. Rows without validation
    have NaN in the ``val_*`` columns. Empty frame if the run has no ``results.csv`` yet.
    """
    path = run_dir(cfg) / "results.csv"
    if not path.is_file():
        return pd.DataFrame(columns=["epoch", "train_loss", "val_mask_ap", "val_mask_ap50", "lr"])
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    if cfg["model"] == "yolo":
        df = raw.rename(columns=YOLO_COLUMNS)
        df["train_loss"] = raw[list(YOLO_LOSSES)].sum(axis=1)
    else:
        df = raw.rename(columns=TORCH_COLUMNS)
    keep = ["epoch", "train_loss", "val_mask_ap", "val_mask_ap50", "lr"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)


def gpu_sessions(cfg: Mapping[str, Any]) -> pd.DataFrame:
    """Rows of the GPU time log for this run id (one per training session)."""
    path = gpu_time_log(cfg)
    if not path.is_file():
        return pd.DataFrame()
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("run_id") == cfg["run_id"]]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["wall_clock_s"] = df["wall_clock_s"].astype(float)
    return df


def run_summary(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Where a run stands: checkpoints, epochs done, GPU time used vs cap, last/best metric."""
    out = run_dir(cfg)
    hist = load_history(cfg)
    sessions = gpu_sessions(cfg)
    used = float(sessions["wall_clock_s"].sum()) if not sessions.empty else 0.0
    cap = cfg.get("time_cap_hours")
    epochs = int(cfg["train"]["epochs"])
    if cfg.get("smoke_mode"):
        epochs = min(epochs, int(cfg["smoke"]["max_epochs"]))
    val = hist.dropna(subset=["val_mask_ap"]) if "val_mask_ap" in hist else hist.iloc[0:0]
    best = val.loc[val["val_mask_ap"].idxmax()] if not val.empty else None
    return {
        "run_id": cfg["run_id"],
        "run_dir": str(out),
        "last_pt": (out / "weights" / "last.pt").is_file(),
        "best_pt": (out / "weights" / "best.pt").is_file(),
        "epochs_done": int(hist["epoch"].max()) if not hist.empty else 0,
        "epochs_planned": epochs,
        "sessions": len(sessions),
        "statuses": ",".join(sessions["status"]) if not sessions.empty else "",
        "gpu_hours_used": round(used / 3600, 3),
        "cap_hours": cap,
        "cap_hit": bool(not sessions.empty and (sessions["cap_hit"] == "True").any()),
        "val_metric": val_metric_source(cfg["model"]),
        "last_val": None if val.empty else round(float(val["val_mask_ap"].iloc[-1]), 4),
        "last_val_epoch": None if val.empty else int(val["epoch"].iloc[-1]),
        "best_val": None if best is None else round(float(best["val_mask_ap"]), 4),
        "best_val_epoch": None if best is None else int(best["epoch"]),
    }


def plot_history(cfg: Mapping[str, Any], save_to: str | Path | None = None) -> plt.Figure:
    """Training loss and validation mask AP per epoch (two panels)."""
    hist = load_history(cfg)
    fig, (ax_loss, ax_ap) = plt.subplots(1, 2, figsize=(11, 3.6))
    if not hist.empty:
        ax_loss.plot(hist["epoch"], hist["train_loss"], color="#3a6ea5")
        val = hist.dropna(subset=["val_mask_ap"])
        ax_ap.plot(val["epoch"], val["val_mask_ap"], marker="o", ms=3, color="#c0504d")
    ax_loss.set(title="train loss", xlabel="epoch")
    ax_ap.set(title=f"valid: {val_metric_source(cfg['model'])}", xlabel="epoch")
    for ax in (ax_loss, ax_ap):
        ax.grid(alpha=0.3)
    fig.suptitle(cfg["run_id"])
    fig.tight_layout()
    if save_to is not None:
        fig.savefig(save_to, dpi=120)
    return fig
