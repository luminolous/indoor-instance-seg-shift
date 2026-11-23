"""Tests for the training-run summaries used by the training notebooks (synthetic files)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from iiss.analysis.training import (  # noqa: E402
    config_table,
    load_history,
    notebook_config,
    plot_history,
    run_summary,
)
from iiss.models.common import run_dir  # noqa: E402

LOG_HEADER = "run_id,start,end,wall_clock_s,gpu_name,peak_vram_mb,cap_hit,status\n"


def _cfg(model: str, tmp_path: Path) -> dict:
    return notebook_config(model, 640, runs_root=(tmp_path / "runs").as_posix())


def test_notebook_config_paths_and_smoke(tmp_path: Path):
    cfg = _cfg("maskrcnn", tmp_path)
    assert cfg["run_id"] == "maskrcnn_s640_seed42"
    assert run_dir(cfg) == tmp_path / "runs" / "maskrcnn_s640_seed42"
    smoke = notebook_config("maskrcnn", 640, smoke=True)
    assert smoke["input_size"] == 320 and smoke["run_id"] == "maskrcnn_s320_seed42"
    assert "train.batch" in config_table(cfg).index


def test_empty_run(tmp_path: Path):
    cfg = _cfg("m2f", tmp_path)
    s = run_summary(cfg)
    assert (s["epochs_done"], s["sessions"], s["last_val"], s["last_pt"]) == (0, 0, None, False)
    plot_history(cfg)  # must not fail on a run that has not started


def test_torch_run_summary(tmp_path: Path):
    cfg = _cfg("maskrcnn", tmp_path)
    out = run_dir(cfg)
    (out / "weights").mkdir(parents=True)
    (out / "weights" / "last.pt").write_bytes(b"x")
    (out / "results.csv").write_text(
        "epoch,loss,val_segm_AP,val_segm_AP50,lr\n1,3.0,,,0.1\n2,2.0,0.10,0.2,0.1\n"
        "3,1.5,,,0.1\n4,1.2,0.08,0.18,0.01\n"
    )
    log = tmp_path / "runs" / "gpu_time_log.csv"
    log.write_text(
        LOG_HEADER
        + "maskrcnn_s640_seed42,a,b,1800,g,1,False,killed\n"
        + "maskrcnn_s640_seed42,a,b,3600,g,1,True,cap_hit\n"
        + "other_run,a,b,999,g,1,False,completed\n"
    )
    s = run_summary(cfg)
    assert s["epochs_done"] == 4 and s["epochs_planned"] == 24
    assert s["gpu_hours_used"] == 1.5 and s["statuses"] == "killed,cap_hit" and s["cap_hit"]
    assert (s["last_val"], s["last_val_epoch"], s["best_val"], s["best_val_epoch"]) == (
        0.08, 4, 0.1, 2
    )  # fmt: skip
    assert s["last_pt"] and not s["best_pt"]


def test_yolo_history_sums_losses(tmp_path: Path):
    cfg = _cfg("yolo", tmp_path)
    out = run_dir(cfg)
    out.mkdir(parents=True)
    (out / "results.csv").write_text(
        "  epoch, train/box_loss, train/seg_loss, train/cls_loss, train/dfl_loss,"
        " metrics/mAP50(M), metrics/mAP50-95(M), lr/pg0\n"
        "1,1,2,3,4,0.3,0.2,0.001\n"
    )
    h = load_history(cfg)
    assert h.loc[0, "train_loss"] == 10 and h.loc[0, "val_mask_ap"] == 0.2
    assert "Ultralytics" in run_summary(cfg)["val_metric"]
