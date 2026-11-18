"""CPU-only tests for the VRAM probe summary (no GPU, synthetic probe results)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from iiss.models.probe import estimate_hours, recommend
from iiss.utils.config import build_config


@pytest.fixture
def gt_files(tmp_path: Path) -> dict[str, str]:
    paths = {}
    for name, n in (("train", 100), ("valid", 40)):
        data = {"images": [{"id": i} for i in range(n)], "annotations": [], "categories": []}
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(data))
        paths[name] = str(path)
    return paths


def _cfg(model: str, size: int, gt_files: dict[str, str]) -> dict:
    cfg = build_config(model, {"input_size": size})
    cfg["eval"]["train_gt"] = gt_files["train"]
    cfg["eval"]["gt_files"]["valid"] = gt_files["valid"]
    return cfg


def _row(model, size, batch, peak, oom=False, s=0.5, ms=100.0):
    return {"model": model, "input_size": size, "batch": batch, "iters_timed": 17, "oom": oom,
            "peak_train_mb": peak, "s_per_iter": s, "infer_ms": ms, "peak_infer_mb": 500.0,
            "note": ""}  # fmt: skip


def test_estimate_hours(gt_files):
    cfg = _cfg("maskrcnn", 640, gt_files)
    cfg["train"].update(epochs=10, eval_every=2)
    # 10 epochs x ceil(100/2)=50 iters x 0.36 s = 180 s; 5 validations x 40 x 0.1 s = 20 s
    assert estimate_hours(cfg, 2, 0.36, 100.0) == pytest.approx(200 / 3600, abs=0.01)


def test_recommend_keeps_effective_batch_and_flags_cap(gt_files):
    configs = {
        "maskrcnn_s640_seed42": _cfg("maskrcnn", 640, gt_files),  # configured batch 2 x 1
        "m2f_s960_seed42": _cfg("m2f", 960, gt_files),
        "yolo_s640_seed42": _cfg("yolo", 640, gt_files),
    }
    results = pd.DataFrame(
        [
            _row("maskrcnn", 640, 1, 3000),
            _row("maskrcnn", 640, 2, 4500),
            _row("maskrcnn", 640, 4, 5500),  # fits, but above the configured effective batch
            _row("m2f", 960, 1, 5900),  # over budget
            _row("m2f", 960, 2, None, oom=True),
            _row("yolo", 640, 8, 4000, s=0.4),
            _row("yolo", 640, 16, 7000, oom=True),
        ]
    )
    rec = recommend(results, configs, budget_mb=5600, cap_hours=2.0).set_index("run_id")
    mr = rec.loc["maskrcnn_s640_seed42"]
    assert (mr["batch"], mr["accumulate"], mr["max_fit_batch"]) == (2, 1, 4)
    assert pd.isna(rec.loc["m2f_s960_seed42"]["batch"])
    assert "Colab" in rec.loc["m2f_s960_seed42"]["note"]
    yl = rec.loc["yolo_s640_seed42"]
    assert (yl["batch"], yl["accumulate"], yl["effective_batch"]) == (8, 1, 64)
    assert bool(yl["cap_risk"]) is False


def test_recommend_raises_accumulation_when_batch_must_shrink(gt_files):
    cfg = _cfg("maskrcnn", 960, gt_files)
    results = pd.DataFrame([_row("maskrcnn", 960, 1, 5000), _row("maskrcnn", 960, 2, 6200)])
    rec = recommend(results, {"maskrcnn_s960_seed42": cfg}).iloc[0]
    assert (rec["batch"], rec["accumulate"], rec["effective_batch"]) == (1, 2, 2)
