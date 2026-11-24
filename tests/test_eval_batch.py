"""Tests for the multi-run predict + evaluate batch (fake predictor, no GPU, tiny GT)."""

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from iiss.eval import batch
from iiss.eval.batch import (
    Job,
    condition_table,
    list_jobs,
    predictions_ready,
    run_jobs,
    status_table,
)
from iiss.eval.results_json import Detection, ResultsWriter

CATS = [{"id": 0, "name": "indoor-objects"}, {"id": 1, "name": "bed"}, {"id": 2, "name": "wall"}]
BOXES = {1: (2, 2, 30, 30), 2: (40, 40, 40, 40), 3: (10, 10, 60, 60)}


def _gt() -> dict:
    """Image 1 and 2 normal, image 3 low_light; one object per image (classes 1, 2, 1)."""
    anns = []
    for i, (x, y, w, h) in BOXES.items():
        poly = [x, y, x + w, y, x + w, y + h, x, y + h]
        anns.append(
            {
                "id": i, "image_id": i, "category_id": 2 if i == 2 else 1, "iscrowd": 0,
                "segmentation": [poly], "bbox": [x, y, w, h], "area": w * h,
            }
        )  # fmt: skip
    images = [
        {"id": i, "file_name": f"../photos/IMG_{i}.jpg", "width": 100, "height": 100} for i in BOXES
    ]
    return {"images": images, "annotations": anns, "categories": CATS}


class FakeModule:
    """Predicts every GT box perfectly, except nothing on low-light image 3."""

    calls = 0
    fail_for: str | None = None

    @classmethod
    def predict(cls, cfg, eval_set, weights, out_path, limit=None):
        cls.calls += 1
        if cls.fail_for == cfg["run_id"]:
            raise RuntimeError("boom")
        gt = json.loads(Path(cfg["eval"]["gt_files"][eval_set]).read_text())
        images = gt["images"][:limit]
        writer = ResultsWriter({i["id"]: (100, 100) for i in images}, [0, 1, 2])
        for img in images:
            x, y, w, h = BOXES[img["id"]]
            mask = np.zeros((100, 100), bool)
            mask[y : y + h, x : x + w] = True
            cat = 2 if img["id"] == 2 else 1
            dets = [] if img["id"] == 3 else [Detection(img["id"], cat, 0.9, mask)]
            writer.add_image(img["id"], dets)
        meta = {
            "weights": Path(weights).name, "images_limited_to": limit,
            "smoke": bool(cfg.get("smoke_mode")), "ms_per_image": 1.0, "peak_vram_mb": None,
        }  # fmt: skip
        return writer.write(out_path, meta=meta)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps(_gt()))
    groups = tmp_path / "metadata.csv"
    groups.write_text(
        "filename,condition\nIMG_1.jpg,normal\nIMG_2.jpg,normal\nIMG_3.jpg,low_light\n"
    )

    def load(path):
        run_id = Path(path).stem
        model, size, _ = run_id.split("_")
        return {
            "run_id": run_id, "model": model, "input_size": int(size[1:]), "seed": 42,
            "paths": {"runs_root": str(tmp_path / "runs"), "gpu_time_log": "x.csv"},
            "eval": {
                "gt_files": {"phone": str(gt), "valid": str(gt)},
                "train_gt": str(gt),
                "image_groups": {"phone": str(groups)},
                "results_root": str(tmp_path / "results"),
            },
            "stuff_classes": ["wall"],
            "smoke": {
                "input_size": 320, "max_minutes": 5, "max_eval_images": 2,
                "output_dir": str(tmp_path / "smoke"),
            },
        }  # fmt: skip

    monkeypatch.setattr(batch, "load_run_config", load)
    monkeypatch.setattr(batch, "model_module", lambda cfg: FakeModule)
    FakeModule.calls, FakeModule.fail_for = 0, None
    for size in (320, 640):
        w = tmp_path / "runs" / f"yolo_s{size}_seed42" / "weights" / "last.pt"
        w.parent.mkdir(parents=True)
        w.write_bytes(b"w")
        os.utime(w, (1_000_000_000, 1_000_000_000))
    return tmp_path


def test_list_jobs_covers_the_full_matrix():
    jobs = list_jobs(["yolo", "maskrcnn", "m2f"], [320, 640, 960], ["valid", "test", "phone"])
    assert len(jobs) == len(set(jobs)) == 27
    assert jobs[0].run_id == "yolo_s320_seed42" and jobs[2].eval_set == "phone"


def test_phone_is_scored_per_condition_and_valid_is_not(env: Path):
    jobs = list_jobs(["yolo"], [640], ["valid", "phone"])
    table = run_jobs(jobs)
    assert table["status"].tolist() == ["ok", "ok"]
    cond = condition_table(jobs).set_index("condition")
    assert set(cond["eval_set"]) == {"phone"}
    assert cond.loc["normal", ["n_images", "n_gt", "n_categories"]].tolist() == [2, 2, 2]
    assert cond.loc["normal", "AP"] == pytest.approx(1.0)
    assert cond.loc["low_light", "AP"] == 0.0 and cond.loc["low_light", "n_categories"] == 1
    out = env / "results" / "yolo_s640_seed42" / "phone"
    per_class = pd.read_csv(out / "per_class_ap_segm_low_light.csv").set_index("name")
    assert per_class.loc["bed", "n_gt"] == 1 and per_class.loc["wall", "n_gt"] == 0
    assert "Image groups" in (out / "summary.md").read_text()


def test_predictions_are_reused_but_always_rescored(env: Path):
    jobs = list_jobs(["yolo"], [640], ["phone"])
    assert not run_jobs(jobs)["reused"][0]
    metrics = env / "results" / "yolo_s640_seed42" / "phone" / "metrics.json"
    metrics.unlink()
    assert run_jobs(jobs)["reused"][0] and FakeModule.calls == 1 and metrics.is_file()
    assert not run_jobs(jobs, overwrite=True)["reused"][0] and FakeModule.calls == 2


def test_newer_checkpoint_or_partial_export_is_not_reused(env: Path):
    jobs = list_jobs(["yolo"], [640], ["phone"])
    run_jobs(jobs)
    pred = env / "runs" / "yolo_s640_seed42" / "predictions" / "phone.json"
    weights = env / "runs" / "yolo_s640_seed42" / "weights" / "last.pt"
    assert predictions_ready(pred, weights, 3)
    assert not predictions_ready(pred, weights, 4)  # GT has more images than were predicted
    os.utime(weights, (4_000_000_000, 4_000_000_000))
    assert not predictions_ready(pred, weights, 3)
    os.utime(weights, (1_000_000_000, 1_000_000_000))
    meta = pred.with_name("phone.meta.json")
    data = json.loads(meta.read_text())
    for change in ({"images_limited_to": 2}, {"smoke": True}, {"weights": "best.pt"}):
        meta.write_text(json.dumps({**data, **change}))
        assert not predictions_ready(pred, weights, 3), change
    meta.unlink()
    assert not predictions_ready(pred, weights, 3)


def test_failure_is_recorded_and_other_jobs_run(env: Path):
    FakeModule.fail_for = "yolo_s320_seed42"
    jobs = [*list_jobs(["yolo"], [320, 640], ["phone"]), Job("m2f", 640, "phone")]
    table = run_jobs(jobs)
    assert table["status"][0] == "failed: RuntimeError: boom"
    assert table["status"][1] == "ok"
    assert "checkpoint not found" in table["status"][2]


def test_interrupted_export_leaves_no_reusable_sidecar(env: Path):
    jobs = list_jobs(["yolo"], [640], ["phone"])
    run_jobs(jobs)
    FakeModule.fail_for = "yolo_s640_seed42"
    run_jobs(jobs, overwrite=True)
    assert not status_table(jobs)["predictions"][0]


def test_smoke_uses_real_weights_and_stays_out_of_results(env: Path):
    jobs = list_jobs(["yolo"], [320], ["phone"])
    table = run_jobs(jobs, smoke=True)
    assert table["status"].tolist() == ["ok"] and table["AP"][0] == pytest.approx(1.0)
    assert not (env / "results").exists()
    assert (env / "smoke" / "yolo_s320_seed42" / "predictions" / "phone.json").is_file()
    assert condition_table(jobs, smoke=True)["n_images"].tolist() == [2]
    with pytest.raises(ValueError, match="only S=320"):
        batch.run_job(Job("yolo", 640, "phone"), smoke=True)


def test_status_table(env: Path):
    jobs = list_jobs(["yolo"], [320, 640], ["phone"])
    run_jobs(jobs[:1])
    t = status_table([*jobs, Job("m2f", 640, "phone")])
    assert t["checkpoint"].tolist() == [True, True, False]
    assert t["predictions"].tolist() == [True, False, False]
    assert t["evaluated"].tolist() == [True, False, False]
