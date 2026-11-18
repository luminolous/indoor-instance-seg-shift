"""CPU-only tests for the YOLO wrapper and shared model helpers (no training, no weights)."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

from iiss.models import yolo
from iiss.models.common import apply_smoke, export_predictions
from iiss.utils.config import ConfigError, build_config
from iiss.utils.timer import RunTimer


def test_build_config_needs_input_size():
    with pytest.raises(ConfigError, match="input_size"):
        build_config("yolo")
    cfg = build_config("yolo", {"input_size": 640})
    assert cfg["run_id"] == "yolo_s640_seed42"
    assert cfg["train"]["epochs"] > 0


def test_apply_smoke_moves_outputs_and_limits():
    cfg = apply_smoke(build_config("yolo", {"input_size": 960}))
    assert cfg["input_size"] == 320
    assert cfg["run_id"] == "yolo_s320_seed42"
    assert cfg["time_cap_hours"] == pytest.approx(5 / 60)
    assert Path(cfg["paths"]["runs_root"]) == Path("runs") / "smoke"
    assert Path(cfg["paths"]["gpu_time_log"]).parent == Path("runs") / "smoke"
    kwargs = yolo.train_kwargs(cfg, Path("data.yaml"))
    assert kwargs["epochs"] == 2
    assert kwargs["imgsz"] == 320
    assert kwargs["name"] == "yolo_s320_seed42"
    assert kwargs["seed"] == 42


def test_train_kwargs_full_run_keeps_budget():
    cfg = build_config("yolo", {"input_size": 640})
    kwargs = yolo.train_kwargs(cfg, Path("data.yaml"))
    assert kwargs["epochs"] == cfg["train"]["epochs"]
    assert kwargs["imgsz"] == 640
    assert kwargs["exist_ok"] is True
    assert "time" not in kwargs  # Ultralytics time= would rescale the schedule


def test_make_smoke_dataset(tmp_path: Path):
    src = tmp_path / "yolo"
    for split in ("train", "valid"):
        (src / "images" / split).mkdir(parents=True)
        (src / "labels" / split).mkdir(parents=True)
        for i in range(5):
            Image.new("RGB", (8, 8)).save(src / "images" / split / f"{split}{i}.jpg")
            (src / "labels" / split / f"{split}{i}.txt").write_text("1 0 0 1 0 1 1\n")
    (src / "data.yaml").write_text(yaml.safe_dump({"nc": 3, "names": {0: "a", 1: "b", 2: "c"}}))

    out = yolo.make_smoke_dataset(src / "data.yaml", tmp_path / "smoke", {"train": 3, "valid": 2})
    data = yaml.safe_load(out.read_text())
    assert data["nc"] == 3 and data["val"] == "images/valid"
    assert len(list((tmp_path / "smoke" / "images" / "train").iterdir())) == 3
    assert len(list((tmp_path / "smoke" / "labels" / "valid").iterdir())) == 2


def test_cap_callback_sets_trainer_stop(tmp_path: Path):
    now = [0.0]
    timer = RunTimer("r", 1 / 3600, tmp_path / "log.csv", clock=lambda: now[0], track_cuda=False)
    trainer = SimpleNamespace(stop=False)
    with timer:
        check = yolo._cap_callback(timer)
        check(trainer)
        assert trainer.stop is False
        now[0] = 2.0
        check(trainer)
        assert trainer.stop is True
    assert "cap_hit" in (tmp_path / "log.csv").read_text()


class _FakeModel:
    """Mimics ``YOLO.predict``: binary masks in the (letterboxed) input frame.

    Without ``masks`` the masks have the image's own size (no padding).
    """

    def __init__(self, masks: torch.Tensor | None = None) -> None:
        self.kwargs: dict = {}
        self.masks = masks

    def predict(self, img: np.ndarray, **kwargs):
        self.kwargs = kwargs
        if self.masks is not None:
            masks = self.masks
        else:
            h, w = img.shape[:2]
            masks = torch.zeros(2, h, w)
            masks[0, :2, :3] = 1
            masks[1, 3:, 3:] = 1
        boxes = _Boxes(torch.tensor([5.0, 7.0]), torch.tensor([0.9, 0.2]))
        result = SimpleNamespace(masks=SimpleNamespace(data=masks), boxes=boxes)
        return [result]


class _Boxes:
    def __init__(self, cls: torch.Tensor, conf: torch.Tensor) -> None:
        self.cls, self.conf = cls, conf

    def __len__(self) -> int:
        return len(self.cls)


def test_predict_image_maps_classes_and_masks():
    cfg = build_config("yolo", {"input_size": 320})
    model = _FakeModel()
    image = Image.new("RGB", (6, 4))
    dets = yolo.predict_image(model, image, {"id": 11, "width": 6, "height": 4}, cfg)
    assert [d.category_id for d in dets] == [5, 7]
    assert dets[0].image_id == 11 and dets[0].mask.shape == (4, 6) and dets[0].mask.dtype == bool
    assert dets[0].mask.sum() == 6
    assert model.kwargs["imgsz"] == 320 and model.kwargs["retina_masks"] is False
    assert model.kwargs["max_det"] == 100


def test_predict_image_crops_letterbox_and_upsamples():
    cfg = build_config("yolo", {"input_size": 320})
    # A 6x4 (w x h) image letterboxed into 8x8: content 8x5 starting at row 1.
    masks = torch.zeros(2, 8, 8)
    masks[0, 1:6, :] = 1  # exactly the content area -> whole image
    masks[1, 0, :] = 1  # padding row only -> nothing
    model = _FakeModel(masks)
    image = Image.new("RGB", (6, 4))
    dets = yolo.predict_image(model, image, {"id": 1, "width": 6, "height": 4}, cfg)
    assert dets[0].mask.shape == (4, 6) and dets[0].mask.all()
    assert not dets[1].mask.any()


def test_predict_image_rejects_retina_masks():
    cfg = build_config("yolo", {"input_size": 320})
    cfg["predict"]["retina_masks"] = True
    with pytest.raises(ValueError, match="retina_masks"):
        yolo.predict_image(
            _FakeModel(), Image.new("RGB", (6, 4)), {"id": 1, "width": 6, "height": 4}, cfg
        )


def test_export_predictions_uses_annotation_frame(tmp_path: Path):
    gt_dir = tmp_path / "valid"
    gt_dir.mkdir()
    Image.new("RGB", (6, 4)).save(gt_dir / "a.png")
    Image.new("RGB", (4, 6)).save(gt_dir / "rot.png")  # stored portrait, annotated landscape
    gt = {
        "images": [
            {"id": 1, "file_name": "a.png", "width": 6, "height": 4},
            {"id": 2, "file_name": "rot.png", "width": 6, "height": 4},
        ],
        "categories": [{"id": 0, "name": "x"}, {"id": 5, "name": "y"}, {"id": 7, "name": "z"}],
        "annotations": [],
    }
    (gt_dir / "gt.json").write_text(json.dumps(gt))
    cfg = build_config("yolo", {"input_size": 320})
    cfg["eval"]["gt_files"] = {"valid": str(gt_dir / "gt.json")}
    cfg["data"]["rotation_fixes"] = {"rot.png": 90}

    model = _FakeModel()
    out = export_predictions(
        cfg,
        "valid",
        lambda image, info: yolo.predict_image(model, image, info, cfg),
        tmp_path / "pred" / "valid.json",
    )
    results = json.loads(out.read_text())
    assert {r["image_id"] for r in results} == {1, 2}
    meta = json.loads((tmp_path / "pred" / "valid.meta.json").read_text())
    assert meta["run_id"] == "yolo_s320_seed42" and meta["num_images_predicted"] == 2
    assert meta["ms_per_image"] is not None
