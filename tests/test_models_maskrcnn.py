"""CPU-only tests for the Mask R-CNN wrapper and the torch dataset (no downloads, no GPU)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from iiss.data.torch_dataset import CocoInstanceDataset, collate, resized_size
from iiss.models import maskrcnn
from iiss.utils.config import build_config


def _write_split(root: Path, anns: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (80, 40), (50, 60, 70)).save(root / "a.png")
    Image.new("RGB", (40, 80)).save(root / "rot.png")  # stored portrait, annotated landscape
    data = {
        "images": [
            {"id": 0, "file_name": "a.png", "width": 80, "height": 40},
            {"id": 1, "file_name": "rot.png", "width": 80, "height": 40},
        ],
        "categories": [{"id": i, "name": f"c{i}"} for i in range(4)],
        "annotations": anns,
    }
    path = root / "gt.json"
    path.write_text(json.dumps(data))
    return path


def _ann(i: int, image_id: int, cat: int, x0: float, y0: float, x1: float, y1: float) -> dict:
    return {
        "id": i,
        "image_id": image_id,
        "category_id": cat,
        "bbox": [x0, y0, x1 - x0, y1 - y0],
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
        "area": (x1 - x0) * (y1 - y0),
        "iscrowd": 0,
    }


def test_resized_size_long_side():
    assert resized_size(4000, 3000, 640) == (640, 480, 0.16)
    assert resized_size(300, 400, 640)[:2] == (480, 640)


def test_dataset_resizes_and_keeps_every_annotation(tmp_path: Path):
    gt = _write_split(
        tmp_path / "train",
        [_ann(0, 0, 2, 8, 4, 48, 24), _ann(1, 0, 3, 10, 10, 10.5, 30), _ann(2, 1, 1, 0, 0, 80, 40)],
    )
    ds = CocoInstanceDataset(gt, 40, {"rot.png": 90}, train=False)
    img, target = ds[0]
    assert img.shape == (3, 20, 40) and img.dtype == torch.float32
    assert target["labels"].tolist() == [2, 3]
    assert target["masks"].shape == (2, 20, 40)
    assert torch.allclose(target["boxes"][0], torch.tensor([4.0, 2.0, 24.0, 12.0]))
    widths = target["boxes"][:, 2] - target["boxes"][:, 0]
    assert (widths >= 1.0).all()  # the 0.25 px wide box is widened, not dropped
    img1, target1 = ds[1]  # rotation fix applied before resizing
    assert img1.shape == (3, 20, 40) and target1["masks"].sum() == 20 * 40


def test_dataset_flip_mirrors_boxes_and_masks(tmp_path: Path):
    gt = _write_split(tmp_path / "train", [_ann(0, 0, 2, 8, 4, 48, 24)])
    plain = CocoInstanceDataset(gt, 40, {"rot.png": 90}, train=False)[0][1]
    ds = CocoInstanceDataset(gt, 40, {"rot.png": 90}, train=True)
    flipped = None
    for seed in range(20):
        torch.manual_seed(seed)
        _, t = ds[0]
        if not torch.equal(t["boxes"], plain["boxes"]):
            flipped = t
            break
    assert flipped is not None
    assert torch.allclose(flipped["boxes"][0], torch.tensor([16.0, 2.0, 36.0, 12.0]))
    assert torch.equal(flipped["masks"], plain["masks"].flip(-1))


def test_dataset_rejects_background_category(tmp_path: Path):
    gt = _write_split(tmp_path / "train", [_ann(0, 0, 0, 8, 4, 48, 24)])
    with pytest.raises(ValueError, match="background"):
        CocoInstanceDataset(gt, 40, {"rot.png": 90})


def test_lr_factor_warmup_and_milestones():
    cfg = {"warmup_iters": 10, "warmup_factor": 0.001, "lr_milestones": [2, 3], "lr_gamma": 0.1}
    assert maskrcnn.lr_factor(0, 100, cfg) == pytest.approx(0.001)
    assert maskrcnn.lr_factor(10, 100, cfg) == pytest.approx(1.0)
    assert maskrcnn.lr_factor(199, 100, cfg) == pytest.approx(1.0)
    assert maskrcnn.lr_factor(200, 100, cfg) == pytest.approx(0.1)
    assert maskrcnn.lr_factor(300, 100, cfg) == pytest.approx(0.01)


def test_model_trains_and_predicts_on_cpu(tmp_path: Path):
    torch.manual_seed(0)
    gt = _write_split(tmp_path / "valid", [_ann(0, 0, 2, 8, 4, 48, 24)])
    cfg = build_config("maskrcnn", {"input_size": 64})
    cfg["amp"] = False
    model = maskrcnn.build_model(cfg, pretrained=False)
    assert model.roi_heads.box_predictor.cls_score.out_features == 75
    assert model.transform.min_size == (64,) and model.transform.max_size == 64

    ds = CocoInstanceDataset(gt, 64, {"rot.png": 90}, train=True)
    images, targets = collate([ds[0], ds[1]])
    model.train()
    losses = model(list(images), list(targets))
    assert set(maskrcnn.LOSS_KEYS) == set(losses)

    model.eval()
    model.roi_heads.score_thresh = 0.0
    timings: list[float] = []
    dets = maskrcnn.predict_image(
        model, Image.new("RGB", (80, 40)), {"id": 5, "width": 80, "height": 40}, cfg, timings
    )
    assert len(timings) == 1
    for det in dets:
        assert det.image_id == 5 and 1 <= det.category_id < 75
        assert det.mask.shape == (40, 80) and det.mask.dtype == np.bool_
