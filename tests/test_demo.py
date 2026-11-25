"""Tests for the live demo: inputs, EXIF orientation, threshold without inference, AP table."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from iiss import demo
from iiss.eval.results_json import Detection

NAMES = {1: "bed", 2: "chair"}
ORIENTATION = 0x0112


def _mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), bool)
    mask[: height // 2, : width // 2] = True
    return mask


class FakeModel:
    """Two instances per image; counts calls and records the sizes it was given."""

    def __init__(self):
        self.sizes = []

    def __call__(self, image):
        self.sizes.append(image.size)
        self.pixel = image.getpixel((0, 0))
        mask = _mask(image.size[1], image.size[0])
        return [Detection(0, 1, 0.3, mask), Detection(0, 2, 0.9, mask)], 12.0, 30.0


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    path = tmp_path / "a.jpg"
    Image.new("RGB", (80, 60), (200, 10, 10)).save(path)
    return path


def test_full_run_id():
    assert demo.full_run_id("yolo_s640") == "yolo_s640_seed42"
    assert demo.full_run_id("m2f_s960_seed42") == "m2f_s960_seed42"
    with pytest.raises(ValueError, match="unknown run"):
        demo.full_run_id("yolo640")


def test_list_images_file_folder_and_errors(photo: Path, tmp_path: Path):
    Image.new("RGB", (5, 5)).save(tmp_path / "B.PNG")
    (tmp_path / "notes.txt").write_text("x")
    assert demo.list_images(photo) == [photo]
    assert [p.name for p in demo.list_images(tmp_path)] == ["a.jpg", "B.PNG"]
    with pytest.raises(ValueError, match="unsupported"):
        demo.list_images(tmp_path / "notes.txt")
    with pytest.raises(FileNotFoundError):
        demo.list_images(tmp_path / "missing")
    Image.new("RGB", (5, 5)).save(tmp_path / "a.png")
    with pytest.raises(ValueError, match="collide"):
        demo.list_images(tmp_path)


def test_exif_portrait_is_turned_before_inference(tmp_path: Path):
    upright = Image.new("RGB", (60, 80), (0, 0, 255))
    upright.paste((255, 0, 0), (0, 0, 60, 20))  # red top
    stored = upright.transpose(Image.Transpose.ROTATE_90)  # sensor frame, landscape
    exif = Image.Exif()
    exif[ORIENTATION] = 6
    path = tmp_path / "portrait.png"
    stored.save(path, exif=exif)
    model = FakeModel()
    result = demo.predict_photo({"yolo_s640_seed42": model}, path, NAMES)
    assert model.sizes == [(60, 80)] and result.size == (60, 80)
    assert model.pixel == (255, 0, 0)
    assert result.photo.shape == (80, 60, 3)
    assert demo.decode_mask(result.models[0].instances[0].rle).shape == (80, 60)


def test_dataset_rotation_fix_is_applied(tmp_path: Path, monkeypatch):
    path = tmp_path / "fixed.png"
    Image.new("RGB", (30, 40)).save(path)
    monkeypatch.setattr("iiss.data.images.default_rotation_fixes", lambda: {"fixed.png": 90})
    result = demo.predict_photo({"yolo_s640_seed42": FakeModel()}, path, NAMES)
    assert result.size == (40, 30)


def test_threshold_needs_no_new_inference(photo: Path, tmp_path: Path):
    models = {"yolo_s640_seed42": FakeModel(), "m2f_s640_seed42": FakeModel()}
    (result,) = demo.predict_photos(models, photo, NAMES)
    colors = demo.photo_colors(result)
    for threshold, n, classes in ((0.25, 2, "chair, bed"), (0.5, 1, "chair"), (0.95, 0, "")):
        table = demo.summary_table([result], threshold)
        assert list(table["instances"]) == [n, n] and table["classes"][0] == classes
        fig = demo.plot_photo(result, threshold, tmp_path / "out")
        assert f"{n} instances" in fig.axes[0].get_title(loc="left")
        assert len(fig.axes[0].texts) == n
        boxes = [t.get_window_extent() for t in fig.axes[0].texts]  # both masks are identical
        assert n < 2 or not boxes[0].overlaps(boxes[1])
        assert (tmp_path / "out" / f"a_yolo_s640-m2f_s640_t{threshold:.2f}.jpg").is_file()
        plt.close(fig)
    assert demo.photo_colors(result) == colors
    assert [len(m.sizes) for m in models.values()] == [1, 1]
    with pytest.raises(ValueError, match="outside"):
        demo.plot_photo(result, 0.01)


def test_masks_are_kept_at_display_size(tmp_path: Path):
    path = tmp_path / "big.png"
    Image.new("RGB", (400, 200)).save(path)
    result = demo.predict_photo({"m2f_s640_seed42": FakeModel()}, path, NAMES, max_side=100)
    assert result.photo.shape == (50, 100, 3)
    assert demo.decode_mask(result.models[0].instances[0].rle).sum() == 25 * 50


def test_per_class_ap_reads_the_evaluation_files(tmp_path: Path):
    for run, ap in (("yolo_s640_seed42", (0.5, 0.1)), ("m2f_s640_seed42", (0.2, 0.4))):
        folder = tmp_path / run / "test"
        folder.mkdir(parents=True)
        pd.DataFrame(
            {
                "category_id": [0, 1, 2],
                "name": ["indoor-objects", "bed", "chair"],
                "n_gt": [0, 3, 7],
                "AP": [None, *ap],
                "AP50": [None, 1, 1],
                "AP75": [None, 1, 1],
            }
        ).to_csv(folder / "per_class_ap_segm.csv", index=False)
        metrics = {
            "run_id": run,
            "eval_set": "test",
            "num_gt_instances": 10,
            "segm": {"summary": {"AP": sum(ap) / 2}},
        }
        (folder / "metrics.json").write_text(json.dumps(metrics))
    table = demo.per_class_ap(["yolo_s640", "m2f_s640"], "test", eval_root=tmp_path)
    assert list(table.index) == ["chair", "bed", "mean"]  # most GT first, no empty class
    assert list(table["n_gt"]) == [7, 3, 10]
    assert list(table["yolo_s640_seed42"]) == [10, 50, 30]
    assert table.loc["mean", "m2f_s640_seed42"] == 30
