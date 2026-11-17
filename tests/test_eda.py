import json
from pathlib import Path

import pandas as pd
import pytest

from iiss.data.coco import load_split
from iiss.data.eda import class_counts, class_shift, object_sizes, run_eda, size_bin

CATS = [
    {"id": 0, "name": "indoor-objects"},
    {"id": 1, "name": "bed"},
    {"id": 2, "name": "chair"},
    {"id": 3, "name": "mouse"},
]


def _ann(i: int, image_id: int, cat: int, area: float) -> dict:
    return {
        "id": i,
        "image_id": image_id,
        "category_id": cat,
        "bbox": [0, 0, 1, 1],
        "area": area,
        "segmentation": [[0, 0, 1, 0, 1, 1]],
    }


def _write(root: Path, name: str, images: list[dict], anns: list[dict]) -> None:
    d = root / name
    d.mkdir(parents=True)
    data = {"categories": CATS, "images": images, "annotations": anns}
    (d / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    _write(
        tmp_path,
        "train",
        [{"id": 0, "file_name": "a.jpg", "width": 640, "height": 480}],
        [_ann(0, 0, 1, 100.0), _ann(1, 0, 1, 2000.0), _ann(2, 0, 2, 20000.0)],
    )
    _write(
        tmp_path,
        "test",
        [
            {"id": 0, "file_name": "b.jpg", "width": 1280, "height": 960},
            {"id": 1, "file_name": "c.jpg", "width": 200, "height": 100},
        ],
        [_ann(0, 0, 2, 4000.0), _ann(1, 1, 3, 500.0)],
    )
    return tmp_path


def test_size_bin_edges():
    assert size_bin(32**2 - 1) == "small"
    assert size_bin(32**2) == "medium"
    assert size_bin(96**2) == "large"


def test_class_counts_and_shift(dataset: Path):
    splits = {n: load_split(dataset, n) for n in ("train", "test")}
    counts = class_counts(splits).set_index("name")
    assert counts.loc["bed", "train_instances"] == 2
    assert counts.loc["bed", "train_images"] == 1
    assert counts.loc["mouse", "test_instances"] == 1

    shift = class_shift(class_counts(splits)).set_index("name")
    assert "indoor-objects" not in shift.index  # absent in both
    assert shift.loc["mouse", "status"] == "absent_in_train"
    assert shift.loc["bed", "status"] == "absent_in_test"
    assert shift.loc["chair", "train_share_pct"] == pytest.approx(100 / 3, abs=1e-3)
    assert shift.loc["chair", "test_share_pct"] == pytest.approx(50.0)


def test_object_sizes_rescale(dataset: Path):
    splits = {"test": load_split(dataset, "test")}
    df = object_sizes(splits).set_index("resolution")
    # native: 4000 px^2 -> medium, 500 px^2 -> small
    assert df.loc["native", "small_pct"] == pytest.approx(50.0)
    # S=320: b.jpg scaled by (320/1280)^2 -> 250 px^2 (small); c.jpg by (320/200)^2 -> 1280 (medium)
    assert df.loc["S=320", "small_pct"] == pytest.approx(50.0)
    assert df.loc["S=320", "medium_pct"] == pytest.approx(50.0)


def test_run_eda_writes_outputs(dataset: Path, tmp_path: Path):
    files = run_eda(dataset, ["train", "test"], tmp_path / "eda")
    for path in files.values():
        assert path.is_file() and path.stat().st_size > 0
    ls = pd.read_csv(files["long_side_stats"]).set_index("split")
    assert ls.loc["test", "upscaled_pct_at_320"] == pytest.approx(50.0)
    assert "absent in train (1)" in files["summary"].read_text(encoding="utf-8")
