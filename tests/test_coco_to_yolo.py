import json
from pathlib import Path

import pytest
import yaml
from PIL import Image

from iiss.data.coco_to_yolo import (
    ConversionError,
    annotation_to_line,
    convert_dataset,
    yolo_to_polygon,
)

CATS = [
    {"id": 0, "name": "indoor-objects"},
    {"id": 1, "name": "bed"},
    {"id": 2, "name": "chair"},
    {"id": 3, "name": "lamp"},
]


def _ann(i: int, image_id: int, cat: int, poly: list[float]) -> dict:
    xs, ys = poly[0::2], poly[1::2]
    return {
        "id": i,
        "image_id": image_id,
        "category_id": cat,
        "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
        "area": 1.0,
        "segmentation": [poly],
        "iscrowd": 0,
    }


@pytest.fixture
def coco_root(tmp_path: Path) -> Path:
    root = tmp_path / "coco"
    for split in ("train", "valid"):
        d = root / split
        d.mkdir(parents=True)
        Image.new("RGB", (64, 48), (10, 20, 30)).save(d / f"{split}_a.jpg")
        # Stored portrait (48x64) but annotated as 64x48 -> needs the rotation fix.
        portrait = Image.new("RGB", (48, 64))
        portrait.putpixel((0, 0), (255, 0, 0))
        portrait.save(d / f"{split}_rot.png")
        Image.new("RGB", (32, 32)).save(d / f"{split}_empty.jpg")
        data = {
            "categories": CATS,
            "images": [
                {"id": 0, "file_name": f"{split}_a.jpg", "width": 64, "height": 48},
                {"id": 1, "file_name": f"{split}_rot.png", "width": 64, "height": 48},
                {"id": 2, "file_name": f"{split}_empty.jpg", "width": 32, "height": 32},
            ],
            "annotations": [
                _ann(0, 0, 1, [1.5, 2.0, 60.25, 2.0, 60.25, 40.0, 1.5, 40.0]),
                _ann(1, 0, 3, [10, 10, 20, 10, 15, 20]),
                _ann(2, 1, 2, [0, 0, 65.0, 0, 65.0, 48, 0, 48]),  # x slightly outside -> clipped
            ],
        }
        (d / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
    return root


def _fixes() -> dict[str, int]:
    return {"train_rot.png": 90, "valid_rot.png": 90}


def test_round_trip_labels(coco_root: Path, tmp_path: Path):
    out = tmp_path / "yolo"
    convert_dataset(coco_root, out, ["train", "valid"], fixes=_fixes())
    src = json.loads((coco_root / "train" / "_annotations.coco.json").read_text())
    lines = (out / "labels" / "train" / "train_a.txt").read_text().splitlines()
    assert len(lines) == 2
    for line, ann in zip(lines, src["annotations"][:2], strict=True):
        parts = line.split()
        assert int(parts[0]) == ann["category_id"]  # class index == category_id
        back = yolo_to_polygon([float(v) for v in parts[1:]], 64, 48)
        assert back == pytest.approx(ann["segmentation"][0], abs=1e-4)


def test_clipping_empty_labels_and_rotation(coco_root: Path, tmp_path: Path):
    out = tmp_path / "yolo"
    convert_dataset(coco_root, out, ["train", "valid"], fixes=_fixes())
    rot_line = (out / "labels" / "train" / "train_rot.txt").read_text().split()
    assert max(float(v) for v in rot_line[1:]) == 1.0
    assert (out / "labels" / "train" / "train_empty.txt").read_text() == ""

    rotated = Image.open(out / "images" / "train" / "train_rot.png")
    assert rotated.size == (64, 48)
    assert rotated.getpixel((0, 47)) == (255, 0, 0)  # 90 deg CCW: top-left -> bottom-left

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["splits"]["train"]["clipped_coords"] == 2
    assert manifest["splits"]["train"]["rotation_fixed"] == ["train_rot.png"]
    assert manifest["splits"]["train"]["empty_label_files"] == 1


def test_data_yaml(coco_root: Path, tmp_path: Path):
    yaml_path = convert_dataset(coco_root, tmp_path / "yolo", ["train", "valid"], fixes=_fixes())
    data = yaml.safe_load(yaml_path.read_text())
    assert "path" not in data
    assert data["train"] == "images/train"
    assert data["val"] == "images/valid"
    assert data["nc"] == 4
    assert data["names"] == {0: "indoor-objects", 1: "bed", 2: "chair", 3: "lamp"}


def test_refuses_to_overwrite(coco_root: Path, tmp_path: Path):
    out = tmp_path / "yolo"
    convert_dataset(coco_root, out, ["train"], fixes=_fixes())
    with pytest.raises(FileExistsError):
        convert_dataset(coco_root, out, ["train"], fixes=_fixes())
    convert_dataset(coco_root, out, ["train"], fixes=_fixes(), overwrite=True)


def test_rejects_multi_polygon():
    ann = {"id": 9, "category_id": 1, "segmentation": [[0, 0, 1, 0, 1, 1], [2, 2, 3, 2, 3, 3]]}
    with pytest.raises(ConversionError, match="multi-part"):
        annotation_to_line(ann, 10, 10)


def test_ultralytics_reads_converted_labels(coco_root: Path, tmp_path: Path):
    from ultralytics.data.utils import check_det_dataset, verify_image_label

    yaml_path = convert_dataset(coco_root, tmp_path / "yolo", ["train", "valid"], fixes=_fixes())
    data = check_det_dataset(str(yaml_path))
    assert data["nc"] == 4
    img = Path(data["train"]) / "train_a.jpg"
    lbl = tmp_path / "yolo" / "labels" / "train" / "train_a.txt"
    result = verify_image_label((str(img), str(lbl), "", False, 4, 0, 0, False))
    lb, segments = result[1], result[3]
    assert lb[:, 0].tolist() == [1.0, 3.0]
    assert len(segments) == 2
