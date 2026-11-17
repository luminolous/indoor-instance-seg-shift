import copy
import json
from pathlib import Path

import pytest
from PIL import Image

from iiss.data.coco import load_split
from iiss.data.validate import (
    ERROR,
    WARNING,
    main,
    polygon_area,
    render_markdown,
    validate_dataset,
)

CATEGORIES = [
    {"id": 0, "name": "indoor-objects", "supercategory": "none"},
    {"id": 1, "name": "bed", "supercategory": "indoor-objects"},
    {"id": 2, "name": "chair", "supercategory": "indoor-objects"},
]


def _good_split() -> dict:
    return {
        "categories": copy.deepcopy(CATEGORIES),
        "images": [
            {"id": 0, "file_name": "a.jpg", "width": 64, "height": 48},
            {"id": 1, "file_name": "b.jpg", "width": 32, "height": 32},
        ],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10, 10, 20, 10],
                "area": 200.0,
                "segmentation": [[10, 10, 30, 10, 30, 20, 10, 20]],
                "iscrowd": 0,
            },
            {
                "id": 1,
                "image_id": 1,
                "category_id": 2,
                "bbox": [0, 0, 8, 8],
                "area": 32.0,
                "segmentation": [[0, 0, 8, 0, 0, 8]],
                "iscrowd": 0,
            },
        ],
    }


def _write_split(root: Path, name: str, data: dict, make_images: bool = True) -> None:
    split_dir = root / name
    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
    if make_images:
        for img in data["images"]:
            Image.new("RGB", (img["width"], img["height"])).save(split_dir / img["file_name"])


def _checks(report, level: str) -> set[str]:
    return {i.check for i in report.all_issues if i.level == level}


def test_polygon_area():
    assert polygon_area([0, 0, 4, 0, 4, 3, 0, 3]) == pytest.approx(12.0)
    assert polygon_area([0, 0, 1, 1, 2, 2]) == pytest.approx(0.0)


def test_clean_dataset_passes(tmp_path: Path):
    _write_split(tmp_path, "train", _good_split())
    _write_split(tmp_path, "valid", _good_split())
    report = validate_dataset(tmp_path, ["train", "valid"])
    assert report.n_errors == 0
    assert report.n_warnings == 0
    stats = report.splits[0].stats
    assert stats["images"] == 2
    assert stats["annotations"] == 2
    assert stats["zero_instance_categories"] == ["indoor-objects"]
    split = load_split(tmp_path, "train")
    assert split.category_names[2] == "chair"
    assert len(split.annotations_by_image[0]) == 1


def test_detects_annotation_errors(tmp_path: Path):
    data = _good_split()
    data["annotations"][0]["segmentation"] = [[10, 10, 30, 10, 30]]  # odd count
    data["annotations"][1]["category_id"] = 99
    data["annotations"].append(
        {
            "id": 1,  # duplicate id
            "image_id": 7,  # unknown image
            "category_id": 1,
            "bbox": [0, 0, 1, 1],
            "area": 1,
            "segmentation": [[0, 0, 1, 0, 1, 1]],
        }
    )
    data["annotations"].append(
        {
            "id": 3,
            "image_id": 0,
            "category_id": 0,  # supercategory
            "bbox": [0, 0, 0, 5],  # zero width
            "area": 0,
            "segmentation": [],
        }
    )
    _write_split(tmp_path, "train", data)
    errors = _checks(validate_dataset(tmp_path, ["train"]), ERROR)
    assert {
        "polygon with odd coordinate count",
        "unknown category_id",
        "duplicate annotation id",
        "annotation -> unknown image_id",
        "annotation on category 0 (supercategory)",
        "bbox with non-positive size",
        "non-positive or missing area",
        "empty or malformed segmentation",
    } <= errors


def test_detects_file_problems_and_warnings(tmp_path: Path):
    data = _good_split()
    data["images"].append({"id": 2, "file_name": "missing.jpg", "width": 10, "height": 10})
    data["annotations"][0]["segmentation"] = [[10, 10, 90, 10, 90, 20, 10, 20]]  # x > width
    _write_split(tmp_path, "train", data)
    split_dir = tmp_path / "train"
    (split_dir / "missing.jpg").unlink()
    Image.new("RGB", (40, 40)).save(split_dir / "b.jpg")  # size mismatch vs 32x32
    Image.new("RGB", (5, 5)).save(split_dir / "extra.png")

    report = validate_dataset(tmp_path, ["train"])
    assert {"image file missing", "image size differs from JSON"} <= _checks(report, ERROR)
    assert {
        "polygon outside image",
        "image file not referenced in JSON",
        "image without annotations",
    } <= _checks(report, WARNING)


def test_detects_exact_duplicate_annotation(tmp_path: Path):
    data = _good_split()
    dup = copy.deepcopy(data["annotations"][0])
    dup["id"] = 99
    data["annotations"].append(dup)
    _write_split(tmp_path, "train", data)
    report = validate_dataset(tmp_path, ["train"])
    assert report.n_errors == 0
    assert "exact duplicate annotation (same image, class, polygon)" in _checks(report, WARNING)
    assert report.splits[0].stats["duplicate_annotations"] == 1


def test_detects_category_mismatch_across_splits(tmp_path: Path):
    _write_split(tmp_path, "train", _good_split())
    other = _good_split()
    other["categories"][2]["name"] = "sofa"
    _write_split(tmp_path, "valid", other)
    report = validate_dataset(tmp_path, ["train", "valid"], check_files=False)
    assert "categories differ from train" in _checks(report, ERROR)


def test_markdown_and_cli(tmp_path: Path):
    _write_split(tmp_path, "train", _good_split())
    report = validate_dataset(tmp_path, ["train"])
    md = render_markdown(report, "cmd")
    assert "**PASS**" in md
    assert "| 2 | chair |" in md

    out = tmp_path / "report.md"
    code = main(["--dataset-root", str(tmp_path), "--splits", "train", "--out", str(out)])
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("# Data validation report")
