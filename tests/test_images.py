import json
from pathlib import Path

import pytest
from PIL import Image

from iiss.data.images import (
    ImageSizeError,
    default_rotation_fixes,
    load_image,
    validate_rotation_fixes,
)
from iiss.data.validate import ERROR, WARNING, validate_dataset


def _portrait_with_marker(path: Path) -> None:
    """A 4x6 (w x h) image, black except a red pixel at the top-left corner."""
    im = Image.new("RGB", (4, 6))
    im.putpixel((0, 0), (255, 0, 0))
    im.save(path)


@pytest.mark.parametrize(("deg", "marker_xy"), [(90, (0, 3)), (270, (5, 0))])
def test_load_image_applies_rotation_fix(tmp_path: Path, deg: int, marker_xy: tuple[int, int]):
    path = tmp_path / "x.png"
    _portrait_with_marker(path)
    im = load_image(path, {"width": 6, "height": 4}, fixes={"x.png": deg})
    assert im.size == (6, 4)
    # 90 CCW moves the top-left corner to bottom-left; 270 CCW (= 90 CW) to top-right.
    assert im.getpixel(marker_xy) == (255, 0, 0)


def test_load_image_size_mismatch_raises(tmp_path: Path):
    path = tmp_path / "x.png"
    _portrait_with_marker(path)
    with pytest.raises(ImageSizeError):
        load_image(path, {"width": 6, "height": 4}, fixes={})
    assert load_image(path, {"width": 4, "height": 6}, fixes={}).size == (4, 6)


def test_invalid_rotation_value():
    with pytest.raises(ValueError):
        validate_rotation_fixes({"x.png": 45})


def test_repo_fix_table():
    fixes = default_rotation_fixes()
    assert len(fixes) == 4
    assert set(fixes.values()) <= {90, 270}


def _split_with_rotated_image(root: Path) -> None:
    split = root / "train"
    split.mkdir(parents=True)
    _portrait_with_marker(split / "x.png")
    data = {
        "categories": [{"id": 0, "name": "indoor-objects"}, {"id": 1, "name": "bed"}],
        "images": [{"id": 0, "file_name": "x.png", "width": 6, "height": 4}],
        "annotations": [
            {
                "id": 0,
                "image_id": 0,
                "category_id": 1,
                "bbox": [0, 0, 6, 4],
                "area": 24,
                "segmentation": [[0, 0, 6, 0, 6, 4, 0, 4]],
                "iscrowd": 0,
            }
        ],
    }
    (split / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")


def test_validator_accepts_listed_fix(tmp_path: Path):
    _split_with_rotated_image(tmp_path)

    without_fix = validate_dataset(tmp_path, ["train"])
    assert "image size differs from JSON" in {i.check for i in without_fix.all_issues}

    with_fix = validate_dataset(tmp_path, ["train"], fixes={"x.png": 90, "gone.jpg": 90})
    assert with_fix.n_errors == 0
    warnings = {i.check for i in with_fix.all_issues if i.level == WARNING}
    assert "rotation fix applied on load (data.rotation_fixes)" in warnings
    assert "rotation fix for a file not in the validated splits" in warnings
    assert with_fix.splits[0].stats["rotation_fixed_images"] == 1


def test_validator_flags_fix_on_unrotated_image(tmp_path: Path):
    _split_with_rotated_image(tmp_path)
    report = validate_dataset(tmp_path, ["train"], fixes={"x.png": 180})
    assert any(
        i.level == ERROR and i.check == "image size differs from JSON" for i in report.all_issues
    )
