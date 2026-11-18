import json
from pathlib import Path

import numpy as np
from PIL import Image

from iiss.analysis.viz import overlay_instances, sample_gallery
from iiss.data.coco import load_split


def _ann(i: int, poly: list[float], area: float) -> dict:
    return {
        "id": i,
        "image_id": 0,
        "category_id": 1,
        "bbox": [poly[0], poly[1], 5, 5],
        "area": area,
        "segmentation": [poly],
    }


def test_overlay_changes_only_masked_pixels():
    img = Image.new("RGB", (40, 30), (255, 255, 255))
    out = overlay_instances(img, [_ann(1, [5, 5, 15, 5, 15, 15, 5, 15], 100)])
    assert out.shape == (30, 40, 3) and out.dtype == np.uint8
    assert tuple(out[0, 0]) == (255, 255, 255)
    assert tuple(out[10, 10]) != (255, 255, 255)


def test_sample_gallery_uses_rotation_fix(tmp_path: Path):
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    Image.new("RGB", (30, 40)).save(split_dir / "rot.png")  # stored portrait
    data = {
        "categories": [{"id": 0, "name": "indoor-objects"}, {"id": 1, "name": "bed"}],
        "images": [{"id": 0, "file_name": "rot.png", "width": 40, "height": 30}],
        "annotations": [_ann(1, [5, 5, 35, 5, 35, 25, 5, 25], 600)],
    }
    (split_dir / "_annotations.coco.json").write_text(json.dumps(data), encoding="utf-8")
    split = load_split(tmp_path, "train")
    fig = sample_gallery(split, file_names=["rot.png"], fixes={"rot.png": 90}, cols=1)
    shown = fig.axes[0].images[0].get_array()
    assert shown.shape[:2] == (30, 40)  # landscape after the fix
