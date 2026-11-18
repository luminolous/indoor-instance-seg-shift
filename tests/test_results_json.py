import json
from pathlib import Path

import numpy as np
import pytest
from pycocotools.coco import COCO

from iiss.eval.results_json import (
    Detection,
    PredictionError,
    ResultsWriter,
    decode_mask,
    detection_to_coco,
    encode_mask,
    polygons_to_mask,
    resize_mask,
)

H, W = 40, 60


def _box_mask(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


@pytest.fixture
def gt() -> dict:
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": W, "height": H},
            {"id": 2, "file_name": "b.jpg", "width": W, "height": H},
        ],
        "categories": [{"id": 0, "name": "indoor-objects"}, {"id": 5, "name": "chair"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 5,
                "bbox": [10, 5, 20, 10],
                "area": 200,
                "segmentation": [[10, 5, 30, 5, 30, 15, 10, 15]],
                "iscrowd": 0,
            }
        ],
    }


def test_rle_round_trip_is_json_safe():
    mask = _box_mask(3, 4, 20, 30)
    rle = encode_mask(mask)
    assert isinstance(rle["counts"], str)
    assert rle["size"] == [H, W]
    json.dumps(rle)
    assert np.array_equal(decode_mask(rle), mask)


def test_detection_to_coco_bbox_area():
    res = detection_to_coco(Detection(1, 5, 0.9, _box_mask(10, 5, 30, 15)))
    assert res["bbox"] == [10.0, 5.0, 20.0, 10.0]
    assert res["area"] == 200.0
    assert detection_to_coco(Detection(1, 5, 0.9, np.zeros((H, W), bool))) is None


def test_polygons_to_mask_matches_box():
    mask = polygons_to_mask([[10, 5, 30, 5, 30, 15, 10, 15]], H, W)
    assert mask.shape == (H, W)
    assert mask.sum() > 0
    assert mask[10, 20] and not mask[0, 0]


def test_resize_mask_soft_and_binary():
    soft = np.zeros((20, 30), dtype=np.float32)
    soft[5:15, 5:25] = 0.9
    out = resize_mask(soft, H, W)
    assert out.shape == (H, W) and out.dtype == bool
    assert out[20, 30] and not out[0, 0]
    binary = resize_mask(soft > 0.5, H, W)
    assert binary.dtype == bool and binary.shape == (H, W)


def test_writer_filters_and_validates(gt: dict, tmp_path: Path):
    writer = ResultsWriter.from_coco_gt(gt, max_dets=2, score_threshold=0.1)
    dets = [
        Detection(1, 5, 0.3, _box_mask(0, 0, 5, 5)),
        Detection(1, 5, 0.9, _box_mask(10, 5, 30, 15)),
        Detection(1, 5, 0.5, _box_mask(0, 0, 10, 10)),
        Detection(1, 5, 0.05, _box_mask(0, 0, 10, 10)),  # below threshold
        Detection(1, 5, 0.8, np.zeros((H, W), bool)),  # empty
    ]
    assert writer.add_image(1, dets) == 2
    assert [r["score"] for r in writer.results] == [0.9, 0.5]
    assert writer.counts == {
        "received": 5,
        "empty_mask": 1,
        "below_threshold": 1,
        "over_max_dets": 1,
    }

    with pytest.raises(PredictionError, match="already added"):
        writer.add_image(1, [])
    with pytest.raises(PredictionError, match="not in the ground truth"):
        writer.add_image(99, [])
    with pytest.raises(PredictionError, match="unknown category_id"):
        writer.add_image(2, [Detection(2, 7, 0.9, _box_mask(0, 0, 5, 5))])

    out = writer.write(tmp_path / "preds" / "test.json", meta={"run_id": "yolo_s640_seed42"})
    meta = json.loads((tmp_path / "preds" / "test.meta.json").read_text())
    assert meta["run_id"] == "yolo_s640_seed42"
    assert meta["images_missing"] == [2]
    assert len(json.loads(out.read_text())) == 2


def test_writer_rejects_wrong_mask_size(gt: dict):
    writer = ResultsWriter.from_coco_gt(gt)
    with pytest.raises(PredictionError, match="resize masks"):
        writer.add_image(1, [Detection(1, 5, 0.9, np.ones((H // 2, W // 2), bool))])


def test_pycocotools_loads_results(gt: dict, tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    writer = ResultsWriter.from_coco_gt(gt)
    writer.add_image(1, [Detection(1, 5, 0.9, _box_mask(10, 5, 30, 15))])
    writer.add_image(2, [])
    res_path = writer.write(tmp_path / "res.json")

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(res_path))
    ann = coco_dt.loadAnns(coco_dt.getAnnIds())[0]
    assert ann["category_id"] == 5
    assert np.array_equal(coco_dt.annToMask(ann).astype(bool), _box_mask(10, 5, 30, 15))
