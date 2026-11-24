import json
from pathlib import Path

import pytest

from iiss.eval.coco_eval import load_gt, load_results, run_cocoeval, summary
from iiss.eval.sanity import gt_as_results, run_sanity


def _gt(n_objects_img1: int, first_ann_id: int = 0) -> dict:
    anns = []
    for k in range(n_objects_img1):
        x = (k % 10) * 8
        y = (k // 10) * 8
        anns.append(
            {
                "id": first_ann_id + k,
                "image_id": 1,
                "category_id": 1,
                "bbox": [x, y, 6, 6],
                "area": 36,
                "segmentation": [[x, y, x + 6, y, x + 6, y + 6, x, y + 6]],
                "iscrowd": 0,
            }
        )
    anns.append(
        {
            "id": first_ann_id + 1000,
            "image_id": 2,
            "category_id": 2,
            "bbox": [10, 10, 50, 30],
            "area": 1500,
            "segmentation": [[10, 10, 60, 10, 60, 40, 10, 40]],
            "iscrowd": 0,
        }
    )
    return {
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": 80, "height": 120},
            {"id": 2, "file_name": "b.jpg", "width": 80, "height": 120},
        ],
        "categories": [
            {"id": 0, "name": "indoor-objects"},
            {"id": 1, "name": "chair"},
            {"id": 2, "name": "wall"},
        ],
        "annotations": anns,
    }


def test_gt_as_results_keeps_all_by_default():
    gt = _gt(120)
    assert len(gt_as_results(gt)) == 121
    assert len(gt_as_results(gt, max_dets=100)) == 101


def test_annotation_id_zero_is_matched():
    """pycocotools counts a match to GT id 0 as a false positive; load_gt must avoid that."""
    gt_dict = _gt(1, first_ann_id=0)
    gt = load_gt(gt_dict)
    assert min(gt.anns) == 1
    ap = summary(run_cocoeval(gt, load_results(gt, gt_as_results(gt_dict)), "segm"))["AP"]
    assert ap == pytest.approx(1.0)


def test_run_sanity_passes_and_reports_ceiling(tmp_path: Path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(_gt(120)), encoding="utf-8")
    report = run_sanity(gt_path, tmp_path / "out", "test", stuff_classes=["wall"])
    assert report["all_passed"]
    assert report["images_over_100_instances"] == 1
    # 120 chairs in one image: the standard protocol cannot reach full recall.
    assert report["standard_ceiling"]["segm"]["AR100"] < 1.0
    assert report["standard_ceiling"]["segm"]["AP"] < 1.0
    assert (tmp_path / "out" / "sanity.json").is_file()
    assert not (tmp_path / "out" / "uncapped" / "gt_as_predictions.json").exists()


def test_group_without_gt_is_skipped(tmp_path: Path):
    """The phone set has no class that is absent from train; that group has no AP."""
    gt = {
        "images": [{"id": 1, "file_name": "a.jpg", "width": 50, "height": 50}],
        "annotations": [
            {
                "id": 1, "image_id": 1, "category_id": 1, "iscrowd": 0, "area": 400,
                "bbox": [5, 5, 20, 20], "segmentation": [[5, 5, 25, 5, 25, 25, 5, 25]],
            }
        ],
        "categories": [{"id": 1, "name": "bed"}, {"id": 2, "name": "wall"}],
    }  # fmt: skip
    path = tmp_path / "gt.json"
    path.write_text(json.dumps(gt))
    report = run_sanity(path, tmp_path / "out", "phone", ["wall"], path)
    assert report["all_passed"]
    assert "segm_AP[in_train]" in report["checks"]
    assert "segm_AP[absent_in_train]" not in report["checks"]
