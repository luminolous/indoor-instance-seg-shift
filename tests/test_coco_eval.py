import contextlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pycocotools.cocoeval import COCOeval

from iiss.eval.coco_eval import (
    category_groups,
    image_groups_from_table,
    load_gt,
    load_results,
    per_class_ap,
    run_cocoeval,
    subset_summary,
    summary,
)
from iiss.eval.results_json import Detection, ResultsWriter, polygons_to_mask
from iiss.evaluate import evaluate_files, main

H, W = 60, 80


def _rect(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _gt() -> dict:
    boxes = {  # image_id -> [(category_id, x, y, w, h)]
        1: [(1, 5, 5, 20, 15), (2, 0, 30, 80, 30)],
        2: [(1, 40, 10, 25, 25), (1, 2, 2, 6, 6)],
        3: [(2, 0, 0, 80, 20)],
    }
    anns, k = [], 0
    for img_id, objs in boxes.items():
        for cat, x, y, w, h in objs:
            k += 1
            anns.append(
                {
                    "id": k,
                    "image_id": img_id,
                    "category_id": cat,
                    "bbox": [x, y, w, h],
                    "area": w * h,
                    "segmentation": [_rect(x, y, w, h)],
                    "iscrowd": 0,
                }
            )
    return {
        "images": [
            {"id": 1, "file_name": "IMG_001_jpg.rf.aaa.jpg", "width": W, "height": H},
            {"id": 2, "file_name": "IMG_002_jpg.rf.bbb.jpg", "width": W, "height": H},
            {"id": 3, "file_name": "IMG_003_jpg.rf.ccc.jpg", "width": W, "height": H},
        ],
        "categories": [
            {"id": 0, "name": "indoor-objects"},
            {"id": 1, "name": "chair"},
            {"id": 2, "name": "wall"},
            {"id": 3, "name": "mouse"},  # no GT anywhere
        ],
        "annotations": anns,
    }


def _perfect_results(gt: dict) -> list[dict]:
    writer = ResultsWriter.from_coco_gt(gt)
    by_img: dict[int, list[Detection]] = {img["id"]: [] for img in gt["images"]}
    for a in gt["annotations"]:
        mask = polygons_to_mask(a["segmentation"], H, W)
        by_img[a["image_id"]].append(Detection(a["image_id"], a["category_id"], 1.0, mask))
    for img_id, dets in by_img.items():
        writer.add_image(img_id, dets)
    return writer.results


def _imperfect_results(gt: dict) -> list[dict]:
    """Shifted masks, a false positive and a missed object."""
    writer = ResultsWriter.from_coco_gt(gt)
    rng = np.random.default_rng(0)
    by_img: dict[int, list[Detection]] = {img["id"]: [] for img in gt["images"]}
    for a in gt["annotations"]:
        if a["id"] == 4:  # missed
            continue
        x, y, w, h = a["bbox"]
        dx, dy = rng.integers(0, 4, size=2)
        mask = polygons_to_mask([_rect(x + dx, y + dy, w, h)], H, W)
        by_img[a["image_id"]].append(
            Detection(a["image_id"], a["category_id"], float(rng.uniform(0.3, 1)), mask)
        )
    fp = polygons_to_mask([_rect(50, 40, 10, 10)], H, W)
    by_img[3].append(Detection(3, 1, 0.95, fp))
    for img_id, dets in by_img.items():
        writer.add_image(img_id, dets)
    return writer.results


@pytest.mark.parametrize("iou_type", ["segm", "bbox"])
def test_perfect_predictions_score_one(iou_type: str):
    gt_dict = _gt()
    gt = load_gt(gt_dict)
    ce = run_cocoeval(gt, load_results(gt, _perfect_results(gt_dict)), iou_type)
    s = summary(ce)
    assert s["AP"] == pytest.approx(1.0)
    assert s["AP50"] == pytest.approx(1.0)
    pc = per_class_ap(ce, gt).set_index("name")
    assert pc.loc["chair", "AP"] == pytest.approx(1.0)
    assert pc.loc["chair", "n_gt"] == 3
    assert np.isnan(pc.loc["mouse", "AP"])  # no GT -> excluded


def test_empty_predictions_score_zero():
    gt = load_gt(_gt())
    s = summary(run_cocoeval(gt, load_results(gt, []), "segm"))
    assert s["AP"] == pytest.approx(0.0)


def _reference_stats(gt, dt, img_ids=None, cat_ids=None) -> np.ndarray:
    ce = COCOeval(gt, dt, iouType="segm")
    if img_ids is not None:
        ce.params.imgIds = sorted(img_ids)
    if cat_ids is not None:
        ce.params.catIds = sorted(cat_ids)
    with contextlib.redirect_stdout(io.StringIO()):
        ce.evaluate()
        ce.accumulate()
        ce.summarize()
    return ce.stats


@pytest.mark.parametrize(
    ("img_ids", "cat_ids"), [(None, None), ([1, 2], None), (None, [1]), ([2, 3], [2])]
)
def test_subset_matches_separate_cocoeval(img_ids, cat_ids):
    gt_dict = _gt()
    gt = load_gt(gt_dict)
    dt = load_results(gt, _imperfect_results(gt_dict))
    ce = run_cocoeval(gt, dt, "segm")
    ours = np.array(list(subset_summary(ce, img_ids=img_ids, cat_ids=cat_ids).values()))
    ref = _reference_stats(gt, dt, img_ids, cat_ids)
    ref = np.where(ref == -1, np.nan, ref)
    np.testing.assert_allclose(ours, ref, equal_nan=True)
    assert 0 < summary(ce)["AP"] < 1


def test_category_groups():
    gt = load_gt(_gt())
    train = load_gt(
        {**_gt(), "annotations": [a for a in _gt()["annotations"] if a["category_id"] == 1]}
    )
    groups = category_groups(gt, ["wall"], train)
    assert groups["stuff"] == [2]
    assert set(groups["things"]) == {0, 1, 3}
    assert groups["in_train"] == [1]
    assert set(groups["absent_in_train"]) == {0, 2, 3}
    with pytest.raises(ValueError, match="not in the category list"):
        category_groups(gt, ["sky"])


def test_image_groups_from_table():
    gt = load_gt(_gt())
    table = pd.DataFrame(
        {
            "filename": ["IMG_001.jpg", "IMG_002.jpg", "IMG_003.jpg"],
            "condition": ["normal", "low_light", "normal"],
        }
    )
    assert image_groups_from_table(gt, table) == {"normal": [1, 3], "low_light": [2]}
    with pytest.raises(ValueError, match="have no group"):
        image_groups_from_table(gt, table.iloc[:2])


def test_evaluate_files_and_cli(tmp_path: Path):
    gt_dict = _gt()
    gt_path = tmp_path / "gt.json"
    gt_path.write_text(json.dumps(gt_dict), encoding="utf-8")
    pred_path = tmp_path / "pred.json"
    pred_path.write_text(json.dumps(_perfect_results(gt_dict)), encoding="utf-8")
    groups_csv = tmp_path / "meta.csv"
    pd.DataFrame(
        {
            "filename": ["IMG_001.jpg", "IMG_002.jpg", "IMG_003.jpg"],
            "condition": ["normal", "backlight", "normal"],
        }
    ).to_csv(groups_csv, index=False)

    out = tmp_path / "eval"
    metrics = evaluate_files(
        gt_path,
        pred_path,
        out,
        run_id="x",
        eval_set="test",
        stuff_classes=["wall"],
        train_gt_file=gt_path,
        image_groups_csv=groups_csv,
    )
    assert metrics["segm"]["summary"]["AP"] == pytest.approx(1.0)
    assert metrics["segm"]["by_category_group"]["stuff"]["AP"] == pytest.approx(1.0)
    assert metrics["segm"]["by_image_group"]["backlight"]["n_images"] == 1
    for name in ("metrics.json", "summary.md", "per_class_ap_segm.csv", "per_class_ap_bbox.csv"):
        assert (out / name).is_file()
    json.loads((out / "metrics.json").read_text())  # valid JSON (NaN replaced)

    code = main(
        [
            "--gt",
            str(gt_path),
            "--pred",
            str(pred_path),
            "--out",
            str(tmp_path / "cli"),
            "--no-breakdowns",
        ]
    )
    assert code == 0
    assert "| segm | 100.0 |" in (tmp_path / "cli" / "summary.md").read_text()
