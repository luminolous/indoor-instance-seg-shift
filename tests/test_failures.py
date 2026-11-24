"""Tests for instance matching, failure tags and the comparison figure (synthetic boxes)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from iiss.analysis import failure_viz as fv
from iiss.analysis import failures as f
from iiss.data.coco import CocoSplit
from iiss.eval.results_json import encode_mask

H, W = 100, 120
LIMITS = f.Limits()
BED, CHAIR = 1, 2


def _poly(x0: int, y0: int, x1: int, y1: int) -> list[list[float]]:
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def _gt(i: int, cat: int, *box: int) -> dict:
    return {"id": i, "image_id": 1, "category_id": cat, "segmentation": _poly(*box)}


def _pred(cat: int, score: float, *box: int) -> dict:
    mask = np.zeros((H, W), bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return {"image_id": 1, "category_id": cat, "score": score, "segmentation": encode_mask(mask)}


def _match(gts: list[dict], preds: list[dict], limits: f.Limits = LIMITS) -> f.ImageMatch:
    return f.match_image(gts, preds, H, W, limits)


GT = _gt(7, BED, 10, 10, 50, 50)


def _tag(*preds: dict) -> str:
    return _match([GT], list(preds)).gt[0]["tag"]


def test_overlap_matches_hand_computed_areas():
    m = _match([GT], [_pred(BED, 0.9, 10, 10, 50, 30)])  # half of the object
    row = m.gt[0]
    assert row["iou"] == pytest.approx(0.5, abs=0.01)
    assert row["precision"] == pytest.approx(1.0, abs=0.01)
    assert row["recall"] == pytest.approx(0.5, abs=0.01)
    assert row["area_fraction"] == pytest.approx(1600 / (H * W), abs=0.005)


@pytest.mark.parametrize(
    ("preds", "tag"),
    [
        ([_pred(BED, 0.9, 10, 10, 50, 50)], "ok"),
        ([_pred(BED, 0.9, 10, 10, 50, 35)], "mask_incomplete"),  # matched, IoU 0.63
        ([_pred(BED, 0.9, 10, 10, 50, 75)], "mask_leak"),  # matched, IoU 0.62
        ([_pred(BED, 0.9, 10, 10, 50, 22)], "mask_incomplete"),  # not matched, IoU 0.3
        ([_pred(BED, 0.9, 0, 0, 110, 90)], "mask_leak"),  # not matched, IoU 0.16
        ([_pred(BED, 0.9, 10, 10, 30, 50), _pred(BED, 0.8, 30, 10, 50, 50)], "split"),
        ([_pred(CHAIR, 0.9, 10, 10, 50, 50)], "wrong_class"),
        ([_pred(BED, 0.1, 10, 10, 50, 50)], "low_score"),
        ([_pred(BED, 0.9, 80, 60, 110, 90)], "missed"),
        ([], "missed"),
    ],
)
def test_single_object_tags(preds: list[dict], tag: str):
    assert _tag(*preds) == tag


def test_duplicates_are_not_split():
    m = _match([GT], [_pred(BED, 0.9, 10, 10, 50, 45), _pred(BED, 0.8, 10, 10, 50, 46)])
    assert m.gt[0]["tag"] == "ok"
    assert [p["kind"] for p in m.pred] == ["matched", "duplicate"]
    rough = _match([GT], [_pred(BED, 0.9, 10, 10, 50, 38), _pred(BED, 0.8, 10, 10, 50, 37)])
    assert rough.gt[0]["tag"] == "mask_incomplete"


def test_wrong_class_beats_low_score_and_records_both():
    row = _match([GT], [_pred(CHAIR, 0.8, 10, 10, 50, 50), _pred(BED, 0.1, 10, 10, 50, 50)]).gt[0]
    assert row["tag"] == "wrong_class"
    assert row["predicted_category_id"] == CHAIR and row["predicted_score"] == 0.8
    assert row["hidden_score"] == 0.1


def test_correct_prediction_of_another_object_is_not_a_wrong_class():
    bed, chair = GT, _gt(8, CHAIR, 10, 10, 50, 48)  # nearly the same region
    m = _match([bed, chair], [_pred(CHAIR, 0.9, 10, 10, 50, 48)])
    assert [r["tag"] for r in m.gt] == ["missed", "ok"]


def test_merged_objects():
    a, b = _gt(1, BED, 10, 10, 40, 40), _gt(2, BED, 40, 10, 70, 40)
    m = _match([a, b], [_pred(BED, 0.9, 10, 10, 70, 40)])
    tags = [r["tag"] for r in m.gt]
    assert "merged" in tags and "ok" not in tags


def test_best_scoring_prediction_is_matched_first_and_one_gt_per_prediction():
    a, b = _gt(1, BED, 10, 10, 40, 40), _gt(2, BED, 60, 10, 90, 40)
    m = _match([a, b], [_pred(BED, 0.5, 10, 10, 40, 40), _pred(BED, 0.9, 10, 10, 40, 39)])
    assert [r["matched"] for r in m.gt] == [True, False]
    assert m.pred[0]["score"] == 0.9 and m.pred[0]["kind"] == "matched"
    assert m.pred[1]["kind"] == "duplicate"


def test_prediction_kinds():
    m = _match(
        [GT],
        [
            _pred(CHAIR, 0.9, 10, 10, 50, 50),
            _pred(BED, 0.8, 10, 10, 50, 22),
            _pred(BED, 0.7, 80, 60, 110, 90),
        ],
    )
    assert [p["kind"] for p in m.pred] == ["wrong_class", "poor_mask", "no_gt"]


def test_image_without_gt():
    m = _match([], [_pred(BED, 0.9, 10, 10, 50, 50)])
    assert m.gt == [] and m.pred[0]["kind"] == "no_gt"


def test_threshold_moves_a_prediction_between_shown_and_hidden():
    preds = [_pred(BED, 0.3, 10, 10, 50, 50)]
    assert _match([GT], preds).gt[0]["tag"] == "ok"
    assert _match([GT], preds, f.Limits(score_threshold=0.5)).gt[0]["tag"] == "low_score"


def test_config_limits_and_overrides():
    assert f.default_limits(score_threshold=None) == f.default_limits()
    assert f.default_limits(score_threshold=0.6).score_threshold == 0.6
    assert set(f.settings()["label_issues"])  # documented annotation defects


@pytest.fixture
def scene(tmp_path: Path):
    Image.new("RGB", (W, H), (120, 120, 120)).save(tmp_path / "a.jpg")
    data = {
        "categories": [{"id": BED, "name": "bed"}, {"id": CHAIR, "name": "chair"}],
        "images": [
            {"id": 1, "file_name": "a.jpg", "width": W, "height": H,
             "condition": "normal", "scene": "room"},
        ],
        "annotations": [GT, _gt(8, CHAIR, 70, 60, 110, 90)],
    }  # fmt: skip
    split = CocoSplit("phone", tmp_path, tmp_path / "x.json", data)
    runs = {
        "yolo_s640_seed42": [_pred(BED, 0.9, 10, 10, 50, 50)],
        "m2f_s640_seed42": [_pred(CHAIR, 0.9, 10, 10, 50, 50)],
    }
    for run_id, preds in runs.items():
        path = tmp_path / "runs" / run_id / "predictions" / "phone.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(preds), encoding="utf-8")
    matches = f.match_runs(runs, split, LIMITS, runs_root=tmp_path / "runs")
    return split, matches


def test_tables(scene):
    split, matches = scene
    inst = f.instance_table(matches, split)
    assert len(inst) == 4
    row = inst[(inst["model"] == "m2f") & (inst["name"] == "bed")].iloc[0]
    assert row["tag"] == "wrong_class" and row["predicted_as"] == "chair"
    counts = f.tag_counts(inst)
    assert list(counts.columns) == list(f.TAGS)
    assert (counts.sum(axis=1) == 2).all() and counts.loc["yolo_s640_seed42", "missed"] == 1
    assert f.prediction_counts(matches).loc["m2f_s640_seed42", "wrong_class"] == 1
    photos = f.photo_table(inst, {"a.jpg": "floor not annotated"})
    assert photos.loc["a.jpg", "n_gt"] == 2
    assert photos.loc["a.jpg", "matched_share"] == pytest.approx(0.25)
    assert photos.loc["a.jpg", "label_issue"] == "floor not annotated"


def test_candidates_prefer_shared_failures_and_skip_label_issues(scene):
    split, matches = scene
    inst = f.instance_table(matches, split)
    c = f.candidates(inst)
    assert set(c["tag"]) == {"missed", "wrong_class"}
    missed = c[c["tag"] == "missed"].iloc[0]
    assert missed["name"] == "chair" and missed["n_runs"] == 2
    assert f.candidates(inst, exclude=["a.jpg"]).empty
    assert f.candidates(inst, min_area=0.9).empty


def test_candidates_show_each_scene_once_per_tag():
    base = {"run_id": "r", "condition": "normal", "name": "bed", "tag": "missed"}
    inst = pd.DataFrame(
        [
            {**base, "file_name": "a", "scene": "x", "gt_id": 1, "area_fraction": 0.1},
            {**base, "file_name": "b", "scene": "x", "gt_id": 2, "area_fraction": 0.3},
            {**base, "file_name": "c", "scene": "y", "gt_id": 3, "area_fraction": 0.2},
        ]
    )
    assert list(f.candidates(inst)["file_name"]) == ["b", "c"]


def test_compare_runs_has_one_panel_per_run(scene, tmp_path: Path):
    split, matches = scene
    out = tmp_path / "fig" / "a.png"
    fig = fv.compare_runs(split, "a.jpg", matches, list(matches), save_to=out)
    titles = [a.get_title(loc="left") for a in fig.axes]
    assert len(titles) == 3 and titles[0].startswith("GT: a.jpg (normal, room)")
    assert "matched 1 of 2 GT, 0 other" in titles[1]
    assert "matched 0 of 2 GT, 1 other" in titles[2]
    labels = [t.get_text() for t in fig.axes[2].texts]
    assert labels == ["chair 0.90\nwrong_class"]
    assert out.is_file()
    plt.close(fig)


def test_label_point_lies_inside_the_mask():
    mask = np.zeros((50, 50), bool)
    mask[5:45, 5:10] = True  # L shape: the centroid would be outside
    mask[40:45, 5:45] = True
    x, y = fv.label_point(mask)
    assert mask[y, x]


def test_reduced_size_never_upscales():
    assert fv.reduced_size(3472, 3472) == (1000, 1000)
    assert fv.reduced_size(40, 30) == (40, 30)


def test_photo_detail_and_written_tables(scene, tmp_path: Path):
    split, matches = scene
    inst = f.instance_table(matches, split)
    detail = f.photo_detail(inst, "a.jpg")
    assert list(detail.columns) == ["yolo_s640_seed42", "m2f_s640_seed42"]
    assert detail.loc[(7, "bed"), "m2f_s640_seed42"] == "wrong_class (chair 0.90)"
    assert detail.loc[(7, "bed"), "yolo_s640_seed42"].startswith("ok (IoU ")
    out = f.write_failure_tables(inst, f.candidates(inst), tmp_path / "out")
    assert {p.name for p in out.iterdir()} == {"instances.csv", "tag_counts.csv", "candidates.csv"}
    assert len(pd.read_csv(out / "instances.csv")) == 4


def test_wrong_class_reports_the_highest_score():
    preds = [_pred(CHAIR, 0.9, 10, 10, 50, 45), _pred(3, 0.4, 10, 10, 50, 50)]
    row = _match([GT], preds).gt[0]
    assert row["predicted_category_id"] == CHAIR and row["predicted_score"] == 0.9
