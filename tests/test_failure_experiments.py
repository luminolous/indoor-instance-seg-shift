"""Tests for the failure experiments: image changes, frame mapping, matching, mask grid bound."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
from PIL import Image

from iiss.analysis import failure_experiments as fe
from iiss.analysis.failures import Limits
from iiss.data.coco import CocoSplit
from iiss.eval.results_json import Detection

H, W = 120, 160
BED, CHAIR = 1, 2
LIMITS = Limits()


def _box_mask(x0: int, y0: int, x1: int, y1: int, h: int = H, w: int = W) -> np.ndarray:
    mask = np.zeros((h, w), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _gt(i: int, cat: int, x0: int, y0: int, x1: int, y1: int) -> dict:
    return {
        "id": i, "image_id": 1, "category_id": cat, "bbox": [x0, y0, x1 - x0, y1 - y0],
        "segmentation": [[x0, y0, x1, y0, x1, y1, x0, y1]],
    }  # fmt: skip


def test_gamma_below_one_brightens_and_keeps_black_and_white():
    img = Image.fromarray(np.array([[[0] * 3, [20] * 3, [255] * 3]], np.uint8))
    out = np.asarray(fe.apply_gamma(img, 0.5))[0, :, 0]
    assert out[0] == 0 and out[2] == 255 and out[1] == round(255 * (20 / 255) ** 0.5)


def test_clahe_raises_local_contrast():
    gray = np.full((128, 128, 3), 100, np.uint8)
    gray[:, 64:] = 108
    out = np.asarray(fe.apply_clahe(Image.fromarray(gray), 4.0)).astype(int)
    assert out.shape == gray.shape and np.ptp(out) > np.ptp(gray)


def test_zoom_out_maps_masks_back():
    img = Image.new("RGB", (W, H), (255, 0, 0))
    v = fe.zoom_out(img, 0.5)
    assert v.image.size == (W, H)
    assert v.image.getpixel((0, 0)) == (fe.GRAY,) * 3
    assert v.image.getpixel((W // 2, H // 2)) == (255, 0, 0)
    content = _box_mask(W // 4, H // 4, 3 * W // 4, 3 * H // 4)  # exactly the shrunk photo
    assert v.to_original(content).all()
    half = _box_mask(W // 4, H // 4, W // 2, 3 * H // 4)
    back = v.to_original(half)
    assert fe.mask_iou(back, _box_mask(0, 0, W // 2, H)) > 0.99


def test_crop_window_stays_inside_and_masks_map_back():
    assert fe.crop_window([0, 0, 10, 10], 0.5, W, H) == (0, 0, 80, 60)
    assert fe.crop_window([150, 110, 10, 10], 0.5, W, H) == (80, 60, 160, 120)
    img = Image.new("RGB", (W, H))
    v = fe.crop_around(img, [70, 50, 20, 20], 0.5)
    assert v.image.size == (80, 60)
    x0, y0, _, _ = fe.crop_window([70, 50, 20, 20], 0.5, W, H)
    back = v.to_original(_box_mask(70 - x0, 50 - y0, 90 - x0, 70 - y0, 60, 80))
    assert (back == _box_mask(70, 50, 90, 70)).all()


def test_boundary_band_follows_the_gap_only():
    a, b = _box_mask(0, 0, 78, 80), _box_mask(80, 0, 160, 80)  # gap of 2 px, both end at y = 80
    band = fe.boundary_band(a, b, 6)
    ys, xs = np.nonzero(band)
    assert xs.min() >= 74 and xs.max() <= 85
    assert ys.min() == 0 and 78 <= ys.max() <= 90
    img = Image.new("RGB", (W, H), (200, 200, 200))
    line = np.asarray(fe.draw_line(img, a, b, 6 / W))
    assert (line[40, 79] == 0).all() and (line[40, 20] == 200).all()


def test_darken_only_changes_the_mask():
    img = Image.new("RGB", (W, H), (200, 100, 50))
    out = np.asarray(fe.darken(img, _box_mask(0, 0, 10, 10), 0.5))
    assert tuple(out[5, 5]) == (100, 50, 25) and tuple(out[50, 50]) == (200, 100, 50)


def test_build_variants_names_and_unknown_kind():
    img = Image.new("RGB", (W, H), (90, 90, 90))
    gts = [_gt(1, BED, 0, 0, 78, 80), _gt(2, BED, 80, 0, 160, 80)]
    spec = {
        "variants": {
            "gamma": [0.5], "clahe": [2], "scale": [0.5], "crop": [0.5], "line": [0.03],
            "darken": [0.7],
        }
    }  # fmt: skip
    names = [v.name for v in fe.build_variants(spec, img, gts)]
    assert names == [
        "original", "gamma 0.5", "CLAHE 2", "scale 0.5", "crop 0.5", "line 0.03", "darken 0.7",
    ]  # fmt: skip
    with pytest.raises(ValueError, match="unknown variant"):
        fe.build_variants({"variants": {"blur": [1]}}, img, gts)


@pytest.fixture
def split(tmp_path: Path) -> CocoSplit:
    Image.new("RGB", (W, H), (30, 30, 30)).save(tmp_path / "a.png")
    return CocoSplit(
        "phone",
        tmp_path,
        tmp_path / "x.json",
        {
            "categories": [{"id": BED, "name": "bed"}, {"id": CHAIR, "name": "chair"}],
            "images": [{"id": 1, "file_name": "a.png", "width": W, "height": H}],
            "annotations": [_gt(5, CHAIR, 20, 20, 80, 80)],
        },
    )


SPECS = {
    "low_light": {"photo": "a.png", "targets": [5], "focus": "yolo", "variants": {"gamma": [0.3]}},
    "mask_grid": {"photo": "a.png", "targets": [5], "focus": "m2f", "variants": {"grid": [28]}},
}


def _fake_predictor(run_id: str):
    def predict(image: Image.Image, info) -> list[Detection]:
        bright = np.asarray(image).mean() > 60  # "sees" the chair only in the brighter variant
        cat, score = (CHAIR, 0.9) if bright else (BED, 0.8)
        mask = _box_mask(20, 20, 80, 80)
        return [
            Detection(info["id"], CHAIR, 0.1, mask),
            Detection(info["id"], cat, score, mask),
        ]

    return predict


def test_run_experiments_table_and_figure(split: CocoSplit, tmp_path: Path):
    runs = ["yolo_s640_seed42"]
    outcomes = fe.run_experiments(SPECS, split, runs, LIMITS, _fake_predictor)
    assert [o.variant.name for o in outcomes] == ["original", "gamma 0.3"]  # no mask_grid
    table = fe.result_table(outcomes).set_index("variant")
    assert table.loc["original", "tag"] == "wrong_class"
    assert table.loc["original", "top_class"] == "bed"
    assert table.loc["original", "correct_score"] == pytest.approx(0.1)
    assert table.loc["gamma 0.3", "tag"] == "ok"
    assert table.loc["gamma 0.3", "top_class"] == "chair"
    text = fe.overview(fe.result_table(outcomes), "low_light")["yolo"]
    assert [k[2] for k in text.index] == ["original", "gamma 0.3"]  # order of the variants
    assert text.iloc[0] == "wrong_class [bed 0.80]"
    assert text.iloc[1] == "ok IoU 1.00 [chair 0.90]"
    fe.write_tables(fe.result_table(outcomes), fe.result_table(outcomes), tmp_path)
    assert (tmp_path / "experiments.csv").is_file() and (tmp_path / "mask_grid.csv").is_file()
    fig = fe.plot_variants(
        outcomes, "low_light", runs[0], split.category_names, save_to=tmp_path / "f" / "x.jpg"
    )
    titles = [a.get_title(loc="left") for a in fig.axes[:2]]
    assert titles[0].endswith("original\nchair: wrong_class")
    assert (tmp_path / "f" / "x.jpg").is_file()
    plt.close(fig)


def test_check_baseline_compares_with_the_export(split: CocoSplit):
    outcomes = fe.run_experiments(SPECS, split, ["yolo_s640_seed42"], LIMITS, _fake_predictor)
    export, _ = fe.predict_variant(
        _fake_predictor(""), fe.Variant("original", Image.new("RGB", (W, H), (30, 30, 30))), 1, 0.25
    )
    same = fe.check_baseline(outcomes, split, LIMITS, lambda run: {1: export}).iloc[0]
    assert same["same_tags"] and same["max_score_diff"] == 0
    other = fe.check_baseline(outcomes, split, LIMITS, lambda run: {1: []}).iloc[0]
    assert not other["same_tags"] and other["n_shown_exported"] == 0


def test_grid_roundtrip_is_near_perfect_for_boxes_and_bad_for_thin_shapes():
    box = np.zeros((640, 640), bool)
    box[100:500, 120:520] = True
    assert fe.mask_iou(fe.grid_roundtrip(box, [120, 100, 520, 500], 28, 640), box) > 0.95
    yy, xx = np.mgrid[:640, :640]
    thin = (np.abs(xx - yy) < 6) & (xx > 50) & (xx < 590)  # thin diagonal
    ys, xs = np.nonzero(thin)
    extent = [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
    coarse = fe.mask_iou(fe.grid_roundtrip(thin, extent, 28, 640), thin)
    fine = fe.mask_iou(fe.grid_roundtrip(thin, extent, 112, 640), thin)
    assert coarse < 0.8 < fine


def test_config_targets_and_focus():
    config = fe.load_experiments()
    assert len(config["experiments"]) == 5
    assert len({e["photo"] for e in config["experiments"].values()}) == 5
    for e in config["experiments"].values():
        assert e["focus"] in config["models"] and e["targets"] and e["refuted_if"]
    assert fe.default_run_ids(config)[0] == "yolo_s640_seed42"
