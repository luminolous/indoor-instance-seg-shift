"""Smoke and content tests for the report figures (synthetic tables, headless)."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from iiss.analysis import result_plots as rp

MODELS, SIZES, CLASSES = ("yolo", "maskrcnn", "m2f"), (320, 640, 960), ("bed", "chair")


@pytest.fixture
def tables() -> dict:
    main, per_class, conditions = [], [], []
    for i, model in enumerate(MODELS):
        for j, size in enumerate(SIZES):
            run = f"{model}_s{size}_seed42"
            for k, eval_set in enumerate(("valid", "test", "phone")):
                ap = 0.05 * (i + 1) + 0.01 * j + 0.02 * (2 - k)
                main.append(
                    {
                        "run_id": run, "model": model, "input_size": size, "eval_set": eval_set,
                        "n_images": 40, "AP": ap, "AP_common": ap,
                        "cap_hit": run == "m2f_s960_seed42",
                    }
                )  # fmt: skip
                for c, n in zip(CLASSES, (3, 9), strict=True):
                    per_class.append(
                        {
                            "model": model, "input_size": size, "eval_set": eval_set,
                            "name": c, "n_gt": n, "AP": ap, "AP50": ap,
                        }
                    )  # fmt: skip
            for cond, ap in (("normal", 0.2), ("low_light", 0.05)):
                conditions.append(
                    {
                        "run_id": run, "model": model, "input_size": size,
                        "condition": cond, "AP": ap, "AP_common": ap, "n_common_classes": 2,
                    }
                )  # fmt: skip
    return {
        "main_results": pd.DataFrame(main),
        "per_class_ap": pd.DataFrame(per_class),
        "conditions": pd.DataFrame(conditions),
        "common_classes": list(CLASSES),
    }


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def test_save_all_writes_every_figure(tables: dict, tmp_path: Path):
    figures = rp.save_all(tables, tmp_path / "fig")
    assert {p.name for p in (tmp_path / "fig").iterdir()} == set(figures)
    assert all(p.stat().st_size > 5000 for p in (tmp_path / "fig").iterdir())


def test_every_model_has_its_own_color_and_marker():
    assert len(set(rp.MODEL_COLORS.values())) == len(set(rp.MODEL_MARKERS.values())) == 3


def test_ap_vs_size_labels_panels_and_marks_the_capped_run(tables: dict):
    fig = rp.plot_ap_vs_size(tables["main_results"])
    assert [a.get_title(loc="left")[:5] for a in fig.axes] == ["valid", "test ", "phone"]
    texts = [t.get_text() for a in fig.axes for t in a.texts]
    assert sum(t.endswith("*") for t in texts) == 3  # m2f 960 in each panel
    assert any("cap" in t.get_text() for t in fig.legends[0].get_texts())


def test_ap_vs_size_shows_the_right_values(tables: dict):
    fig = rp.plot_ap_vs_size(tables["main_results"], "AP")
    line = fig.axes[0].get_lines()[0]  # yolo on valid
    assert list(line.get_xdata()) == [320, 640, 960]
    assert list(line.get_ydata()) == pytest.approx([9.0, 10.0, 11.0])


def test_domain_shift_panels_follow_sizes_and_set_order(tables: dict):
    fig = rp.plot_domain_shift(tables["main_results"])
    assert [a.get_title(loc="left") for a in fig.axes] == ["S = 320", "S = 640", "S = 960"]
    assert [t.get_text() for t in fig.axes[0].get_xticklabels()] == ["valid", "test", "phone"]
    assert list(fig.axes[0].get_lines()[0].get_ydata()) == pytest.approx([9.0, 7.0, 5.0])


def test_no_cap_column_is_fine(tables: dict):
    main = tables["main_results"].drop(columns="cap_hit")
    fig = rp.plot_ap_vs_size(main)
    assert not any("*" in t.get_text() for a in fig.axes for t in a.texts)


def test_conditions_show_both_values_per_run(tables: dict):
    fig = rp.plot_conditions(tables["conditions"], tables["main_results"])
    labels = [t.get_text() for t in fig.axes[0].texts]
    assert len(labels) == 9 and labels[0] == "5.0 | 20.0"
    assert fig.axes[0].get_yticklabels()[-1].get_text() == "Mask2Former 960*"


def test_heatmap_orders_classes_by_phone_count(tables: dict):
    fig = rp.plot_per_class_heatmap(tables["per_class_ap"], tables["common_classes"])
    ticks = [t.get_text() for t in fig.axes[0].get_yticklabels()]
    assert ticks[:2] == ["chair (9 / 9 / 9)", "bed (3 / 3 / 3)"]
    assert not fig.axes[1].get_yticklabels()  # the names are printed once
    grid, counts = rp.heatmap_table(tables["per_class_ap"], CLASSES, "phone")
    assert grid.shape == (2, 9) and counts["bed"] == 3
    assert list(grid.columns[:3]) == [("yolo", 320), ("yolo", 640), ("yolo", 960)]


def test_heatmap_stars_only_the_capped_run(tables: dict):
    fig = rp.plot_per_class_heatmap(
        tables["per_class_ap"], tables["common_classes"], capped=[("m2f", 960)]
    )
    labels = [t.get_text() for t in fig.axes[2].get_xticklabels()]
    assert labels[-1] == "Mask2Former 960*" and sum("*" in x for x in labels) == 1


def test_end_labels_do_not_overlap(tables: dict):
    main = tables["main_results"].assign(AP_common=0.15)  # every line ends at one point
    for fig in (rp.plot_ap_vs_size(main), rp.plot_domain_shift(main)):
        fig.canvas.draw()
        for ax in fig.axes:
            boxes = sorted((t.get_window_extent() for t in ax.texts), key=lambda b: b.y0)
            assert len(boxes) == 3
            assert all(a.y1 <= b.y0 + 1 for a, b in zip(boxes, boxes[1:], strict=False))


def test_wide_table_keeps_run_order_and_stars_the_capped_run(tables: dict):
    wide = rp.wide_table(tables["main_results"], "AP")
    assert list(wide.columns) == ["valid", "test", "phone"]
    assert wide.index[0] == "YOLO11s-seg 320" and wide.index[-1] == "Mask2Former 960*"
    assert list(wide.iloc[0]) == pytest.approx([9.0, 7.0, 5.0])
    no_cap = tables["main_results"].drop(columns="cap_hit")
    assert rp.capped_runs(no_cap) == [] and rp.wide_table(no_cap, "AP").index[-1][-1] == "0"
