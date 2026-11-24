"""Tests for the results aggregation (hand-made evaluation files with known numbers)."""

import json
from pathlib import Path

import pandas as pd
import pytest

from iiss.analysis.domain_shift import (
    AggregationError,
    build_tables,
    common_classes,
    load_results,
    load_tables,
    parse_run_id,
    run_ids,
    write_tables,
)

RUN = "yolo_s640_seed42"
CLASSES = ["bed", "chair", "wall"]


def _write_set(
    root: Path,
    run_id: str,
    eval_set: str,
    ap: dict[str, float],
    n_gt: dict[str, int] | None = None,
    conditions: dict[str, dict[str, float]] | None = None,
    summary_ap: float | None = None,
    claimed_run: str | None = None,
) -> None:
    """``ap``: class -> AP; AP50 is 2 x AP. Classes without an entry have no GT."""
    folder = root / run_id / eval_set
    folder.mkdir(parents=True, exist_ok=True)
    n_gt = n_gt or dict.fromkeys(ap, 3)
    mean = sum(ap.values()) / len(ap)

    def table(values: dict[str, float], counts: dict[str, int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "category_id": range(len(CLASSES)),
                "name": CLASSES,
                "n_gt": [counts.get(c, 0) if c in values else 0 for c in CLASSES],
                "AP": [values.get(c) for c in CLASSES],
                "AP50": [2 * values[c] if c in values else None for c in CLASSES],
                "AP75": [values.get(c) for c in CLASSES],
            }
        )

    table(ap, n_gt).to_csv(folder / "per_class_ap_segm.csv", index=False)
    by_group = {}
    for name, values in (conditions or {}).items():
        table(values, dict.fromkeys(values, 2)).to_csv(
            folder / f"per_class_ap_segm_{name}.csv", index=False
        )
        by_group[name] = {
            "n_images": 5, "n_gt": 2 * len(values), "n_categories": len(values),
            "AP": sum(values.values()) / len(values), "AP50": 0.5,
        }  # fmt: skip
    ap_value = mean if summary_ap is None else summary_ap
    summary = dict.fromkeys(("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100"), ap_value)
    summary["APs"] = None
    group = {"AP": 0.1, "AP50": 0.2}
    segm = {
        "summary": summary,
        "by_category_group": {
            "things": group, "stuff": group, "in_train": group,
            "absent_in_train": {"AP": None, "AP50": None},
        },
        "by_image_group": by_group,
    }  # fmt: skip
    metrics = {
        "run_id": claimed_run or run_id, "eval_set": eval_set, "num_images": 10,
        "num_gt_instances": sum(n_gt.values()), "segm": segm, "bbox": {"summary": {"AP": 0.3}},
    }  # fmt: skip
    (folder / "metrics.json").write_text(json.dumps(metrics))


@pytest.fixture
def root(tmp_path: Path) -> Path:
    _write_set(tmp_path, RUN, "valid", {"bed": 0.4, "chair": 0.2, "wall": 0.6})
    _write_set(tmp_path, RUN, "test", {"bed": 0.2, "chair": 0.1})
    _write_set(
        tmp_path, RUN, "phone", {"bed": 0.1, "chair": 0.1},
        conditions={"normal": {"bed": 0.3, "chair": 0.1}, "low_light": {"bed": 0.05}},
    )  # fmt: skip
    return tmp_path


def test_run_ids_and_parsing():
    assert len(run_ids()) == 9 and run_ids()[0] == "yolo_s320_seed42"
    assert parse_run_id("m2f_s960_seed42") == ("m2f", 960)


def test_common_classes_and_main_table(root: Path):
    t = build_tables(root, ids=[RUN])
    assert t["common_classes"] == ["bed", "chair"]  # wall has GT only in valid
    main = t["main_results"].set_index("eval_set")
    assert main.loc["valid", "n_classes"] == 3
    assert main.loc["valid", "AP"] == pytest.approx(0.4)
    assert main.loc["valid", "AP_common"] == pytest.approx(0.3)  # mean of bed, chair
    assert main.loc["valid", "AP50_common"] == pytest.approx(0.6)
    assert main.loc["phone", "bbox_AP"] == 0.3


def test_shift_deltas(root: Path):
    s = build_tables(root, ids=[RUN])["domain_shift"]
    full = s[(s["metric"] == "AP") & (s["from"] == "valid") & (s["to"] == "test")].iloc[0]
    assert full["delta"] == pytest.approx(0.15 - 0.4)
    assert full["relative"] == pytest.approx(-0.625)
    both = s[(s["metric"] == "AP_common") & (s["from"] == "valid") & (s["to"] == "phone")].iloc[0]
    assert both["delta"] == pytest.approx(0.1 - 0.3)
    assert len(s) == 6


def test_conditions_use_shared_classes(root: Path):
    c = build_tables(root, ids=[RUN])["conditions"].set_index("condition")
    assert c.loc["normal", "AP"] == pytest.approx(0.2) and c.loc["normal", "n_gt"] == 4
    assert c.loc["normal", "AP_common"] == pytest.approx(0.3)  # only bed is shared
    assert c.loc["low_light", "AP_common"] == pytest.approx(0.05)
    assert c.loc["normal", "n_common_classes"] == 1 and c.loc["normal", "n_gt_common"] == 2


def test_zero_ap_has_no_relative_change(tmp_path: Path):
    for eval_set in ("valid", "test", "phone"):
        _write_set(tmp_path, RUN, eval_set, {"bed": 0.0})
    s = build_tables(tmp_path, ids=[RUN])["domain_shift"]
    assert s["relative"].isna().all() and (s["delta"] == 0).all()


def test_missing_set_is_an_error_not_a_gap(root: Path):
    (root / RUN / "test" / "metrics.json").unlink()
    with pytest.raises(AggregationError, match=f"missing {RUN}/test"):
        load_results(root, [RUN])


def test_files_of_another_job_are_rejected(root: Path):
    _write_set(root, RUN, "test", {"bed": 0.2, "chair": 0.1}, claimed_run="other")
    with pytest.raises(AggregationError, match="another job"):
        load_results(root, [RUN])


def test_summary_must_match_per_class_mean(root: Path):
    _write_set(root, RUN, "test", {"bed": 0.2, "chair": 0.1}, summary_ap=0.5)
    with pytest.raises(AggregationError, match="mean per-class AP"):
        load_results(root, [RUN])


def test_different_ground_truth_is_rejected(root: Path):
    other = "m2f_s640_seed42"
    for eval_set, ap in (
        ("valid", {"bed": 0.4, "chair": 0.2, "wall": 0.6}),
        ("test", {"bed": 0.2, "chair": 0.1}),
        ("phone", {"bed": 0.1, "chair": 0.1}),
    ):
        _write_set(root, other, eval_set, ap, n_gt=dict.fromkeys(ap, 9))
    with pytest.raises(AggregationError, match="different ground truth"):
        load_results(root, [RUN, other])


def test_classes_with_gt_must_agree_between_runs(root: Path):
    other = "m2f_s640_seed42"
    _write_set(root, other, "valid", {"bed": 0.4, "chair": 0.2, "wall": 0.6})
    _write_set(root, other, "test", {"bed": 0.2, "wall": 0.1})  # same count, other classes
    _write_set(root, other, "phone", {"bed": 0.1, "chair": 0.1})
    results = load_results(root, [RUN, other])
    with pytest.raises(AggregationError, match="differ between runs"):
        common_classes(results, ("valid", "test", "phone"))


def test_write_tables_marks_capped_runs(root: Path, tmp_path: Path):
    status = {RUN: {"epochs_done": 10, "epochs_planned": 50, "cap_hit": True}}
    out = write_tables(build_tables(root, status, [RUN]), tmp_path / "out")
    assert {p.name for p in out.iterdir()} == {
        "main_results.csv", "domain_shift.csv", "conditions.csv", "category_groups.csv",
        "per_class_ap.csv", "summary.md",
    }  # fmt: skip
    text = (out / "summary.md").read_text(encoding="utf-8")
    assert "YOLO11s-seg 640*" in text
    assert "-20.0 (-67%)" in text  # common classes, valid -> phone: 0.3 -> 0.1
    main = pd.read_csv(out / "main_results.csv")
    assert main["cap_hit"].all() and (main["epochs_done"] == 10).all()
    groups = pd.read_csv(out / "category_groups.csv")
    assert groups.loc[groups["group"] == "absent_in_train", "AP"].isna().all()


def test_load_tables_round_trip_and_stale_check(root: Path, tmp_path: Path):
    status = {RUN: {"epochs_done": 10, "epochs_planned": 50, "cap_hit": True}}
    out = write_tables(build_tables(root, status, [RUN]), tmp_path / "out")
    tables = load_tables(out, root, [RUN])
    assert tables["common_classes"] == ["bed", "chair"]
    assert tables["main_results"]["cap_hit"].all()
    valid = tables["main_results"].set_index("eval_set").loc["valid"]
    assert valid["AP_common"] == pytest.approx(0.3, abs=1e-12) and valid["epochs_done"] == 10
    _write_set(root, RUN, "test", {"bed": 0.25, "chair": 0.1})  # scored again
    with pytest.raises(AggregationError, match="does not match"):
        load_tables(out, root, [RUN])
