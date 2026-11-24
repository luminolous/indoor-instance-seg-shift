"""Tests for the phone-set importer on a tiny synthetic export."""

from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from iiss.data.coco import CocoSplit
from iiss.data.phone import (
    MergedFragments,
    PhoneImportError,
    category_map,
    count_non_large,
    import_phone_set,
    index_originals,
    load_metadata,
    merge_annotations,
    original_stem,
    render_report,
    save_overlays,
)
from iiss.eval.coco_eval import image_groups_from_table, load_gt
from iiss.utils.config import load_yaml
from iiss.utils.paths import repo_root

V5 = [
    {"id": 0, "name": "indoor-objects"},
    {"id": 1, "name": "bed"},
    {"id": 2, "name": "chair"},
    {"id": 3, "name": "pillow"},
]
EXPORT_CATS = [
    {"id": 0, "name": "my-project"},
    {"id": 1, "name": "chair"},
    {"id": 2, "name": "pillow"},
]


def _box(x: float, y: float, w: float, h: float) -> list[float]:
    return [x, y, x + w, y, x + w, y + h, x, y + h]


def _ann(i: int, image: int, cat: int, box: tuple[float, float, float, float]) -> dict:
    return {
        "id": i, "image_id": image, "category_id": cat, "segmentation": [_box(*box)],
        "area": box[2] * box[3], "bbox": list(box), "iscrowd": 0,
    }  # fmt: skip


def _split(name: str, image_names: list[str], anns: list[dict], cats=None) -> CocoSplit:
    images = [
        {"id": i, "file_name": f"{n}_jpg.rf.abc{i}.jpg", "width": 100, "height": 100}
        for i, n in enumerate(image_names)
    ]
    data = {"images": images, "annotations": anns, "categories": cats or EXPORT_CATS}
    return CocoSplit(name, Path("unused"), Path("unused"), data)


@pytest.fixture
def phone(tmp_path: Path) -> dict:
    originals = tmp_path / "manual-benchmark" / "originals"
    originals.mkdir(parents=True)
    for n in ("IMG_A", "IMG_B"):
        Image.new("RGB", (100, 100)).save(originals / f"{n}.jpg")
    meta = tmp_path / "metadata.csv"
    meta.write_text(
        "filename,location,condition,scene\n"
        "IMG_A.jpg,bedroom,normal,chair\nIMG_B.jpg,bedroom,low_light,bed\n"
    )
    # IMG_A: one chair in three pieces (two overlap) and two separate pillows; IMG_B: one chair
    splits = [
        _split("valid", ["IMG_B"], [_ann(1, 0, 1, (0, 0, 20, 20))]),
        _split(
            "test",
            ["IMG_A"],
            [
                _ann(1, 0, 1, (0, 0, 10, 10)),
                _ann(2, 0, 1, (5, 0, 10, 10)),
                _ann(3, 0, 2, (50, 50, 10, 10)),
                _ann(4, 0, 1, (80, 80, 10, 10)),
                _ann(5, 0, 2, (70, 50, 10, 10)),
            ],
        ),
    ]
    out = tmp_path / "derived" / "phone"
    out.mkdir(parents=True)
    return {
        "export_splits": splits,
        "originals": index_originals(originals),
        "metadata": load_metadata(meta),
        "v5_categories": V5,
        "merge_classes": ["chair"],
        "out_dir": out,
    }


def test_import_merges_splits_remaps_and_merges_fragments(phone: dict):
    data, merged = import_phone_set(**phone)
    assert [Path(i["file_name"]).name for i in data["images"]] == ["IMG_A.jpg", "IMG_B.jpg"]
    assert [(i["condition"], i["scene"]) for i in data["images"]] == [
        ("normal", "chair"), ("low_light", "bed"),
    ]  # fmt: skip
    for img in data["images"]:
        assert "\\" not in img["file_name"] and not Path(img["file_name"]).is_absolute()
        assert (phone["out_dir"] / img["file_name"]).is_file()
        assert (img["width"], img["height"]) == (100, 100)
    assert data["categories"] == V5
    assert [a["id"] for a in data["annotations"]] == [1, 2, 3, 4]

    assert merged == [MergedFragments("IMG_A.jpg", "chair", 3)]
    chair = [a for a in data["annotations"] if a["image_id"] == 1 and a["category_id"] == 2]
    assert len(chair) == 1 and len(chair[0]["segmentation"]) == 3
    assert chair[0]["area"] == 250  # union: 15x10 (overlap counted once) + 10x10
    assert chair[0]["bbox"] == [0, 0, 90, 90]
    # pillow is not a merge class; a single chair is copied unchanged
    assert sum(a["category_id"] == 3 for a in data["annotations"]) == 2
    (single,) = [a for a in data["annotations"] if a["image_id"] == 2]
    assert single["category_id"] == 2 and single["bbox"] == [0, 0, 20, 20]


def test_merge_annotations_does_not_change_input():
    anns = [_ann(1, 0, 1, (0, 0, 10, 10)), _ann(2, 0, 1, (20, 0, 10, 10))]
    merge_annotations(anns, 100, 100)
    assert len(anns[0]["segmentation"]) == 1


def test_unknown_category_name_fails():
    with pytest.raises(PhoneImportError, match="rice cooker"):
        category_map([*EXPORT_CATS, {"id": 3, "name": "rice cooker"}], V5)


def test_v5_supercategory_is_not_a_match():
    with pytest.raises(PhoneImportError, match="indoor-objects"):
        category_map([{"id": 4, "name": "indoor-objects"}], V5)


def test_annotation_on_supercategory_fails(phone: dict):
    phone["export_splits"][1].data["annotations"].append(_ann(9, 0, 0, (0, 0, 5, 5)))
    with pytest.raises(PhoneImportError, match="category id 0"):
        import_phone_set(**phone)


def test_unknown_merge_class_fails(phone: dict):
    with pytest.raises(PhoneImportError, match="sofa"):
        import_phone_set(**{**phone, "merge_classes": ["sofa"]})


def test_missing_original_or_metadata_fails(phone: dict):
    originals = {k: v for k, v in phone["originals"].items() if k != "IMG_A"}
    with pytest.raises(PhoneImportError, match="No original"):
        import_phone_set(**{**phone, "originals": originals})
    metadata = {k: v for k, v in phone["metadata"].items() if k != "IMG_A.jpg"}
    with pytest.raises(PhoneImportError, match="not listed"):
        import_phone_set(**{**phone, "metadata": metadata})


def test_photo_that_was_not_exported_fails(phone: dict):
    with pytest.raises(PhoneImportError, match="not exported.*IMG_B"):
        import_phone_set(**{**phone, "export_splits": phone["export_splits"][1:]})


def test_photo_exported_twice_fails(phone: dict):
    splits = [*phone["export_splits"], phone["export_splits"][0]]
    with pytest.raises(PhoneImportError, match="more than once"):
        import_phone_set(**{**phone, "export_splits": splits})


def test_original_stem():
    name = "IMG_20261124_172304_jpg.rf.9ee139aae70ff37b84b371ff40f11081.jpg"
    assert original_stem(name) == "IMG_20261124_172304"
    with pytest.raises(PhoneImportError):
        original_stem("IMG_20261124_172304.jpg")


def test_report_lists_pairs_and_counts(phone: dict):
    data, merged = import_phone_set(**phone)
    text = render_report(data, merged, 6, ["chair"])
    assert "6 exported -> 4 after merging" in text
    assert "| chair | IMG_A.jpg | 3 |" in text
    assert "| chair | 4 | 2 |" in text
    assert count_non_large(data) == 4  # fixture objects are all below 96x96


def test_evaluator_finds_condition_for_relative_file_names(phone: dict):
    data, _ = import_phone_set(**phone)
    table = pd.DataFrame(list(phone["metadata"].values()))
    assert image_groups_from_table(load_gt(data), table) == {"normal": [1], "low_light": [2]}


def test_overlay_is_written_and_ambiguous_key_fails(phone: dict, tmp_path: Path):
    data, _ = import_phone_set(**phone)
    gt_path = phone["out_dir"] / "annotations.coco.json"
    save_overlays(gt_path, data, ["IMG_A"], tmp_path / "ov")
    assert (tmp_path / "ov" / "IMG_A.jpg").is_file()
    with pytest.raises(PhoneImportError, match="matches 2"):
        save_overlays(gt_path, data, ["IMG"], tmp_path / "ov")


def test_repo_merge_list():
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    assert base["phone_import"]["merge_fragments"] == [
        "chair", "door-stuff", "floor-other-merged",
    ]  # fmt: skip
