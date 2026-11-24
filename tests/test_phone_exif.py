"""Tests for the phone EXIF extractor and the metadata validator (synthetic photos)."""

import re
from pathlib import Path

import pandas as pd
import pytest
from PIL import ExifTags, Image

from iiss.data.phone_exif import (
    EXIF_COLUMNS,
    exif_summary,
    exif_table,
    name_time_offset_s,
    read_exif,
)
from iiss.data.phone_metadata import (
    condition_exposure,
    condition_table,
    render_report,
    validate_metadata,
)

RULES = {"conditions": ["normal", "low_light", "backlight"], "min_photos": 2, "min_hard": 1}
BASE = ExifTags.Base


def _photo(path: Path, exif: bool = True, gps: bool = False, iso: int = 800) -> None:
    ex = Image.Exif()
    if exif:
        ex[BASE.Make] = "Maker"
        ex[BASE.Model] = "Phone\x00\x00"
        ex[BASE.Orientation] = 1
        ex.get_ifd(ExifTags.IFD.Exif).update(
            {
                BASE.DateTimeOriginal: "2026:11:24 17:19:20",
                BASE.ExposureTime: 0.05,
                BASE.FNumber: 1.79,
                BASE.ISOSpeedRatings: iso,
                BASE.Flash: 17,
            }
        )
    if gps:
        ex.get_ifd(ExifTags.IFD.GPSInfo).update({ExifTags.GPS.GPSLatitudeRef: "S"})
    Image.new("RGB", (8, 6)).save(path, exif=ex)


@pytest.fixture
def originals(tmp_path: Path) -> Path:
    root = tmp_path / "originals"
    root.mkdir()
    _photo(root / "IMG_20261124_171919.jpg")
    _photo(root / "IMG_20261124_171920.jpg", iso=3200)
    return root


def _meta(**changes: list[str]) -> pd.DataFrame:
    data = {
        "filename": ["IMG_20261124_171919.jpg", "IMG_20261124_171920.jpg"],
        "location": ["bedroom"] * 2,
        "condition": ["normal", "low_light"],
        "scene": ["bed", "bed"],
    }
    return pd.DataFrame({**data, **changes})


def test_read_exif(originals: Path):
    row = read_exif(originals / "IMG_20261124_171919.jpg")
    assert set(row) == set(EXIF_COLUMNS)
    assert (row["make"], row["model"]) == ("Maker", "Phone")  # NUL padding removed
    assert (row["width"], row["height"], row["iso"]) == (8, 6, 800)
    assert row["exposure_time_s"] == pytest.approx(0.05)
    assert row["f_number"] == pytest.approx(1.79)
    assert row["flash_fired"] is True and row["has_exif"] and not row["has_gps"]


def test_photo_without_exif_and_gps_flag_only(tmp_path: Path):
    _photo(tmp_path / "a.jpg", exif=False)
    _photo(tmp_path / "b.jpg", gps=True)
    a, b = read_exif(tmp_path / "a.jpg"), read_exif(tmp_path / "b.jpg")
    assert not a["has_exif"] and a["model"] is None and a["flash_fired"] is None
    assert b["has_gps"] is True
    assert [c for c in EXIF_COLUMNS if "gps" in c] == ["has_gps"]


def test_exif_table_and_summary(originals: Path):
    table = exif_table(originals)
    assert list(table.columns) == list(EXIF_COLUMNS) and len(table) == 2
    s = exif_summary(table)
    assert (s["photos"], s["with EXIF"], s["with GPS"], s["flash fired"]) == (2, 2, 0, 2)
    assert s["unique capture times"] == 1 and s["ISO range"] == "800-3200"


def test_name_time_offset():
    assert name_time_offset_s("IMG_20261124_171919.jpg", "2026:11:24 17:19:20") == 1
    assert name_time_offset_s("photo.jpg", "2026:11:24 17:19:20") is None
    assert name_time_offset_s("IMG_20261124_171919.jpg", None) is None
    assert name_time_offset_s("IMG_20261124_171919.jpg", "garbage") is None


def test_valid_metadata_passes(originals: Path):
    rep = validate_metadata(_meta(), exif_table(originals), RULES)
    assert rep.ok and (rep.n_photos, rep.n_hard) == (2, 1)
    assert rep.by_condition == {"normal": 1, "low_light": 1, "backlight": 0}
    assert rep.warnings == ["hard conditions without photos: ['backlight']"]
    assert condition_table(rep)["photos"].tolist() == [1, 1, 0]
    assert "**PASS**" in render_report(rep, RULES)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"condition": ["normal", "dark"]}, "unknown conditions"),
        ({"condition": ["normal", "normal"]}, "0 hard photos"),
        ({"scene": ["bed", ""]}, r"empty scene in CSV rows \[3\]"),
        ({"location": ["bedroom", "bedroom "]}, "spaces around location"),
        ({"filename": ["IMG_20261124_171919.jpg"] * 2}, "duplicate filenames"),
        ({"filename": ["IMG_20261124_171919.jpg", "x.jpg"]}, "rows without a photo"),
        ({"filename": ["IMG_20261124_171919.jpg", "x.jpg"]}, "photos without a metadata row"),
        ({"filename": ["IMG_20261124_171919.jpg", "x.jpg"]}, "1 photos, at least 2"),
    ],
)
def test_invalid_metadata_fails(originals: Path, changes: dict, message: str):
    rep = validate_metadata(_meta(**changes), exif_table(originals), RULES)
    assert not rep.ok and any(re.search(message, e) for e in rep.errors), rep.errors
    assert "**FAIL**" in render_report(rep, RULES)


def test_missing_column_fails(originals: Path):
    rep = validate_metadata(_meta().drop(columns="scene"), exif_table(originals), RULES)
    assert rep.errors == ["missing columns: ['scene']"]


def test_exif_problems_are_reported(tmp_path: Path):
    _photo(tmp_path / "IMG_20261124_171919.jpg", exif=False)
    _photo(tmp_path / "IMG_20261124_181920.jpg", gps=True)
    meta = _meta(filename=["IMG_20261124_171919.jpg", "IMG_20261124_181920.jpg"])
    rep = validate_metadata(meta, exif_table(tmp_path), RULES)
    assert any("without camera model" in e for e in rep.errors)
    assert any("-3600 s" in w for w in rep.warnings)
    assert any("GPS" in w for w in rep.warnings)


def test_condition_exposure(originals: Path):
    t = condition_exposure(_meta(), exif_table(originals)).set_index("condition")
    assert t.loc["low_light", "iso"] == "3200 (3200-3200)"
    assert t.loc["normal", "flash_fired"] == 1
