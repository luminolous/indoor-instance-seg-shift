"""Extract EXIF from the phone originals into ``results/phone_exif.csv``.

EXIF is read from ``originals/`` because the Roboflow export strips it. Only camera and capture
fields are written. GPS coordinates are never written, only the ``has_gps`` flag.

Usage::

    python -m iiss.data.phone_exif            # -> results/phone_exif.csv
    python -m iiss.data.phone_exif --dry-run  # print the summary only
"""

from __future__ import annotations

import argparse
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import ExifTags, Image

from iiss.data.phone import index_originals
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

EXIF_COLUMNS = (
    "filename",
    "width",
    "height",
    "file_bytes",
    "has_exif",
    "make",
    "model",
    "software",
    "datetime_original",
    "exposure_time_s",
    "f_number",
    "iso",
    "focal_length_mm",
    "flash_fired",
    "orientation",
    "has_gps",
)
_NAME_TIME = re.compile(r"(\d{8})_(\d{6})")
_EXIF_TIME = "%Y:%m:%d %H:%M:%S"


def _text(value: Any) -> str | None:
    """EXIF string without NUL padding, or None when missing or empty."""
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    return str(value).replace("\x00", "").strip() or None


def _number(value: Any) -> float | None:
    """EXIF number as float, or None when missing or invalid (e.g. a 0/0 rational)."""
    try:
        number = float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return round(number, 6) if math.isfinite(number) else None


def read_exif(path: str | Path) -> dict[str, Any]:
    """Selected EXIF fields of one photo. Missing fields are None."""
    path = Path(path)
    with Image.open(path) as im:
        width, height = im.size
        exif = im.getexif()
        sub = exif.get_ifd(ExifTags.IFD.Exif)
        has_gps = bool(exif.get_ifd(ExifTags.IFD.GPSInfo))
    base = ExifTags.Base
    flash = sub.get(base.Flash)
    return {
        "filename": path.name,
        "width": width,
        "height": height,
        "file_bytes": path.stat().st_size,
        "has_exif": bool(exif),
        "make": _text(exif.get(base.Make)),
        "model": _text(exif.get(base.Model)),
        "software": _text(exif.get(base.Software)),
        "datetime_original": _text(sub.get(base.DateTimeOriginal)),
        "exposure_time_s": _number(sub.get(base.ExposureTime)),
        "f_number": _number(sub.get(base.FNumber)),
        "iso": _number(sub.get(base.ISOSpeedRatings)),
        "focal_length_mm": _number(sub.get(base.FocalLength)),
        "flash_fired": None if flash is None else bool(int(flash) & 1),
        "orientation": exif.get(base.Orientation),
        "has_gps": has_gps,
    }


def exif_table(originals_dir: str | Path) -> pd.DataFrame:
    """One row per original photo, sorted by file name."""
    paths = sorted(index_originals(originals_dir).values(), key=lambda p: p.name)
    return pd.DataFrame([read_exif(p) for p in paths], columns=list(EXIF_COLUMNS))


def name_time_offset_s(filename: str, datetime_original: str | None) -> float | None:
    """Seconds from the timestamp in the file name to the EXIF capture time (None if unknown)."""
    match = _NAME_TIME.search(filename)
    if match is None or not datetime_original:
        return None
    try:
        named = datetime.strptime("".join(match.groups()), "%Y%m%d%H%M%S")
        taken = datetime.strptime(datetime_original, _EXIF_TIME)
    except ValueError:
        return None
    return (taken - named).total_seconds()


def exif_summary(table: pd.DataFrame) -> pd.Series:
    """Short summary of an EXIF table for the notebook and the log."""
    times = table["datetime_original"].dropna()
    iso = table["iso"].dropna()
    sizes = {f"{w}x{h}" for w, h in zip(table["width"], table["height"], strict=True)}
    return pd.Series(
        {
            "photos": len(table),
            "with EXIF": int(table["has_exif"].sum()),
            "with capture time": len(times),
            "unique capture times": times.nunique(),
            "first capture": times.min() if len(times) else None,
            "last capture": times.max() if len(times) else None,
            "cameras": ", ".join(sorted(set(table["model"].dropna()))) or None,
            "image sizes": ", ".join(sorted(sizes)),
            "ISO range": f"{iso.min():g}-{iso.max():g}" if len(iso) else None,
            "flash fired": sum(v is True for v in table["flash_fired"].tolist()),
            "with GPS": int(table["has_gps"].sum()),
        }
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract EXIF of the phone originals")
    parser.add_argument("--dry-run", action="store_true", help="print the summary only")
    args = parser.parse_args(argv)
    setup_logging()
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    root = resolve_path(base["paths"]["phone_root"])
    table = exif_table(root / base["phone_import"]["originals_dir"])
    print(exif_summary(table).to_string())  # CLI output
    if not args.dry_run:
        out = resolve_path(base["paths"]["results_root"]) / "phone_exif.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out, index=False)
        logger.info("%d rows -> %s", len(table), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
