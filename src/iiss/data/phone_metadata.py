"""Validate ``metadata.csv`` of the phone test set against the originals and the assignment.

Errors: missing columns, empty or padded cells, duplicate names, unknown condition, photo without
a row or row without a photo, fewer photos or hard photos than required, photo without EXIF
camera model or capture time.
Warnings: hard conditions that are not represented, file-name time far from the EXIF time,
photos with GPS data.

Limits and allowed conditions come from ``phone_import`` in ``configs/base.yaml``. The validator
only reads; ``metadata.csv`` is the user's file.

Usage::

    python -m iiss.data.phone_metadata            # -> results/phone_metadata_report.md
    python -m iiss.data.phone_metadata --dry-run  # print the report only
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from iiss.data.phone_exif import exif_table, name_time_offset_s
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

COLUMNS = ("filename", "location", "condition", "scene")
NORMAL = "normal"
MAX_NAME_TIME_OFFSET_S = 5.0


@dataclass
class MetadataReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n_photos: int = 0
    n_hard: int = 0
    by_condition: dict[str, int] = field(default_factory=dict)
    by_scene: dict[str, int] = field(default_factory=dict)
    by_location: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def read_metadata(path: str | Path) -> pd.DataFrame:
    """``metadata.csv`` as strings; spaces around values are kept so they can be reported."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def validate_metadata(
    table: pd.DataFrame, exif: pd.DataFrame, rules: Mapping[str, Any]
) -> MetadataReport:
    """Check the metadata table against the EXIF table of the originals and ``rules``."""
    rep = MetadataReport()
    missing = [c for c in COLUMNS if c not in table.columns]
    if missing:
        rep.errors.append(f"missing columns: {missing}")
        return rep

    for col in COLUMNS:
        values = table[col]
        empty = table.index[values.str.strip() == ""]
        if len(empty):
            rep.errors.append(f"empty {col} in CSV rows {[int(i) + 2 for i in empty]}")
        padded = values[(values != values.str.strip()) & (values.str.strip() != "")]
        if len(padded):
            rep.errors.append(f"spaces around {col}: {sorted(padded)}")
    duplicates = sorted(set(table.loc[table["filename"].duplicated(), "filename"]))
    if duplicates:
        rep.errors.append(f"duplicate filenames: {duplicates}")

    named, photos = set(table["filename"]), set(exif["filename"])
    if photos - named:
        rep.errors.append(f"photos without a metadata row: {sorted(photos - named)}")
    if named - photos:
        rep.errors.append(f"metadata rows without a photo: {sorted(named - photos)}")

    allowed = list(rules["conditions"])
    unknown = sorted(set(table["condition"]) - set(allowed))
    if unknown:
        rep.errors.append(f"unknown conditions: {unknown} (allowed: {allowed})")

    # Counts use rows that belong to a real photo, so a bad row cannot fulfil a requirement.
    valid = table[table["filename"].isin(photos) & table["condition"].isin(allowed)]
    valid = valid.drop_duplicates("filename")
    rep.n_photos = len(valid)
    rep.n_hard = int((valid["condition"] != NORMAL).sum())
    rep.by_condition = {c: int((valid["condition"] == c).sum()) for c in allowed}
    rep.by_scene = _counts(valid["scene"])
    rep.by_location = _counts(valid["location"])
    if rep.n_photos < rules["min_photos"]:
        rep.errors.append(f"{rep.n_photos} photos, at least {rules['min_photos']} required")
    if rep.n_hard < rules["min_hard"]:
        rep.errors.append(f"{rep.n_hard} hard photos, at least {rules['min_hard']} required")
    absent = [c for c in allowed if c != NORMAL and rep.by_condition[c] == 0]
    if absent:
        rep.warnings.append(f"hard conditions without photos: {absent}")

    no_exif = exif.loc[exif[["model", "datetime_original"]].isna().any(axis=1), "filename"]
    if len(no_exif):
        rep.errors.append(f"photos without camera model or capture time: {sorted(no_exif)}")
    far = []
    for name, taken in zip(exif["filename"], exif["datetime_original"], strict=True):
        offset = name_time_offset_s(name, taken if isinstance(taken, str) else None)
        if offset is not None and abs(offset) > MAX_NAME_TIME_OFFSET_S:
            far.append(f"{name} ({offset:+.0f} s)")
    if far:
        rep.warnings.append(f"file-name time differs from EXIF time: {far}")
    gps = sorted(exif.loc[exif["has_gps"], "filename"])
    if gps:
        rep.warnings.append(f"photos with GPS data (do not publish the originals): {gps}")
    return rep


def _counts(values: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in values.value_counts().sort_index().items()}


def condition_table(rep: MetadataReport) -> pd.DataFrame:
    """Photo counts per condition, with the hard flag."""
    return pd.DataFrame(
        [(c, c != NORMAL, n) for c, n in rep.by_condition.items()],
        columns=["condition", "hard", "photos"],
    )


def condition_exposure(table: pd.DataFrame, exif: pd.DataFrame) -> pd.DataFrame:
    """Median (min-max) EXIF exposure time and ISO, and flash count, per condition."""
    rows = []
    for condition, group in table.merge(exif, on="filename").groupby("condition"):
        row: dict[str, Any] = {"condition": condition, "photos": len(group)}
        for col in ("exposure_time_s", "iso"):
            v = group[col].dropna()
            row[col] = f"{v.median():g} ({v.min():g}-{v.max():g})" if len(v) else "n/a"
        row["flash_fired"] = sum(v is True for v in group["flash_fired"].tolist())
        rows.append(row)
    return pd.DataFrame(rows)


def render_report(rep: MetadataReport, rules: Mapping[str, Any]) -> str:
    """Markdown validation report."""
    lines = [
        "# Phone test set metadata validation",
        "",
        "Generated by `python -m iiss.data.phone_metadata`.",
        "",
        f"- Result: **{'PASS' if rep.ok else 'FAIL'}**"
        f" ({len(rep.errors)} errors, {len(rep.warnings)} warnings)",
        f"- Photos: {rep.n_photos} (required >= {rules['min_photos']})",
        f"- Hard photos: {rep.n_hard} (required >= {rules['min_hard']})",
        "",
    ]
    for title, items in (("Errors", rep.errors), ("Warnings", rep.warnings)):
        if items:
            lines += [f"## {title}", "", *[f"- {i}" for i in items], ""]
    for title, counts in (
        ("Condition", rep.by_condition),
        ("Scene", rep.by_scene),
        ("Location", rep.by_location),
    ):
        lines += [f"## {title}", "", f"| {title} | Photos |", "|---|---|"]
        lines += [f"| {k} | {v} |" for k, v in counts.items()]
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the phone metadata.csv")
    parser.add_argument("--dry-run", action="store_true", help="print the report only")
    args = parser.parse_args(argv)
    setup_logging()
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    rules = base["phone_import"]
    root = resolve_path(base["paths"]["phone_root"])
    rep = validate_metadata(
        read_metadata(root / rules["metadata_file"]),
        exif_table(root / rules["originals_dir"]),
        rules,
    )
    text = render_report(rep, rules)
    print(text)  # CLI output
    if not args.dry_run:
        out = resolve_path(base["paths"]["results_root"]) / "phone_metadata_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        logger.info("report -> %s", out)
    return 0 if rep.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
