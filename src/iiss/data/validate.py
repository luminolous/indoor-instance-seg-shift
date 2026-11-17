"""Validate the COCO segmentation export and write a Markdown report.

Checks per split: unique ids, annotation -> image / category references, polygon structure and
bounds, bbox and area sanity, image files on disk, actual image size vs JSON, EXIF orientation.
Across splits: identical, contiguous category lists, and no annotations on category 0.

Images listed in ``data.rotation_fixes`` (base config) are checked after the rotation that
:func:`iiss.data.images.load_image` applies; they are reported as warnings, not errors.

The dataset is read-only; this module only reports. Known quirks of v5 (near-duplicate classes,
label noise) are kept by policy and are not flagged.

Usage::

    python -m iiss.data.validate                      # all splits -> results/data_validation.md
    python -m iiss.data.validate --splits train --no-files
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from iiss.data.coco import DEFAULT_ANNOTATION_FILE, CocoSplit, load_split
from iiss.data.images import validate_rotation_fixes
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

ERROR = "error"
WARNING = "warning"
MAX_EXAMPLES = 5
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXIF_ORIENTATION_TAG = 274


@dataclass
class Issue:
    """All occurrences of one failed check in one split."""

    level: str
    split: str
    check: str
    count: int = 0
    examples: list[str] = field(default_factory=list)

    def add(self, example: str) -> None:
        self.count += 1
        if len(self.examples) < MAX_EXAMPLES:
            self.examples.append(example)


@dataclass
class SplitReport:
    name: str
    stats: dict[str, Any] = field(default_factory=dict)
    issues: dict[str, Issue] = field(default_factory=dict)

    def flag(self, level: str, check: str, example: str) -> None:
        if check not in self.issues:
            self.issues[check] = Issue(level=level, split=self.name, check=check)
        self.issues[check].add(example)


@dataclass
class ValidationReport:
    dataset_root: Path
    splits: list[SplitReport] = field(default_factory=list)
    global_issues: list[Issue] = field(default_factory=list)
    category_table: list[tuple[int, str]] = field(default_factory=list)

    @property
    def all_issues(self) -> list[Issue]:
        issues = list(self.global_issues)
        for split in self.splits:
            issues.extend(split.issues.values())
        return issues

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.all_issues if i.level == ERROR)

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.all_issues if i.level == WARNING)


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------


def polygon_area(coords: list[float]) -> float:
    """Shoelace area of a flat ``[x1, y1, x2, y2, ...]`` polygon."""
    pts = np.asarray(coords, dtype=float).reshape(-1, 2)
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


# ---------------------------------------------------------------------------------------------
# Per-split checks
# ---------------------------------------------------------------------------------------------


def _check_ids(split: CocoSplit, rep: SplitReport) -> None:
    for kind, items in (("image", split.images), ("annotation", split.annotations)):
        counts = Counter(item.get("id") for item in items)
        for item_id, n in counts.items():
            if n > 1:
                rep.flag(ERROR, f"duplicate {kind} id", f"id={item_id} x{n}")


def _check_annotation(
    ann: dict[str, Any],
    split: CocoSplit,
    rep: SplitReport,
    tol: float,
    counters: Counter,
) -> None:
    ann_id = ann.get("id")
    img = split.images_by_id.get(ann.get("image_id"))
    if img is None:
        rep.flag(ERROR, "annotation -> unknown image_id", f"ann {ann_id}: {ann.get('image_id')}")
        return
    where = f"ann {ann_id} ({img['file_name']})"

    if ann.get("category_id") not in split.category_names:
        rep.flag(ERROR, "unknown category_id", f"{where}: {ann.get('category_id')}")
    if ann.get("category_id") == 0:
        rep.flag(ERROR, "annotation on category 0 (supercategory)", where)
    if ann.get("iscrowd", 0) != 0:
        counters["iscrowd_nonzero"] += 1
        rep.flag(WARNING, "iscrowd != 0", where)

    w_img, h_img = img["width"], img["height"]

    bbox = ann.get("bbox")
    if not (isinstance(bbox, list) and len(bbox) == 4 and all(_is_number(v) for v in bbox)):
        rep.flag(ERROR, "malformed bbox", f"{where}: {bbox}")
    else:
        x, y, w, h = bbox
        if w <= 0 or h <= 0:
            rep.flag(ERROR, "bbox with non-positive size", f"{where}: {bbox}")
        elif x < -tol or y < -tol or x + w > w_img + tol or y + h > h_img + tol:
            rep.flag(WARNING, "bbox outside image", f"{where}: {bbox} vs {w_img}x{h_img}")

    area = ann.get("area")
    if not _is_number(area) or area <= 0:
        rep.flag(ERROR, "non-positive or missing area", f"{where}: {area}")

    seg = ann.get("segmentation")
    if isinstance(seg, dict):
        counters["rle"] += 1
        rep.flag(WARNING, "RLE segmentation (expected polygons)", where)
        return
    if not isinstance(seg, list) or len(seg) == 0:
        rep.flag(ERROR, "empty or malformed segmentation", where)
        return

    counters["polygons"] += len(seg)
    if len(seg) > 1:
        counters["multi_polygon_annotations"] += 1
    total_area = 0.0
    for poly in seg:
        if not isinstance(poly, list) or not all(_is_number(v) for v in poly):
            rep.flag(ERROR, "polygon with non-numeric values", where)
            continue
        if len(poly) % 2 != 0:
            rep.flag(ERROR, "polygon with odd coordinate count", f"{where}: len={len(poly)}")
            continue
        if len(poly) < 6:
            rep.flag(ERROR, "polygon with < 3 points", f"{where}: len={len(poly)}")
            continue
        xs, ys = poly[0::2], poly[1::2]
        if min(xs) < -tol or min(ys) < -tol or max(xs) > w_img + tol or max(ys) > h_img + tol:
            rep.flag(
                WARNING,
                "polygon outside image",
                f"{where}: x[{min(xs):.1f},{max(xs):.1f}] y[{min(ys):.1f},{max(ys):.1f}]"
                f" vs {w_img}x{h_img}",
            )
        poly_area = polygon_area(poly)
        total_area += poly_area
        if poly_area < 1.0:
            rep.flag(WARNING, "degenerate polygon (area < 1 px)", f"{where}: {poly_area:.2f}")
    if total_area < 1.0:
        counters["degenerate_annotations"] += 1


def _check_files(
    split: CocoSplit, rep: SplitReport, counters: Counter, fixes: Mapping[str, int]
) -> None:
    listed = set()
    for img in split.images:
        path = split.image_path(img)
        listed.add(img["file_name"])
        if not path.is_file():
            rep.flag(ERROR, "image file missing", img["file_name"])
            continue
        try:
            with Image.open(path) as im:
                actual = im.size
                orientation = im.getexif().get(EXIF_ORIENTATION_TAG, 1)
        except OSError as exc:
            rep.flag(ERROR, "image file unreadable", f"{img['file_name']}: {exc}")
            continue
        fix = fixes.get(img["file_name"])
        if fix is not None:
            counters["rotation_fixed"] += 1
            rep.flag(
                WARNING,
                "rotation fix applied on load (data.rotation_fixes)",
                f"{img['file_name']}: file {actual[0]}x{actual[1]}, rotate {fix} deg CCW",
            )
            if fix in (90, 270):
                actual = (actual[1], actual[0])
        if actual != (img["width"], img["height"]):
            rep.flag(
                ERROR,
                "image size differs from JSON",
                f"{img['file_name']}: file {actual[0]}x{actual[1]}"
                f"{' (after rotation fix)' if fix is not None else ''}"
                f" vs json {img['width']}x{img['height']}",
            )
        if orientation not in (1, None):
            counters["exif_rotated"] += 1
            rep.flag(WARNING, "EXIF orientation != 1", f"{img['file_name']}: {orientation}")

    on_disk = {p.name for p in split.image_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    for name in sorted(on_disk - listed):
        rep.flag(WARNING, "image file not referenced in JSON", name)
    counters["image_files_on_disk"] = len(on_disk)


def validate_split(
    split: CocoSplit,
    check_files: bool = True,
    tol: float = 1.0,
    fixes: Mapping[str, int] | None = None,
) -> SplitReport:
    """Run all per-split checks. ``fixes`` is the ``file_name -> degrees_ccw`` table."""
    rep = SplitReport(name=split.name)
    counters: Counter = Counter()

    _check_ids(split, rep)
    for ann in split.annotations:
        _check_annotation(ann, split, rep, tol, counters)

    without_anns = [img for img in split.images if img["id"] not in split.annotations_by_image]
    for img in without_anns:
        rep.flag(WARNING, "image without annotations", img["file_name"])

    if check_files:
        _check_files(split, rep, counters, fixes or {})

    sizes = Counter(f"{img['width']}x{img['height']}" for img in split.images)
    cat_counts = Counter(ann.get("category_id") for ann in split.annotations)
    rep.stats = {
        "images": len(split.images),
        "annotations": len(split.annotations),
        "categories": len(split.categories),
        "categories_with_instances": len([c for c in split.category_names if cat_counts[c] > 0]),
        "images_without_annotations": len(without_anns),
        "max_annotations_per_image": max(
            (len(v) for v in split.annotations_by_image.values()), default=0
        ),
        "polygons": counters["polygons"],
        "multi_polygon_annotations": counters["multi_polygon_annotations"],
        "rle_annotations": counters["rle"],
        "iscrowd_nonzero": counters["iscrowd_nonzero"],
        "image_files_on_disk": counters["image_files_on_disk"] if check_files else None,
        "exif_rotated_images": counters["exif_rotated"] if check_files else None,
        "rotation_fixed_images": counters["rotation_fixed"] if check_files else None,
        "distinct_image_sizes": len(sizes),
        "top_image_sizes": sizes.most_common(5),
        "zero_instance_categories": [
            split.category_names[c] for c in sorted(split.category_names) if cat_counts[c] == 0
        ],
    }
    return rep


# ---------------------------------------------------------------------------------------------
# Cross-split checks
# ---------------------------------------------------------------------------------------------


def _check_categories(splits: list[CocoSplit], report: ValidationReport) -> None:
    reference = [(c["id"], c["name"]) for c in sorted(splits[0].categories, key=lambda c: c["id"])]
    report.category_table = reference

    ids = [cid for cid, _ in reference]
    if ids != list(range(len(ids))):
        issue = Issue(ERROR, "all", "category ids not contiguous from 0")
        issue.add(f"ids: {ids}")
        report.global_issues.append(issue)

    for split in splits[1:]:
        other = [(c["id"], c["name"]) for c in sorted(split.categories, key=lambda c: c["id"])]
        if other != reference:
            issue = Issue(ERROR, split.name, f"categories differ from {splits[0].name}")
            for a, b in zip(reference, other, strict=False):
                if a != b:
                    issue.add(f"{splits[0].name} {a} vs {split.name} {b}")
            if len(reference) != len(other):
                issue.add(f"count {len(reference)} vs {len(other)}")
            report.global_issues.append(issue)


def validate_dataset(
    dataset_root: str | Path,
    splits: list[str],
    annotation_file: str = DEFAULT_ANNOTATION_FILE,
    check_files: bool = True,
    tol: float = 1.0,
    fixes: Mapping[str, int] | None = None,
) -> ValidationReport:
    """Validate every split and the category consistency across them.

    ``fixes`` is the ``file_name -> degrees_ccw`` rotation table (``data.rotation_fixes``).
    """
    fixes = dict(fixes or {})
    validate_rotation_fixes(fixes)
    root = resolve_path(dataset_root)
    report = ValidationReport(dataset_root=root)
    loaded = []
    for name in splits:
        split = load_split(root, name, annotation_file)
        loaded.append(split)
        logger.info("Validating %s (%d images)", name, len(split.images))
        report.splits.append(validate_split(split, check_files=check_files, tol=tol, fixes=fixes))
    if loaded:
        _check_categories(loaded, report)
        known = {img["file_name"] for split in loaded for img in split.images}
        unused = sorted(set(fixes) - known)
        if unused:
            issue = Issue(WARNING, "all", "rotation fix for a file not in the validated splits")
            for name in unused:
                issue.add(name)
            report.global_issues.append(issue)
    return report


# ---------------------------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------------------------

_STAT_ROWS = [
    ("images", "Images"),
    ("annotations", "Annotations"),
    ("categories", "Categories in JSON"),
    ("categories_with_instances", "Categories with >= 1 instance"),
    ("images_without_annotations", "Images without annotations"),
    ("max_annotations_per_image", "Max annotations per image"),
    ("polygons", "Polygons"),
    ("multi_polygon_annotations", "Annotations with > 1 polygon"),
    ("rle_annotations", "RLE annotations"),
    ("iscrowd_nonzero", "iscrowd != 0"),
    ("image_files_on_disk", "Image files on disk"),
    ("exif_rotated_images", "Images with EXIF orientation != 1"),
    ("rotation_fixed_images", "Images rotated on load (data.rotation_fixes)"),
    ("distinct_image_sizes", "Distinct image sizes"),
]


def _rel(path: Path) -> str:
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def render_markdown(report: ValidationReport, command: str) -> str:
    names = [s.name for s in report.splits]
    lines = [
        "# Data validation report",
        "",
        f"- Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Dataset root: `{_rel(report.dataset_root)}`",
        f"- Command: `{command}`",
        f"- Result: **{'PASS' if report.n_errors == 0 else 'FAIL'}**"
        f" ({report.n_errors} error checks, {report.n_warnings} warning checks)",
        "",
        "Generated by `iiss.data.validate`. Do not edit by hand.",
        "",
        "## Summary",
        "",
        "| Metric | " + " | ".join(names) + " |",
        "|---|" + "---|" * len(names),
    ]
    for key, label in _STAT_ROWS:
        values = [s.stats.get(key) for s in report.splits]
        cells = ["n/a" if v is None else str(v) for v in values]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## Image sizes (top 5 per split, width x height)", ""]
    for s in report.splits:
        sizes = ", ".join(f"{size} ({n})" for size, n in s.stats["top_image_sizes"])
        lines.append(f"- **{s.name}**: {sizes}")

    lines += ["", "## Categories with zero instances", ""]
    for s in report.splits:
        zero = s.stats["zero_instance_categories"]
        text = ", ".join(f"`{n}`" for n in zero) if zero else "none"
        lines.append(f"- **{s.name}** ({len(zero)}): {text}")

    lines += ["", "## Issues", ""]
    issues = sorted(report.all_issues, key=lambda i: (i.level != ERROR, i.split, i.check))
    if not issues:
        lines.append("No issues found.")
    else:
        lines += ["| Level | Split | Check | Count | Examples |", "|---|---|---|---|---|"]
        for i in issues:
            examples = "<br>".join(e.replace("|", "\\|") for e in i.examples)
            lines.append(f"| {i.level} | {i.split} | {i.check} | {i.count} | {examples} |")

    lines += [
        "",
        f"## Category list ({len(report.category_table)} entries, from `{names[0]}`)",
        "",
        "| id | name |",
        "|---|---|",
    ]
    lines += [f"| {cid} | {name} |" for cid, name in report.category_table]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    default_root = base["paths"]["dataset_root"]
    default_splits = list(base["data"]["splits"].values())
    default_out = Path(base["paths"]["results_root"]) / "data_validation.md"

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", default=default_root)
    parser.add_argument("--splits", nargs="+", default=default_splits)
    parser.add_argument("--annotation-file", default=base["data"]["annotation_file"])
    parser.add_argument("--out", default=str(default_out), help="Markdown report path")
    parser.add_argument("--no-files", action="store_true", help="skip image file checks")
    parser.add_argument("--tol", type=float, default=1.0, help="bounds tolerance in pixels")
    args = parser.parse_args(argv)

    setup_logging()
    report = validate_dataset(
        args.dataset_root,
        args.splits,
        annotation_file=args.annotation_file,
        check_files=not args.no_files,
        tol=args.tol,
        fixes=base["data"].get("rotation_fixes") or {},
    )
    command = "python -m iiss.data.validate " + " ".join(argv if argv is not None else sys.argv[1:])
    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_markdown(report, command.strip()), encoding="utf-8")

    for issue in report.all_issues:
        log = logger.error if issue.level == ERROR else logger.warning
        log("[%s] %s: %d", issue.split, issue.check, issue.count)
    logger.info(
        "Wrote %s (%d error checks, %d warning checks)", out, report.n_errors, report.n_warnings
    )
    return 1 if report.n_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
