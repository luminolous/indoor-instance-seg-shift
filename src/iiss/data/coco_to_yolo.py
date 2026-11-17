"""Convert the COCO segmentation export to YOLO-seg format.

This is the only transformation applied to the training data. It keeps every class, split and
annotation: **YOLO class index = COCO ``category_id``** (75 names, index 0 = ``indoor-objects``,
which has no annotations).

Output layout (``dataset/derived/yolo/`` by default)::

    data.yaml                  # no 'path' key: Ultralytics resolves paths from the YAML's folder
    manifest.json              # source files, counts, rotation fixes, clipping stats
    images/<split>/<file>      # copies of the source images (rotation-fixed ones re-saved)
    labels/<split>/<stem>.txt  # one line per instance: cls x1 y1 x2 y2 ... (normalised)

Polygon coordinates are normalised by the image width/height recorded in the COCO JSON and
clipped to [0, 1]. Images listed in ``data.rotation_fixes`` are written already rotated, so
their pixels match the labels (the source files are untouched).

Usage::

    python -m iiss.data.coco_to_yolo            # all splits -> dataset/derived/yolo
    python -m iiss.data.coco_to_yolo --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from iiss.data.coco import DEFAULT_ANNOTATION_FILE, CocoSplit, load_split
from iiss.data.images import default_rotation_fixes, load_image, validate_rotation_fixes
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

DECIMALS = 6
JPEG_QUALITY = 95


class ConversionError(ValueError):
    """Raised when an annotation cannot be converted without changing its meaning."""


def polygon_to_yolo(poly: list[float], width: int, height: int) -> tuple[list[float], int]:
    """Normalise a flat COCO polygon to [0, 1]. Returns the coords and how many were clipped."""
    out: list[float] = []
    clipped = 0
    for i, v in enumerate(poly):
        n = v / (width if i % 2 == 0 else height)
        if n < 0.0 or n > 1.0:
            clipped += 1
            n = min(max(n, 0.0), 1.0)
        out.append(round(n, DECIMALS))
    return out, clipped


def yolo_to_polygon(coords: list[float], width: int, height: int) -> list[float]:
    """Inverse of :func:`polygon_to_yolo` (without the clipping)."""
    return [v * (width if i % 2 == 0 else height) for i, v in enumerate(coords)]


def annotation_to_line(ann: Mapping[str, Any], width: int, height: int) -> tuple[str, int]:
    """Format one COCO annotation as a YOLO-seg label line."""
    seg = ann.get("segmentation")
    if not isinstance(seg, list) or len(seg) == 0:
        raise ConversionError(f"annotation {ann.get('id')}: segmentation is not a polygon list")
    if len(seg) > 1:
        # v5 has none (see results/data_validation.md). Merging parts would change the mask
        # topology, so refuse instead of guessing.
        raise ConversionError(
            f"annotation {ann.get('id')}: {len(seg)} polygons; multi-part masks are not supported"
        )
    coords, clipped = polygon_to_yolo(seg[0], width, height)
    if len(coords) < 6:
        raise ConversionError(f"annotation {ann.get('id')}: polygon has < 3 points")
    cls = int(ann["category_id"])
    return f"{cls} " + " ".join(f"{v:.{DECIMALS}f}" for v in coords), clipped


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def convert_split(
    split: CocoSplit,
    out_root: Path,
    fixes: Mapping[str, int],
) -> dict[str, Any]:
    """Write images and labels for one split. Returns per-split statistics."""
    img_dir = out_root / "images" / split.name
    lbl_dir = out_root / "labels" / split.name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "images": 0,
        "instances": 0,
        "empty_label_files": 0,
        "clipped_coords": 0,
        "rotation_fixed": [],
    }
    stems: set[str] = set()
    for img in split.images:
        src = split.image_path(img)
        stem = Path(img["file_name"]).stem
        if stem in stems:
            raise ConversionError(f"{split.name}: duplicate file stem {stem!r}")
        stems.add(stem)

        dst = img_dir / img["file_name"]
        if img["file_name"] in fixes:
            im = load_image(src, img, fixes)  # rotated and size-checked against the JSON
            im.save(dst, quality=JPEG_QUALITY)
            stats["rotation_fixed"].append(img["file_name"])
        else:
            shutil.copy2(src, dst)

        lines = []
        for ann in split.annotations_by_image.get(img["id"], []):
            line, clipped = annotation_to_line(ann, img["width"], img["height"])
            lines.append(line)
            stats["clipped_coords"] += clipped
        (lbl_dir / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        stats["images"] += 1
        stats["instances"] += len(lines)
        stats["empty_label_files"] += int(not lines)

    logger.info(
        "%s: %d images, %d instances, %d clipped coords, %d rotation-fixed",
        split.name,
        stats["images"],
        stats["instances"],
        stats["clipped_coords"],
        len(stats["rotation_fixed"]),
    )
    return stats


def build_data_yaml(categories: list[dict[str, Any]], split_names: list[str]) -> dict[str, Any]:
    """Ultralytics data YAML with names indexed by COCO category id."""
    cats = sorted(categories, key=lambda c: c["id"])
    ids = [c["id"] for c in cats]
    if ids != list(range(len(ids))):
        raise ConversionError(f"category ids must be contiguous from 0, got {ids}")
    data: dict[str, Any] = {}
    for name in split_names:
        key = "val" if name == "valid" else name
        data[key] = f"images/{name}"
    data["nc"] = len(cats)
    data["names"] = {c["id"]: c["name"] for c in cats}
    return data


def convert_dataset(
    dataset_root: str | Path,
    out_dir: str | Path,
    split_names: list[str],
    annotation_file: str = DEFAULT_ANNOTATION_FILE,
    fixes: Mapping[str, int] | None = None,
    overwrite: bool = False,
) -> Path:
    """Convert all splits and write ``data.yaml`` and ``manifest.json``. Returns the YAML path."""
    src_root = resolve_path(dataset_root)
    out_root = resolve_path(out_dir)
    fixes = dict(default_rotation_fixes() if fixes is None else fixes)
    validate_rotation_fixes(fixes)

    if out_root.exists() and any(out_root.iterdir()):
        if not overwrite:
            raise FileExistsError(f"{out_root} is not empty; pass --overwrite to rebuild it")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    splits = [load_split(src_root, name, annotation_file) for name in split_names]
    reference = [(c["id"], c["name"]) for c in splits[0].categories]
    for split in splits[1:]:
        if [(c["id"], c["name"]) for c in split.categories] != reference:
            raise ConversionError(f"categories of {split.name} differ from {splits[0].name}")

    split_stats = {split.name: convert_split(split, out_root, fixes) for split in splits}

    data = build_data_yaml(splits[0].categories, split_names)
    yaml_path = out_root / "data.yaml"
    header = (
        "# Generated by iiss.data.coco_to_yolo. Do not edit by hand.\n"
        "# Class index = COCO category_id. Index 0 (indoor-objects) has no annotations.\n"
        "# No 'path' key: Ultralytics resolves the split folders relative to this file.\n"
    )
    yaml_path.write_text(
        header + yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    manifest = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_root": _display(src_root),
        "sources": {
            s.name: {
                "annotation_file": _display(s.annotation_path),
                "sha256": _sha256(s.annotation_path),
            }
            for s in splits
        },
        "class_index": "coco_category_id",
        "nc": data["nc"],
        "normalisation": "polygon coords / JSON width|height, clipped to [0, 1]",
        "decimals": DECIMALS,
        "rotation_fixes": fixes,
        "splits": split_stats,
    }
    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote %s", yaml_path)
    return yaml_path


def _display(path: Path) -> str:
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    parser = argparse.ArgumentParser(description="Convert COCO segmentation to YOLO-seg")
    parser.add_argument("--dataset-root", default=base["paths"]["dataset_root"])
    parser.add_argument("--splits", nargs="+", default=list(base["data"]["splits"].values()))
    parser.add_argument("--annotation-file", default=base["data"]["annotation_file"])
    parser.add_argument("--out-dir", default=str(Path(base["paths"]["derived_root"]) / "yolo"))
    parser.add_argument("--overwrite", action="store_true", help="rebuild a non-empty out-dir")
    args = parser.parse_args(argv)
    setup_logging()
    convert_dataset(
        args.dataset_root,
        args.out_dir,
        args.splits,
        annotation_file=args.annotation_file,
        fixes=base["data"].get("rotation_fixes") or {},
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
