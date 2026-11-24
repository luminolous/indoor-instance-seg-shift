"""Import the self-made phone test set (Roboflow COCO export) into one COCO file with v5 ids.

Steps:

1. Merge the Roboflow splits. The split has no meaning for this test-only set.
2. Match every exported image to ``originals/`` and ``metadata.csv`` by its original file name
   (the part before ``_<ext>.rf.<hash>``). ``file_name`` in the output is the path of the original
   photo relative to the output JSON's folder, so ``<json folder>/<file_name>`` opens it.
3. Remap category ids to v5 by exact class name. Unknown names fail. The empty Roboflow
   supercategory (id 0) is ignored and fails if it has annotations.
4. Merge fragments: for the classes in ``phone_import.merge_fragments`` all annotations of the
   same class in one image become one annotation with several polygons. ``area`` and ``bbox`` are
   recomputed from the mask union. All other annotations keep their exported geometry.

Note on ``area``: Roboflow exports the *bounding-box* area (v5 too). Only merged annotations carry
the mask area. At 3472x3472 every object is "large" under both definitions (checked per run and
written to the report), so COCO size buckets are unaffected.

Nothing under ``dataset/manual-benchmark/`` is modified.

Usage::

    python -m iiss.data.phone                 # -> dataset/derived/phone/annotations.coco.json
                                              #    + results/phone_import_report.md
    python -m iiss.data.phone --overlays 172304 172021
    python -m iiss.data.phone --dry-run       # check everything, write nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pycocotools import mask as mask_utils

from iiss.data.coco import CocoSplit, load_coco_json, load_split
from iiss.data.images import load_image
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

SUPERCATEGORY_ID = 0
METADATA_COLUMNS = ("filename", "condition", "scene")
LARGE_AREA = 96**2
_EXPORT_NAME = re.compile(r"^(?P<stem>.+)_(?P<ext>[A-Za-z]+)\.rf\.[^.]+\.[A-Za-z]+$")


class PhoneImportError(ValueError):
    """Raised when the export cannot be mapped to the originals, the metadata or v5."""


@dataclass
class MergedFragments:
    """Fragments of one class in one image that became a single annotation."""

    file_name: str
    category: str
    pieces: int


def original_stem(exported_name: str) -> str:
    """``IMG_1_jpg.rf.<hash>.jpg`` -> ``IMG_1``."""
    match = _EXPORT_NAME.match(exported_name)
    if match is None:
        raise PhoneImportError(f"Not a Roboflow export file name: {exported_name!r}")
    return match["stem"]


def index_originals(originals_dir: str | Path) -> dict[str, Path]:
    """``stem -> path`` of the photos in ``originals/``. Duplicate stems fail."""
    index: dict[str, Path] = {}
    for path in sorted(Path(originals_dir).iterdir()):
        if not path.is_file():
            continue
        if path.stem in index:
            raise PhoneImportError(f"Two originals share the stem {path.stem!r}")
        index[path.stem] = path
    return index


def category_map(
    export_categories: Iterable[Mapping[str, Any]], v5_categories: Iterable[Mapping[str, Any]]
) -> dict[int, int]:
    """Export category id -> v5 category id by exact name, skipping the supercategory (id 0)."""
    v5_ids = {c["name"]: int(c["id"]) for c in v5_categories if c["id"] != SUPERCATEGORY_ID}
    used = [c for c in export_categories if c["id"] != SUPERCATEGORY_ID]
    unknown = sorted(c["name"] for c in used if c["name"] not in v5_ids)
    if unknown:
        raise PhoneImportError(
            f"Category names in the phone export that do not exist in v5: {unknown}"
        )
    return {int(c["id"]): v5_ids[c["name"]] for c in used}


def merge_annotations(
    annotations: Sequence[Mapping[str, Any]], height: int, width: int
) -> dict[str, Any]:
    """One annotation with all polygons of ``annotations``; area and bbox of the mask union."""
    polygons = [poly for ann in annotations for poly in ann["segmentation"]]
    rle = mask_utils.merge(mask_utils.frPyObjects(polygons, height, width))
    return {
        **annotations[0],
        "segmentation": polygons,
        "area": float(mask_utils.area(rle)),
        "bbox": [float(v) for v in mask_utils.toBbox(rle)],
        "iscrowd": 0,
    }


def load_metadata(path: str | Path) -> dict[str, dict[str, str]]:
    """``filename -> row`` of ``metadata.csv``."""
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in METADATA_COLUMNS if c not in table.columns]
    if missing:
        raise PhoneImportError(f"{path}: missing columns {missing}")
    duplicates = sorted(table.loc[table["filename"].duplicated(), "filename"])
    if duplicates:
        raise PhoneImportError(f"{path}: duplicate filenames {duplicates}")
    return {row["filename"]: row for row in table.to_dict("records")}


def import_phone_set(
    export_splits: Sequence[CocoSplit],
    originals: Mapping[str, Path],
    metadata: Mapping[str, Mapping[str, str]],
    v5_categories: Sequence[Mapping[str, Any]],
    merge_classes: Iterable[str],
    out_dir: str | Path,
) -> tuple[dict[str, Any], list[MergedFragments]]:
    """Build the phone COCO dict (see the module docstring) and the list of merged fragments."""
    merge_classes = list(merge_classes)
    v5_names = {int(c["id"]): c["name"] for c in v5_categories}
    unknown = sorted(set(merge_classes) - set(v5_names.values()))
    if unknown:
        raise PhoneImportError(f"merge_fragments classes that do not exist in v5: {unknown}")

    # (original path, exported image entry, annotations with v5 category ids)
    entries: dict[str, tuple[Path, Mapping[str, Any], list[dict[str, Any]]]] = {}
    for split in export_splits:
        cat_map = category_map(split.categories, v5_categories)
        for ann in split.annotations:
            if ann["image_id"] not in split.images_by_id:
                raise PhoneImportError(f"{split.name}: annotation {ann['id']} has no image")
            if ann["category_id"] not in cat_map:
                raise PhoneImportError(
                    f"{split.name}: annotation {ann['id']} uses category id {ann['category_id']}"
                    " (the Roboflow supercategory or an undeclared id)"
                )
            if not isinstance(ann["segmentation"], list) or not ann["segmentation"]:
                raise PhoneImportError(f"{split.name}: annotation {ann['id']} has no polygons")
        for img in split.images:
            stem = original_stem(img["file_name"])
            if stem not in originals:
                raise PhoneImportError(f"No original photo for {img['file_name']!r}")
            path = originals[stem]
            if path.name in entries:
                raise PhoneImportError(f"{path.name} is exported more than once")
            if path.name not in metadata:
                raise PhoneImportError(f"{path.name} is not listed in metadata.csv")
            load_image(path, img, fixes={})  # readable, and same size as the export says
            anns = [
                {**ann, "category_id": cat_map[ann["category_id"]]}
                for ann in split.annotations_by_image.get(img["id"], [])
            ]
            entries[path.name] = (path, img, anns)

    exported = set(entries)
    for label, names in (
        ("originals/", {p.name for p in originals.values()}),
        ("metadata.csv", set(metadata)),
    ):
        if names - exported:
            raise PhoneImportError(f"In {label} but not exported: {sorted(names - exported)}")

    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    merged: list[MergedFragments] = []
    for image_id, name in enumerate(sorted(entries), start=1):
        path, img, anns = entries[name]
        height, width = int(img["height"]), int(img["width"])
        images.append(
            {
                "id": image_id,
                "file_name": Path(os.path.relpath(path, out_dir)).as_posix(),
                "width": width,
                "height": height,
                "condition": metadata[name]["condition"],
                "scene": metadata[name]["scene"],
                "exported_file_name": img["file_name"],
            }
        )
        kept: list[dict[str, Any]] = []
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in anns:
            if v5_names[ann["category_id"]] in merge_classes:
                groups[ann["category_id"]].append(ann)
            else:
                kept.append(ann)
        for category_id, pieces in groups.items():
            if len(pieces) == 1:
                kept.append(pieces[0])
                continue
            kept.append(merge_annotations(pieces, height, width))
            merged.append(MergedFragments(name, v5_names[category_id], len(pieces)))
        for ann in kept:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": ann["category_id"],
                    "segmentation": ann["segmentation"],
                    "area": ann["area"],
                    "bbox": ann["bbox"],
                    "iscrowd": 0,
                }
            )
    merged.sort(key=lambda m: (m.category, m.file_name))
    data = {
        "info": {"description": "Self-made phone test set, category ids remapped to v5"},
        "images": images,
        "annotations": annotations,
        "categories": [dict(c) for c in v5_categories],
    }
    return data, merged


def count_non_large(data: Mapping[str, Any]) -> int:
    """Annotations that are not COCO-"large" under the stored area or under the mask area."""
    sizes = {i["id"]: (i["height"], i["width"]) for i in data["images"]}
    count = 0
    for ann in data["annotations"]:
        rle = mask_utils.merge(mask_utils.frPyObjects(ann["segmentation"], *sizes[ann["image_id"]]))
        count += min(float(ann["area"]), float(mask_utils.area(rle))) < LARGE_AREA
    return count


def render_report(
    data: Mapping[str, Any],
    merged: Sequence[MergedFragments],
    n_exported: int,
    merge_classes: Sequence[str],
) -> str:
    """Markdown import report."""
    names = {c["id"]: c["name"] for c in data["categories"]}
    per_class = Counter(names[a["category_id"]] for a in data["annotations"])
    before = Counter(per_class)
    for m in merged:
        before[m.category] += m.pieces - 1
    sizes = Counter((i["width"], i["height"]) for i in data["images"])
    lines = [
        "# Phone test set import",
        "",
        "Generated by `python -m iiss.data.phone`.",
        "",
        f"- Images: {len(data['images'])} ("
        + ", ".join(f"{n} at {w}x{h}" for (w, h), n in sorted(sizes.items()))
        + ", no resize)",
        f"- Annotations: {n_exported} exported -> {len(data['annotations'])} after merging",
        "- Conditions: "
        + ", ".join(
            f"{k} {v}" for k, v in sorted(Counter(i["condition"] for i in data["images"]).items())
        ),
        "- Scenes: "
        + ", ".join(
            f"{k} {v}" for k, v in sorted(Counter(i["scene"] for i in data["images"]).items())
        ),
        f"- Annotations that are not COCO-large (area < 96x96): {count_non_large(data)}",
        "",
        "Roboflow stores the bounding-box area in `area` (v5 does too). Merged annotations store"
        " the mask area instead.",
        "",
        "## Merged fragments",
        "",
        "Classes merged per image (`phone_import.merge_fragments`): "
        + ", ".join(f"`{c}`" for c in merge_classes)
        + ".",
        f"{len(merged)} image/class pairs, {sum(m.pieces for m in merged)} pieces ->"
        f" {len(merged)} annotations.",
        "",
        "| Class | Image | Pieces -> 1 |",
        "|---|---|---|",
    ]
    lines += [f"| {m.category} | {m.file_name} | {m.pieces} |" for m in merged]
    lines += [
        "",
        "## Annotations per class",
        "",
        "| Class | Exported | Imported |",
        "|---|---|---|",
    ]
    lines += [
        f"| {c} | {before[c]} | {per_class[c]} |" for c in sorted(before, key=lambda c: -before[c])
    ]
    return "\n".join(lines) + "\n"


def save_overlays(gt_path: Path, data: dict[str, Any], keys: Sequence[str], out_dir: Path) -> None:
    """Save a before-scoring sanity overlay for every image whose name contains a key."""
    import matplotlib.pyplot as plt

    from iiss.analysis.viz import show_instances

    split = CocoSplit("phone", gt_path.parent, gt_path, data)
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in keys:
        hits = [i for i in split.images if key in Path(i["file_name"]).stem]
        if len(hits) != 1:
            raise PhoneImportError(f"Overlay key {key!r} matches {len(hits)} images, expected 1")
        stem = Path(hits[0]["file_name"]).stem
        fig, ax = plt.subplots(figsize=(8, 8))
        n = len(split.annotations_by_image.get(hits[0]["id"], []))
        show_instances(ax, split, hits[0], fixes={}, title=f"phone: {stem} ({n} inst.)")
        fig.tight_layout()
        fig.savefig(out_dir / f"{stem}.jpg", dpi=110)
        plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import the phone test set")
    parser.add_argument("--dry-run", action="store_true", help="check only, write nothing")
    parser.add_argument("--overlays", nargs="*", default=[], help="name fragments to overlay")
    args = parser.parse_args(argv)
    setup_logging()

    base = load_yaml(repo_root() / "configs" / "base.yaml")
    pcfg, paths = base["phone_import"], base["paths"]
    root = resolve_path(paths["phone_root"])
    gt_path = resolve_path(base["eval"]["gt_files"]["phone"])
    splits = [load_split(root / pcfg["export_dir"], s) for s in pcfg["export_splits"]]
    data, merged = import_phone_set(
        splits,
        index_originals(root / pcfg["originals_dir"]),
        load_metadata(root / pcfg["metadata_file"]),
        load_coco_json(resolve_path(base["eval"]["train_gt"]))["categories"],
        pcfg["merge_fragments"],
        gt_path.parent,
    )
    n_exported = sum(len(s.annotations) for s in splits)
    report = render_report(data, merged, n_exported, pcfg["merge_fragments"])
    if args.dry_run:
        print(report)  # CLI output
        return 0
    results = resolve_path(paths["results_root"])
    gt_path.parent.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    gt_path.write_text(json.dumps(data), encoding="utf-8")
    (results / "phone_import_report.md").write_text(report, encoding="utf-8")
    logger.info(
        "%d images, %d -> %d annotations, %d merged pairs -> %s",
        len(data["images"]), n_exported, len(data["annotations"]), len(merged), gt_path,
    )  # fmt: skip
    if args.overlays:
        save_overlays(gt_path, data, args.overlays, gt_path.parent / "overlays")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
