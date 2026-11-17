"""Minimal COCO annotation IO for the Roboflow export.

Each split directory holds the images and one ``_annotations.coco.json``. Class index equals the
COCO ``category_id`` everywhere in this project (see the dataset policy in the README).
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from iiss.utils.paths import resolve_path

logger = logging.getLogger(__name__)

DEFAULT_ANNOTATION_FILE = "_annotations.coco.json"


@dataclass
class CocoSplit:
    """One split of a COCO dataset: the raw JSON plus lookup indexes."""

    name: str
    image_dir: Path
    annotation_path: Path
    data: dict[str, Any] = field(repr=False)

    @property
    def images(self) -> list[dict[str, Any]]:
        return self.data.get("images", [])

    @property
    def annotations(self) -> list[dict[str, Any]]:
        return self.data.get("annotations", [])

    @property
    def categories(self) -> list[dict[str, Any]]:
        return self.data.get("categories", [])

    @cached_property
    def images_by_id(self) -> dict[int, dict[str, Any]]:
        return {img["id"]: img for img in self.images}

    @cached_property
    def annotations_by_image(self) -> dict[int, list[dict[str, Any]]]:
        index: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in self.annotations:
            index[ann["image_id"]].append(ann)
        return dict(index)

    @cached_property
    def category_names(self) -> dict[int, str]:
        return {cat["id"]: cat["name"] for cat in self.categories}

    def image_path(self, image: dict[str, Any]) -> Path:
        return self.image_dir / image["file_name"]


def load_coco_json(path: str | Path) -> dict[str, Any]:
    """Load a COCO JSON file."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: COCO file must be a JSON object")
    return data


def load_split(
    dataset_root: str | Path,
    split: str,
    annotation_file: str = DEFAULT_ANNOTATION_FILE,
) -> CocoSplit:
    """Load ``<dataset_root>/<split>/<annotation_file>``. Relative roots resolve from the repo."""
    image_dir = resolve_path(dataset_root) / split
    annotation_path = image_dir / annotation_file
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    data = load_coco_json(annotation_path)
    logger.debug(
        "Loaded %s: %d images, %d annotations",
        annotation_path,
        len(data.get("images", [])),
        len(data.get("annotations", [])),
    )
    return CocoSplit(name=split, image_dir=image_dir, annotation_path=annotation_path, data=data)
