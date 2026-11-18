"""Write model predictions as a COCO results JSON (one format for every model).

Every model wrapper converts its raw output into :class:`Detection` objects whose mask is a
binary array **at the original image resolution, in the annotation frame** (the frame produced
by :func:`iiss.data.images.load_image`, i.e. after the dataset rotation fixes). This module
encodes those masks as RLE and writes a list that ``COCO.loadRes`` accepts directly::

    [{"image_id": 3, "category_id": 17, "score": 0.91,
      "segmentation": {"size": [H, W], "counts": "..."}, "bbox": [x, y, w, h], "area": 1234}]

``category_id`` is the v5 category id (= class index in this project).

A sidecar ``<name>.meta.json`` records the run, eval set, detection limits and counts.

Selection policy, identical for all models (recorded in the sidecar):

- detections with an empty mask are dropped (they cannot be scored as segments);
- detections below ``score_threshold`` are dropped;
- at most ``max_dets`` detections per image are kept, highest score first. 100 matches the
  largest ``maxDets`` used by COCOeval, so nothing COCOeval would read is lost.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pycocotools import mask as mask_utils

logger = logging.getLogger(__name__)

DEFAULT_MAX_DETS = 100
DEFAULT_SCORE_THRESHOLD = 0.0


class PredictionError(ValueError):
    """Raised when a prediction is inconsistent with the ground truth it will be scored on."""


@dataclass
class Detection:
    """One predicted instance. ``mask`` is a boolean ``(H, W)`` array at original resolution."""

    image_id: int
    category_id: int
    score: float
    mask: np.ndarray = field(repr=False)


# ---------------------------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------------------------


def encode_mask(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary ``(H, W)`` mask as COCO RLE with ``counts`` as a JSON-safe string."""
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {mask.shape}")
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def decode_mask(rle: Mapping[str, Any]) -> np.ndarray:
    """Decode a COCO RLE (string or bytes counts) to a boolean ``(H, W)`` mask."""
    rle = dict(rle)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return mask_utils.decode(rle).astype(bool)


def polygons_to_mask(segmentation: list[list[float]], height: int, width: int) -> np.ndarray:
    """Rasterise COCO polygons to a boolean mask, exactly as pycocotools does for GT."""
    rles = mask_utils.frPyObjects(segmentation, height, width)
    return mask_utils.decode(mask_utils.merge(rles)).astype(bool)


def resize_mask(mask: np.ndarray, height: int, width: int, threshold: float = 0.5) -> np.ndarray:
    """Resize a soft or binary mask to ``(height, width)`` and binarise it.

    Soft masks (probabilities/logits after sigmoid) are resized bilinearly, then thresholded.
    Binary masks are resized with nearest-neighbour so they stay binary.
    """
    if mask.shape == (height, width):
        return mask >= threshold if mask.dtype != bool else mask
    if mask.dtype == bool:
        out = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        return out.astype(bool)
    out = cv2.resize(mask.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    return out >= threshold


# ---------------------------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------------------------


def detection_to_coco(det: Detection) -> dict[str, Any] | None:
    """Convert one detection to a COCO result dict. Returns ``None`` for an empty mask."""
    mask = det.mask.astype(bool)
    rle = encode_mask(mask)
    area = float(mask_utils.area(_bytes_rle(rle)))
    if area == 0:
        return None
    bbox = [float(v) for v in mask_utils.toBbox(_bytes_rle(rle))]
    return {
        "image_id": int(det.image_id),
        "category_id": int(det.category_id),
        "score": round(float(det.score), 6),
        "segmentation": rle,
        "bbox": [round(v, 2) for v in bbox],
        "area": area,
    }


def _bytes_rle(rle: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(rle)
    if isinstance(out["counts"], str):
        out["counts"] = out["counts"].encode("ascii")
    return out


class ResultsWriter:
    """Collect detections per image, check them against the GT, and write the results JSON.

    Args:
        gt_images: ``image_id -> (height, width)`` of the GT images this prediction set covers.
        valid_category_ids: Category ids that exist in the GT category list.
        max_dets: Maximum detections kept per image (highest score first).
        score_threshold: Detections with a lower score are dropped.
    """

    def __init__(
        self,
        gt_images: Mapping[int, tuple[int, int]],
        valid_category_ids: Iterable[int],
        max_dets: int = DEFAULT_MAX_DETS,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    ) -> None:
        self.gt_images = dict(gt_images)
        self.valid_category_ids = set(valid_category_ids)
        self.max_dets = max_dets
        self.score_threshold = score_threshold
        self.results: list[dict[str, Any]] = []
        self.images_done: set[int] = set()
        self.counts = {"received": 0, "empty_mask": 0, "below_threshold": 0, "over_max_dets": 0}

    @classmethod
    def from_coco_gt(cls, gt: Mapping[str, Any], **kwargs: Any) -> ResultsWriter:
        """Build a writer from a loaded COCO GT dict."""
        images = {img["id"]: (img["height"], img["width"]) for img in gt["images"]}
        cats = [c["id"] for c in gt["categories"]]
        return cls(images, cats, **kwargs)

    def add_image(self, image_id: int, detections: Iterable[Detection]) -> int:
        """Add all detections of one image. Returns how many were kept."""
        if image_id not in self.gt_images:
            raise PredictionError(f"image_id {image_id} is not in the ground truth")
        if image_id in self.images_done:
            raise PredictionError(f"image_id {image_id} was already added")
        height, width = self.gt_images[image_id]

        kept: list[dict[str, Any]] = []
        for det in detections:
            self.counts["received"] += 1
            if det.image_id != image_id:
                raise PredictionError(f"detection for image {det.image_id} passed as {image_id}")
            if det.category_id not in self.valid_category_ids:
                raise PredictionError(f"image {image_id}: unknown category_id {det.category_id}")
            if det.mask.shape != (height, width):
                raise PredictionError(
                    f"image {image_id}: mask shape {det.mask.shape} != GT size {(height, width)};"
                    " resize masks to the original resolution first"
                )
            if det.score < self.score_threshold:
                self.counts["below_threshold"] += 1
                continue
            result = detection_to_coco(det)
            if result is None:
                self.counts["empty_mask"] += 1
                continue
            kept.append(result)

        kept.sort(key=lambda r: r["score"], reverse=True)
        if len(kept) > self.max_dets:
            self.counts["over_max_dets"] += len(kept) - self.max_dets
            kept = kept[: self.max_dets]
        self.results.extend(kept)
        self.images_done.add(image_id)
        return len(kept)

    def write(self, path: str | Path, meta: Mapping[str, Any] | None = None) -> Path:
        """Write ``path`` (results list) and ``<stem>.meta.json`` next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        missing = sorted(set(self.gt_images) - self.images_done)
        if missing:
            logger.warning(
                "%d GT images have no predictions added (e.g. %s)", len(missing), missing[:5]
            )
        path.write_text(json.dumps(self.results), encoding="utf-8")

        sidecar = {
            "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
            "results_file": path.name,
            "num_results": len(self.results),
            "num_images_gt": len(self.gt_images),
            "num_images_predicted": len(self.images_done),
            "images_missing": missing,
            "max_dets": self.max_dets,
            "score_threshold": self.score_threshold,
            "mask_resolution": "original image size, annotation frame",
            "category_id": "v5 COCO category id",
            "counts": self.counts,
            **(dict(meta) if meta else {}),
        }
        meta_path = path.with_name(f"{path.stem}.meta.json")
        meta_path.write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
        logger.info("Wrote %d results to %s", len(self.results), path)
        return path
