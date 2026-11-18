"""PyTorch dataset over a COCO split for torchvision-style instance segmentation training.

Each item is ``(image, target)`` with the image already resized so that its **longest side is
S** (aspect ratio kept). This matches what ``GeneralizedRCNNTransform(min_size=S, max_size=S)``
would do, but rasterises the masks directly at that size instead of at the (often much larger)
original resolution, which keeps memory and CPU cost low.

Target keys: ``boxes`` (float32 ``[N, 4]`` xyxy), ``labels`` (int64, = v5 category id),
``masks`` (uint8 ``[N, h, w]``), ``image_id`` (int).

No annotation is dropped. A box that collapses below 1 px after resizing is widened to 1 px,
because torchvision rejects boxes with zero width or height.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from iiss.data.coco import load_coco_json
from iiss.data.images import load_image
from iiss.eval.results_json import polygons_to_mask

MIN_BOX_SIZE = 1.0


def resized_size(width: int, height: int, input_size: int) -> tuple[int, int, float]:
    """``(new_width, new_height, scale)`` with the longest side equal to ``input_size``."""
    scale = input_size / max(width, height)
    return max(1, round(width * scale)), max(1, round(height * scale)), scale


def collate(batch: list[tuple[Any, Any]]) -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """Keep images and targets as lists (variable image sizes)."""
    images, targets = zip(*batch, strict=True)
    return images, targets


class CocoInstanceDataset(Dataset):
    """Instance segmentation samples from one COCO annotation file.

    Args:
        ann_file: COCO JSON; images are read from the same folder (Roboflow layout).
        input_size: S, the longest image side after resizing.
        fixes: Dataset rotation fixes (``data.rotation_fixes``).
        train: Enables the random horizontal flip (torchvision reference default augmentation).
        limit: Keep only the first N images by id (smoke tests).
    """

    def __init__(
        self,
        ann_file: str | Path,
        input_size: int,
        fixes: Mapping[str, int],
        train: bool = False,
        limit: int | None = None,
    ) -> None:
        self.ann_file = Path(ann_file)
        self.root = self.ann_file.parent
        data = load_coco_json(self.ann_file)
        self.images = sorted(data["images"], key=lambda i: i["id"])
        if limit is not None:
            self.images = self.images[:limit]
        self.anns: dict[int, list[dict[str, Any]]] = {img["id"]: [] for img in self.images}
        for ann in data["annotations"]:
            if ann["image_id"] in self.anns:
                if ann["category_id"] == 0:
                    raise ValueError(
                        f"annotation {ann['id']} uses category 0, which is the background slot"
                    )
                self.anns[ann["image_id"]].append(ann)
        self.input_size = int(input_size)
        self.fixes = dict(fixes)
        self.train = train

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        info = self.images[index]
        image = load_image(self.root / info["file_name"], info, self.fixes)
        width, height = image.size
        new_w, new_h, scale = resized_size(width, height, self.input_size)
        image = image.resize((new_w, new_h), Image.Resampling.BILINEAR)

        boxes, labels, masks = [], [], []
        for ann in self.anns[info["id"]]:
            polys = [[v * scale for v in poly] for poly in ann["segmentation"]]
            masks.append(polygons_to_mask(polys, new_h, new_w))
            x, y, w, h = (v * scale for v in ann["bbox"])
            x1, y1 = min(max(x, 0.0), new_w - MIN_BOX_SIZE), min(max(y, 0.0), new_h - MIN_BOX_SIZE)
            x2 = min(max(x + w, x1 + MIN_BOX_SIZE), float(new_w))
            y2 = min(max(y + h, y1 + MIN_BOX_SIZE), float(new_h))
            boxes.append([x1, y1, x2, y2])
            labels.append(int(ann["category_id"]))

        img_t = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": torch.from_numpy(np.stack(masks).astype(np.uint8))
            if masks
            else torch.zeros((0, new_h, new_w), dtype=torch.uint8),
            "image_id": int(info["id"]),
        }
        if self.train and torch.rand(1).item() < 0.5:
            img_t = img_t.flip(-1)
            target["masks"] = target["masks"].flip(-1)
            b = target["boxes"].clone()
            target["boxes"][:, 0] = new_w - b[:, 2]
            target["boxes"][:, 2] = new_w - b[:, 0]
        return img_t.contiguous(), target
