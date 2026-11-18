"""Instance mask overlays and sample galleries.

All images are loaded with :func:`iiss.data.images.load_image`, so the dataset rotation fixes
are applied and pixels line up with the annotations. Masks are rasterised with pycocotools
(:func:`iiss.eval.results_json.polygons_to_mask`), the same way the evaluator sees them.

Used by ``notebooks/data_exploration.ipynb`` now and by the failure analysis later.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from iiss.data.coco import CocoSplit
from iiss.data.images import load_image
from iiss.eval.results_json import polygons_to_mask

logger = logging.getLogger(__name__)

# Instance colours: the 8-slot reference categorical palette, reused in order. Instances are
# not series, so cycling is only for telling neighbouring masks apart; the class name label
# carries the identity.
INSTANCE_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"


def _rgb(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    return np.array([int(h[i : i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)


def overlay_instances(
    image: Image.Image,
    annotations: Sequence[Mapping[str, Any]],
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend every annotation mask onto ``image``. Returns an ``(H, W, 3)`` uint8 array.

    Larger masks are drawn first so small objects stay visible on top.
    """
    canvas = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    height, width = canvas.shape[:2]
    order = sorted(range(len(annotations)), key=lambda i: -annotations[i].get("area", 0))
    for rank, i in enumerate(order):
        mask = polygons_to_mask(annotations[i]["segmentation"], height, width)
        color = _rgb(INSTANCE_COLORS[rank % len(INSTANCE_COLORS)])
        canvas[mask] = (1 - alpha) * canvas[mask] + alpha * color
    return canvas.clip(0, 255).astype(np.uint8)


def show_instances(
    ax: plt.Axes,
    split: CocoSplit,
    image_info: Mapping[str, Any],
    fixes: Mapping[str, int] | None = None,
    max_labels: int = 12,
    title: str | None = None,
) -> None:
    """Draw one image with its GT masks and the class names of its largest instances."""
    img = load_image(split.image_path(image_info), image_info, fixes)
    anns = split.annotations_by_image.get(image_info["id"], [])
    ax.imshow(overlay_instances(img, anns))
    for ann in sorted(anns, key=lambda a: -a.get("area", 0))[:max_labels]:
        x, y = ann["bbox"][:2]
        ax.text(
            x + 2,
            y + 2,
            split.category_names[ann["category_id"]],
            fontsize=6,
            color=INK,
            va="top",
            bbox={"facecolor": SURFACE, "alpha": 0.75, "pad": 1, "linewidth": 0},
        )
    ax.set_title(
        title or f"{split.name}: {image_info['file_name'][:28]} ({len(anns)} inst.)",
        fontsize=7,
        loc="left",
        color=INK,
    )
    ax.set_axis_off()


def sample_gallery(
    split: CocoSplit,
    n: int = 6,
    seed: int = 42,
    file_names: Sequence[str] | None = None,
    fixes: Mapping[str, int] | None = None,
    cols: int = 3,
) -> plt.Figure:
    """Grid of GT overlays: ``file_names`` if given, else ``n`` images sampled with ``seed``."""
    if file_names is not None:
        by_name = {img["file_name"]: img for img in split.images}
        images = [by_name[name] for name in file_names]
    else:
        images = random.Random(seed).sample(split.images, k=min(n, len(split.images)))
    rows = max(1, -(-len(images) // cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.2 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, info in zip(axes.flat, images, strict=False):
        show_instances(ax, split, info, fixes)
    fig.tight_layout()
    return fig
