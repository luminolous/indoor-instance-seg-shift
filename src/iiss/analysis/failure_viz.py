"""Side-by-side GT and prediction overlays for the failure analysis.

Fill color = class (the same in all panels of a figure). Outline: white = GT or matched
prediction, black = prediction without a matching GT. Labels carry the class, the score and the
matching result, so nothing depends on color alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from iiss.analysis.domain_shift import parse_run_id
from iiss.analysis.failures import ImageMatch, photo_name, to_rle
from iiss.analysis.result_plots import SHORT_NAMES
from iiss.analysis.viz import INK, INSTANCE_COLORS, SURFACE, _rgb
from iiss.data.coco import CocoSplit
from iiss.data.images import load_image
from iiss.eval.results_json import decode_mask

MAX_SIDE = 1000
ALPHA = 0.45


@dataclass
class Item:
    """One mask to draw, at the size of the (reduced) photo."""

    mask: np.ndarray
    color: str
    label: str
    matched: bool


def reduced_size(height: int, width: int, max_side: int = MAX_SIDE) -> tuple[int, int]:
    scale = min(1.0, max_side / max(height, width))
    return max(1, round(height * scale)), max(1, round(width * scale))


def small_mask(segmentation: Any, info: Mapping[str, Any], size: tuple[int, int]) -> np.ndarray:
    full = decode_mask(to_rle(segmentation, info["height"], info["width"]))
    out = cv2.resize(full.astype(np.uint8), size[::-1], interpolation=cv2.INTER_NEAREST)
    return out.astype(bool)


def class_colors(names: Sequence[str]) -> dict[str, str]:
    """Color per class name, cycling through the instance palette (labels carry the identity)."""
    return {
        n: INSTANCE_COLORS[i % len(INSTANCE_COLORS)] for i, n in enumerate(dict.fromkeys(names))
    }


def label_point(mask: np.ndarray) -> tuple[int, int]:
    """(x, y) of the pixel deepest inside the mask."""
    padded = np.pad(mask.astype(np.uint8), 1)
    depth = cv2.distanceTransform(padded, cv2.DIST_L2, 3)[1:-1, 1:-1]
    y, x = np.unravel_index(int(depth.argmax()), depth.shape)
    return int(x), int(y)


def draw_panel(ax: plt.Axes, photo: np.ndarray, items: Sequence[Item], title: str) -> None:
    canvas = photo.astype(np.float32).copy()
    ordered = sorted((i for i in items if i.mask.any()), key=lambda i: -int(i.mask.sum()))
    for item in ordered:
        canvas[item.mask] = (1 - ALPHA) * canvas[item.mask] + ALPHA * _rgb(item.color)
    canvas = canvas.clip(0, 255).astype(np.uint8).copy()
    for item in ordered:
        contours, _ = cv2.findContours(
            item.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        edge = (255, 255, 255) if item.matched else (0, 0, 0)
        cv2.drawContours(canvas, contours, -1, edge, 3)
    ax.imshow(canvas)
    height, width = canvas.shape[:2]
    for item in ordered:
        x, y = label_point(item.mask)
        align = "left" if x < 0.18 * width else "right" if x > 0.82 * width else "center"
        ax.text(
            {"left": 4, "right": width - 4}.get(align, x),
            min(max(y, 0.03 * height), 0.97 * height),
            item.label,
            fontsize=6,
            color=INK,
            ha=align,
            va="center",
            clip_on=True,
            bbox={"facecolor": SURFACE, "alpha": 0.8, "pad": 1, "linewidth": 0},
        )
    ax.set_title(title, fontsize=8, loc="left", color=INK)
    ax.set_axis_off()


def compare_runs(
    split: CocoSplit,
    file_name: str,
    matches: Mapping[str, Mapping[int, ImageMatch]],
    run_ids: Sequence[str],
    max_side: int = MAX_SIDE,
    save_to: str | Path | None = None,
) -> plt.Figure:
    """GT panel and one panel per run for one photo."""
    info = next(i for i in split.images if photo_name(i) == file_name)
    size = reduced_size(info["height"], info["width"], max_side)
    photo = np.asarray(load_image(split.image_path(info), info).resize(size[::-1]))
    gts = split.annotations_by_image.get(info["id"], [])
    shown = {r: matches[r][info["id"]] for r in run_ids}
    names = split.category_names
    colors = class_colors(
        [names[a["category_id"]] for a in gts]
        + [names[p["category_id"]] for m in shown.values() for p in m.pred]
    )

    def item(segmentation: Any, category_id: int, text: str, matched: bool) -> Item:
        name = names[category_id]
        return Item(small_mask(segmentation, info, size), colors[name], name + text, matched)

    fig, axes = plt.subplots(
        1,
        1 + len(run_ids),
        figsize=(4.4 * (1 + len(run_ids)), 4.4 * size[0] / size[1] + 0.5),
        squeeze=False,
    )
    fig.patch.set_facecolor(SURFACE)
    context = ", ".join(str(info[k]) for k in ("condition", "scene") if info.get(k))
    draw_panel(
        axes[0][0],
        photo,
        [item(a["segmentation"], a["category_id"], "", True) for a in gts],
        f"GT: {file_name} ({context})\n{len(gts)} instances",
    )
    for ax, run_id in zip(axes[0][1:], run_ids, strict=True):
        model, input_size = parse_run_id(run_id)
        m = shown[run_id]
        items = []
        for p in m.pred:
            result = f"IoU {p['iou']:.2f}" if p["kind"] == "matched" else p["kind"]
            items.append(
                item(
                    p["segmentation"],
                    p["category_id"],
                    f" {p['score']:.2f}\n{result}",
                    p["kind"] == "matched",
                )
            )
        n_matched = sum(r["matched"] for r in m.gt)
        draw_panel(
            ax,
            photo,
            items,
            f"{SHORT_NAMES[model]} {input_size}\n"
            f"matched {n_matched} of {len(m.gt)} GT, {len(m.pred) - n_matched} other predictions",
        )
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=110, facecolor=SURFACE)
    return fig
