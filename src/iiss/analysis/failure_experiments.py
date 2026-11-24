"""Small inference-only experiments that test one failure hypothesis each (assignment part 3).

Configured in ``configs/failure_experiments.yaml``. A *variant* is a changed copy of one phone
photo. Every model predicts each variant with the same code as the evaluation
(``predict_image``), the masks are mapped back to the frame of the original photo, and the
target instances are matched and tagged with :mod:`iiss.analysis.failures`. The unchanged photo
(``original``) is predicted too and must reproduce the exported predictions
(:func:`check_baseline`).

``mask_grid`` needs no model: it sends the GT mask through a G x G grid with torchvision's own
RoIAlign and mask pasting, which gives the best IoU a mask head of that resolution can reach.

Every result is for a single photo. It supports or contradicts one explanation and does not
measure how often the explanation holds.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from iiss.analysis.domain_shift import parse_run_id
from iiss.analysis.failure_viz import Item, class_colors, draw_panel, reduced_size
from iiss.analysis.failures import (
    ImageMatch,
    Limits,
    load_predictions,
    match_image,
    overlap,
    photo_name,
    to_rle,
)
from iiss.analysis.result_plots import SHORT_NAMES
from iiss.analysis.viz import SURFACE
from iiss.data.coco import CocoSplit
from iiss.data.images import load_image
from iiss.eval.results_json import Detection, decode_mask, encode_mask, resize_mask
from iiss.utils.config import load_yaml, make_run_id
from iiss.utils.paths import repo_root

logger = logging.getLogger(__name__)

GRAY = 114  # padding value of Ultralytics' letterbox
Predictor = Callable[[Image.Image, Mapping[str, Any]], list[Detection]]
MaskMap = Callable[[np.ndarray], np.ndarray]


@dataclass
class Variant:
    """A changed photo. ``to_original`` maps a mask of ``image`` to the original photo frame."""

    name: str
    image: Image.Image
    to_original: MaskMap = field(default=lambda mask: mask, repr=False)


@dataclass
class Outcome:
    """Predictions of one run on one variant, matched in the original frame."""

    experiment: str
    variant: Variant
    run_id: str
    match: ImageMatch
    # shown predictions in the frame of the variant, in the order of ``match.pred``
    shown: list[dict[str, Any]]
    rows: list[dict[str, Any]]


def load_experiments() -> dict[str, Any]:
    return load_yaml(repo_root() / "configs" / "failure_experiments.yaml")


# ---------------------------------------------------------------------------------------------
# Image changes
# ---------------------------------------------------------------------------------------------


def apply_gamma(image: Image.Image, gamma: float) -> Image.Image:
    table = (255 * (np.arange(256) / 255) ** gamma).round().astype(np.uint8)
    return Image.fromarray(table[np.asarray(image)])


def apply_clahe(image: Image.Image, clip: float, tiles: int = 8) -> Image.Image:
    """Local contrast enhancement on the L channel of Lab."""
    lab = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clip, (tiles, tiles)).apply(lab[:, :, 0])
    return Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))


def zoom_out(image: Image.Image, scale: float) -> Variant:
    """The photo shrunk by ``scale`` in the middle of a gray canvas of the original size."""
    w, h = image.size
    sw, sh = max(1, round(w * scale)), max(1, round(h * scale))
    x, y = (w - sw) // 2, (h - sh) // 2
    canvas = Image.new("RGB", (w, h), (GRAY, GRAY, GRAY))
    canvas.paste(image.resize((sw, sh), Image.Resampling.LANCZOS), (x, y))

    def back(mask: np.ndarray) -> np.ndarray:
        return resize_mask(mask[y : y + sh, x : x + sw].astype(bool), h, w)

    return Variant(f"scale {scale:g}", canvas, back)


def crop_window(box: Sequence[float], share: float, width: int, height: int) -> tuple[int, ...]:
    """``(x0, y0, x1, y1)`` of a window with ``share`` of the photo's size, centered on ``box``
    (x, y, w, h) and moved inside the photo."""
    cw, ch = max(1, round(width * share)), max(1, round(height * share))
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    x0 = int(min(max(cx - cw / 2, 0), width - cw))
    y0 = int(min(max(cy - ch / 2, 0), height - ch))
    return x0, y0, x0 + cw, y0 + ch


def crop_around(image: Image.Image, box: Sequence[float], share: float) -> Variant:
    w, h = image.size
    x0, y0, x1, y1 = crop_window(box, share, w, h)

    def back(mask: np.ndarray) -> np.ndarray:
        full = np.zeros((h, w), bool)
        full[y0:y1, x0:x1] = mask
        return full

    return Variant(f"crop {share:g}", image.crop((x0, y0, x1, y1)), back)


def boundary_band(a: np.ndarray, b: np.ndarray, width: int) -> np.ndarray:
    """Band of ``width`` pixels along the common boundary of two masks.

    Annotated neighbors leave a small gap, so the band follows the line with the same distance
    to both masks. It ends where the masks are further apart than about 1 % of the image.
    """
    size = reduced_size(*a.shape)
    far = [
        cv2.distanceTransform(
            1 - cv2.resize(m.astype(np.uint8), size[::-1], interpolation=cv2.INTER_NEAREST),
            cv2.DIST_L2,
            5,
        )
        for m in (a, b)
    ]
    half = max(1.0, width * size[1] / a.shape[1] / 2)
    max_gap = 0.012 * max(size)
    band = (np.abs(far[0] - far[1]) / 2 <= half) & (far[0] + far[1] <= 2 * half + max_gap)
    return resize_mask(band, *a.shape)


def draw_line(image: Image.Image, a: np.ndarray, b: np.ndarray, share: float) -> Image.Image:
    """Black line along the boundary of two masks; ``share`` = line width / image width."""
    out = np.asarray(image).copy()
    out[boundary_band(a, b, max(1, round(share * image.size[0])))] = 0
    return Image.fromarray(out)


def darken(image: Image.Image, mask: np.ndarray, factor: float) -> Image.Image:
    out = np.asarray(image).astype(np.float32)
    out[mask] *= factor
    return Image.fromarray(out.round().astype(np.uint8))


def build_variants(
    spec: Mapping[str, Any], image: Image.Image, gts: Sequence[Mapping[str, Any]]
) -> list[Variant]:
    """``original`` and every variant of one experiment. ``gts`` = the target annotations."""
    h, w = image.size[1], image.size[0]
    out = [Variant("original", image)]
    for kind, values in spec["variants"].items():
        for v in values:
            if kind == "gamma":
                out.append(Variant(f"gamma {v:g}", apply_gamma(image, v)))
            elif kind == "clahe":
                out.append(Variant(f"CLAHE {v:g}", apply_clahe(image, v)))
            elif kind == "scale":
                out.append(zoom_out(image, v))
            elif kind == "crop":
                out.append(crop_around(image, gts[0]["bbox"], v))
            elif kind in ("line", "darken"):
                a, b = (decode_mask(to_rle(g["segmentation"], h, w)) for g in gts)
                changed = draw_line(image, a, b, v) if kind == "line" else darken(image, b, v)
                out.append(Variant(f"{kind} {v:g}", changed))
            else:
                raise ValueError(f"unknown variant kind {kind!r}")
    return out


# ---------------------------------------------------------------------------------------------
# Prediction and matching
# ---------------------------------------------------------------------------------------------


def load_predictor(run_id: str) -> Predictor:
    """Prediction function of a trained run (its evaluated checkpoint)."""
    from iiss.cli_common import model_module
    from iiss.eval.batch import Job, job_config, job_weights

    model, size = parse_run_id(run_id)
    job = Job(model, size, "phone")
    if job.run_id != run_id:
        raise ValueError(f"unexpected run id {run_id!r}, expected {job.run_id!r}")
    cfg, weights = job_config(job), job_weights(job)
    module = model_module(cfg)
    net = module.load_model(weights) if model == "yolo" else module.load_model(cfg, weights)
    return lambda image, info: module.predict_image(net, image, info, cfg)


def release_gpu() -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def predict_variant(
    predictor: Predictor, variant: Variant, image_id: int, score_threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """All predictions in the original frame, and the shown ones in the variant frame.

    Both lists are sorted by score, so the second one lines up with the start of the first.
    """
    info = {"id": image_id, "width": variant.image.size[0], "height": variant.image.size[1]}
    dets = sorted(predictor(variant.image, info), key=lambda d: -d.score)
    original, view = [], []
    for d in dets:
        mask = variant.to_original(d.mask)
        if not mask.any():
            continue
        row = {"image_id": image_id, "category_id": d.category_id, "score": d.score}
        original.append({**row, "segmentation": encode_mask(mask)})
        if d.score >= score_threshold:
            view.append({**row, "segmentation": encode_mask(d.mask)})
    return original, view


def target_rows(
    targets: Sequence[int],
    gts: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    match: ImageMatch,
    names: Mapping[int, str],
    height: int,
    width: int,
    limits: Limits,
) -> list[dict[str, Any]]:
    """One row per target: tag, IoU, score, and the best-scoring class on the object.

    ``correct_score`` and ``correct_iou`` look at all predictions of the right class with IoU >=
    ``match_iou``, also those below the score threshold. ``top_class`` is the class of the
    highest-scoring prediction of any class with that IoU.
    """
    rows = []
    by_id = {r["gt_id"]: r for r in match.gt}
    rles = [to_rle(p["segmentation"], height, width) for p in preds]
    for gt_id in targets:
        ann = next(a for a in gts if a["id"] == gt_id)
        iou = overlap(rles, [to_rle(ann["segmentation"], height, width)]).iou
        on_object = [i for i in range(len(preds)) if iou[i, 0] >= limits.match_iou]
        right = [i for i in on_object if preds[i]["category_id"] == ann["category_id"]]
        top = max(on_object, key=lambda i: preds[i]["score"], default=None)
        best = max(right, key=lambda i: preds[i]["score"], default=None)
        row = by_id[gt_id]
        rows.append(
            {
                "gt_id": gt_id,
                "name": names[ann["category_id"]],
                "tag": row["tag"],
                "matched": row["matched"],
                "iou": row["iou"],
                "correct_score": None if best is None else preds[best]["score"],
                "correct_iou": None if best is None else float(iou[best, 0]),
                "top_class": None if top is None else names[preds[top]["category_id"]],
                "top_score": None if top is None else preds[top]["score"],
            }
        )
    return rows


def run_experiments(
    specs: Mapping[str, Mapping[str, Any]],
    split: CocoSplit,
    run_ids: Sequence[str],
    limits: Limits,
    get_predictor: Callable[[str], Predictor] = load_predictor,
) -> list[Outcome]:
    """Predict every variant of every experiment (except ``mask_grid``) with every run."""
    by_name = {photo_name(i): i for i in split.images}
    outcomes = []
    prepared = {}
    for name, spec in specs.items():
        if "grid" in spec["variants"]:
            continue
        info = by_name[spec["photo"]]
        gts = split.annotations_by_image[info["id"]]
        targets = [next(a for a in gts if a["id"] == t) for t in spec["targets"]]
        photo = load_image(split.image_path(info), info)
        prepared[name] = (info, gts, build_variants(spec, photo, targets))
    for run_id in run_ids:
        predictor = get_predictor(run_id)
        for name, (info, gts, variants) in prepared.items():
            for variant in variants:
                logger.info("%s: %s on %s", run_id, name, variant.name)
                preds, view = predict_variant(
                    predictor, variant, info["id"], limits.score_threshold
                )
                match = match_image(gts, preds, info["height"], info["width"], limits)
                model, size = parse_run_id(run_id)
                base = {
                    "experiment": name,
                    "photo": specs[name]["photo"],
                    "model": model,
                    "input_size": size,
                    "run_id": run_id,
                    "variant": variant.name,
                }
                rows = target_rows(
                    specs[name]["targets"], gts, preds, match, split.category_names,
                    info["height"], info["width"], limits,
                )  # fmt: skip
                outcomes.append(
                    Outcome(
                        name, variant, run_id, match, view,
                        [{**base, **r} for r in rows],
                    )
                )  # fmt: skip
        del predictor
        release_gpu()
    return outcomes


def result_table(outcomes: Sequence[Outcome]) -> pd.DataFrame:
    return pd.DataFrame([r for o in outcomes for r in o.rows])


def overview(table: pd.DataFrame, experiment: str) -> pd.DataFrame:
    """Tag, IoU and strongest class on the object of one experiment, one column per model."""
    part = table[table["experiment"] == experiment]
    text = part.apply(
        lambda r: (
            r["tag"]
            + (f" IoU {r['iou']:.2f}" if r["matched"] else "")
            + (f" [{r['top_class']} {r['top_score']:.2f}]" if pd.notna(r["top_score"]) else "")
        ),
        axis=1,
    )
    rows = part.assign(text=text).set_index(["gt_id", "name", "variant", "model"])["text"]
    keys = list(dict.fromkeys(k[:3] for k in rows.index))
    return rows.unstack("model").reindex(keys)[list(dict.fromkeys(part["model"]))]


def write_tables(table: pd.DataFrame, grid: pd.DataFrame, out_dir: str | Path) -> None:
    """``experiments.csv`` and ``mask_grid.csv``, for later use in the README."""
    out = repo_root() / out_dir
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("experiments", table), ("mask_grid", grid)):
        rounded = part.round(4)
        rounded.to_csv(out / f"{name}.csv", index=False, lineterminator="\n")


def check_baseline(
    outcomes: Sequence[Outcome],
    split: CocoSplit,
    limits: Limits,
    predictions: Callable[[str], Mapping[int, list[dict[str, Any]]]] = load_predictions,
) -> pd.DataFrame:
    """Compare the ``original`` variant with the exported predictions of the evaluation."""
    by_name = {photo_name(i): i for i in split.images}
    cache: dict[str, Any] = {}
    rows = []
    for o in outcomes:
        if o.variant.name != "original":
            continue
        info = by_name[o.rows[0]["photo"]]
        if o.run_id not in cache:
            cache[o.run_id] = predictions(o.run_id)
        stored = match_image(
            split.annotations_by_image[info["id"]],
            cache[o.run_id].get(info["id"], []),
            info["height"], info["width"], limits,
        )  # fmt: skip
        now = [r["tag"] for r in o.match.gt]
        before = [r["tag"] for r in stored.gt]
        a, b = ([p["score"] for p in m.pred] for m in (o.match, stored))
        same_n = len(a) == len(b)
        diff = max((abs(x - y) for x, y in zip(a, b, strict=True)), default=0.0) if same_n else None
        rows.append(
            {
                "experiment": o.experiment,
                "run_id": o.run_id,
                "same_tags": now == before,
                "n_shown": len(a),
                "n_shown_exported": len(b),
                "max_score_diff": diff,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Mask grid bound
# ---------------------------------------------------------------------------------------------


def grid_roundtrip(
    mask: np.ndarray, box: Sequence[float], grid: int, input_size: int
) -> np.ndarray:
    """``mask`` at original size -> ``grid`` x ``grid`` inside ``box`` -> back to original size.

    Same steps as Mask R-CNN training and inference: RoIAlign of the mask on the box (xyxy,
    original pixels) at input size, torchvision mask pasting at input size, bilinear upsampling and
    threshold 0.5.
    """
    import torch
    from torchvision.models.detection.roi_heads import paste_masks_in_image
    from torchvision.ops import roi_align

    h, w = mask.shape
    scale = input_size / max(h, w)
    sh, sw = round(h * scale), round(w * scale)
    small = torch.from_numpy(
        cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA)
    )
    boxes = torch.tensor([[float(v) * scale for v in box]], dtype=torch.float32)
    coarse = roi_align(small[None, None], [boxes], (grid, grid))
    pasted = paste_masks_in_image(coarse, boxes, (sh, sw))[0, 0].numpy()
    return resize_mask(pasted, h, w)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def _xyxy(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def grid_bound_table(
    spec: Mapping[str, Any], split: CocoSplit, run_id: str, limits: Limits
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Best IoU per grid size for every GT of the photo, and the model's own result.

    ``iou_gt_box`` uses the box of the GT mask. For instances the model matched, ``iou_pred_box``
    uses the extent of the predicted mask instead of its box regression output (the exported
    results hold masks only), so it also contains the localisation error.
    """
    info = next(i for i in split.images if photo_name(i) == spec["photo"])
    h, w = info["height"], info["width"]
    size = parse_run_id(run_id)[1]
    gts = split.annotations_by_image[info["id"]]
    match = match_image(gts, load_predictions(run_id).get(info["id"], []), h, w, limits)
    shown = {p["gt_id"]: p for p in match.pred if p["kind"] == "matched"}
    rows, masks = [], {}
    for ann in gts:
        gt = decode_mask(to_rle(ann["segmentation"], h, w))
        pred = shown.get(ann["id"])
        pred_mask = None if pred is None else decode_mask(to_rle(pred["segmentation"], h, w))
        if ann["id"] in spec["targets"]:
            masks["GT"] = gt
            if pred_mask is not None:
                masks["prediction"] = pred_mask
        for grid in spec["variants"]["grid"]:
            bound = grid_roundtrip(gt, _xyxy(gt), grid, size)
            row = {
                "gt_id": ann["id"],
                "name": split.category_names[ann["category_id"]],
                "target": ann["id"] in spec["targets"],
                "grid": grid,
                "iou_gt_box": mask_iou(bound, gt),
                "iou_pred_box": None,
                "model_iou": None if pred is None else pred["iou"],
            }
            if pred_mask is not None:
                row["iou_pred_box"] = mask_iou(grid_roundtrip(gt, _xyxy(pred_mask), grid, size), gt)
            if ann["id"] in spec["targets"]:
                masks[f"GT through {grid}x{grid}"] = bound
            rows.append(row)
    return pd.DataFrame(rows), masks


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------


def _grid(n: int, cols: int) -> tuple[plt.Figure, list[plt.Axes]]:
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.6 * rows), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.set_axis_off()
    return fig, list(axes.flat)


def _save(fig: plt.Figure, save_to: str | Path | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=80, facecolor=SURFACE, bbox_inches="tight")
    return fig


def plot_variants(
    outcomes: Sequence[Outcome],
    experiment: str,
    run_id: str,
    names: Mapping[int, str],
    cols: int = 4,
    save_to: str | Path | None = None,
) -> plt.Figure:
    """One panel per variant with the shown predictions of one run, in the variant's frame."""
    part = [o for o in outcomes if o.experiment == experiment and o.run_id == run_id]
    colors = class_colors([names[p["category_id"]] for o in part for p in o.shown])
    fig, axes = _grid(len(part), cols)
    model, size = parse_run_id(run_id)
    for ax, o in zip(axes, part, strict=False):
        h, w = o.variant.image.size[::-1]
        small = reduced_size(h, w, 700)
        items = []
        for p, m in zip(o.shown, o.match.pred, strict=True):
            mask = resize_mask(decode_mask(p["segmentation"]), *small)
            text = f"IoU {m['iou']:.2f}" if m["kind"] == "matched" else m["kind"]
            name = names[p["category_id"]]
            items.append(
                Item(mask, colors[name], f"{name} {p['score']:.2f}\n{text}", m["kind"] == "matched")
            )
        tags = "; ".join(f"{r['name']}: {r['tag']}" for r in o.rows)
        photo = np.asarray(o.variant.image.resize(small[::-1]))
        draw_panel(ax, photo, items, f"{SHORT_NAMES[model]} {size}, {o.variant.name}\n{tags}")
    return _save(fig, save_to)


def plot_grid_bound(
    spec: Mapping[str, Any],
    split: CocoSplit,
    table: pd.DataFrame,
    masks: Mapping[str, np.ndarray],
    save_to: str | Path | None = None,
) -> plt.Figure:
    """GT, prediction and the GT after each grid round trip, with IoU to the GT."""
    info = next(i for i in split.images if photo_name(i) == spec["photo"])
    photo = load_image(split.image_path(info), info)
    small = reduced_size(info["height"], info["width"], 700)
    view = np.asarray(photo.resize(small[::-1]))
    color = class_colors([table.loc[table["target"], "name"].iloc[0]]).popitem()[1]
    fig, axes = _grid(len(masks), 3)
    for ax, (label, mask) in zip(axes, masks.items(), strict=False):
        iou = "" if label == "GT" else f" (IoU {mask_iou(mask, masks['GT']):.3f})"
        draw_panel(ax, view, [Item(resize_mask(mask, *small), color, "", True)], label + iou)
    return _save(fig, save_to)


def default_run_ids(config: Mapping[str, Any], seed: int = 42) -> list[str]:
    return [make_run_id(m, int(config["input_size"]), seed) for m in config["models"]]
