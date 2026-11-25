"""Live demo: run trained checkpoints on any photo and compare the overlays.

Used by ``notebooks/live_demo.ipynb``. Every model predicts with the code of the evaluation
(``predict_image`` with the evaluated checkpoint). Images are opened with
:func:`iiss.data.images.load_image`, which applies the EXIF orientation (phone photos in portrait)
and the dataset rotation fixes.

Predictions are kept in memory down to ``MIN_SCORE``. The score threshold is applied only when a
figure or table is made, so changing it needs no new inference. The floor is also handed to the
models. This skips the mask upsampling of predictions that are never shown and does not change the
predictions above it, because greedy NMS and the top-100 cut keep the higher scores.

Two times are reported per image. An untimed gray image of the same size runs before the first
photo of every size, because the first pass of a new input shape can take seconds. ``model_ms`` is
the model itself at input size S (Ultralytics: preprocess + inference + postprocess; torch models:
forward pass, for Mask2Former including the instance selection). ``total_ms`` also includes the
resize of the photo and the mask upsampling to the original size. Both depend on other GPU load
and are not a benchmark.

The scores per object compare predictions with the ground truth of the photo, if it belongs to an
annotated set. Matching and tags are those of :mod:`iiss.analysis.failures`, but on the masks at
display size, so an IoU can differ slightly from the evaluation.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from iiss.analysis.domain_shift import load_results, parse_run_id
from iiss.analysis.failure_experiments import release_gpu
from iiss.analysis.failure_viz import Item, class_colors, draw_panel, reduced_size, small_mask
from iiss.analysis.failures import default_limits, load_gt_set, match_image, overlap
from iiss.analysis.result_plots import SHORT_NAMES
from iiss.analysis.viz import SURFACE
from iiss.data.coco import CocoSplit, load_coco_json
from iiss.data.images import load_image
from iiss.eval.results_json import Detection, decode_mask, encode_mask, resize_mask
from iiss.utils.config import load_yaml
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

DEFAULT_RUNS = ("yolo_s640", "maskrcnn_s640", "m2f_s640")
DEFAULT_THRESHOLD = 0.25
MIN_SCORE = 0.05  # predictions below are not kept
MAX_SIDE = 1000  # longest side of the displayed photo and of the kept masks
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

# image -> (detections, model time in ms, total time in ms)
PredictFn = Callable[[Image.Image], tuple[list[Detection], float, float]]


@dataclass
class Instance:
    """One prediction. ``rle`` is the mask at display size."""

    name: str
    score: float
    rle: dict[str, Any] = field(repr=False)


@dataclass
class ModelResult:
    run_id: str
    instances: list[Instance]  # highest score first
    model_ms: float
    total_ms: float

    def shown(self, threshold: float) -> list[Instance]:
        return [i for i in self.instances if i.score >= threshold]


@dataclass
class PhotoResult:
    path: Path
    size: tuple[int, int]  # (width, height) after EXIF orientation and rotation fix
    photo: np.ndarray = field(repr=False)  # display size
    models: list[ModelResult] = field(repr=False)


# ---------------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------------


def full_run_id(run: str, seed: int = 42) -> str:
    """``yolo_s640`` -> ``yolo_s640_seed42``. Full run ids are returned unchanged."""
    if not re.fullmatch(r"(yolo|maskrcnn|m2f)_s\d+(_seed\d+)?", run):
        raise ValueError(f"unknown run {run!r}, expected e.g. 'yolo_s640'")
    return run if "_seed" in run else f"{run}_seed{seed}"


def list_images(image_path: str | Path) -> list[Path]:
    """The image file, or all images directly inside the folder (sorted by name)."""
    path = resolve_path(image_path)
    if path.is_file():
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"unsupported image type {path.suffix!r}, use {IMAGE_SUFFIXES}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"no such file or folder: {path}")
    files = (p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    images = sorted(files, key=lambda p: p.name.lower())
    if not images:
        raise FileNotFoundError(f"no images in {path}")
    stems = [p.stem.lower() for p in images]
    if len(set(stems)) < len(stems):
        raise ValueError("two images share a name without extension, their figures would collide")
    return images


def category_names() -> dict[int, str]:
    """``category_id -> name`` of the training dataset."""
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    coco = load_coco_json(resolve_path(base["eval"]["train_gt"]))
    return {c["id"]: c["name"] for c in coco["categories"]}


# ---------------------------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------------------------


def load_model(run_id: str, min_score: float = MIN_SCORE) -> PredictFn:
    """Load the evaluated checkpoint of a run."""
    from iiss.cli_common import model_module
    from iiss.eval.batch import Job, job_config, job_weights

    model, size = parse_run_id(run_id)
    job = Job(model, size, "demo", seed=int(run_id.rsplit("_seed", 1)[1]))
    cfg, weights = job_config(job), job_weights(job)
    if not weights.is_file():
        raise FileNotFoundError(f"no checkpoint for {run_id}: {weights}")
    cfg["predict"] = {
        **(cfg.get("predict") or {}),
        "conf" if model == "yolo" else "score_thresh": min_score,
    }
    module = model_module(cfg)
    net = module.load_model(weights) if model == "yolo" else module.load_model(cfg, weights)

    warm: set[tuple[int, int]] = set()

    def predict(image: Image.Image) -> tuple[list[Detection], float, float]:
        info = {"id": 0, "width": image.size[0], "height": image.size[1]}
        if image.size not in warm:
            module.predict_image(net, Image.new("RGB", image.size, (114, 114, 114)), info, cfg)
            warm.add(image.size)
        times: list[Any] = []
        start = time.perf_counter()
        dets = module.predict_image(net, image, info, cfg, times)
        total_ms = 1000 * (time.perf_counter() - start)
        ms = times[0]
        return dets, float(sum(ms.values()) if isinstance(ms, dict) else ms), total_ms

    return predict


def load_models(runs: Sequence[str] = DEFAULT_RUNS) -> dict[str, PredictFn]:
    """``run_id -> prediction function``, all models stay in memory."""
    models = {}
    for run in runs:
        run_id = full_run_id(run)
        logger.info("loading %s", run_id)
        models[run_id] = load_model(run_id)
    return models


def predict_photo(
    models: Mapping[str, PredictFn],
    path: str | Path,
    names: Mapping[int, str],
    max_side: int = MAX_SIDE,
) -> PhotoResult:
    """Run every model on one photo and keep its raw predictions."""
    image = load_image(path)
    width, height = image.size
    small = reduced_size(height, width, max_side)
    results = []
    for run_id, predict in models.items():
        dets, model_ms, total_ms = predict(image)
        instances = []
        for d in sorted(dets, key=lambda d: -d.score):
            mask = resize_mask(d.mask, *small)
            if d.mask.any():
                instances.append(Instance(names[d.category_id], d.score, encode_mask(mask)))
        results.append(ModelResult(run_id, instances, model_ms, total_ms))
    photo = np.asarray(image.resize(small[::-1], Image.Resampling.BILINEAR))
    return PhotoResult(Path(path), image.size, photo, results)


def predict_photos(
    models: Mapping[str, PredictFn], image_path: str | Path, names: Mapping[int, str]
) -> list[PhotoResult]:
    return [predict_photo(models, p, names) for p in list_images(image_path)]


def predict_sizes(
    families: Sequence[str],
    sizes: Sequence[int],
    photos: Sequence[str | Path],
    names: Mapping[int, str],
    loaded: Mapping[str, PredictFn] | None = None,
) -> list[PhotoResult]:
    """Predict with every input size. One result per photo and model family, sizes in order.

    ``loaded`` models are reused. The others are loaded one at a time and removed from memory
    again, so that nine models never share the GPU.
    """
    loaded = loaded or {}
    parts: dict[tuple[int, str], PhotoResult] = {}
    for family in families:
        for size in sizes:
            run_id = full_run_id(f"{family}_s{size}")
            predict = loaded[run_id] if run_id in loaded else load_model(run_id)
            for k, path in enumerate(photos):
                one = predict_photo({run_id: predict}, path, names)
                if (k, family) in parts:
                    parts[k, family].models.extend(one.models)
                else:
                    parts[k, family] = one
            del predict
            if run_id not in loaded:
                release_gpu()
    return [parts[k, f] for k in range(len(photos)) for f in families]


# ---------------------------------------------------------------------------------------------
# Output (no inference)
# ---------------------------------------------------------------------------------------------


def _check_threshold(threshold: float) -> None:
    if not MIN_SCORE <= threshold <= 1:
        raise ValueError(
            f"threshold {threshold} is outside [{MIN_SCORE}, 1]: predictions below {MIN_SCORE} "
            "are not kept"
        )


def photo_colors(result: PhotoResult) -> dict[str, str]:
    """Color per class. The order does not depend on the threshold, so colors stay the same."""
    instances = sorted((i for m in result.models for i in m.instances), key=lambda i: -i.score)
    return class_colors([i.name for i in instances])


def spread_labels(ax: plt.Axes) -> None:
    """Move labels down until none covers another one (two masks can share their center)."""
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()
    placed = []
    for text in sorted(ax.texts, key=lambda t: t.get_position()[1]):
        box = text.get_window_extent(renderer).expanded(1.0, 1.35)
        while any(box.overlaps(b) for b in placed):
            box = box.translated(0, -box.height / 4)  # display y points up
        _, y = to_data.transform((box.x0, (box.y0 + box.y1) / 2))
        text.set_y(y)
        placed.append(box)


def _finish(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
    result: PhotoResult,
    threshold: float,
    out_dir: str | Path | None,
    suffix: str = "",
) -> None:
    fig.tight_layout()
    fig.canvas.draw()
    for ax in axes:
        spread_labels(ax)
    if out_dir is not None:
        runs = "-".join(m.run_id.split("_seed")[0] for m in result.models)
        path = resolve_path(out_dir) / f"{result.path.stem}_{runs}_t{threshold:.2f}{suffix}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100, facecolor=SURFACE, bbox_inches="tight")
        logger.info("saved %s", path)


def plot_photo(
    result: PhotoResult,
    threshold: float = DEFAULT_THRESHOLD,
    out_dir: str | Path | None = None,
) -> plt.Figure:
    """One panel per model: predictions with score >= ``threshold``, count and times.

    Saved as ``<out_dir>/<photo>_<runs>_t<threshold>.jpg`` if ``out_dir`` is given.
    """
    _check_threshold(threshold)
    height, width = result.photo.shape[:2]
    colors = photo_colors(result)
    n = len(result.models)
    fig, axes = plt.subplots(
        1, n, figsize=(6 * n, 6 * height / width + 0.6), squeeze=False, facecolor=SURFACE
    )
    for ax, m in zip(axes[0], result.models, strict=True):
        model, size = parse_run_id(m.run_id)
        shown = m.shown(threshold)
        items = [
            Item(decode_mask(i.rle), colors[i.name], f"{i.name} {i.score:.2f}", True) for i in shown
        ]
        title = (
            f"{SHORT_NAMES[model]} {size}: {len(shown)} instances (score >= {threshold:g})\n"
            f"model {m.model_ms:.0f} ms, with mask upsampling {m.total_ms:.0f} ms"
        )
        draw_panel(ax, result.photo, items, title, label_size=9, title_size=12)
    _finish(fig, axes[0], result, threshold, out_dir)
    return fig


def summary_table(
    results: Sequence[PhotoResult], threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """Instance count, classes and times per photo and model."""
    _check_threshold(threshold)
    rows = []
    for r in results:
        for m in r.models:
            shown = m.shown(threshold)
            counts = pd.Series([i.name for i in shown], dtype=object).value_counts()
            rows.append(
                {
                    "photo": r.path.name,
                    "size": f"{r.size[0]}x{r.size[1]}",
                    "run_id": m.run_id,
                    "instances": len(shown),
                    "classes": ", ".join(f"{n} x{c}" if c > 1 else n for n, c in counts.items()),
                    "model_ms": round(m.model_ms),
                    "total_ms": round(m.total_ms),
                }
            )
    return pd.DataFrame(rows)


def per_class_ap(
    runs: Sequence[str],
    eval_set: str,
    metric: str = "AP",
    eval_root: str | Path | None = None,
) -> pd.DataFrame:
    """Mask AP x 100 per class, one column per run, read from ``results/eval/``.

    Only classes with GT in the set. The last row (``mean``) equals the reported AP.
    """
    if eval_root is None:
        eval_root = load_yaml(repo_root() / "configs" / "base.yaml")["eval"]["results_root"]
    ids = [full_run_id(r) for r in runs]
    loaded = load_results(eval_root, ids, [eval_set])
    parts = [loaded[i, eval_set].per_class for i in ids]
    table = pd.DataFrame({i: p[metric] for i, p in zip(ids, parts, strict=True)}) * 100
    table.insert(0, "n_gt", parts[0]["n_gt"])
    table = table[table["n_gt"] > 0].sort_values("n_gt", ascending=False, kind="stable")
    mean = table[ids].mean()
    mean["n_gt"] = table["n_gt"].sum()
    table.loc["mean"] = mean
    return table.astype({"n_gt": int}).round(1)


# ---------------------------------------------------------------------------------------------
# Scores per object (no inference)
# ---------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _annotated_images() -> dict[Path, tuple[CocoSplit, dict[str, Any]]]:
    """``image path -> (split, image entry)`` of all evaluation sets."""
    index = {}
    for eval_set in load_yaml(repo_root() / "configs" / "base.yaml")["eval"]["gt_files"]:
        split = load_gt_set(eval_set)
        index.update({split.image_path(i).resolve(): (split, i) for i in split.images})
    return index


def ground_truth(result: PhotoResult) -> list[dict[str, Any]] | None:
    """Annotations of the photo with masks at display size, ``None`` if it is not annotated."""
    found = _annotated_images().get(resolve_path(result.path).resolve())
    if found is None:
        return None
    split, info = found
    if (info["width"], info["height"]) != result.size:
        raise ValueError(f"{result.path.name}: the photo and its annotations differ in size")
    size = result.photo.shape[:2]
    return [
        {
            "id": a["id"],
            "name": split.category_names[a["category_id"]],
            "segmentation": encode_mask(small_mask(a["segmentation"], info, size)),
        }
        for a in split.annotations_by_image.get(info["id"], [])
    ]


def _match(
    result: PhotoResult, run: ModelResult, gts: Sequence[Mapping[str, Any]], threshold: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Rows per GT and per shown prediction (in the order of ``run.shown``)."""
    all_names = [g["name"] for g in gts] + [i.name for i in run.instances]
    ids = {name: k for k, name in enumerate(dict.fromkeys(all_names))}
    limits = default_limits(score_threshold=threshold)
    found = match_image(
        [{**g, "category_id": ids[g["name"]]} for g in gts],
        [
            {"category_id": ids[i.name], "score": i.score, "segmentation": i.rle}
            for i in run.instances
        ],
        *result.photo.shape[:2],
        limits,
    )
    shown = run.shown(threshold)
    ov = overlap([i.rle for i in shown], [g["segmentation"] for g in gts])
    index = {g["id"]: k for k, g in enumerate(gts)}
    rows = []
    for p, (instance, row) in enumerate(zip(shown, found.pred, strict=True)):
        g = index.get(row["gt_id"])
        if g is None and gts and ov.iou[p].max() >= limits.partial_iou:
            g = int(ov.iou[p].argmax())
        rows.append(
            {
                "object": instance.name,
                "score": instance.score,
                "result": row["kind"],
                "gt_object": None if g is None else gts[g]["name"],
                "iou": None if g is None else ov.iou[p, g],
                "mask_precision": None if g is None else ov.precision[p, g],
                "mask_recall": None if g is None else ov.recall[p, g],
            }
        )
    return found.gt, rows


def object_scores(
    results: Sequence[PhotoResult],
    threshold: float = DEFAULT_THRESHOLD,
    ap_set: str = "phone",
    eval_root: str | Path | None = None,
) -> pd.DataFrame:
    """One row per detected object: score, class AP of the run, and the comparison with the GT.

    ``class_AP`` is the mask AP x 100 of the class on ``ap_set`` (empty if the set has no GT of the
    class). ``iou``, ``mask_precision`` and ``mask_recall`` compare the mask with ``gt_object``: the
    matched GT, else the GT of any class with the highest IoU (at least ``partial_iou``). They are
    empty without annotations.
    """
    _check_threshold(threshold)
    runs = list(dict.fromkeys(m.run_id for r in results for m in r.models))
    class_ap = per_class_ap(runs, ap_set, eval_root=eval_root).drop(index="mean")
    rows = []
    for r in results:
        gts = ground_truth(r)
        for m in r.models:
            if gts is None:
                objects = [
                    {"object": i.name, "score": i.score, "result": "not annotated"}
                    for i in m.shown(threshold)
                ]
            else:
                objects = _match(r, m, gts, threshold)[1]
            for o in objects:
                known = o["object"] in class_ap.index
                ap = class_ap.at[o["object"], m.run_id] if known else None
                rows.append({"photo": r.path.name, "run_id": m.run_id, **o, "class_AP": ap})
    numbers = ["score", "iou", "mask_precision", "mask_recall"]
    columns = ["photo", "run_id", "object", "score", "class_AP", "result", "gt_object"]
    table = pd.DataFrame(rows, columns=[*columns, *numbers[1:]])
    table[numbers] = table[numbers].astype(float).round(3)
    return table.astype({"class_AP": float})


def photo_scores(
    results: Sequence[PhotoResult], threshold: float = DEFAULT_THRESHOLD
) -> pd.DataFrame:
    """One row per photo and run: matched objects, precision, recall and mean IoU of the matches.

    These are values of one photo at one threshold, not AP. Photos without annotations are left
    out.
    """
    _check_threshold(threshold)
    rows = []
    for r in results:
        gts = ground_truth(r)
        for m in r.models if gts is not None else []:
            objects = _match(r, m, gts, threshold)[1]
            ious = [o["iou"] for o in objects if o["result"] == "matched"]
            rows.append(
                {
                    "photo": r.path.name,
                    "run_id": m.run_id,
                    "gt": len(gts),
                    "predicted": len(objects),
                    "matched": len(ious),
                    "precision": len(ious) / len(objects) if objects else None,
                    "recall": len(ious) / len(gts) if gts else None,
                    "mean_iou": float(np.mean(ious)) if ious else None,
                }
            )
    numbers = ["precision", "recall", "mean_iou"]
    columns = ["photo", "run_id", "gt", "predicted", "matched", *numbers]
    table = pd.DataFrame(rows, columns=columns)
    table[numbers] = table[numbers].astype(float).round(2)
    return table


def gt_scores(results: Sequence[PhotoResult], threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """One row per annotated object, one column per run: best IoU within the class and the tag.

    Photos without annotations are left out.
    """
    _check_threshold(threshold)
    rows: dict[tuple[Path, int], dict[str, Any]] = {}
    for r in results:
        gts = ground_truth(r)
        for m in r.models if gts is not None else []:
            for gt, row in zip(gts, _match(r, m, gts, threshold)[0], strict=True):
                entry = rows.setdefault(
                    (r.path, gt["id"]),
                    {
                        "photo": r.path.name,
                        "object": gt["name"],
                        "area_%": round(100 * row["area_fraction"], 1),
                    },
                )
                entry[m.run_id] = f"{row['iou']:.3f} {row['tag']}"
    return pd.DataFrame(list(rows.values()))


def plot_scores(
    result: PhotoResult,
    threshold: float = DEFAULT_THRESHOLD,
    out_dir: str | Path | None = None,
) -> plt.Figure | None:
    """GT panel and one panel per model, labels with score and IoU. ``None`` without annotations.

    White outline: GT or matched prediction. Black outline: prediction without a matching GT.
    Saved as ``<out_dir>/<photo>_<runs>_t<threshold>_scores.jpg`` if ``out_dir`` is given.
    """
    _check_threshold(threshold)
    gts = ground_truth(result)
    if gts is None:
        logger.warning("%s is not annotated, there is nothing to compare", result.path.name)
        return None
    height, width = result.photo.shape[:2]
    colors = class_colors([*photo_colors(result), *(g["name"] for g in gts)])
    n = 1 + len(result.models)
    fig, axes = plt.subplots(
        1, n, figsize=(6 * n, 6 * height / width + 0.6), squeeze=False, facecolor=SURFACE
    )
    items = [Item(decode_mask(g["segmentation"]), colors[g["name"]], g["name"], True) for g in gts]
    title = f"Ground truth: {len(gts)} objects\n{result.path.name}"
    draw_panel(axes[0][0], result.photo, items, title, label_size=9, title_size=12)
    for ax, m in zip(axes[0][1:], result.models, strict=True):
        model, size = parse_run_id(m.run_id)
        objects = _match(result, m, gts, threshold)[1]
        items = []
        for i, o in zip(m.shown(threshold), objects, strict=True):
            matched = o["result"] == "matched"
            text = f"IoU {o['iou']:.2f}" if matched else o["result"]
            label = f"{i.name} {i.score:.2f}, {text}"
            items.append(Item(decode_mask(i.rle), colors[i.name], label, matched))
        n_matched = sum(i.matched for i in items)
        title = (
            f"{SHORT_NAMES[model]} {size} (score >= {threshold:g})\n"
            f"{n_matched} of {len(gts)} matched, {len(items) - n_matched} other predictions"
        )
        draw_panel(ax, result.photo, items, title, label_size=9, title_size=12)
    _finish(fig, axes[0], result, threshold, out_dir, "_scores")
    return fig
