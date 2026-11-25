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
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from iiss.analysis.domain_shift import load_results, parse_run_id
from iiss.analysis.failure_viz import Item, class_colors, draw_panel, reduced_size
from iiss.analysis.result_plots import SHORT_NAMES
from iiss.analysis.viz import SURFACE
from iiss.data.coco import load_coco_json
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
    fig.tight_layout()
    fig.canvas.draw()
    for ax in axes[0]:
        spread_labels(ax)
    if out_dir is not None:
        runs = "-".join(m.run_id.split("_seed")[0] for m in result.models)
        path = resolve_path(out_dir) / f"{result.path.stem}_{runs}_t{threshold:.2f}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=100, facecolor=SURFACE, bbox_inches="tight")
        logger.info("saved %s", path)
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
