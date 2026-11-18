"""Helpers shared by all model wrappers: run directories, smoke mode, prediction export."""

from __future__ import annotations

import json
import logging
import shutil
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from tqdm import tqdm

from iiss.data.coco import load_coco_json
from iiss.data.images import load_image
from iiss.eval.coco_eval import load_gt, load_results, run_cocoeval, summary
from iiss.eval.results_json import Detection, ResultsWriter
from iiss.utils.config import deep_merge, make_run_id, save_config
from iiss.utils.env import write_env_json
from iiss.utils.paths import resolve_path

logger = logging.getLogger(__name__)

# (PIL image in the annotation frame, COCO image entry) -> detections at original resolution
Predictor = Callable[[Image.Image, Mapping[str, Any]], list[Detection]]


def apply_smoke(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``cfg`` with the smoke-test limits from ``base.yaml`` applied.

    Sets ``S`` to the smoke size, the time cap to the smoke minutes, and moves the run
    directory and GPU time log under ``smoke.output_dir`` so real logs stay clean. The epoch
    and image limits are applied by each wrapper.
    """
    smoke = cfg["smoke"]
    out_dir = Path(smoke["output_dir"])
    cfg = deep_merge(
        cfg,
        {
            "input_size": int(smoke["input_size"]),
            "time_cap_hours": float(smoke["max_minutes"]) / 60.0,
            "paths": {
                "runs_root": out_dir.as_posix(),
                "gpu_time_log": (out_dir / "gpu_time_log.csv").as_posix(),
            },
            "smoke_mode": True,
        },
    )
    cfg["run_id"] = make_run_id(cfg["model"], cfg["input_size"], int(cfg["seed"]))
    return cfg


def run_dir(cfg: Mapping[str, Any]) -> Path:
    """``<runs_root>/<run_id>`` (absolute)."""
    return resolve_path(cfg["paths"]["runs_root"]) / cfg["run_id"]


def gpu_time_log(cfg: Mapping[str, Any]) -> Path:
    return resolve_path(cfg["paths"]["gpu_time_log"])


def prepare_run_dir(cfg: Mapping[str, Any], fresh: bool = False) -> Path:
    """Create the run directory and write ``config.yaml`` and ``env.json`` into it.

    ``fresh=True`` deletes an existing directory first (smoke runs only).
    """
    path = run_dir(cfg)
    if fresh and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    save_config(cfg, path / "config.yaml")
    write_env_json(path)
    return path


def predictions_path(cfg: Mapping[str, Any], eval_set: str) -> Path:
    """Default output: ``<run_dir>/predictions/<eval_set>.json``."""
    return run_dir(cfg) / "predictions" / f"{eval_set}.json"


def _cuda_peak_mb(reset: bool = False) -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if reset:
        torch.cuda.reset_peak_memory_stats()
        return None
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_allocated() / 2**20, 1)


def export_predictions(
    cfg: Mapping[str, Any],
    eval_set: str,
    predictor: Predictor,
    out_path: str | Path,
    limit: int | None = None,
    meta: Mapping[str, Any] | Callable[[], Mapping[str, Any]] | None = None,
    track_vram: bool = True,
) -> Path:
    """Run ``predictor`` on every image of ``eval_set`` and write a COCO results JSON.

    Images are read from the folder that holds the GT file (Roboflow layout) with
    :func:`load_image`, so EXIF orientation and the dataset rotation fixes are applied and
    predictions are in the annotation frame. ``limit`` keeps the first N images by id (smoke).
    ``track_vram=False`` leaves the CUDA peak-memory counter alone (validation during training,
    where :class:`~iiss.utils.timer.RunTimer` owns it).

    ``ms_per_image`` in the sidecar is the mean wall-clock time of the predictor call (batch 1,
    including the framework's pre/post-processing and upsampling masks to the original
    resolution, excluding RLE encoding), skipping the first image as warm-up. Wrappers may add
    framework-reported model timing to ``meta``; pass a callable to build it after the loop.
    """
    gt_file = resolve_path(cfg["eval"]["gt_files"][eval_set])
    gt = load_coco_json(gt_file)
    images = eval_images(gt, limit)
    writer = ResultsWriter(
        {img["id"]: (img["height"], img["width"]) for img in images},
        [c["id"] for c in gt["categories"]],
        max_dets=int(cfg["eval"]["max_dets"]),
        score_threshold=float(cfg["eval"]["score_threshold"]),
    )
    fixes = cfg["data"].get("rotation_fixes") or {}

    if track_vram:
        _cuda_peak_mb(reset=True)
    times: list[float] = []
    for img in tqdm(images, desc=f"predict {eval_set}", unit="img"):
        pil = load_image(gt_file.parent / img["file_name"], img, fixes)
        t0 = time.perf_counter()
        dets = predictor(pil, img)
        times.append(time.perf_counter() - t0)
        writer.add_image(img["id"], dets)

    timed = times[1:] if len(times) > 1 else times
    info = {
        "run_id": cfg["run_id"],
        "model": cfg["model"],
        "input_size": cfg["input_size"],
        "eval_set": eval_set,
        "gt_file": gt_file.name,
        "images_limited_to": limit,
        "ms_per_image": round(1000 * float(np.mean(timed)), 2) if timed else None,
        "peak_vram_mb": _cuda_peak_mb() if track_vram else None,
        "smoke": bool(cfg.get("smoke_mode", False)),
    }
    if callable(meta):
        meta = meta()
    info.update(meta or {})
    return writer.write(out_path, meta=info)


def eval_images(gt: Mapping[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    """GT images sorted by id; the first ``limit`` of them if given (same rule everywhere)."""
    images = sorted(gt["images"], key=lambda i: i["id"])
    return images if limit is None else images[:limit]


def quick_eval(
    gt_file: str | Path, results_file: str | Path, limit: int | None = None
) -> dict[str, dict[str, float]]:
    """Segm and bbox COCO metrics of a results file, restricted to the same image subset.

    Used for checkpoint selection on ``valid`` during training (the reported numbers come
    from ``iiss.evaluate``).
    """
    gt = load_coco_json(resolve_path(gt_file))
    keep = {img["id"] for img in eval_images(gt, limit)}
    subset = {
        **gt,
        "images": [i for i in gt["images"] if i["id"] in keep],
        "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
    }
    coco_gt = load_gt(subset)
    results = json.loads(Path(results_file).read_text(encoding="utf-8"))
    coco_dt = load_results(coco_gt, results)
    return {t: summary(run_cocoeval(coco_gt, coco_dt, t)) for t in ("segm", "bbox")}
