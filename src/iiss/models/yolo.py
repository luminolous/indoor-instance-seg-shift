"""YOLO11-seg wrapper (Ultralytics): train with the 2-hour cap and resume, export predictions.

Training reads the YOLO-seg copy of the dataset (``python -m iiss.data.coco_to_yolo``), where
the rotation fixes are already baked into the images and class index = COCO category id.

Run directory (``<runs_root>/<run_id>/``), written by Ultralytics plus this wrapper::

    config.yaml, env.json      # resolved project config and environment (this wrapper)
    args.yaml, results.csv     # Ultralytics training args and per-epoch metrics
    weights/last.pt, best.pt   # last = evaluated checkpoint; best (Ultralytics fitness on
                               # valid) is kept and logged only
    events.out.tfevents.*      # TensorBoard (if logging.tensorboard)

Time cap: a callback checks :class:`~iiss.utils.timer.RunTimer` after every batch and every
epoch and sets ``trainer.stop``. Ultralytics then validates, saves ``last.pt``/``best.pt`` and
ends normally. Ultralytics' own ``time=`` argument is not used because it rescales the
epoch count and LR schedule, which would make the schedule depend on ``S``.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image
from ultralytics.utils.ops import letterbox_span

from iiss.eval.results_json import Detection, resize_mask
from iiss.models.common import (
    export_predictions,
    gpu_time_log,
    prepare_run_dir,
    run_dir,
)
from iiss.utils.paths import resolve_path
from iiss.utils.seed import seed_everything
from iiss.utils.timer import RunTimer

logger = logging.getLogger(__name__)

SMOKE_SPLITS = {"train": "max_train_images", "valid": "max_eval_images"}


# ---------------------------------------------------------------------------------------------
# data


def data_yaml_path(cfg: Mapping[str, Any]) -> Path:
    """``<derived_root>/yolo/data.yaml``; raises if the conversion has not been run."""
    path = resolve_path(cfg["paths"]["derived_root"]) / "yolo" / "data.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `python -m iiss.data.coco_to_yolo` first")
    return path


def make_smoke_dataset(src_yaml: Path, out_dir: Path, limits: Mapping[str, int]) -> Path:
    """Copy the first N images (sorted by name) and labels of train/valid into ``out_dir``.

    ``limits`` maps split folder name -> number of images. Returns the new ``data.yaml``.
    """
    src_root = src_yaml.parent
    src = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split, n in limits.items():
        img_out = out_dir / "images" / split
        lbl_out = out_dir / "labels" / split
        img_out.mkdir(parents=True)
        lbl_out.mkdir(parents=True)
        for img in sorted((src_root / "images" / split).iterdir())[:n]:
            shutil.copy2(img, img_out / img.name)
            label = src_root / "labels" / split / f"{img.stem}.txt"
            shutil.copy2(label, lbl_out / label.name)
    data = {"train": "images/train", "val": "images/valid", "nc": src["nc"], "names": src["names"]}
    path = out_dir / "data.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Smoke dataset %s: %s", out_dir, dict(limits))
    return path


# ---------------------------------------------------------------------------------------------
# training


def pretrained_weights(cfg: Mapping[str, Any]) -> Path:
    """Local path for the COCO-pretrained weights (Ultralytics downloads it there if missing)."""
    return resolve_path(cfg["paths"]["pretrained_root"]) / cfg["weights"]


def train_kwargs(cfg: Mapping[str, Any], data_yaml: Path) -> dict[str, Any]:
    """Arguments for ``YOLO.train`` built from the resolved config."""
    kwargs = dict(cfg.get("train") or {})
    if cfg.get("smoke_mode"):
        kwargs["epochs"] = min(int(kwargs.get("epochs", 1)), int(cfg["smoke"]["max_epochs"]))
        kwargs["plots"] = False
        kwargs["workers"] = min(int(kwargs.get("workers", 0)), 2)
    kwargs.update(
        data=str(data_yaml),
        imgsz=int(cfg["input_size"]),
        seed=int(cfg["seed"]),
        deterministic=bool(cfg.get("deterministic", False)),
        amp=bool(cfg.get("amp", True)),
        project=str(resolve_path(cfg["paths"]["runs_root"])),
        name=cfg["run_id"],
        exist_ok=True,
    )
    return kwargs


def _configure_ultralytics(cfg: Mapping[str, Any]) -> None:
    from ultralytics import settings

    logging_cfg = cfg.get("logging") or {}
    wanted = {
        "tensorboard": bool(logging_cfg.get("tensorboard", True)),
        "wandb": bool(logging_cfg.get("wandb", False)),
    }
    if any(settings.get(k) != v for k, v in wanted.items()):
        settings.update(wanted)  # persisted in the Ultralytics user settings file
        logger.info("Ultralytics settings updated: %s", wanted)


def _cap_callback(timer: RunTimer):
    def check(trainer: Any) -> None:
        if timer.should_stop():
            trainer.stop = True

    return check


def train(cfg: Mapping[str, Any], resume: bool = False) -> Path:
    """Train (or resume) one run. Returns the run directory.

    A fresh run refuses to start if ``weights/last.pt`` already exists (use ``resume``).
    Smoke runs (``cfg["smoke_mode"]``) always start fresh on a 16/8-image subset.
    """
    from ultralytics import YOLO

    smoke = bool(cfg.get("smoke_mode"))
    out = run_dir(cfg)
    last = out / "weights" / "last.pt"
    if resume and not last.is_file():
        raise FileNotFoundError(f"cannot resume: {last} does not exist")
    if not resume and last.is_file() and not smoke:
        raise FileExistsError(f"{last} exists; pass --resume or delete {out} to start over")

    data_yaml = data_yaml_path(cfg)
    if smoke and not resume:
        limits = {split: int(cfg["smoke"][key]) for split, key in SMOKE_SPLITS.items()}
        prepare_run_dir(cfg, fresh=True)
        data_yaml = make_smoke_dataset(data_yaml, out / "data", limits)
    else:
        prepare_run_dir(cfg)

    seed_everything(int(cfg["seed"]), bool(cfg.get("deterministic", False)))
    _configure_ultralytics(cfg)

    # The budget is cumulative per run id, except for a fresh smoke run (always 5 min).
    cumulative = resume or not smoke
    log_path = gpu_time_log(cfg)
    with RunTimer(cfg["run_id"], cfg.get("time_cap_hours"), log_path, resume=cumulative) as timer:
        if timer.remaining() <= 0:
            logger.warning("No time budget left for %s; not training", cfg["run_id"])
            timer.cap_hit = True
            return out
        if resume:
            logger.info("Resuming %s from %s", cfg["run_id"], last)
            model = YOLO(str(last))
        else:
            weights = pretrained_weights(cfg)
            weights.parent.mkdir(parents=True, exist_ok=True)
            model = YOLO(str(weights))
        check = _cap_callback(timer)
        model.add_callback("on_train_batch_end", check)
        model.add_callback("on_fit_epoch_end", check)
        if resume:
            model.train(resume=True)
        else:
            model.train(**train_kwargs(cfg, data_yaml))
    return out


# ---------------------------------------------------------------------------------------------
# prediction


def load_model(weights: str | Path) -> Any:
    from ultralytics import YOLO

    return YOLO(str(weights))


def predict_image(
    model: Any,
    image: Image.Image,
    image_info: Mapping[str, Any],
    cfg: Mapping[str, Any],
    speeds: list[dict[str, float]] | None = None,
) -> list[Detection]:
    """Predict one image (annotation frame, RGB). Masks are returned at the original size.

    Mask resolution policy (same for every model): masks are produced at input size S and
    upsampled bilinearly to the original resolution, then thresholded at 0.5. Ultralytics'
    ``retina_masks`` (masks computed at the original size) is never used. Ultralytics returns
    the S-resolution masks already binarised (prototype logits > 0) in the letterboxed frame,
    so the padding is cropped first, then the 0/1 mask is upsampled bilinearly.

    If ``speeds`` is given, Ultralytics' per-stage timing (ms) of this image is appended.
    """
    import torch

    pred_cfg = dict(cfg.get("predict") or {})
    if pred_cfg.pop("retina_masks", False):
        raise ValueError("retina_masks is not allowed: masks must be predicted at S")
    bgr = np.ascontiguousarray(np.asarray(image)[:, :, ::-1])
    result = model.predict(
        bgr,
        imgsz=int(cfg["input_size"]),
        max_det=int(cfg["eval"]["max_dets"]),
        quantize=16 if bool(cfg.get("amp", True)) and torch.cuda.is_available() else None,
        retina_masks=False,
        verbose=False,
        **pred_cfg,
    )[0]
    if speeds is not None:
        speeds.append(dict(result.speed))
    if result.masks is None or len(result.boxes) == 0:
        return []
    height, width = int(image_info["height"]), int(image_info["width"])
    masks = result.masks.data.cpu().numpy()  # (N, h_in, w_in), letterboxed input frame at S
    top, left, content_h, content_w = letterbox_span(masks.shape[1:], (height, width))
    classes = result.boxes.cls.cpu().numpy().astype(int)
    scores = result.boxes.conf.cpu().numpy()
    dets = []
    for mask, cls, score in zip(masks, classes, scores, strict=True):
        content = mask[top : top + content_h, left : left + content_w].astype(np.float32)
        full = resize_mask(content, height, width)  # bilinear, then >= 0.5
        dets.append(Detection(image_info["id"], int(cls), float(score), full))
    return dets


def predict(
    cfg: Mapping[str, Any],
    eval_set: str,
    weights: str | Path,
    out_path: str | Path,
    limit: int | None = None,
) -> Path:
    """Export a COCO results JSON for ``eval_set`` with the given checkpoint."""
    model = load_model(weights)
    speeds: list[dict[str, float]] = []
    path = export_predictions(
        cfg,
        eval_set,
        lambda image, info: predict_image(model, image, info, cfg, speeds),
        out_path,
        limit=limit,
        meta=lambda: {
            "weights": Path(weights).name,
            "predict_args": dict(cfg.get("predict") or {}),
            "ultralytics_speed_ms": _mean_speed(speeds[1:] or speeds),
        },
    )
    return path


def _mean_speed(speeds: list[dict[str, float]]) -> dict[str, float]:
    """Mean Ultralytics preprocess/inference/postprocess ms per image (warm-up skipped)."""
    keys = sorted({k for s in speeds for k in s})
    return {k: round(float(np.mean([s[k] for s in speeds if k in s])), 2) for k in keys}
