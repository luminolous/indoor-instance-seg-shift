"""Mask R-CNN R50-FPN v2 wrapper (torchvision): train with the 2-hour cap and resume, predict.

Input size: the dataset resizes every image so its longest side is ``S``, and the model is
built with ``min_size=max_size=S`` so ``GeneralizedRCNNTransform`` keeps that size (scale 1).

Run directory (``<runs_root>/<run_id>/``)::

    config.yaml, env.json      # resolved project config and environment
    results.csv                # one row per epoch: losses, lr, valid segm/bbox AP
    events.out.tfevents.*      # TensorBoard (if logging.tensorboard)
    weights/last.pt            # full training state after the last finished epoch (resume)
    weights/best.pt            # model weights with the best valid mask AP
    val/last.json              # predictions of the latest validation

Time cap: :class:`~iiss.utils.timer.RunTimer` is checked after every iteration. When it is
reached the epoch stops, the model is validated, ``last.pt``/``best.pt`` are written and
training ends (the session is logged with ``cap_hit=True``). A resumed run restarts the epoch
that was interrupted, from the state saved after the previous epoch.

Prediction: the image is resized to long side ``S``, masks come out at that size and are
upsampled bilinearly to the original resolution before thresholding at 0.5. This avoids
pasting up to 100 full-resolution float masks on the GPU (about 5 GB for a 12 MP test photo).
"""

from __future__ import annotations

import contextlib
import csv
import logging
import math
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from iiss.data.torch_dataset import CocoInstanceDataset, collate, resized_size
from iiss.eval.results_json import Detection, resize_mask
from iiss.models.common import (
    export_predictions,
    gpu_time_log,
    prepare_run_dir,
    quick_eval,
    run_dir,
)
from iiss.utils.paths import resolve_path
from iiss.utils.seed import seed_everything
from iiss.utils.timer import RunTimer

logger = logging.getLogger(__name__)

LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg")
CSV_COLUMNS = (
    "epoch",
    "epoch_complete",
    "iterations",
    "time_s",
    "lr",
    "loss",
    *LOSS_KEYS,
    "val_segm_AP",
    "val_segm_AP50",
    "val_bbox_AP",
    "best_segm_AP",
)


# ---------------------------------------------------------------------------------------------
# model


@contextlib.contextmanager
def _hub_dir(path: Path) -> Iterator[None]:
    """Download torchvision weights into ``path`` instead of the global torch hub cache."""
    old = torch.hub.get_dir()
    path.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(path))
    try:
        yield
    finally:
        torch.hub.set_dir(old)


def build_model(cfg: Mapping[str, Any], pretrained: bool = True) -> torch.nn.Module:
    """Mask R-CNN v2 with COCO weights and new box/mask predictors for ``data.num_classes``.

    The COCO weights are always loaded first (also before loading a fine-tuned checkpoint),
    because torchvision uses frozen BatchNorm in the backbone only when built with weights.
    ``pretrained=False`` (tests) builds the same architecture without downloading anything.
    """
    from torchvision.models.detection import (
        MaskRCNN_ResNet50_FPN_V2_Weights,
        maskrcnn_resnet50_fpn_v2,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    size = int(cfg["input_size"])
    pred = cfg.get("predict") or {}
    kwargs = {
        "min_size": size,
        "max_size": size,
        "box_detections_per_img": int(cfg["eval"]["max_dets"]),
        "box_score_thresh": float(pred.get("score_thresh", 0.05)),
        "box_nms_thresh": float(pred.get("nms_thresh", 0.5)),
        "trainable_backbone_layers": int(
            (cfg.get("train") or {}).get("trainable_backbone_layers", 3)
        ),
    }
    if pretrained:
        with _hub_dir(resolve_path(cfg["paths"]["pretrained_root"]) / "torch"):
            model = maskrcnn_resnet50_fpn_v2(
                weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1, **kwargs
            )
    else:
        kwargs.pop("trainable_backbone_layers")
        model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None, **kwargs)

    num_classes = int(cfg["data"]["num_classes"])
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, num_classes)
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, 256, num_classes)
    return model


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _use_amp(cfg: Mapping[str, Any], device: torch.device) -> bool:
    return bool(cfg.get("amp", True)) and device.type == "cuda"


# ---------------------------------------------------------------------------------------------
# schedule


def lr_factor(step: int, steps_per_epoch: int, train_cfg: Mapping[str, Any]) -> float:
    """Linear warm-up over ``warmup_iters`` optimizer steps, then step decay at milestones."""
    epoch = step // max(1, steps_per_epoch)
    factor = float(train_cfg.get("lr_gamma", 0.1)) ** sum(
        epoch >= m for m in train_cfg.get("lr_milestones", [])
    )
    warmup = int(train_cfg.get("warmup_iters", 0))
    if step < warmup:
        alpha = step / warmup
        start = float(train_cfg.get("warmup_factor", 0.001))
        factor *= start * (1 - alpha) + alpha
    return factor


# ---------------------------------------------------------------------------------------------
# training


def _atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def _append_csv(path: Path, row: Mapping[str, Any]) -> None:
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in CSV_COLUMNS})


def _smoke_limit(cfg: Mapping[str, Any], key: str) -> int | None:
    return int(cfg["smoke"][key]) if cfg.get("smoke_mode") else None


def train_dataset(cfg: Mapping[str, Any]) -> CocoInstanceDataset:
    return CocoInstanceDataset(
        resolve_path(cfg["eval"]["train_gt"]),
        int(cfg["input_size"]),
        cfg["data"].get("rotation_fixes") or {},
        train=True,
        limit=_smoke_limit(cfg, "max_train_images"),
    )


def validate(cfg: Mapping[str, Any], model: torch.nn.Module, out_path: Path) -> dict[str, Any]:
    """Predict ``valid`` with the current weights and return segm/bbox COCO metrics."""
    limit = _smoke_limit(cfg, "max_eval_images")
    was_training = model.training
    model.eval()
    try:
        export_predictions(
            cfg,
            "valid",
            lambda image, info: predict_image(model, image, info, cfg),
            out_path,
            limit=limit,
            track_vram=False,
        )
    finally:
        model.train(was_training)
    return quick_eval(cfg["eval"]["gt_files"]["valid"], out_path, limit)


def train(cfg: Mapping[str, Any], resume: bool = False) -> Path:
    """Train (or resume) one run. Returns the run directory."""
    smoke = bool(cfg.get("smoke_mode"))
    out = run_dir(cfg)
    last, best = out / "weights" / "last.pt", out / "weights" / "best.pt"
    if resume and not last.is_file():
        raise FileNotFoundError(f"cannot resume: {last} does not exist")
    if not resume and last.is_file() and not smoke:
        raise FileExistsError(f"{last} exists; pass --resume or delete {out} to start over")
    prepare_run_dir(cfg, fresh=smoke and not resume)

    tcfg = dict(cfg["train"])
    epochs = int(tcfg["epochs"])
    max_iters = None
    if smoke:
        epochs = min(epochs, int(cfg["smoke"]["max_epochs"]))
        max_iters = int(cfg["smoke"]["max_iters"])
        tcfg["workers"] = min(int(tcfg.get("workers", 0)), 2)

    seed = int(cfg["seed"])
    seed_everything(seed, bool(cfg.get("deterministic", False)))
    device = _device()
    amp = _use_amp(cfg, device)

    dataset = train_dataset(cfg)
    workers = int(tcfg.get("workers", 0))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(tcfg["batch"]),
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    accumulate = max(1, int(tcfg.get("accumulate", 1)))
    steps_per_epoch = math.ceil(len(loader) / accumulate)

    model = build_model(cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=float(tcfg["lr"]),
        momentum=float(tcfg.get("momentum", 0.9)),
        weight_decay=float(tcfg.get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, steps_per_epoch, tcfg)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    start_epoch, best_ap, iteration = 0, -1.0, 0
    if resume:
        state = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_ap, iteration = state["epoch"] + 1, state["best_ap"], state["iteration"]
        logger.info(
            "Resuming %s at epoch %d (best AP %.4f)", cfg["run_id"], start_epoch + 1, best_ap
        )
        if start_epoch >= epochs:
            logger.info("Run %s already finished %d epochs", cfg["run_id"], epochs)
            return out

    tb = None
    if (cfg.get("logging") or {}).get("tensorboard", True):
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(out))
    log_every = int(tcfg.get("log_every", 20))
    eval_every = max(1, int(tcfg.get("eval_every", 1)))
    logger.info(
        "Training %s: %d images, S=%d, %d epochs, batch %d x accumulate %d, %d steps/epoch, AMP=%s",
        cfg["run_id"], len(dataset), cfg["input_size"], epochs, tcfg["batch"], accumulate,
        steps_per_epoch, amp,
    )  # fmt: skip

    cumulative = resume or not smoke
    with RunTimer(
        cfg["run_id"], cfg.get("time_cap_hours"), gpu_time_log(cfg), resume=cumulative
    ) as timer:
        if timer.remaining() <= 0:
            logger.warning("No time budget left for %s; not training", cfg["run_id"])
            timer.cap_hit = True
            return out
        model.train()
        for epoch in range(start_epoch, epochs):
            t0 = time.monotonic()
            sums = dict.fromkeys(("loss", *LOSS_KEYS), 0.0)
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)
            for i, (images, targets) in enumerate(loader):
                images = [img.to(device, non_blocking=True) for img in images]
                targets = [
                    {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in t.items()}
                    for t in targets
                ]
                with torch.autocast(device.type, enabled=amp):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch + 1} iter {i}: "
                        f"{ {k: float(v) for k, v in loss_dict.items()} }"
                    )
                scaler.scale(loss / accumulate).backward()
                if (i + 1) % accumulate == 0 or i + 1 == len(loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                n_batches += 1
                iteration += 1
                sums["loss"] += float(loss.detach())
                for k in LOSS_KEYS:
                    sums[k] += float(loss_dict[k].detach())
                if tb is not None and iteration % log_every == 0:
                    tb.add_scalar("train/loss", float(loss.detach()), iteration)
                    for k in LOSS_KEYS:
                        tb.add_scalar(f"train/{k}", float(loss_dict[k].detach()), iteration)
                    tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
                if timer.should_stop() or (max_iters is not None and iteration >= max_iters):
                    break

            complete = n_batches == len(loader)
            final = epoch + 1 == epochs or timer.cap_hit or not complete
            metrics = None
            if (epoch + 1) % eval_every == 0 or final:
                metrics = validate(cfg, model, out / "val" / "last.json")
                ap = metrics["segm"]["AP"]
                if ap > best_ap:
                    best_ap = ap
                    _atomic_save(
                        {"model": model.state_dict(), "epoch": epoch, "val": metrics,
                         "run_id": cfg["run_id"]},
                        best,
                    )  # fmt: skip
                    logger.info("New best valid mask AP %.4f at epoch %d", ap, epoch + 1)
            _atomic_save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch if complete else epoch - 1,
                    "best_ap": best_ap,
                    "iteration": iteration,
                    "run_id": cfg["run_id"],
                },
                last,
            )
            row = {
                "epoch": epoch + 1,
                "epoch_complete": complete,
                "iterations": iteration,
                "time_s": round(time.monotonic() - t0, 1),
                "lr": optimizer.param_groups[0]["lr"],
                **{k: round(v / max(1, n_batches), 5) for k, v in sums.items()},
                "best_segm_AP": round(best_ap, 5) if best_ap >= 0 else None,
            }
            if metrics is not None:
                row.update(
                    val_segm_AP=round(metrics["segm"]["AP"], 5),
                    val_segm_AP50=round(metrics["segm"]["AP50"], 5),
                    val_bbox_AP=round(metrics["bbox"]["AP"], 5),
                )
                if tb is not None:
                    tb.add_scalar("val/segm_AP", metrics["segm"]["AP"], epoch + 1)
                    tb.add_scalar("val/bbox_AP", metrics["bbox"]["AP"], epoch + 1)
            _append_csv(out / "results.csv", row)
            logger.info(
                "Epoch %d/%d: loss %.4f, valid mask AP %s, %.0f s",
                epoch + 1, epochs, row["loss"], row.get("val_segm_AP", "-"), row["time_s"],
            )  # fmt: skip
            if final:
                break
    if tb is not None:
        tb.close()
    return out


# ---------------------------------------------------------------------------------------------
# prediction


def load_model(cfg: Mapping[str, Any], weights: str | Path) -> torch.nn.Module:
    """Build the model and load a ``best.pt``/``last.pt`` checkpoint (``model`` key)."""
    device = _device()
    model = build_model(cfg).to(device)
    state = torch.load(weights, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return model


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_info: Mapping[str, Any],
    cfg: Mapping[str, Any],
    model_ms: list[float] | None = None,
) -> list[Detection]:
    """Predict one image (annotation frame, RGB). Masks are returned at the original size.

    If ``model_ms`` is given, the forward-pass time (ms, CUDA-synchronised, including the
    model's own pre/post-processing at size S) is appended.
    """
    device = next(model.parameters()).device
    width, height = image.size
    new_w, new_h, _ = resized_size(width, height, int(cfg["input_size"]))
    small = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)
    tensor = tensor.to(device)
    _sync(device)
    t0 = time.perf_counter()
    with torch.autocast(device.type, enabled=_use_amp(cfg, device)):
        out = model([tensor])[0]
    _sync(device)
    if model_ms is not None:
        model_ms.append(1000 * (time.perf_counter() - t0))
    threshold = float((cfg.get("predict") or {}).get("mask_threshold", 0.5))
    masks = out["masks"][:, 0].float().cpu().numpy()
    labels = out["labels"].cpu().numpy().astype(int)
    scores = out["scores"].float().cpu().numpy()
    return [
        Detection(
            image_info["id"], int(label), float(score), resize_mask(m, height, width, threshold)
        )
        for m, label, score in zip(masks, labels, scores, strict=True)
    ]


def predict(
    cfg: Mapping[str, Any],
    eval_set: str,
    weights: str | Path,
    out_path: str | Path,
    limit: int | None = None,
) -> Path:
    """Export a COCO results JSON for ``eval_set`` with the given checkpoint."""
    model = load_model(cfg, weights)
    model_ms: list[float] = []
    return export_predictions(
        cfg,
        eval_set,
        lambda image, info: predict_image(model, image, info, cfg, model_ms),
        out_path,
        limit=limit,
        meta=lambda: {
            "weights": Path(weights).name,
            "predict_args": dict(cfg.get("predict") or {}),
            "model_ms_per_image": round(float(np.mean(model_ms[1:] or model_ms)), 2)
            if model_ms
            else None,
        },
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
