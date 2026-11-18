"""Training loop shared by the plain-PyTorch wrappers (Mask R-CNN, Mask2Former).

A wrapper supplies a :class:`TrainParts` (model, optimizer, scheduler, loss and predict
functions); this module handles everything that must be identical across models:

- data: :class:`~iiss.data.torch_dataset.CocoInstanceDataset` (long side ``S``, rotation fixes,
  hflip), seeded shuffling, smoke limits;
- AMP (fp16 with ``GradScaler``, or bf16 without), gradient accumulation, optional gradient
  clipping, non-finite loss check;
- the 2-hour cap (checked after every iteration), resume from ``weights/last.pt`` (full state
  after the last finished epoch), atomic checkpoint writes;
- validation on ``valid`` every ``train.eval_every`` epochs with pycocotools mask AP;
  ``best.pt`` is kept and logged, ``last.pt`` is the evaluated checkpoint;
- ``results.csv`` (one row per epoch) and TensorBoard.

Learning rate: ``train.base_lr`` is defined for ``train.base_batch`` images per optimizer step
and scaled linearly to ``batch * accumulate``; the scaled ``lr`` and ``effective_batch`` are
written into the run's ``config.yaml``.
"""

from __future__ import annotations

import csv
import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from iiss.data.torch_dataset import CocoInstanceDataset, collate
from iiss.eval.results_json import Detection
from iiss.models.common import (
    export_predictions,
    gpu_time_log,
    prepare_run_dir,
    quick_eval,
    run_dir,
)
from iiss.utils.config import deep_merge
from iiss.utils.paths import resolve_path
from iiss.utils.seed import seed_everything
from iiss.utils.timer import RunTimer

logger = logging.getLogger(__name__)

LossFn = Callable[[torch.nn.Module, list[torch.Tensor], list[dict[str, Any]]], dict[str, Any]]
PredictFn = Callable[[torch.nn.Module, Image.Image, Mapping[str, Any]], list[Detection]]

VAL_COLUMNS = ("val_segm_AP", "val_segm_AP50", "val_bbox_AP", "best_segm_AP")


@dataclass
class TrainParts:
    """Model-specific pieces of a training run.

    ``loss_fn`` returns a dict with the total ``"loss"`` (the tensor that is back-propagated)
    plus the components named in ``loss_keys`` (logged only).
    """

    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    loss_fn: LossFn
    predict_fn: PredictFn
    loss_keys: tuple[str, ...] = ()
    amp_dtype: torch.dtype | None = torch.float16
    clip_grad_norm: float | None = None


def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def amp_dtype_from_cfg(cfg: Mapping[str, Any], dev: torch.device) -> torch.dtype | None:
    """``None`` (no autocast) unless ``amp`` is on and the device is CUDA."""
    if not bool(cfg.get("amp", True)) or dev.type != "cuda":
        return None
    name = str((cfg.get("train") or {}).get("amp_dtype", "float16"))
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)


def effective_lr(train_cfg: Mapping[str, Any]) -> tuple[int, float]:
    """``(effective_batch, lr)`` with ``base_lr`` scaled linearly from ``base_batch``."""
    effective = int(train_cfg["batch"]) * max(1, int(train_cfg.get("accumulate", 1)))
    base_batch = int(train_cfg.get("base_batch", effective))
    return effective, float(train_cfg["base_lr"]) * effective / base_batch


def smoke_limit(cfg: Mapping[str, Any], key: str) -> int | None:
    return int(cfg["smoke"][key]) if cfg.get("smoke_mode") else None


def train_dataset(cfg: Mapping[str, Any]) -> CocoInstanceDataset:
    return CocoInstanceDataset(
        resolve_path(cfg["eval"]["train_gt"]),
        int(cfg["input_size"]),
        cfg["data"].get("rotation_fixes") or {},
        train=True,
        limit=smoke_limit(cfg, "max_train_images"),
    )


def atomic_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def append_csv(path: Path, row: Mapping[str, Any], columns: tuple[str, ...]) -> None:
    new = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in columns})


def csv_columns(loss_keys: tuple[str, ...]) -> tuple[str, ...]:
    return (
        "epoch",
        "epoch_complete",
        "iterations",
        "time_s",
        "lr",
        "loss",
        *loss_keys,
        *VAL_COLUMNS,
    )


def validate(
    cfg: Mapping[str, Any], model: torch.nn.Module, predict_fn: PredictFn, out_path: Path
) -> dict[str, Any]:
    """Predict ``valid`` with the current weights and return segm/bbox COCO metrics."""
    limit = smoke_limit(cfg, "max_eval_images")
    was_training = model.training
    model.eval()
    try:
        export_predictions(
            cfg,
            "valid",
            lambda image, info: predict_fn(model, image, info),
            out_path,
            limit=limit,
            track_vram=False,
        )
    finally:
        model.train(was_training)
    return quick_eval(cfg["eval"]["gt_files"]["valid"], out_path, limit)


def run_training(
    cfg: Mapping[str, Any],
    build_parts: Callable[[Mapping[str, Any], torch.device, int], TrainParts],
    resume: bool = False,
) -> Path:
    """Train (or resume) one run. ``build_parts(cfg, device, steps_per_epoch)`` builds the
    model-specific parts (the scheduler needs the number of optimizer steps per epoch).
    Returns the run directory.
    """
    smoke = bool(cfg.get("smoke_mode"))
    out = run_dir(cfg)
    last, best = out / "weights" / "last.pt", out / "weights" / "best.pt"
    if resume and not last.is_file():
        raise FileNotFoundError(f"cannot resume: {last} does not exist")
    if not resume and last.is_file() and not smoke:
        raise FileExistsError(f"{last} exists; pass --resume or delete {out} to start over")
    effective_batch, lr = effective_lr(cfg["train"])
    cfg = deep_merge(cfg, {"train": {"effective_batch": effective_batch, "lr": lr}})
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
    dev = device()

    dataset = train_dataset(cfg)
    workers = int(tcfg.get("workers", 0))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=int(tcfg["batch"]),
        shuffle=True,
        num_workers=workers,
        collate_fn=collate,
        pin_memory=dev.type == "cuda",
        persistent_workers=workers > 0,
        generator=torch.Generator().manual_seed(seed),
    )
    accumulate = max(1, int(tcfg.get("accumulate", 1)))
    steps_per_epoch = math.ceil(len(loader) / accumulate)

    parts = build_parts(cfg, dev, steps_per_epoch)
    model, optimizer, scheduler = parts.model, parts.optimizer, parts.scheduler
    amp_dtype = parts.amp_dtype if dev.type == "cuda" else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    columns = csv_columns(parts.loss_keys)

    start_epoch, best_ap, iteration = 0, -1.0, 0
    if resume:
        state = torch.load(last, map_location=dev, weights_only=False)
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
        "Training %s: %d images, S=%d, %d epochs, batch %d x accumulate %d (lr %.3g), "
        "%d steps/epoch, AMP=%s",
        cfg["run_id"], len(dataset), cfg["input_size"], epochs, tcfg["batch"], accumulate,
        tcfg["lr"], steps_per_epoch, amp_dtype,
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
            sums = dict.fromkeys(("loss", *parts.loss_keys), 0.0)
            n_batches = 0
            optimizer.zero_grad(set_to_none=True)
            for i, (images, targets) in enumerate(loader):
                images = [img.to(dev, non_blocking=True) for img in images]
                targets = [
                    {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()}
                    for t in targets
                ]
                with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                    losses = parts.loss_fn(model, images, targets)
                loss = losses["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at epoch {epoch + 1} iter {i}: "
                        f"{ {k: float(v) for k, v in losses.items()} }"
                    )
                scaler.scale(loss / accumulate).backward()
                if (i + 1) % accumulate == 0 or i + 1 == len(loader):
                    if parts.clip_grad_norm is not None:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), parts.clip_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    scheduler.step()

                n_batches += 1
                iteration += 1
                values = {k: float(losses[k].detach()) for k in ("loss", *parts.loss_keys)}
                for k, v in values.items():
                    sums[k] += v
                if tb is not None and iteration % log_every == 0:
                    for k, v in values.items():
                        tb.add_scalar(f"train/{k}", v, iteration)
                    tb.add_scalar("train/lr", optimizer.param_groups[0]["lr"], iteration)
                if timer.should_stop() or (max_iters is not None and iteration >= max_iters):
                    break

            complete = n_batches == len(loader)
            final = epoch + 1 == epochs or timer.cap_hit or not complete
            metrics = None
            if (epoch + 1) % eval_every == 0 or final:
                metrics = validate(cfg, model, parts.predict_fn, out / "val" / "last.json")
                ap = metrics["segm"]["AP"]
                if ap > best_ap:
                    best_ap = ap
                    atomic_save(
                        {"model": model.state_dict(), "epoch": epoch, "val": metrics,
                         "run_id": cfg["run_id"]},
                        best,
                    )  # fmt: skip
                    logger.info("New best valid mask AP %.4f at epoch %d", ap, epoch + 1)
            atomic_save(
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
            append_csv(out / "results.csv", row, columns)
            logger.info(
                "Epoch %d/%d: loss %.4f, valid mask AP %s, %.0f s",
                epoch + 1, epochs, row["loss"], row.get("val_segm_AP", "-"), row["time_s"],
            )  # fmt: skip
            if final:
                break
    if tb is not None:
        tb.close()
    return out


def load_checkpoint(model: torch.nn.Module, weights: str | Path) -> torch.nn.Module:
    """Load a ``last.pt``/``best.pt`` (``model`` key) into ``model`` and set eval mode."""
    dev = next(model.parameters()).device
    state = torch.load(weights, map_location=dev, weights_only=False)
    model.load_state_dict(state["model"])
    return model.eval()


def predict_with_timing(
    cfg: Mapping[str, Any],
    eval_set: str,
    model: torch.nn.Module,
    predict_fn: Callable[..., list[Detection]],
    weights: str | Path,
    out_path: str | Path,
    limit: int | None = None,
) -> Path:
    """Export predictions; ``predict_fn(model, image, info, model_ms)`` appends forward ms."""
    model_ms: list[float] = []
    return export_predictions(
        cfg,
        eval_set,
        lambda image, info: predict_fn(model, image, info, model_ms),
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
