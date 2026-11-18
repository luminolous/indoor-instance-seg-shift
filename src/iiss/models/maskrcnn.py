"""Mask R-CNN R50-FPN v2 wrapper (torchvision): train with the 2-hour cap and resume, predict.

Input size: the dataset resizes every image so its longest side is ``S``, and the model is
built with ``min_size=max_size=S`` so ``GeneralizedRCNNTransform`` keeps that size (scale 1).

Run directory (``<runs_root>/<run_id>/``)::

    config.yaml, env.json      # resolved project config and environment
    results.csv                # one row per epoch: losses, lr, valid segm/bbox AP
    events.out.tfevents.*      # TensorBoard (if logging.tensorboard)
    weights/last.pt            # full training state after the last epoch: resume, and the
                               # checkpoint that is evaluated (fixed schedule)
    weights/best.pt            # weights with the best valid mask AP; kept and logged only
    val/last.json              # predictions of the latest validation

Training uses the shared loop in :mod:`iiss.models.torch_loop` (SGD, warm-up + step decay,
fp16 AMP). Time cap: :class:`~iiss.utils.timer.RunTimer` is checked after every iteration.
When it is reached the epoch stops, the model is validated, ``last.pt``/``best.pt`` are written and
training ends (the session is logged with ``cap_hit=True``). A resumed run restarts the epoch
that was interrupted, from the state saved after the previous epoch.

Learning rate: ``train.base_lr`` is defined for ``train.base_batch`` images per optimizer step
and scaled linearly to the effective batch ``batch * accumulate``. The scaled ``lr`` and
``effective_batch`` are written into the run's ``config.yaml``.

Prediction: the image is resized to long side ``S``, masks come out at that size and are
upsampled bilinearly to the original resolution before thresholding at 0.5. This avoids
pasting up to 100 full-resolution float masks on the GPU (about 5 GB for a 12 MP test photo).
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from iiss.data.torch_dataset import resized_size
from iiss.eval.results_json import Detection, resize_mask
from iiss.models.torch_loop import (
    TrainParts,
    amp_dtype_from_cfg,
    device,
    effective_lr,  # noqa: F401  (re-exported for tests and notebooks)
    load_checkpoint,
    predict_with_timing,
    run_training,
    sync,
)
from iiss.utils.paths import resolve_path

logger = logging.getLogger(__name__)

LOSS_KEYS = ("loss_classifier", "loss_box_reg", "loss_mask", "loss_objectness", "loss_rpn_box_reg")


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


def loss_fn(
    model: torch.nn.Module, images: list[torch.Tensor], targets: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    losses = model(images, targets)
    return {"loss": sum(losses.values()), **losses}


def build_parts(cfg: Mapping[str, Any], dev: torch.device, steps_per_epoch: int) -> TrainParts:
    tcfg = cfg["train"]
    model = build_model(cfg).to(dev)
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
    return TrainParts(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        predict_fn=lambda m, image, info: predict_image(m, image, info, cfg),
        loss_keys=LOSS_KEYS,
        amp_dtype=amp_dtype_from_cfg(cfg, dev),
    )


def train(cfg: Mapping[str, Any], resume: bool = False) -> Path:
    """Train (or resume) one run with :func:`iiss.models.torch_loop.run_training`."""
    return run_training(cfg, build_parts, resume=resume)


# ---------------------------------------------------------------------------------------------
# prediction


def load_model(cfg: Mapping[str, Any], weights: str | Path) -> torch.nn.Module:
    """Build the model and load a ``last.pt``/``best.pt`` checkpoint."""
    return load_checkpoint(build_model(cfg).to(device()), weights)


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
    dev = next(model.parameters()).device
    width, height = image.size
    new_w, new_h, _ = resized_size(width, height, int(cfg["input_size"]))
    small = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)
    tensor = tensor.to(dev)
    amp_dtype = amp_dtype_from_cfg(cfg, dev)
    sync(dev)
    t0 = time.perf_counter()
    with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model([tensor])[0]
    sync(dev)
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
    return predict_with_timing(
        cfg,
        eval_set,
        model,
        lambda m, image, info, ms: predict_image(m, image, info, cfg, ms),
        weights,
        out_path,
        limit=limit,
    )
