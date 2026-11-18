"""Mask2Former Swin-T wrapper (Hugging Face transformers): capped, resumable training; predict.

Uses the shared loop in :mod:`iiss.models.torch_loop` (data, AMP, accumulation, cap, resume,
validation, logging). This module supplies the model, optimizer, schedule, loss and inference.

Labels: class index = v5 category id (``num_labels = 75``). Id 0 (``indoor-objects``) has no
annotations, so it is never a training target; its score is zeroed at inference.

Input: the dataset resizes each image to long side ``S``; it is normalised with the ImageNet
mean/std (as ``Mask2FormerImageProcessor``) and zero-padded at the bottom/right to a multiple
of 32, with ``pixel_mask`` marking the real pixels. Target masks are padded the same way.

Optimizer (approximating the official Mask2Former recipe): AdamW, weight decay 0.05 (0 for
norms, biases and embeddings), backbone lr x ``backbone_lr_multiplier`` (0.1), full-model
gradient-norm clipping (0.01), poly schedule (power 0.9) over the fixed ``epochs`` budget,
bf16 autocast by default.

Inference (official instance post-processing, with this project's mask resolution policy):
mask logits are upsampled bilinearly from S/4 to the padded input, cropped to the image (size
S), then ``softmax`` class scores of all query x class pairs are ranked and the top 100 kept;
score = class probability x mean mask probability inside the binary mask. Each kept mask
probability map (at S) is upsampled bilinearly to the original size and thresholded at 0.5.
Hugging Face's ``post_process_instance_segmentation`` is not used: it resizes masks to a fixed
384x384 and then to the target size with nearest-neighbour interpolation.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from PIL import Image

from iiss.data.torch_dataset import resized_size
from iiss.eval.results_json import Detection, resize_mask
from iiss.models.torch_loop import (
    TrainParts,
    amp_dtype_from_cfg,
    device,
    load_checkpoint,
    predict_with_timing,
    run_training,
    sync,
)
from iiss.utils.paths import resolve_path

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIZE_DIVISOR = 32
BACKBONE_PREFIX = "model.pixel_level_module.encoder."
NO_DECAY_KEYS = (
    "level_embed",
    "queries_embedder",
    "queries_features",
    "relative_position_bias_table",
    "position_embeddings",
)


# ---------------------------------------------------------------------------------------------
# model


def _labels(cfg: Mapping[str, Any]) -> dict[int, str]:
    from iiss.data.coco import load_coco_json

    gt = load_coco_json(resolve_path(cfg["eval"]["train_gt"]))
    return {int(c["id"]): c["name"] for c in gt["categories"]}


def build_model(cfg: Mapping[str, Any], pretrained: bool = True) -> torch.nn.Module:
    """Mask2Former with COCO-instance weights and a new 75-class classifier.

    ``pretrained=False`` (tests) builds a small randomly initialised Swin model from
    ``cfg["model_config"]`` overrides without downloading anything.
    """
    from transformers import Mask2FormerConfig, Mask2FormerForUniversalSegmentation

    num_labels = int(cfg["data"]["num_classes"])
    if not pretrained:
        config = Mask2FormerConfig(num_labels=num_labels, **(cfg.get("model_config") or {}))
        return Mask2FormerForUniversalSegmentation(config)
    try:
        id2label = _labels(cfg)
    except FileNotFoundError:
        id2label = {i: str(i) for i in range(num_labels)}
    if len(id2label) != num_labels:
        raise ValueError(f"{len(id2label)} categories but data.num_classes = {num_labels}")
    return Mask2FormerForUniversalSegmentation.from_pretrained(
        cfg["weights"],
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        ignore_mismatched_sizes=True,  # new class head (75 + no-object)
        cache_dir=str(resolve_path(cfg["paths"]["pretrained_root"]) / "hf"),
    )


# ---------------------------------------------------------------------------------------------
# inputs


def pad_batch(
    images: list[torch.Tensor], masks: list[torch.Tensor] | None = None
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor] | None]:
    """Normalise, zero-pad to a common size divisible by 32; returns pixels, pixel_mask, masks."""
    h = max(int(i.shape[1]) for i in images)
    w = max(int(i.shape[2]) for i in images)
    h, w = -(-h // SIZE_DIVISOR) * SIZE_DIVISOR, -(-w // SIZE_DIVISOR) * SIZE_DIVISOR
    dev = images[0].device
    mean = torch.tensor(IMAGENET_MEAN, device=dev).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=dev).view(3, 1, 1)
    pixels = torch.zeros((len(images), 3, h, w), device=dev)
    pixel_mask = torch.zeros((len(images), h, w), dtype=torch.long, device=dev)
    padded_masks = [] if masks is not None else None
    for k, img in enumerate(images):
        ih, iw = img.shape[1:]
        pixels[k, :, :ih, :iw] = (img - mean) / std
        pixel_mask[k, :ih, :iw] = 1
        if masks is not None:
            m = torch.zeros((masks[k].shape[0], h, w), dtype=torch.float32, device=dev)
            m[:, :ih, :iw] = masks[k].float()
            padded_masks.append(m)
    return pixels, pixel_mask, padded_masks


def loss_fn(
    model: torch.nn.Module, images: list[torch.Tensor], targets: list[dict[str, Any]]
) -> dict[str, torch.Tensor]:
    pixels, pixel_mask, masks = pad_batch(images, [t["masks"] for t in targets])
    out = model(
        pixel_values=pixels,
        pixel_mask=pixel_mask,
        mask_labels=masks,
        class_labels=[t["labels"] for t in targets],
    )
    return {"loss": out.loss}


# ---------------------------------------------------------------------------------------------
# optimizer and schedule


def param_groups(model: torch.nn.Module, train_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """AdamW groups: backbone lr multiplier, no weight decay for norms/biases/embeddings."""
    lr = float(train_cfg["lr"])
    wd = float(train_cfg.get("weight_decay", 0.05))
    mult = float(train_cfg.get("backbone_lr_multiplier", 0.1))
    groups: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        backbone = name.startswith(BACKBONE_PREFIX)
        no_decay = p.ndim <= 1 or any(k in name for k in NO_DECAY_KEYS)
        groups.setdefault((backbone, no_decay), []).append(p)
    return [
        {"params": ps, "lr": lr * (mult if bb else 1.0), "weight_decay": 0.0 if nd else wd}
        for (bb, nd), ps in sorted(groups.items())
    ]


def poly_factor(step: int, total_steps: int, train_cfg: Mapping[str, Any]) -> float:
    """``(1 - step/total) ** power`` with an optional linear warm-up."""
    power = float(train_cfg.get("poly_power", 0.9))
    factor = max(0.0, 1.0 - step / max(1, total_steps)) ** power
    warmup = int(train_cfg.get("warmup_iters", 0))
    if step < warmup:
        alpha = step / warmup
        start = float(train_cfg.get("warmup_factor", 1.0))
        factor *= start * (1 - alpha) + alpha
    return factor


def build_parts(cfg: Mapping[str, Any], dev: torch.device, steps_per_epoch: int) -> TrainParts:
    tcfg = cfg["train"]
    model = build_model(cfg).to(dev)
    optimizer = torch.optim.AdamW(param_groups(model, tcfg), lr=float(tcfg["lr"]))
    total = int(tcfg["epochs"]) * steps_per_epoch  # fixed schedule, independent of the cap
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: poly_factor(step, total, tcfg)
    )
    clip = tcfg.get("clip_grad_norm")
    return TrainParts(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        predict_fn=lambda m, image, info: predict_image(m, image, info, cfg),
        amp_dtype=amp_dtype_from_cfg(cfg, dev),
        clip_grad_norm=None if clip is None else float(clip),
    )


def train(cfg: Mapping[str, Any], resume: bool = False) -> Path:
    """Train (or resume) one run with :func:`iiss.models.torch_loop.run_training`."""
    return run_training(cfg, build_parts, resume=resume)


# ---------------------------------------------------------------------------------------------
# prediction


def instances_from_logits(
    class_logits: torch.Tensor,
    mask_logits: torch.Tensor,
    padded_hw: tuple[int, int],
    image_hw: tuple[int, int],
    top_k: int = 100,
    ignore_class: int | None = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Official Mask2Former instance inference for one image, masks kept at input size S.

    Args:
        class_logits: ``(Q, C + 1)``, last column = no-object.
        mask_logits: ``(Q, h, w)`` at 1/4 of the padded input.
        padded_hw: Padded input size; ``image_hw``: real image size inside it (at S).

    Returns:
        ``(mask_probs [K, H_s, W_s], labels [K], scores [K])`` sorted by score.
    """
    masks = F.interpolate(
        mask_logits[None].float(), size=padded_hw, mode="bilinear", align_corners=False
    )[0, :, : image_hw[0], : image_hw[1]]
    scores = class_logits.float().softmax(-1)[:, :-1]
    if ignore_class is not None:
        scores[:, ignore_class] = 0.0
    num_queries, num_classes = scores.shape
    k = min(top_k, num_queries * num_classes)
    top_scores, idx = scores.flatten().topk(k)
    labels = idx % num_classes
    queries = torch.div(idx, num_classes, rounding_mode="floor")
    probs = masks[queries].sigmoid()
    binary = (masks[queries] > 0).float()
    mask_scores = (probs * binary).flatten(1).sum(1) / (binary.flatten(1).sum(1) + 1e-6)
    final = top_scores * mask_scores
    order = final.argsort(descending=True)
    return probs[order], labels[order], final[order]


@torch.inference_mode()
def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    image_info: Mapping[str, Any],
    cfg: Mapping[str, Any],
    model_ms: list[float] | None = None,
) -> list[Detection]:
    """Predict one image (annotation frame, RGB). Masks are returned at the original size."""
    dev = next(model.parameters()).device
    width, height = image.size
    new_w, new_h, _ = resized_size(width, height, int(cfg["input_size"]))
    small = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(small, dtype=np.float32) / 255.0).permute(2, 0, 1)
    pixels, pixel_mask, _ = pad_batch([tensor.to(dev)])
    amp_dtype = amp_dtype_from_cfg(cfg, dev)
    sync(dev)
    t0 = time.perf_counter()
    with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        out = model(pixel_values=pixels, pixel_mask=pixel_mask)
    probs, labels, scores = instances_from_logits(
        out.class_queries_logits[0],
        out.masks_queries_logits[0],
        tuple(pixels.shape[2:]),
        (new_h, new_w),
        top_k=int(cfg["eval"]["max_dets"]),
    )
    sync(dev)
    if model_ms is not None:
        model_ms.append(1000 * (time.perf_counter() - t0))
    pred_cfg = cfg.get("predict") or {}
    threshold = float(pred_cfg.get("mask_threshold", 0.5))
    min_score = float(pred_cfg.get("score_thresh", 0.0))
    dets = []
    for p, label, score in zip(
        probs.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy(), strict=True
    ):
        if score < min_score:
            continue
        mask = resize_mask(p, height, width, threshold)  # bilinear from S, then >= 0.5
        dets.append(Detection(image_info["id"], int(label), float(score), mask))
    return dets


def load_model(cfg: Mapping[str, Any], weights: str | Path) -> torch.nn.Module:
    """Build the model and load a ``last.pt``/``best.pt`` checkpoint."""
    return load_checkpoint(build_model(cfg).to(device()), weights)


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
