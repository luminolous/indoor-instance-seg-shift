"""VRAM and speed probe: a few real training iterations per model x S x batch size.

Used by ``notebooks/vram_probe.ipynb`` (run by the user on the GPU) to pick a batch size per
run before the 2-hour training runs, and to estimate whether the epoch budget fits the cap.

For every combination the probe

1. trains ``iters`` iterations on shuffled ``train`` images with the run's real settings
   (AMP, optimizer, augmentation, data loader) and records the peak CUDA memory and the
   seconds per iteration (the first ``warmup`` iterations are not timed);
2. predicts ``infer_images`` ``valid`` images one at a time and records ms per image and the
   inference peak memory.

Peak memory is ``torch.cuda.max_memory_reserved`` (what ``nvidia-smi`` roughly shows). A
combination "fits" if it does not run out of memory and its training peak is at most
``budget_mb``. YOLO: Ultralytics halves the batch automatically after an out-of-memory error;
the probe reports that as ``oom`` instead of hiding it.

Estimated training time = epochs x iterations per epoch x seconds per iteration + validation
time (valid images x inference ms, once per validation). It is an estimate for planning only.
"""

from __future__ import annotations

import gc
import importlib
import itertools
import logging
import math
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from iiss.data.coco import load_coco_json
from iiss.data.images import load_image
from iiss.models.common import eval_images
from iiss.utils.config import deep_merge
from iiss.utils.paths import resolve_path

logger = logging.getLogger(__name__)

DEFAULT_BUDGET_MB = 5600  # 6 GB card: leave ~0.5 GB for the display and fragmentation


@dataclass
class ProbeResult:
    model: str
    input_size: int
    batch: int
    iters_timed: int
    oom: bool
    peak_train_mb: float | None
    s_per_iter: float | None
    infer_ms: float | None
    peak_infer_mb: float | None
    note: str = ""


# ---------------------------------------------------------------------------------------------
# helpers


def _cuda_reset() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _peak_mb() -> float | None:
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    return round(torch.cuda.max_memory_reserved() / 2**20, 1)


def _valid_images(cfg: Mapping[str, Any], n: int) -> list[tuple[Any, dict[str, Any]]]:
    gt_file = resolve_path(cfg["eval"]["gt_files"]["valid"])
    gt = load_coco_json(gt_file)
    fixes = cfg["data"].get("rotation_fixes") or {}
    return [
        (load_image(gt_file.parent / info["file_name"], info, fixes), info)
        for info in eval_images(gt, n)
    ]


def _time_inference(predict, images: list[tuple[Any, dict[str, Any]]]) -> tuple[float, float]:
    """Mean ms per image (first image = warm-up) and peak MB of ``predict(image, info)``."""
    _cuda_reset()
    times = []
    for image, info in images:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        predict(image, info)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    timed = times[1:] or times
    return round(1000 * sum(timed) / len(timed), 1), _peak_mb()


def _with_batch(cfg: Mapping[str, Any], batch: int) -> dict[str, Any]:
    return deep_merge(cfg, {"train": {"batch": int(batch)}})


# ---------------------------------------------------------------------------------------------
# torch models (Mask R-CNN, Mask2Former)


def probe_torch(
    cfg: Mapping[str, Any], batch: int, iters: int, warmup: int, infer_images: int
) -> ProbeResult:
    from iiss.models import torch_loop

    module = importlib.import_module(f"iiss.models.{cfg['model']}")
    cfg = _with_batch(cfg, batch)
    _, lr = torch_loop.effective_lr(cfg["train"])
    cfg = deep_merge(cfg, {"train": {"lr": lr}})
    dev = torch_loop.device()
    result = ProbeResult(
        cfg["model"], int(cfg["input_size"]), batch, 0, False, None, None, None, None
    )
    _cuda_reset()
    parts = None
    try:
        dataset = torch_loop.train_dataset(cfg)
        workers = int(cfg["train"].get("workers", 0))
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch,
            shuffle=True,
            num_workers=workers,
            collate_fn=torch_loop.collate,
            pin_memory=dev.type == "cuda",
            generator=torch.Generator().manual_seed(int(cfg["seed"])),
        )
        parts = module.build_parts(cfg, dev, max(1, len(loader)))
        amp = parts.amp_dtype if dev.type == "cuda" else None
        scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
        parts.model.train()
        t_start = None
        batches = itertools.chain.from_iterable(itertools.repeat(loader))
        for i, (images, targets) in enumerate(itertools.islice(batches, iters)):
            if i == warmup:
                torch_loop.sync(dev)
                t_start = time.perf_counter()
            images = [img.to(dev) for img in images]
            targets = [
                {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in t.items()} for t in targets
            ]
            with torch.autocast(dev.type, dtype=amp, enabled=amp is not None):
                loss = parts.loss_fn(parts.model, images, targets)["loss"]
            scaler.scale(loss).backward()
            if parts.clip_grad_norm is not None:
                scaler.unscale_(parts.optimizer)
                torch.nn.utils.clip_grad_norm_(parts.model.parameters(), parts.clip_grad_norm)
            scaler.step(parts.optimizer)
            scaler.update()
            parts.optimizer.zero_grad(set_to_none=True)
        torch_loop.sync(dev)
        result.peak_train_mb = _peak_mb()
        if t_start is not None:
            result.iters_timed = iters - warmup
            result.s_per_iter = round((time.perf_counter() - t_start) / result.iters_timed, 3)

        parts.optimizer.zero_grad(set_to_none=True)
        parts.model.eval()
        images = _valid_images(cfg, infer_images)
        net, predict_fn = parts.model, parts.predict_fn
        result.infer_ms, result.peak_infer_mb = _time_inference(
            lambda image, info: predict_fn(net, image, info), images
        )
        net = predict_fn = None
    except torch.OutOfMemoryError as exc:
        result.oom = True
        result.peak_train_mb = _peak_mb()
        result.note = f"CUDA out of memory: {str(exc).splitlines()[0][:120]}"
        logger.warning("%s S=%d batch=%d: out of memory", cfg["model"], cfg["input_size"], batch)
    finally:
        parts = None  # release the model and optimizer before the next probe
        _cuda_reset()
    return result


# ---------------------------------------------------------------------------------------------
# YOLO (Ultralytics trainer)


def probe_yolo(
    cfg: Mapping[str, Any], batch: int, iters: int, warmup: int, infer_images: int
) -> ProbeResult:
    """Run the Ultralytics trainer and stop it after ``iters`` batches via a callback.

    After the stop Ultralytics still validates once on ``valid`` (it always does when stopped);
    that time is not part of the measurement. Outputs go to ``runs/vram_probe/``.
    """
    from ultralytics import YOLO

    from iiss.models import yolo

    cfg = _with_batch(cfg, batch)
    result = ProbeResult("yolo", int(cfg["input_size"]), batch, 0, False, None, None, None, None)
    probe_root = resolve_path(cfg["paths"]["runs_root"]) / "vram_probe"
    data_yaml = yolo.data_yaml_path(cfg)
    if cfg.get("smoke_mode"):
        limits = {split: int(cfg["smoke"][key]) for split, key in yolo.SMOKE_SPLITS.items()}
        data_yaml = yolo.make_smoke_dataset(data_yaml, probe_root / "smoke_data", limits)

    state: dict[str, Any] = {"times": [], "batch_seen": None, "peak": None}

    def on_batch_end(trainer: Any) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        state["times"].append(time.perf_counter())
        state["batch_seen"] = trainer.batch_size
        if len(state["times"]) >= iters and not trainer.stop:
            state["peak"] = _peak_mb()
            trainer.stop = True

    weights = yolo.pretrained_weights(cfg)
    weights.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    model.add_callback("on_train_batch_end", on_batch_end)
    kwargs = yolo.train_kwargs(cfg, data_yaml)
    kwargs.update(
        epochs=1,
        project=str(probe_root),
        name=f"{cfg['run_id']}_b{batch}",
        plots=False,
        save=False,
        verbose=False,
        close_mosaic=0,
    )
    _cuda_reset()
    try:
        model.train(**kwargs)
    except torch.OutOfMemoryError as exc:
        result.oom = True
        result.note = f"CUDA out of memory: {str(exc).splitlines()[0][:120]}"
    times = state["times"]
    result.peak_train_mb = state["peak"] or _peak_mb()
    if state["batch_seen"] is not None and int(state["batch_seen"]) != batch:
        result.oom = True
        result.note = f"Ultralytics reduced the batch to {state['batch_seen']} after an OOM"
    if len(times) > warmup + 1:
        timed = times[warmup:]
        result.iters_timed = len(timed) - 1
        result.s_per_iter = round((timed[-1] - timed[0]) / result.iters_timed, 3)
    elif not result.oom:
        result.note = f"only {len(times)} iterations ran (dataset smaller than iters x batch?)"
    del model
    _cuda_reset()

    if not result.oom:
        net = yolo.load_model(weights)  # COCO head; speed is close to the 75-class head
        images = _valid_images(cfg, infer_images)
        result.infer_ms, result.peak_infer_mb = _time_inference(
            lambda image, info: yolo.predict_image(net, image, info, cfg), images
        )
        del net
        _cuda_reset()
    return result


# ---------------------------------------------------------------------------------------------
# driver and summary


def probe(
    cfg: Mapping[str, Any], batch: int, iters: int = 20, warmup: int = 3, infer_images: int = 10
) -> ProbeResult:
    """Probe one model x S x batch (``cfg`` = resolved run config)."""
    fn = probe_yolo if cfg["model"] == "yolo" else probe_torch
    logger.info("Probing %s S=%s batch=%d", cfg["model"], cfg["input_size"], batch)
    t0 = time.monotonic()
    res = fn(cfg, batch, iters, warmup, infer_images)
    logger.info("-> %s (%.0f s)", asdict(res), time.monotonic() - t0)
    return res


def probe_matrix(
    configs: Iterable[Mapping[str, Any]],
    batches: Mapping[str, Iterable[int]],
    out_csv: str | Path | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Probe every config with every candidate batch of its model; append rows to ``out_csv``.

    Batches are tried from small to large; once one runs out of memory the larger ones for
    that model x S are skipped (recorded as ``oom`` with a note).
    """
    rows = []
    for cfg in configs:
        oom_seen = False
        for batch in sorted(batches[cfg["model"]]):
            if oom_seen:
                res = ProbeResult(
                    cfg["model"], int(cfg["input_size"]), batch, 0, True,
                    None, None, None, None, "skipped: a smaller batch ran out of memory",
                )  # fmt: skip
            else:
                res = probe(cfg, batch, **kwargs)
                oom_seen = res.oom
            row = asdict(res)
            if out_csv is not None:
                path = resolve_path(out_csv)
                path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([row]).to_csv(path, mode="a", header=not path.is_file(), index=False)
            rows.append(row)
    return pd.DataFrame(rows)


def estimate_hours(
    cfg: Mapping[str, Any], batch: int, s_per_iter: float, infer_ms: float | None
) -> float:
    """Planned training hours for the run's epoch budget (train + periodic validation)."""
    tcfg = cfg["train"]
    n_train = len(load_coco_json(resolve_path(cfg["eval"]["train_gt"]))["images"])
    n_valid = len(load_coco_json(resolve_path(cfg["eval"]["gt_files"]["valid"]))["images"])
    epochs = int(tcfg["epochs"])
    train_s = epochs * math.ceil(n_train / batch) * s_per_iter
    every = 1 if cfg["model"] == "yolo" else max(1, int(tcfg.get("eval_every", 1)))
    val_s = math.ceil(epochs / every) * n_valid * (infer_ms or 0.0) / 1000
    return round((train_s + val_s) / 3600, 2)


def recommend(
    results: pd.DataFrame,
    configs: Mapping[str, Mapping[str, Any]],
    budget_mb: float = DEFAULT_BUDGET_MB,
    cap_hours: float = 2.0,
) -> pd.DataFrame:
    """Per run: batch, accumulation and estimated training time against the cap.

    The configured effective batch (``batch * accumulate`` of the model config; ``nbs`` for
    YOLO) is kept: the recommended batch is the largest fitting batch not above it, and
    ``accumulate`` makes up the rest. ``max_fit_batch`` is the largest fitting batch overall,
    for deciding whether to raise the effective batch of a model (for all S together).

    ``configs`` maps ``run_id`` -> resolved run config.
    """
    rows = []
    for run_id, cfg in configs.items():
        sub = results[
            (results["model"] == cfg["model"]) & (results["input_size"] == cfg["input_size"])
        ]
        ok = sub[(~sub["oom"]) & (sub["peak_train_mb"] <= budget_mb)]
        tcfg = cfg["train"]
        target = int(tcfg["batch"]) * max(1, int(tcfg.get("accumulate", 1)))
        if cfg["model"] == "yolo":
            target = int(tcfg.get("nbs", 64))  # Ultralytics accumulates to nbs by itself
        if ok.empty:
            rows.append({"run_id": run_id, "batch": None, "accumulate": None,
                         "peak_train_mb": None, "est_train_h": None, "cap_risk": None,
                         "note": "no probed batch fits: try a smaller batch or Colab"})  # fmt: skip
            continue
        max_fit = int(ok["batch"].max())
        usable = ok[ok["batch"] <= target]
        if usable.empty:
            usable = ok[ok["batch"] == ok["batch"].min()]
        best = usable.sort_values("batch").iloc[-1]
        batch = int(best["batch"])
        accumulate = 1 if cfg["model"] == "yolo" else max(1, math.ceil(target / batch))
        hours = estimate_hours(cfg, batch, float(best["s_per_iter"]), best["infer_ms"])
        rows.append(
            {
                "run_id": run_id,
                "batch": batch,
                "accumulate": accumulate,
                "effective_batch": target if cfg["model"] == "yolo" else batch * accumulate,
                "max_fit_batch": max_fit,
                "peak_train_mb": best["peak_train_mb"],
                "s_per_iter": best["s_per_iter"],
                "infer_ms": best["infer_ms"],
                "est_train_h": hours,
                "cap_risk": hours > cap_hours,
                "note": ""
                if cfg["model"] == "yolo" or target % batch == 0
                else f"effective batch {batch * accumulate} != configured {target}",
            }
        )
    return pd.DataFrame(rows)
