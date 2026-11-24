"""Predict and score many runs on many eval sets (used by ``predict_and_evaluate.ipynb``).

One job = one run id on one eval set: export the predictions of ``weights/last.pt`` (as
``iiss.predict``), then score them (as ``iiss.evaluate``). Jobs run one after another; a failing
job is recorded and does not stop the others.

The whole batch can be stopped and started again: finished predictions are reused, and scoring
always runs again (so a changed GT or evaluator never leaves old numbers behind). Predictions are
reused only if their sidecar says they are complete and they are newer than the checkpoint.

Phone photos are additionally scored per capture condition (``eval.image_groups``). Each
condition's AP is the mean over the classes that have GT in that condition, and the class sets
differ, so the group APs are not directly comparable. ``n_categories`` and the per-class CSV of
every group are written to make a same-class comparison possible.

With ``smoke=True``: S=320 checkpoints, 8 images per set, everything written to ``runs/smoke/``.
"""

from __future__ import annotations

import gc
import json
import logging
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from iiss.cli_common import model_module
from iiss.data.coco import load_coco_json
from iiss.evaluate import evaluate_files
from iiss.models.common import apply_smoke, eval_images, predictions_path, run_dir
from iiss.utils.config import load_run_config
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100")


@dataclass(frozen=True)
class Job:
    model: str
    size: int
    eval_set: str
    seed: int = 42

    @property
    def run_id(self) -> str:
        return f"{self.model}_s{self.size}_seed{self.seed}"


def list_jobs(
    models: Iterable[str], sizes: Iterable[int], eval_sets: Iterable[str], seed: int = 42
) -> list[Job]:
    """All combinations, grouped by run so each checkpoint's sets come together."""
    sizes, eval_sets = list(sizes), list(eval_sets)
    return [Job(m, s, e, seed) for m in models for s in sizes for e in eval_sets]


def job_config(job: Job, smoke: bool = False) -> dict[str, Any]:
    """Resolved config of the job's run."""
    cfg = load_run_config(repo_root() / "configs" / "runs" / f"{job.run_id}.yaml")
    if smoke and job.size != int(cfg["smoke"]["input_size"]):
        raise ValueError(f"smoke mode allows only S={cfg['smoke']['input_size']}, got {job.size}")
    return apply_smoke(cfg) if smoke else cfg


def job_weights(job: Job) -> Path:
    """``last.pt`` of the real run (also in smoke mode)."""
    return run_dir(job_config(job)) / "weights" / "last.pt"


def predictions_ready(pred: Path, weights: Path, n_images: int) -> bool:
    """True if ``pred`` is a complete export that is newer than the checkpoint."""
    meta_path = pred.with_name(f"{pred.stem}.meta.json")
    if not (pred.is_file() and meta_path.is_file() and weights.is_file()):
        return False
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return (
        meta.get("images_limited_to") is None
        and not meta.get("smoke", False)
        and meta.get("num_images_predicted") == n_images
        and meta.get("num_images_gt") == n_images
        and meta.get("weights") == weights.name
        and meta_path.stat().st_mtime >= weights.stat().st_mtime
    )


def status_table(jobs: Iterable[Job]) -> pd.DataFrame:
    """What exists for every job: checkpoint, finished predictions, evaluation."""
    rows = []
    n_images: dict[str, int | None] = {}
    for job in jobs:
        cfg = job_config(job)
        gt_file = resolve_path(cfg["eval"]["gt_files"][job.eval_set])
        if job.eval_set not in n_images:
            n_images[job.eval_set] = (
                len(load_coco_json(gt_file)["images"]) if gt_file.is_file() else None
            )
        n = n_images[job.eval_set]
        weights = job_weights(job)
        pred = predictions_path(cfg, job.eval_set)
        rows.append(
            {
                "run_id": job.run_id,
                "eval_set": job.eval_set,
                "images": n,
                "checkpoint": weights.is_file(),
                "predictions": n is not None and predictions_ready(pred, weights, n),
                "evaluated": (_eval_dir(cfg, job) / "metrics.json").is_file(),
            }
        )
    return pd.DataFrame(rows)


def _eval_dir(cfg: Mapping[str, Any], job: Job) -> Path:
    if cfg.get("smoke_mode"):
        return resolve_path(cfg["paths"]["runs_root"]) / "eval" / job.run_id / job.eval_set
    return resolve_path(cfg["eval"]["results_root"]) / job.run_id / job.eval_set


def _gt_for_scoring(cfg: Mapping[str, Any], job: Job, limit: int | None) -> Path:
    """The GT file, or in smoke mode a copy restricted to the predicted images."""
    gt_file = resolve_path(cfg["eval"]["gt_files"][job.eval_set])
    if limit is None:
        return gt_file
    gt = load_coco_json(gt_file)
    images = eval_images(gt, limit)
    keep = {img["id"] for img in images}
    subset = {
        **gt,
        "images": images,
        "annotations": [a for a in gt["annotations"] if a["image_id"] in keep],
    }
    out = _eval_dir(cfg, job) / "gt_subset.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(subset), encoding="utf-8")
    return out


def _release_gpu() -> None:
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_job(job: Job, smoke: bool = False, overwrite: bool = False) -> dict[str, Any]:
    """Export (or reuse) predictions and score them. Returns one result row."""
    cfg = job_config(job, smoke)
    weights = job_weights(job)
    if not weights.is_file():
        raise FileNotFoundError(f"checkpoint not found: {weights}")
    gt_file = resolve_path(cfg["eval"]["gt_files"][job.eval_set])
    if not gt_file.is_file():
        raise FileNotFoundError(f"ground truth not found: {gt_file}")
    limit = int(cfg["smoke"]["max_eval_images"]) if smoke else None
    n_images = len(load_coco_json(gt_file)["images"])
    pred = predictions_path(cfg, job.eval_set)

    t0 = time.monotonic()
    reused = not smoke and not overwrite and predictions_ready(pred, weights, n_images)
    if reused:
        logger.info("%s on %s: reusing %s", job.run_id, job.eval_set, pred)
    else:
        logger.info("%s on %s: predicting", job.run_id, job.eval_set)
        # an interrupted export must not leave an old sidecar next to new results
        pred.with_name(f"{pred.stem}.meta.json").unlink(missing_ok=True)
        try:
            model_module(cfg).predict(cfg, job.eval_set, weights, pred, limit=limit)
        finally:
            _release_gpu()
    t1 = time.monotonic()
    metrics = evaluate_files(
        _gt_for_scoring(cfg, job, limit),
        pred,
        _eval_dir(cfg, job),
        run_id=job.run_id,
        eval_set=job.eval_set,
        stuff_classes=cfg["stuff_classes"],
        train_gt_file=cfg["eval"]["train_gt"],
        image_groups_csv=cfg["eval"].get("image_groups", {}).get(job.eval_set),
    )
    meta = json.loads(pred.with_name(f"{pred.stem}.meta.json").read_text(encoding="utf-8"))
    return {
        "run_id": job.run_id,
        "eval_set": job.eval_set,
        "status": "ok",
        "reused": reused,
        **{m: metrics["segm"]["summary"][m] for m in METRICS},
        "bbox_AP": metrics["bbox"]["summary"]["AP"],
        "ms_per_image": meta.get("ms_per_image"),
        "peak_vram_mb": meta.get("peak_vram_mb"),
        "predict_s": round(t1 - t0, 1),
        "eval_s": round(time.monotonic() - t1, 1),
        "out_dir": str(_eval_dir(cfg, job)),
    }


def run_jobs(jobs: Iterable[Job], smoke: bool = False, overwrite: bool = False) -> pd.DataFrame:
    """Run every job; a failure is logged and recorded, the remaining jobs still run."""
    rows = []
    for job in jobs:
        try:
            rows.append(run_job(job, smoke=smoke, overwrite=overwrite))
        except Exception as exc:  # the batch must survive one bad run; status says what failed
            logger.exception("%s on %s failed", job.run_id, job.eval_set)
            error = f"{type(exc).__name__}: {exc}"
            rows.append(
                {"run_id": job.run_id, "eval_set": job.eval_set, "status": f"failed: {error}"}
            )
    return pd.DataFrame(rows)


def condition_table(jobs: Iterable[Job], smoke: bool = False) -> pd.DataFrame:
    """Segm AP per image group (phone condition) of every evaluated job that has groups."""
    rows = []
    for job in jobs:
        path = _eval_dir(job_config(job, smoke), job) / "metrics.json"
        if not path.is_file():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        for name, g in (metrics["segm"].get("by_image_group") or {}).items():
            rows.append(
                {
                    "run_id": job.run_id,
                    "eval_set": job.eval_set,
                    "condition": name,
                    **{k: g[k] for k in ("n_images", "n_gt", "n_categories", "AP", "AP50")},
                }
            )
    return pd.DataFrame(rows)
