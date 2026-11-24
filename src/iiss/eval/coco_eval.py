"""Scoring with pycocotools ``COCOeval`` plus per-class AP and subset breakdowns.

One evaluator for every model: predictions come from :mod:`iiss.eval.results_json`, ground
truth is the COCO JSON of the eval set. Framework-internal metrics are never used.

Breakdowns (classes in/absent from train, things vs stuff, phone-set conditions) do **not**
re-run matching. ``COCOeval.evaluate()`` matches detections per (image, category), so a subset
of images or categories only needs a re-accumulation over that subset (see
:func:`accumulate_subset` for why ``accumulate(p)`` with subset params cannot be used directly).
This gives exactly the numbers a separate COCOeval restricted to the subset would give (tested).

Categories without any GT instance in the evaluated images are excluded from every mean, as in
``COCOeval.summarize`` (their precision stays -1).
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

logger = logging.getLogger(__name__)

AREA_INDEX = {"all": 0, "small": 1, "medium": 2, "large": 3}
METRIC_NAMES = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
    "AR1",
    "AR10",
    "AR100",
    "ARs",
    "ARm",
    "ARl",
)


def load_gt(path_or_dict: str | Path | Mapping[str, Any]) -> COCO:
    """Load a COCO GT object quietly from a path or an in-memory dict.

    Annotation ids are shifted in memory so that the smallest id is 1. pycocotools stores the
    matched GT id in ``dtMatches`` and ``accumulate()`` treats that value as a boolean, so a
    detection matched to GT annotation id 0 is counted as a false positive. The v5 export
    numbers annotations from 0. The shift only renames ids; files on disk are not modified.
    """
    if isinstance(path_or_dict, Mapping):
        data = copy.deepcopy(dict(path_or_dict))
    else:
        data = json.loads(Path(path_or_dict).read_text(encoding="utf-8"))
    anns = data.get("annotations", [])
    if anns:
        min_id = min(a["id"] for a in anns)
        if min_id < 1:
            offset = 1 - min_id
            for a in anns:
                a["id"] += offset
            logger.debug("Shifted GT annotation ids by +%d (pycocotools id-0 issue)", offset)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = data
        gt.createIndex()
    return gt


def load_results(gt: COCO, results: str | list[dict[str, Any]]) -> COCO:
    """``gt.loadRes`` without the console noise. An empty result list is allowed."""
    if isinstance(results, list) and not results:
        dt = COCO()
        dt.dataset = {
            "images": gt.dataset["images"],
            "categories": gt.dataset["categories"],
            "annotations": [],
        }
        dt.createIndex()
        return dt
    with contextlib.redirect_stdout(io.StringIO()):
        return gt.loadRes(results if isinstance(results, list) else str(results))


def run_cocoeval(
    gt: COCO, dt: COCO, iou_type: str, max_dets: tuple[int, int, int] | None = None
) -> COCOeval:
    """Run ``evaluate()`` and ``accumulate()`` over all images and categories.

    ``max_dets`` overrides COCOeval's ``(1, 10, 100)``. Note that COCOeval applies the cap per
    image **and per category**. Only the sanity check changes it; reported metrics use the
    default.
    """
    if iou_type not in ("segm", "bbox"):
        raise ValueError(f"iou_type must be 'segm' or 'bbox', got {iou_type!r}")
    ce = COCOeval(gt, dt, iouType=iou_type)
    if max_dets is not None:
        ce.params.maxDets = list(max_dets)
    with contextlib.redirect_stdout(io.StringIO()):
        ce.evaluate()
        ce.accumulate()
    return ce


# ---------------------------------------------------------------------------------------------
# Accumulation over subsets
# ---------------------------------------------------------------------------------------------


def accumulate_subset(
    coco_eval: COCOeval,
    img_ids: Iterable[int] | None = None,
    cat_ids: Iterable[int] | None = None,
) -> tuple[dict[str, Any], Any]:
    """Accumulate an already evaluated ``COCOeval`` over a subset of images and/or categories.

    ``COCOeval.accumulate(p)`` with a subset ``p.imgIds``/``p.catIds`` indexes ``evalImgs`` by
    position *within the subset*, which is wrong unless the subset is a prefix. So instead:

    - images: ``evalImgs`` entries of images outside the subset are set to ``None`` (which
      ``accumulate`` skips) and accumulation runs with the original params;
    - categories: precision/recall are computed per category, so the category axis of the
      accumulated arrays is sliced.

    Returns ``(eval_dict, params)``; ``params.imgIds``/``catIds`` describe the subset.
    ``coco_eval`` itself is left unchanged.
    """
    params = coco_eval.params
    out_params = copy.deepcopy(params)

    if img_ids is None:
        ev = coco_eval.eval
    else:
        keep = set(img_ids)
        out_params.imgIds = [i for i in params.imgIds if i in keep]
        n_img = len(params.imgIds)
        drop = {pos for pos, i in enumerate(params.imgIds) if i not in keep}
        masked = [None if (idx % n_img) in drop else e for idx, e in enumerate(coco_eval.evalImgs)]
        saved_imgs, saved_eval = coco_eval.evalImgs, coco_eval.eval
        coco_eval.evalImgs = masked
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                coco_eval.accumulate()
            ev = coco_eval.eval
        finally:
            coco_eval.evalImgs, coco_eval.eval = saved_imgs, saved_eval

    if cat_ids is not None:
        keep_c = set(cat_ids)
        k_idx = [k for k, c in enumerate(params.catIds) if c in keep_c]
        out_params.catIds = [params.catIds[k] for k in k_idx]
        ev = {
            **ev,
            "precision": ev["precision"][:, :, k_idx],
            "recall": ev["recall"][:, k_idx],
            "scores": ev["scores"][:, :, k_idx],
        }
    return ev, out_params


def _mean_valid(values: np.ndarray) -> float:
    v = values[values > -1]
    return float(np.mean(v)) if v.size else float("nan")


def stats_from_eval(ev: Mapping[str, Any], params: Any) -> dict[str, float]:
    """The 12 standard COCO metrics from an accumulated eval dict (``nan`` if undefined)."""
    precision, recall = ev["precision"], ev["recall"]  # [T,R,K,A,M], [T,K,A,M]
    iou = np.asarray(params.iouThrs)
    m_last = len(params.maxDets) - 1
    md = {d: i for i, d in enumerate(params.maxDets)}

    def ap(area: str = "all", thr: float | None = None) -> float:
        s = precision[:, :, :, AREA_INDEX[area], m_last]
        if thr is not None:
            s = s[np.where(np.isclose(iou, thr))[0]]
        return _mean_valid(s)

    def ar(area: str = "all", max_det: int | None = None) -> float:
        m = m_last if max_det is None else md[max_det]
        return _mean_valid(recall[:, :, AREA_INDEX[area], m])

    values = [
        ap(),
        ap(thr=0.5),
        ap(thr=0.75),
        ap("small"),
        ap("medium"),
        ap("large"),
        ar(max_det=params.maxDets[0]),
        ar(max_det=params.maxDets[1]),
        ar(max_det=params.maxDets[2]),
        ar("small"),
        ar("medium"),
        ar("large"),
    ]
    return dict(zip(METRIC_NAMES, values, strict=True))


def summary(coco_eval: COCOeval) -> dict[str, float]:
    """Standard COCO metrics for all images and categories of an evaluated ``COCOeval``."""
    return stats_from_eval(coco_eval.eval, coco_eval.params)


def subset_summary(
    coco_eval: COCOeval,
    img_ids: Iterable[int] | None = None,
    cat_ids: Iterable[int] | None = None,
) -> dict[str, float]:
    ev, p = accumulate_subset(coco_eval, img_ids=img_ids, cat_ids=cat_ids)
    return stats_from_eval(ev, p)


def per_class_ap(
    coco_eval: COCOeval, gt: COCO, img_ids: Iterable[int] | None = None
) -> pd.DataFrame:
    """AP, AP50, AP75 and GT instance count per category (``nan`` where there is no GT)."""
    if img_ids is None:
        ev, p = coco_eval.eval, coco_eval.params
    else:
        ev, p = accumulate_subset(coco_eval, img_ids=img_ids)
    precision = ev["precision"]
    iou = np.asarray(p.iouThrs)
    i50 = np.where(np.isclose(iou, 0.5))[0]
    i75 = np.where(np.isclose(iou, 0.75))[0]
    img_set = set(p.imgIds)
    rows = []
    for k, cat_id in enumerate(p.catIds):
        s = precision[:, :, k, AREA_INDEX["all"], -1]
        n_gt = sum(
            1
            for a in gt.loadAnns(gt.getAnnIds(catIds=[cat_id], iscrowd=False))
            if a["image_id"] in img_set
        )
        rows.append(
            {
                "category_id": cat_id,
                "name": gt.cats[cat_id]["name"],
                "n_gt": n_gt,
                "AP": _mean_valid(s),
                "AP50": _mean_valid(s[i50]),
                "AP75": _mean_valid(s[i75]),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Breakdown helpers
# ---------------------------------------------------------------------------------------------


def category_groups(
    gt: COCO,
    stuff_names: Iterable[str],
    train_gt: COCO | None = None,
) -> dict[str, list[int]]:
    """Category-id groups for breakdowns: things/stuff and (if ``train_gt``) in/absent train."""
    stuff = set(stuff_names)
    names = {cid: c["name"] for cid, c in gt.cats.items()}
    unknown = stuff - set(names.values())
    if unknown:
        raise ValueError(f"stuff_classes not in the category list: {sorted(unknown)}")
    groups = {
        "things": [cid for cid, n in names.items() if n not in stuff],
        "stuff": [cid for cid, n in names.items() if n in stuff],
    }
    if train_gt is not None:
        in_train = {a["category_id"] for a in train_gt.dataset["annotations"]}
        groups["in_train"] = [cid for cid in names if cid in in_train]
        groups["absent_in_train"] = [cid for cid in names if cid not in in_train]
    return groups


def image_groups_from_table(
    gt: COCO,
    table: pd.DataFrame,
    file_col: str = "filename",
    group_col: str = "condition",
) -> dict[str, list[int]]:
    """Map a ``file name -> group`` table (e.g. phone ``metadata.csv``) to image-id groups.

    Matching is by file name without directories (the phone GT stores relative paths to
    ``originals/``), falling back to the file stem (Roboflow renames uploads to
    ``<stem>_<ext>.rf.<hash>.<ext>``, so the original stem is the prefix before ``_<ext>.rf.``).
    Fails loudly on GT images that cannot be matched.
    """
    lookup = {str(r[file_col]): str(r[group_col]) for _, r in table.iterrows()}
    by_stem = {name.rsplit(".", 1)[0]: grp for name, grp in lookup.items()}
    groups: dict[str, list[int]] = {}
    unmatched = []
    for img in gt.dataset["images"]:
        fname = PurePosixPath(img["file_name"]).name
        grp = lookup.get(fname)
        if grp is None and ".rf." in fname:
            stem = fname.split(".rf.")[0].rsplit("_", 1)[0]
            grp = by_stem.get(stem)
        if grp is None:
            unmatched.append(fname)
            continue
        groups.setdefault(grp, []).append(img["id"])
    if unmatched:
        raise ValueError(f"{len(unmatched)} GT images have no group, e.g. {unmatched[:5]}")
    return groups


def evaluate_all(
    gt: COCO,
    dt: COCO,
    category_groups_: Mapping[str, list[int]] | None = None,
    image_groups: Mapping[str, list[int]] | None = None,
    iou_types: tuple[str, ...] = ("segm", "bbox"),
    max_dets: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Full evaluation: summaries per IoU type, per-class AP and breakdowns.

    Returns ``{"segm": {"summary": ..., "per_class": DataFrame, "by_category_group": ...,
    "by_image_group": ..., "per_class_by_image_group": {name: DataFrame}}, "bbox": {...}}``.
    """
    out: dict[str, Any] = {}
    for iou_type in iou_types:
        ce = run_cocoeval(gt, dt, iou_type, max_dets=max_dets)
        entry: dict[str, Any] = {"summary": summary(ce), "per_class": per_class_ap(ce, gt)}
        if category_groups_:
            entry["by_category_group"] = {
                name: {"n_categories": len(ids), **subset_summary(ce, cat_ids=ids)}
                for name, ids in category_groups_.items()
            }
        if image_groups:
            per_class = {n: per_class_ap(ce, gt, ids) for n, ids in image_groups.items()}
            entry["by_image_group"] = {
                name: {
                    "n_images": len(ids),
                    "n_gt": int(per_class[name]["n_gt"].sum()),
                    "n_categories": int((per_class[name]["n_gt"] > 0).sum()),
                    **subset_summary(ce, img_ids=ids),
                }
                for name, ids in image_groups.items()
            }
            entry["per_class_by_image_group"] = per_class
        out[iou_type] = entry
    return out
