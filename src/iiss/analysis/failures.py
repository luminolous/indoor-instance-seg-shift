"""Per-instance matching and failure tags for the phone set (assignment part 3).

Works on the exported predictions (``runs/<run_id>/predictions/<eval_set>.json``), so no model is
loaded. AP integrates over all score thresholds; a failure gallery needs one operating point, so
only predictions with ``score >= score_threshold`` count as *shown*. The threshold and the IoU
limits live in ``configs/base.yaml`` under ``failure_analysis``.

Matching (per image and class, as in COCOeval): shown predictions in score order each take the
free GT of their class with the highest IoU, if that is at least ``match_iou``.

Every GT instance gets exactly one tag, the first that applies:

- ``ok``              matched with IoU >= ``good_iou``
- ``split``           two or more predictions of the class lie on it (each mostly inside) and
                      together cover at least half of it, and no single one does most of that
- ``mask_leak`` / ``mask_incomplete``  matched with IoU < ``good_iou``: the mask is bigger than the
                      object (precision < recall) or misses part of it
- ``wrong_class``     not matched, an unmatched prediction of another class has IoU >= ``match_iou``
- ``merged``          not matched, one prediction of the class covers at least half of it and of
                      another GT of the class
- ``mask_leak`` / ``mask_incomplete``  not matched, best free prediction of the class has
                      ``partial_iou`` <= IoU < ``match_iou``
- ``low_score``       not matched, but a prediction below the score threshold has the right class
                      and IoU >= ``match_iou``
- ``missed``          none of the above

The tags describe what is visible in the masks. They say nothing about the cause.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from iiss.analysis.domain_shift import parse_run_id
from iiss.data.coco import CocoSplit, load_coco_json
from iiss.utils.config import load_yaml
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

TAGS = (
    "ok",
    "mask_leak",
    "mask_incomplete",
    "split",
    "merged",
    "wrong_class",
    "low_score",
    "missed",
)
FAILURES = TAGS[1:]
# A predicted piece belongs to an object when at least this share of the piece lies inside it.
PIECE_PRECISION = 0.5
PIECE_MIN_RECALL = 0.05
# Share of the object that split pieces (together) or a merging mask must cover.
COVER = 0.5
# The biggest piece must cover less than this share of what all pieces cover together.
DOMINANT_PIECE = 0.8


@dataclass(frozen=True)
class Limits:
    score_threshold: float = 0.25
    match_iou: float = 0.5
    good_iou: float = 0.75
    partial_iou: float = 0.1


def settings() -> dict[str, Any]:
    """The ``failure_analysis`` section of ``configs/base.yaml``."""
    return dict(load_yaml(repo_root() / "configs" / "base.yaml")["failure_analysis"])


def default_limits(**overrides: float | None) -> Limits:
    """Limits from the config. Overrides that are ``None`` are ignored."""
    cfg = settings()
    values = {k: float(cfg[k]) for k in Limits.__dataclass_fields__}
    return Limits(**{**values, **{k: v for k, v in overrides.items() if v is not None}})


def load_gt_set(eval_set: str = "phone") -> CocoSplit:
    """GT of an eval set as configured in ``eval.gt_files``."""
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    path = resolve_path(base["eval"]["gt_files"][eval_set])
    return CocoSplit(eval_set, path.parent, path, load_coco_json(path))


def load_predictions(
    run_id: str, eval_set: str = "phone", runs_root: str | Path = "runs"
) -> dict[int, list[dict[str, Any]]]:
    """Exported predictions of one run, grouped by image id."""
    path = resolve_path(runs_root) / run_id / "predictions" / f"{eval_set}.json"
    grouped: dict[int, list[dict[str, Any]]] = {}
    for p in json.loads(path.read_text(encoding="utf-8")):
        grouped.setdefault(p["image_id"], []).append(p)
    return grouped


def photo_name(image: Mapping[str, Any]) -> str:
    return Path(image["file_name"]).name


# ---------------------------------------------------------------------------------------------
# Geometry on RLE masks
# ---------------------------------------------------------------------------------------------


def to_rle(segmentation: Any, height: int, width: int) -> dict[str, Any]:
    """COCO segmentation (polygons or RLE with str/bytes counts) as a compressed RLE."""
    if isinstance(segmentation, list):
        return mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
    rle = dict(segmentation)
    if isinstance(rle["counts"], list):
        return mask_utils.frPyObjects(rle, height, width)
    if isinstance(rle["counts"], str):
        rle["counts"] = rle["counts"].encode("ascii")
    return rle


@dataclass
class Overlap:
    """Areas and intersections of predictions (rows) and GT instances (columns), in pixels."""

    pred_area: np.ndarray
    gt_area: np.ndarray
    inter: np.ndarray

    @cached_property
    def iou(self) -> np.ndarray:
        union = self.pred_area[:, None] + self.gt_area[None, :] - self.inter
        return np.divide(self.inter, union, out=np.zeros_like(self.inter), where=union > 0)

    @cached_property
    def precision(self) -> np.ndarray:
        """Share of the prediction that lies inside the GT."""
        area = np.broadcast_to(self.pred_area[:, None], self.inter.shape)
        return np.divide(self.inter, area, out=np.zeros_like(self.inter), where=area > 0)

    @cached_property
    def recall(self) -> np.ndarray:
        """Share of the GT that the prediction covers."""
        area = np.broadcast_to(self.gt_area[None, :], self.inter.shape)
        return np.divide(self.inter, area, out=np.zeros_like(self.inter), where=area > 0)


def overlap(pred_rles: Sequence[Any], gt_rles: Sequence[Any]) -> Overlap:
    def areas(rles: Sequence[Any]) -> np.ndarray:
        return np.asarray(mask_utils.area(list(rles)), dtype=float) if rles else np.zeros(0)

    pred_area, gt_area = areas(pred_rles), areas(gt_rles)
    inter = np.zeros((len(pred_rles), len(gt_rles)))
    if len(pred_rles) and len(gt_rles):
        # with iscrowd = 1 pycocotools returns intersection / prediction area
        share = mask_utils.iou(list(pred_rles), list(gt_rles), [1] * len(gt_rles))
        inter = np.rint(np.asarray(share) * pred_area[:, None])
    return Overlap(pred_area, gt_area, inter)


# ---------------------------------------------------------------------------------------------
# Matching and tagging of one image
# ---------------------------------------------------------------------------------------------


@dataclass
class ImageMatch:
    """Result for one image and run: one row per GT and one per shown prediction."""

    gt: list[dict[str, Any]]
    pred: list[dict[str, Any]]


def greedy_match(
    iou: np.ndarray, pred_cat: np.ndarray, gt_cat: np.ndarray, match_iou: float
) -> dict[int, int]:
    """``{gt index: prediction index}``. Predictions must be in descending score order."""
    taken: dict[int, int] = {}
    for p in range(len(pred_cat)):
        free = [g for g in range(len(gt_cat)) if gt_cat[g] == pred_cat[p] and g not in taken]
        if free:
            best = max(free, key=lambda g: iou[p, g])
            if iou[p, best] >= match_iou:
                taken[best] = p
    return taken


def _is_split(
    g: int, pieces: Sequence[int], ov: Overlap, pred_rles: Sequence[Any], gt_rle: Any
) -> bool:
    if len(pieces) < 2:
        return False
    union = mask_utils.merge([pred_rles[p] for p in pieces])
    covered = mask_utils.area(mask_utils.merge([union, gt_rle], intersect=True)) / ov.gt_area[g]
    biggest = max(ov.recall[p, g] for p in pieces)
    return covered >= COVER and biggest < DOMINANT_PIECE * covered


def match_image(
    gts: Sequence[Mapping[str, Any]],
    preds: Sequence[Mapping[str, Any]],
    height: int,
    width: int,
    limits: Limits,
) -> ImageMatch:
    """Match and tag all instances of one image (see the module docstring)."""
    ordered = sorted(preds, key=lambda p: -p["score"])
    shown = [p for p in ordered if p["score"] >= limits.score_threshold]
    hidden = ordered[len(shown) :]
    gt_rles = [to_rle(a["segmentation"], height, width) for a in gts]
    rles = [to_rle(p["segmentation"], height, width) for p in shown]
    gt_cat = np.array([a["category_id"] for a in gts], dtype=int)
    cat = np.array([p["category_id"] for p in shown], dtype=int)
    ov = overlap(rles, gt_rles)
    taken = greedy_match(ov.iou, cat, gt_cat, limits.match_iou)
    owner = {p: g for g, p in taken.items()}
    free = [p for p in range(len(shown)) if p not in owner]
    low = overlap([to_rle(p["segmentation"], height, width) for p in hidden], gt_rles)
    low_cat = np.array([p["category_id"] for p in hidden], dtype=int)

    gt_rows = []
    for g, ann in enumerate(gts):
        row: dict[str, Any] = {
            "gt_id": ann["id"],
            "category_id": int(gt_cat[g]),
            "area_fraction": ov.gt_area[g] / (height * width),
            "matched": g in taken,
        }
        same = [p for p in range(len(shown)) if cat[p] == gt_cat[g] and owner.get(p, g) == g]
        pieces = [
            p
            for p in same
            if ov.precision[p, g] >= PIECE_PRECISION and ov.recall[p, g] >= PIECE_MIN_RECALL
        ]
        split = _is_split(g, pieces, ov, rles, gt_rles[g])
        other = [p for p in free if cat[p] != gt_cat[g] and ov.iou[p, g] >= limits.match_iou]
        hit = [
            p
            for p in range(len(hidden))
            if low_cat[p] == gt_cat[g] and low.iou[p, g] >= limits.match_iou
        ]
        merging = [
            p
            for p in range(len(shown))
            if cat[p] == gt_cat[g]
            and ov.recall[p, g] >= COVER
            and any(
                ov.recall[p, h] >= COVER
                for h in range(len(gts))
                if h != g and gt_cat[h] == gt_cat[g]
            )
        ]
        # the prediction that the IoU / precision / recall columns describe
        best = taken.get(g)
        if best is None and same:
            best = max(same, key=lambda p: ov.iou[p, g])
        if best is not None and ov.iou[best, g] > 0:
            mask_error = (
                "mask_leak" if ov.precision[best, g] < ov.recall[best, g] else "mask_incomplete"
            )
            row.update(
                score=shown[best]["score"],
                iou=ov.iou[best, g],
                precision=ov.precision[best, g],
                recall=ov.recall[best, g],
            )
        else:
            mask_error = None
            row.update(score=None, iou=0.0, precision=None, recall=None)
        if row["matched"] and row["iou"] >= limits.good_iou:
            tag = "ok"
        elif split:
            tag = "split"
        elif row["matched"]:
            tag = mask_error
        elif other:
            tag = "wrong_class"
        elif merging:
            tag = "merged"
        elif row["iou"] >= limits.partial_iou:
            tag = mask_error
        elif hit:
            tag = "low_score"
        else:
            tag = "missed"
        wrong = other[0] if other else None  # highest score
        row.update(
            tag=tag,
            n_pieces=len(pieces),
            predicted_category_id=int(cat[wrong]) if wrong is not None else None,
            predicted_score=shown[wrong]["score"] if wrong is not None else None,
            hidden_score=max((hidden[p]["score"] for p in hit), default=None),
        )
        gt_rows.append(row)

    pred_rows = []
    for p, pred in enumerate(shown):
        same_iou = max((ov.iou[p, g] for g in range(len(gts)) if gt_cat[g] == cat[p]), default=0.0)
        other_iou = max((ov.iou[p, g] for g in range(len(gts)) if gt_cat[g] != cat[p]), default=0.0)
        if p in owner:
            kind = "matched"
        elif same_iou >= limits.match_iou:
            kind = "duplicate"
        elif other_iou >= limits.match_iou:
            kind = "wrong_class"
        elif same_iou >= limits.partial_iou:
            kind = "poor_mask"
        else:  # background, or an object that is not annotated
            kind = "no_gt"
        pred_rows.append(
            {
                "category_id": int(cat[p]),
                "score": pred["score"],
                "kind": kind,
                "gt_id": gts[owner[p]]["id"] if p in owner else None,
                "iou": ov.iou[p, owner[p]] if p in owner else same_iou,
                "segmentation": pred["segmentation"],
            }
        )
    return ImageMatch(gt_rows, pred_rows)


# ---------------------------------------------------------------------------------------------
# Tables over runs
# ---------------------------------------------------------------------------------------------


def match_run(
    split: CocoSplit, predictions: Mapping[int, Sequence[Mapping[str, Any]]], limits: Limits
) -> dict[int, ImageMatch]:
    return {
        img["id"]: match_image(
            split.annotations_by_image.get(img["id"], []),
            predictions.get(img["id"], []),
            img["height"],
            img["width"],
            limits,
        )
        for img in split.images
    }


def match_runs(
    run_ids: Iterable[str],
    split: CocoSplit,
    limits: Limits,
    eval_set: str = "phone",
    runs_root: str | Path = "runs",
) -> dict[str, dict[int, ImageMatch]]:
    """``{run_id: {image_id: ImageMatch}}``. Loads one prediction file at a time."""
    return {r: match_run(split, load_predictions(r, eval_set, runs_root), limits) for r in run_ids}


def instance_table(
    matches: Mapping[str, Mapping[int, ImageMatch]], split: CocoSplit
) -> pd.DataFrame:
    """One row per run and GT instance."""
    rows = []
    for run_id, images in matches.items():
        model, size = parse_run_id(run_id)
        for image_id, m in images.items():
            info = split.images_by_id[image_id]
            for r in m.gt:
                wrong = r["predicted_category_id"]
                rows.append(
                    {
                        "run_id": run_id,
                        "model": model,
                        "input_size": size,
                        "file_name": photo_name(info),
                        "condition": info.get("condition"),
                        "scene": info.get("scene"),
                        "gt_id": r["gt_id"],
                        "name": split.category_names[r["category_id"]],
                        "area_fraction": r["area_fraction"],
                        "tag": r["tag"],
                        "matched": r["matched"],
                        "iou": r["iou"],
                        "precision": r["precision"],
                        "recall": r["recall"],
                        "score": r["score"],
                        "n_pieces": r["n_pieces"],
                        "predicted_as": split.category_names[wrong] if wrong is not None else None,
                        "predicted_score": r["predicted_score"],
                        "hidden_score": r["hidden_score"],
                    }
                )
    return pd.DataFrame(rows)


def tag_counts(instances: pd.DataFrame, by: str | Sequence[str] = "run_id") -> pd.DataFrame:
    """Number of GT instances per tag (columns in the order of :data:`TAGS`)."""
    keys = [by] if isinstance(by, str) else list(by)
    counts = instances.groupby([*keys, "tag"], sort=False).size().unstack(fill_value=0)
    return counts.reindex(columns=list(TAGS), fill_value=0)


def prediction_counts(matches: Mapping[str, Mapping[int, ImageMatch]]) -> pd.DataFrame:
    """Number of shown predictions per kind and run."""
    kinds = ("matched", "duplicate", "wrong_class", "poor_mask", "no_gt")
    rows = {}
    for run_id, images in matches.items():
        c = Counter(p["kind"] for m in images.values() for p in m.pred)
        rows[run_id] = {k: c.get(k, 0) for k in kinds}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("run_id")


def photo_table(
    instances: pd.DataFrame, label_issues: Mapping[str, str] | None = None
) -> pd.DataFrame:
    """Matched GT per photo (rows) and run (columns), worst photos first."""
    matched = instances.pivot_table(
        index="file_name", columns="run_id", values="matched", aggfunc="sum", sort=False
    ).astype(int)
    info = instances.drop_duplicates("file_name").set_index("file_name")[["condition", "scene"]]
    n_gt = instances.groupby("file_name")["gt_id"].nunique().rename("n_gt")
    table = info.join(n_gt).join(matched)
    table["matched_share"] = matched.mean(axis=1) / n_gt
    issues = dict(label_issues or {})
    table["label_issue"] = [issues.get(f, "") for f in table.index]
    return table.sort_values("matched_share", kind="stable")


def photo_detail(instances: pd.DataFrame, file_name: str) -> pd.DataFrame:
    """Tag and IoU (or the predicted class) per GT instance (rows) and run (columns) of a photo."""
    part = instances[instances["file_name"] == file_name]
    text = [
        f"wrong_class ({r.predicted_as} {r.predicted_score:.2f})"
        if r.tag == "wrong_class"
        else f"low_score ({r.hidden_score:.2f})"
        if r.tag == "low_score"
        else r.tag + (f" (IoU {r.iou:.2f})" if r.iou > 0 else "")
        for r in part.itertuples()
    ]
    return part.assign(text=text).pivot(index=["gt_id", "name"], columns="run_id", values="text")[
        list(dict.fromkeys(part["run_id"]))
    ]


def candidates(
    instances: pd.DataFrame,
    per_tag: int = 3,
    exclude: Iterable[str] = (),
    min_area: float = 0.005,
) -> pd.DataFrame:
    """Photos worth looking at: up to ``per_tag`` instances for every failure tag.

    ``instances`` should hold one input size. Instances with the same failure in several models
    come first, then the larger ones. Within a tag every scene appears once, so the list does not
    repeat near-identical photos. ``exclude`` (photos with label issues) and objects smaller than
    ``min_area`` of the image are skipped.
    """
    skip = set(exclude)
    pool = instances[
        ~instances["file_name"].isin(skip)
        & (instances["tag"] != "ok")
        & (instances["area_fraction"] >= min_area)
    ]
    rows = []
    for tag in FAILURES:
        part = pool[pool["tag"] == tag]
        if part.empty:
            continue
        g = part.groupby(["file_name", "gt_id"], sort=False)
        one = g.agg(
            scene=("scene", "first"),
            condition=("condition", "first"),
            name=("name", "first"),
            area_fraction=("area_fraction", "first"),
            runs=("run_id", lambda s: ", ".join(s)),
            n_runs=("run_id", "nunique"),
        ).reset_index()
        one = one.sort_values(
            ["n_runs", "area_fraction", "file_name"], ascending=[False, False, True]
        )
        one = one.drop_duplicates("scene").head(per_tag)
        rows.append(one.assign(tag=tag))
    columns = [
        "tag", "file_name", "scene", "condition", "gt_id", "name", "area_fraction", "n_runs",
        "runs",
    ]  # fmt: skip
    return pd.concat(rows, ignore_index=True)[columns] if rows else pd.DataFrame(columns=columns)


def write_failure_tables(
    instances: pd.DataFrame, candidate_table: pd.DataFrame, out_dir: str | Path
) -> Path:
    """Write ``instances.csv``, ``tag_counts.csv`` and ``candidates.csv``."""
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    instances.to_csv(out / "instances.csv", index=False, float_format="%.4f")
    tag_counts(instances).to_csv(out / "tag_counts.csv")
    candidate_table.to_csv(out / "candidates.csv", index=False, float_format="%.4f")
    return out
