"""Evaluator sanity check: ground truth fed back as predictions must score mask AP = 1.0.

Each GT polygon is rasterised to a mask exactly as pycocotools does, wrapped in a
:class:`~iiss.eval.results_json.Detection` with score 1.0, written through
:class:`~iiss.eval.results_json.ResultsWriter` (the same path every model uses) and scored with
:func:`iiss.evaluate.evaluate_files`. This checks the whole chain: mask encoding, RLE at the
original resolution, category ids, GT loading and the breakdown code.

Two evaluations are written for each eval set:

``uncapped/``
    No detection is dropped: the writer keeps every GT-as-prediction and COCOeval's largest
    ``maxDets`` is raised above the largest per-image instance count. **Pass criterion:** mask
    AP, AP50 and every category-group AP are exactly 1.0. Groups without GT instances (e.g.
    ``absent_in_train`` on the phone set) have no AP and are skipped.

``standard/``
    The protocol used for every model (writer: top 100 per image; COCOeval: maxDets 1/10/100,
    applied per image and category). Images with more than 100 instances cannot reach full
    recall, so these numbers are the **best score any model can reach** on the eval set. Not a
    pass/fail check.

Box AP is reported but not checked: predicted boxes are the tight boxes of the rasterised mask,
while GT boxes come from the polygon's float extents, so small objects can fall below IoU 0.95.

Usage::

    python -m iiss.eval.sanity --eval-set valid
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from iiss.eval.results_json import DEFAULT_MAX_DETS, Detection, ResultsWriter, polygons_to_mask
from iiss.evaluate import evaluate_files
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

SANITY_RUN_ID = "sanity_gt_as_pred"
TOLERANCE = 1e-9


def gt_as_results(gt: Mapping[str, Any], max_dets: int | None = None) -> list[dict[str, Any]]:
    """Convert every GT annotation into a score-1.0 prediction in COCO results format.

    ``max_dets`` is the writer's per-image cap; ``None`` keeps every instance.
    """
    images = {img["id"]: img for img in gt["images"]}
    by_image: dict[int, list[Detection]] = defaultdict(list)
    for ann in gt["annotations"]:
        img = images[ann["image_id"]]
        mask = polygons_to_mask(ann["segmentation"], img["height"], img["width"])
        by_image[ann["image_id"]].append(Detection(ann["image_id"], ann["category_id"], 1.0, mask))
    cap = max_dets if max_dets is not None else max((len(v) for v in by_image.values()), default=1)
    writer = ResultsWriter.from_coco_gt(gt, max_dets=cap)
    for img_id in images:
        writer.add_image(img_id, by_image.get(img_id, []))
    return writer.results


def _score(
    gt_path: Path,
    gt: Mapping[str, Any],
    out: Path,
    eval_set: str,
    writer_cap: int | None,
    cocoeval_max_dets: tuple[int, int, int] | None,
    stuff_classes: list[str] | None,
    train_gt_file: str | Path | None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    pred_path = out / "gt_as_predictions.json"
    pred_path.write_text(json.dumps(gt_as_results(gt, max_dets=writer_cap)), encoding="utf-8")
    try:
        return evaluate_files(
            gt_path,
            pred_path,
            out,
            run_id=SANITY_RUN_ID,
            eval_set=eval_set,
            stuff_classes=stuff_classes,
            train_gt_file=train_gt_file,
            max_dets=cocoeval_max_dets,
        )
    finally:
        pred_path.unlink(missing_ok=True)  # large and reproducible; keep only the scores


def run_sanity(
    gt_file: str | Path,
    out_dir: str | Path,
    eval_set: str,
    stuff_classes: list[str] | None = None,
    train_gt_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run the uncapped (pass/fail) and standard (ceiling) evaluations; write ``sanity.json``."""
    gt_path = resolve_path(gt_file)
    gt = json.loads(gt_path.read_text(encoding="utf-8"))
    out = resolve_path(out_dir)

    per_image = Counter(a["image_id"] for a in gt["annotations"])
    per_image_cat = Counter((a["image_id"], a["category_id"]) for a in gt["annotations"])
    max_inst = max(per_image.values(), default=1)
    big_cap = max(max_inst, DEFAULT_MAX_DETS) + 1

    uncapped = _score(
        gt_path,
        gt,
        out / "uncapped",
        eval_set,
        None,
        (1, 10, big_cap),
        stuff_classes,
        train_gt_file,
    )
    standard = _score(
        gt_path,
        gt,
        out / "standard",
        eval_set,
        DEFAULT_MAX_DETS,
        None,
        stuff_classes,
        train_gt_file,
    )

    checks = {
        "segm_AP": uncapped["segm"]["summary"]["AP"],
        "segm_AP50": uncapped["segm"]["summary"]["AP50"],
    }
    for name, g in (uncapped["segm"].get("by_category_group") or {}).items():
        if g["AP"] is None:
            logger.info("Skipping group %s: no GT instances in %s", name, eval_set)
            continue
        checks[f"segm_AP[{name}]"] = g["AP"]
    passed = {
        k: v is not None and math.isclose(v, 1.0, abs_tol=TOLERANCE) for k, v in checks.items()
    }
    report = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_set": eval_set,
        "gt_instances": len(gt["annotations"]),
        "max_instances_per_image": max_inst,
        "images_over_100_instances": sum(1 for n in per_image.values() if n > DEFAULT_MAX_DETS),
        "image_category_pairs_over_100": sum(
            1 for n in per_image_cat.values() if n > DEFAULT_MAX_DETS
        ),
        "uncapped_cocoeval_max_dets": [1, 10, big_cap],
        "checks": checks,
        "passed": passed,
        "all_passed": all(passed.values()),
        "uncapped_bbox_AP": uncapped["bbox"]["summary"]["AP"],
        "standard_ceiling": {
            "segm": standard["segm"]["summary"],
            "bbox": standard["bbox"]["summary"],
        },
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "sanity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for k, v in checks.items():
        (logger.info if passed[k] else logger.error)(
            "%s %s = %r", "PASS" if passed[k] else "FAIL", k, v
        )
    logger.info(
        "Standard-protocol ceiling: segm AP %.4f, bbox AP %.4f",
        standard["segm"]["summary"]["AP"],
        standard["bbox"]["summary"]["AP"],
    )
    return report


def main(argv: list[str] | None = None) -> int:
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    ecfg = base["eval"]
    parser = argparse.ArgumentParser(description="GT-as-prediction sanity check (mask AP = 1)")
    parser.add_argument("--eval-set", required=True, choices=sorted(ecfg["gt_files"]))
    parser.add_argument("--out", help="default results/eval/sanity_gt_as_pred/<eval_set>")
    args = parser.parse_args(argv)
    setup_logging()
    out = args.out or str(Path(ecfg["results_root"]) / SANITY_RUN_ID / args.eval_set)
    report = run_sanity(
        ecfg["gt_files"][args.eval_set],
        out,
        args.eval_set,
        stuff_classes=base.get("stuff_classes", []),
        train_gt_file=ecfg.get("train_gt"),
    )
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
