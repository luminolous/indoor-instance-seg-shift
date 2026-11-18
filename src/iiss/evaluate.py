"""Score a COCO results JSON with pycocotools and write metrics, per-class AP and breakdowns.

Usage::

    python -m iiss.evaluate --pred runs/<run_id>/predictions/test.json --eval-set test \
        --run-id <run_id>

Outputs in ``results/eval/<run_id>/<eval_set>/`` (or ``--out``):

- ``metrics.json``         segm (primary) and bbox summaries + category / image-group breakdowns
- ``per_class_ap_segm.csv`` and ``per_class_ap_bbox.csv``
- ``summary.md``           the main numbers in Markdown

Breakdowns: things vs stuff (``stuff_classes`` in the base config), classes present vs absent
in train (``eval.train_gt``), and, with ``--image-groups``, image groups such as the phone-set
capture conditions (CSV with ``filename`` and ``condition`` columns).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from iiss.eval.coco_eval import (
    METRIC_NAMES,
    category_groups,
    evaluate_all,
    image_groups_from_table,
    load_gt,
    load_results,
)
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)


def _clean(value: Any) -> Any:
    """Replace NaN with None so the JSON stays valid."""
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    return value


def _fmt(v: float | None) -> str:
    return "n/a" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f}"


def render_summary_md(metrics: dict[str, Any]) -> str:
    lines = [
        f"# Evaluation: {metrics['run_id'] or 'unnamed'} on {metrics['eval_set'] or 'custom'}",
        "",
        f"- Generated: {metrics['generated']}",
        f"- Predictions: `{metrics['pred_file']}`",
        f"- Ground truth: `{metrics['gt_file']}`",
        f"- Images: {metrics['num_images']}, GT instances: {metrics['num_gt_instances']},"
        f" predictions: {metrics['num_predictions']}",
        "- Scorer: pycocotools COCOeval. Values are AP/AR x 100.",
        "",
        "## Summary",
        "",
        "| IoU type | " + " | ".join(METRIC_NAMES) + " |",
        "|---|" + "---|" * len(METRIC_NAMES),
    ]
    for iou_type in ("segm", "bbox"):
        s = metrics[iou_type]["summary"]
        lines.append(f"| {iou_type} | " + " | ".join(_fmt(s[m]) for m in METRIC_NAMES) + " |")

    for key, label, count in (
        ("by_category_group", "Category groups (segm)", "n_categories"),
        ("by_image_group", "Image groups (segm)", "n_images"),
    ):
        groups = metrics["segm"].get(key)
        if not groups:
            continue
        lines += [
            "",
            f"## {label}",
            "",
            f"| Group | {count} | AP | AP50 | APs | APm | APl |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, g in groups.items():
            lines.append(
                f"| {name} | {g[count]} | "
                + " | ".join(_fmt(g[m]) for m in ("AP", "AP50", "APs", "APm", "APl"))
                + " |"
            )
    return "\n".join(lines) + "\n"


def evaluate_files(
    gt_file: str | Path,
    pred_file: str | Path,
    out_dir: str | Path,
    run_id: str | None = None,
    eval_set: str | None = None,
    stuff_classes: list[str] | None = None,
    train_gt_file: str | Path | None = None,
    image_groups_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate one prediction file against one GT file and write all outputs."""
    gt_path, pred_path, out = resolve_path(gt_file), resolve_path(pred_file), resolve_path(out_dir)
    gt = load_gt(gt_path)
    results = json.loads(pred_path.read_text(encoding="utf-8"))
    dt = load_results(gt, results)

    cat_groups = None
    if stuff_classes is not None:
        train_gt = load_gt(resolve_path(train_gt_file)) if train_gt_file else None
        cat_groups = category_groups(gt, stuff_classes, train_gt)
    img_groups = None
    if image_groups_csv:
        img_groups = image_groups_from_table(gt, pd.read_csv(resolve_path(image_groups_csv)))

    ev = evaluate_all(gt, dt, cat_groups, img_groups)

    out.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_id": run_id,
        "eval_set": eval_set,
        "gt_file": _display(gt_path),
        "pred_file": _display(pred_path),
        "num_images": len(gt.getImgIds()),
        "num_gt_instances": len(gt.getAnnIds()),
        "num_predictions": len(results),
        "category_groups": {k: len(v) for k, v in (cat_groups or {}).items()},
    }
    for iou_type, entry in ev.items():
        entry["per_class"].to_csv(out / f"per_class_ap_{iou_type}.csv", index=False)
        metrics[iou_type] = _clean({k: v for k, v in entry.items() if k != "per_class"})
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "summary.md").write_text(render_summary_md(metrics), encoding="utf-8")
    logger.info(
        "segm AP %s / bbox AP %s -> %s",
        _fmt(metrics["segm"]["summary"]["AP"]),
        _fmt(metrics["bbox"]["summary"]["AP"]),
        out,
    )
    return metrics


def _display(path: Path) -> str:
    try:
        return path.relative_to(repo_root()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    ecfg = base["eval"]
    parser = argparse.ArgumentParser(description="Score a COCO results JSON with pycocotools")
    parser.add_argument("--pred", required=True, help="COCO results JSON")
    parser.add_argument("--eval-set", choices=sorted(ecfg["gt_files"]), help="valid/test/phone")
    parser.add_argument("--gt", help="GT COCO JSON (overrides --eval-set)")
    parser.add_argument("--run-id", help="run id, used for the default output folder")
    parser.add_argument("--out", help="output folder (default results/eval/<run_id>/<eval_set>)")
    parser.add_argument("--image-groups", help="CSV with filename,condition columns")
    parser.add_argument("--no-breakdowns", action="store_true", help="skip category groups")
    args = parser.parse_args(argv)

    if not args.gt and not args.eval_set:
        parser.error("give --eval-set or --gt")
    gt_file = args.gt or ecfg["gt_files"][args.eval_set]
    out = args.out or str(
        Path(ecfg["results_root"]) / (args.run_id or "unnamed") / (args.eval_set or "custom")
    )
    setup_logging()
    evaluate_files(
        gt_file,
        args.pred,
        out,
        run_id=args.run_id,
        eval_set=args.eval_set,
        stuff_classes=None if args.no_breakdowns else base.get("stuff_classes", []),
        train_gt_file=None if args.no_breakdowns else ecfg.get("train_gt"),
        image_groups_csv=args.image_groups,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
