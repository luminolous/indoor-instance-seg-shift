"""Aggregate ``results/eval/`` into the main tables of the report.

Reads only the files written by ``iiss.evaluate`` (plus the run status for the cap flag) and
writes to ``results/summary/``:

- ``main_results.csv``   one row per run and eval set: mask AP metrics over all classes with GT
  and over the *common classes* (GT in valid, test and phone), box AP, epochs, cap flag
- ``domain_shift.csv``   absolute and relative change valid -> test -> phone, full and common
- ``conditions.csv``     phone set per capture condition, full and common classes
- ``category_groups.csv`` things/stuff and classes in/absent from train, per eval set
- ``per_class_ap.csv``   mask AP per class, run and eval set (classes with GT only)
- ``summary.md``         the main tables in Markdown

Why common classes: the sets have GT for different classes, so their full AP averages over
different class lists. Neither the full AP nor the common-class AP is a pure "difficulty" effect:
label habits and instance counts per class differ between the sets too.

Values in the CSV files are fractions in [0, 1]; ``summary.md`` shows AP x 100. Nothing is typed by
hand, and a missing run or set is reported instead of filled in.

Usage::

    python -m iiss.analysis.domain_shift
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from iiss.analysis.training import notebook_config, run_summary
from iiss.utils.config import VALID_MODELS, load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

SETS = ("valid", "test", "phone")
SHIFTS = (("valid", "test"), ("test", "phone"), ("valid", "phone"))
METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR100")
MODEL_NAMES = {"yolo": "YOLO11s-seg", "maskrcnn": "Mask R-CNN R50-FPN v2", "m2f": "Mask2Former"}
TOLERANCE = 1e-6


class AggregationError(ValueError):
    """Raised when evaluation files are missing or inconsistent with each other."""


@dataclass
class SetResult:
    """Evaluation of one run on one set."""

    run_id: str
    model: str
    size: int
    eval_set: str
    metrics: dict[str, Any]
    per_class: pd.DataFrame  # index: class name; columns n_gt, AP, AP50 (segm)
    conditions: dict[str, pd.DataFrame]

    @property
    def classes(self) -> set[str]:
        return set(self.per_class.index[self.per_class["n_gt"] > 0])

    def mean_ap(self, classes: Sequence[str], metric: str = "AP") -> float:
        return float(self.per_class.loc[list(classes), metric].mean())


def _read_per_class(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).set_index("name")[["n_gt", "AP", "AP50"]]


def load_results(
    eval_root: str | Path,
    run_ids: Iterable[str],
    eval_sets: Sequence[str] = SETS,
) -> dict[tuple[str, str], SetResult]:
    """Load every ``<eval_root>/<run_id>/<set>/`` and check it is consistent.

    Raises :class:`AggregationError` listing every missing or inconsistent file.
    """
    root = resolve_path(eval_root)
    out: dict[tuple[str, str], SetResult] = {}
    problems: list[str] = []
    for run_id in run_ids:
        model, size = parse_run_id(run_id)
        for eval_set in eval_sets:
            folder = root / run_id / eval_set
            metrics_file = folder / "metrics.json"
            if not metrics_file.is_file():
                problems.append(f"missing {run_id}/{eval_set}/metrics.json")
                continue
            metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
            if (metrics["run_id"], metrics["eval_set"]) != (run_id, eval_set):
                problems.append(f"{run_id}/{eval_set}: metrics.json belongs to another job")
                continue
            per_class = _read_per_class(folder / "per_class_ap_segm.csv")
            conditions = {
                name: _read_per_class(folder / f"per_class_ap_segm_{name}.csv")
                for name in metrics["segm"].get("by_image_group") or {}
            }
            res = SetResult(run_id, model, size, eval_set, metrics, per_class, conditions)
            expected = metrics["segm"]["summary"]["AP"]
            actual = res.mean_ap(sorted(res.classes))
            if not math.isclose(expected, actual, abs_tol=TOLERANCE):
                problems.append(
                    f"{run_id}/{eval_set}: AP {expected:.6f} != mean per-class AP {actual:.6f}"
                )
            out[run_id, eval_set] = res
    for eval_set in eval_sets:
        loaded = [r for (_, e), r in out.items() if e == eval_set]
        if len({r.metrics["num_gt_instances"] for r in loaded}) > 1:
            problems.append(f"{eval_set}: runs were scored against different ground truth")
    if problems:
        raise AggregationError("; ".join(problems))
    return out


def parse_run_id(run_id: str) -> tuple[str, int]:
    """``yolo_s640_seed42`` -> ``("yolo", 640)``."""
    model, size, _ = run_id.split("_")
    return model, int(size.removeprefix("s"))


def common_classes(
    results: Mapping[tuple[str, str], SetResult], eval_sets: Sequence[str]
) -> list[str]:
    """Classes with GT in every given set (identical for all runs)."""
    per_set = []
    for eval_set in eval_sets:
        sets = [r.classes for (_, e), r in results.items() if e == eval_set]
        if any(s != sets[0] for s in sets):
            raise AggregationError(f"{eval_set}: classes with GT differ between runs")
        per_set.append(sets[0])
    return sorted(set.intersection(*per_set))


def main_table(results: Mapping[tuple[str, str], SetResult], common: Sequence[str]) -> pd.DataFrame:
    """One row per run and eval set."""
    rows = []
    for (run_id, eval_set), r in results.items():
        summary = r.metrics["segm"]["summary"]
        rows.append(
            {
                "run_id": run_id,
                "model": r.model,
                "input_size": r.size,
                "eval_set": eval_set,
                "n_images": r.metrics["num_images"],
                "n_gt": r.metrics["num_gt_instances"],
                "n_classes": len(r.classes),
                **{m: summary[m] for m in METRICS},
                "bbox_AP": r.metrics["bbox"]["summary"]["AP"],
                "n_common_classes": len(common),
                "AP_common": r.mean_ap(common),
                "AP50_common": r.mean_ap(common, "AP50"),
            }
        )
    return pd.DataFrame(rows)


def add_run_status(table: pd.DataFrame, status: Mapping[str, Mapping[str, Any]]) -> pd.DataFrame:
    """Add epochs and the cap flag per run (``status[run_id]`` as from ``run_summary``)."""
    table = table.copy()
    for col in ("epochs_done", "epochs_planned", "cap_hit"):
        table[col] = [status[r].get(col) for r in table["run_id"]]
    return table


def shift_table(main: pd.DataFrame) -> pd.DataFrame:
    """Change between sets per run. ``delta`` = later - earlier, ``relative`` = delta / earlier."""
    rows = []
    for run_id, g in main.groupby("run_id", sort=False):
        by_set = g.set_index("eval_set")
        for metric in ("AP", "AP_common"):
            for a, b in SHIFTS:
                first, second = float(by_set.loc[a, metric]), float(by_set.loc[b, metric])
                rows.append(
                    {
                        "run_id": run_id,
                        "metric": metric,
                        "from": a,
                        "to": b,
                        "ap_from": first,
                        "ap_to": second,
                        "delta": second - first,
                        "relative": (second - first) / first if first > 0 else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def condition_rows(results: Mapping[tuple[str, str], SetResult]) -> pd.DataFrame:
    """Per condition of every run and set: AP over all its classes and over shared classes."""
    rows = []
    for r in results.values():
        if not r.conditions:
            continue
        shared = sorted(
            set.intersection(*(set(t.index[t["n_gt"] > 0]) for t in r.conditions.values()))
        )
        for name, t in r.conditions.items():
            g = r.metrics["segm"]["by_image_group"][name]
            rows.append(
                {
                    "run_id": r.run_id,
                    "model": r.model,
                    "input_size": r.size,
                    "eval_set": r.eval_set,
                    "condition": name,
                    **{k: g[k] for k in ("n_images", "n_gt", "n_categories", "AP", "AP50")},
                    "n_common_classes": len(shared),
                    "n_gt_common": int(t.loc[shared, "n_gt"].sum()),
                    "AP_common": float(t.loc[shared, "AP"].mean()),
                    "AP50_common": float(t.loc[shared, "AP50"].mean()),
                }
            )
    return pd.DataFrame(rows)


def category_group_rows(results: Mapping[tuple[str, str], SetResult]) -> pd.DataFrame:
    """Things/stuff and in/absent-from-train AP per run and set (NaN where there is no GT)."""
    rows = []
    for r in results.values():
        for name, g in (r.metrics["segm"].get("by_category_group") or {}).items():
            rows.append(
                {
                    "run_id": r.run_id,
                    "model": r.model,
                    "input_size": r.size,
                    "eval_set": r.eval_set,
                    "group": name,
                    "AP": g["AP"],
                    "AP50": g["AP50"],
                }
            )
    return pd.DataFrame(rows).astype({"AP": float, "AP50": float})


def per_class_rows(results: Mapping[tuple[str, str], SetResult]) -> pd.DataFrame:
    """Long per-class table, classes with GT only."""
    frames = []
    for r in results.values():
        t = r.per_class[r.per_class["n_gt"] > 0].reset_index()
        t.insert(0, "eval_set", r.eval_set)
        t.insert(0, "input_size", r.size)
        t.insert(0, "model", r.model)
        frames.append(t)
    return pd.concat(frames, ignore_index=True)


def _x100(v: Any) -> str:
    return "n/a" if v is None or pd.isna(v) else f"{100 * v:.1f}"


def _signed(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{100 * v:+.1f}"


def _percent(v: float) -> str:
    return "n/a" if pd.isna(v) else f"{100 * v:+.0f}%"


def _md_table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return [*lines, *("| " + " | ".join(r) + " |" for r in rows), ""]


def render_summary(
    main: pd.DataFrame,
    shift: pd.DataFrame,
    conditions: pd.DataFrame,
    groups: pd.DataFrame,
    common: Sequence[str],
) -> str:
    """Markdown report of the tables (AP x 100)."""
    sets = main.drop_duplicates("eval_set").set_index("eval_set")
    label = _run_labels(main)
    counts = ", ".join(
        f"{s}: {sets.loc[s, 'n_classes']} classes / {sets.loc[s, 'n_gt']} instances /"
        f" {sets.loc[s, 'n_images']} images"
        for s in SETS
    )
    lines = [
        "# Results summary",
        "",
        f"Generated {datetime.now().astimezone().isoformat(timespec='seconds')} by"
        " `python -m iiss.analysis.domain_shift` from `results/eval/`. All values are mask"
        " AP x 100 (pycocotools, `last` checkpoint), unless noted.",
        "",
        f"- Ground truth: {counts}.",
        f"- *Common* columns average over the {len(common)} classes with GT in all three sets.",
        "- A run marked `*` hit the 2-hour training cap and is undertrained.",
        "- APs/APm/APl use the `area` stored in the GT (bounding-box area, as Roboflow exports"
        " it). On the phone set every object is large, so APs and APm are undefined.",
        "- The phone set has few instances per class, so small differences are not meaningful.",
        "- Inference time is not tabulated: the recorded times were affected by other load.",
        "",
        "## Mask AP by model, input size and eval set",
        "",
    ]
    rows = []
    for run_id, g in main.groupby("run_id", sort=False):
        b = g.set_index("eval_set")
        rows.append(
            [label[run_id], *(_x100(b.loc[s, "AP"]) for s in SETS)]
            + [_x100(b.loc[s, "AP50"]) for s in SETS]
            + [_x100(b.loc[s, "AP_common"]) for s in SETS]
        )
    lines += _md_table(
        ["Run"]
        + [f"AP {s}" for s in SETS]
        + [f"AP50 {s}" for s in SETS]
        + [f"AP {s} (common)" for s in SETS],
        rows,
    )
    for metric, title in (("AP_common", "common classes"), ("AP", "all classes with GT")):
        lines += [
            f"## Domain shift ({title})",
            "",
            "Change from the earlier to the later set: AP points (relative).",
            "",
        ]
        part = shift[shift["metric"] == metric]
        rows = [
            [label[run_id]]
            + [f"{_signed(r['delta'])} ({_percent(r['relative'])})" for _, r in g.iterrows()]
            for run_id, g in part.groupby("run_id", sort=False)
        ]
        lines += _md_table(["Run"] + [f"{a} -> {b}" for a, b in SHIFTS], rows)

    if not conditions.empty:
        first = conditions.drop_duplicates("condition").set_index("condition")
        names = list(first.index)
        desc = "; ".join(
            f"{n}: {first.loc[n, 'n_images']} images, {first.loc[n, 'n_gt']} instances,"
            f" {first.loc[n, 'n_categories']} classes"
            for n in names
        )
        n_common = int(first["n_common_classes"].iloc[0])
        lines += [
            "## Phone set by condition",
            "",
            f"{desc}. *Common* = the {n_common} classes with GT in every condition.",
            "",
        ]
        rows = []
        for run_id, g in conditions.groupby("run_id", sort=False):
            b = g.set_index("condition")
            rows.append(
                [label[run_id]]
                + [_x100(b.loc[n, "AP"]) for n in names]
                + [_x100(b.loc[n, "AP_common"]) for n in names]
            )
        lines += _md_table(
            ["Run"] + [f"AP {n}" for n in names] + [f"AP {n} (common)" for n in names], rows
        )

    lines += ["## Category groups (mask AP)", ""]
    rows = []
    for run_id, g in groups.groupby("run_id", sort=False):
        b = g.set_index(["eval_set", "group"])["AP"]
        rows.append(
            [label[run_id]]
            + [_x100(b.loc[s, n]) for s in SETS for n in ("things", "stuff")]
            + [_x100(b.loc["test", n]) for n in ("in_train", "absent_in_train")]
        )
    lines += _md_table(
        ["Run"]
        + [f"{s} {n}" for s in SETS for n in ("things", "stuff")]
        + ["test in train", "test absent in train"],
        rows,
    )
    return "\n".join(lines)


def _run_labels(main: pd.DataFrame) -> dict[str, str]:
    capped = main["cap_hit"] if "cap_hit" in main else pd.Series(False, index=main.index)
    out = {}
    for row, cap in zip(main.itertuples(), capped, strict=True):
        name = MODEL_NAMES.get(row.model, row.model)
        out[row.run_id] = f"{name} {row.input_size}{'*' if cap else ''}"
    return out


def run_ids(sizes: Iterable[int] = (320, 640, 960), seed: int = 42) -> list[str]:
    return [f"{m}_s{s}_seed{seed}" for m in VALID_MODELS for s in sizes]


def build_tables(
    eval_root: str | Path,
    status: Mapping[str, Mapping[str, Any]] | None = None,
    ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Load all results and build every table (see the module docstring)."""
    results = load_results(eval_root, ids or run_ids())
    common = common_classes(results, SETS)
    main = main_table(results, common)
    if status is not None:
        main = add_run_status(main, status)
    return {
        "common_classes": common,
        "main_results": main,
        "domain_shift": shift_table(main),
        "conditions": condition_rows(results),
        "category_groups": category_group_rows(results),
        "per_class_ap": per_class_rows(results),
    }


def write_tables(tables: Mapping[str, Any], out_dir: str | Path) -> Path:
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("main_results", "domain_shift", "conditions", "category_groups", "per_class_ap"):
        tables[name].to_csv(out / f"{name}.csv", index=False, float_format="%.6f")
    text = render_summary(
        tables["main_results"],
        tables["domain_shift"],
        tables["conditions"],
        tables["category_groups"],
        tables["common_classes"],
    )
    (out / "summary.md").write_text(text, encoding="utf-8")
    return out


def training_status(ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    """Epochs and cap flag of each run from its run directory and the GPU time log."""
    out = {}
    for run_id in ids:
        model, size = parse_run_id(run_id)
        out[run_id] = run_summary(notebook_config(model, size))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate evaluation results into tables")
    parser.add_argument("--out", help="output folder (default <results_root>/summary)")
    args = parser.parse_args(argv)
    setup_logging()
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    ids = run_ids()
    tables = build_tables(base["eval"]["results_root"], training_status(ids), ids)
    out = write_tables(tables, args.out or Path(base["paths"]["results_root"]) / "summary")
    logger.info("%d rows -> %s", len(tables["main_results"]), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
