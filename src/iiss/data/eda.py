"""Exploratory data analysis of the COCO export -> ``results/eda/``.

Outputs (all derived from the annotation JSON only; no pixels are read):

- ``class_counts.csv``      instances and images per class and split
- ``class_shift.csv``       train vs test class distribution shift
- ``image_sizes.csv``       width x height counts per split
- ``long_side_stats.csv``   longest image side per split, and share of images up/down-scaled at
  each input size S
- ``instances_per_image.csv`` distribution of instances per image
- ``object_sizes.csv``      COCO size bins (small/medium/large) at native resolution and after
  resizing the longest side to S
- ``summary.md``            key numbers from the tables above
- ``*.png``                 figures

Usage::

    python -m iiss.data.eda
"""

from __future__ import annotations

import argparse
import logging
import math
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iiss.data.coco import DEFAULT_ANNOTATION_FILE, CocoSplit, load_split
from iiss.utils.config import load_yaml
from iiss.utils.log import setup_logging
from iiss.utils.paths import repo_root, resolve_path

logger = logging.getLogger(__name__)

INPUT_SIZES = (320, 640, 960)
# COCOeval area bins (pixels^2 at the evaluated resolution).
SMALL_MAX = 32**2
MEDIUM_MAX = 96**2

# Reference categorical palette (validated slots 1-3) and neutral inks.
SPLIT_COLORS = {"train": "#2a78d6", "valid": "#eb6834", "test": "#1baf7a"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"


# ---------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------


def size_bin(area: float) -> str:
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def class_counts(splits: dict[str, CocoSplit]) -> pd.DataFrame:
    """Instances and images-containing-class per category and split."""
    first = next(iter(splits.values()))
    df = pd.DataFrame(
        {"id": list(first.category_names), "name": list(first.category_names.values())}
    )
    for name, split in splits.items():
        inst = pd.Series([a["category_id"] for a in split.annotations]).value_counts()
        imgs = (
            pd.DataFrame(
                {(a["image_id"], a["category_id"]) for a in split.annotations},
                columns=["image_id", "category_id"],
            )
            .groupby("category_id")
            .size()
        )
        df[f"{name}_instances"] = df["id"].map(inst).fillna(0).astype(int)
        df[f"{name}_images"] = df["id"].map(imgs).fillna(0).astype(int)
    return df


def class_shift(counts: pd.DataFrame, ref: str = "train", target: str = "test") -> pd.DataFrame:
    """Share of instances per class in ``ref`` vs ``target``, sorted by absolute share change."""
    a, b = counts[f"{ref}_instances"], counts[f"{target}_instances"]
    df = counts[["id", "name"]].copy()
    df[f"{ref}_instances"] = a
    df[f"{target}_instances"] = b
    df[f"{ref}_share_pct"] = (100 * a / a.sum()).round(3)
    df[f"{target}_share_pct"] = (100 * b / b.sum()).round(3)
    df["share_diff_pp"] = (df[f"{target}_share_pct"] - df[f"{ref}_share_pct"]).round(3)
    ratio = df[f"{target}_share_pct"] / df[f"{ref}_share_pct"].replace(0, np.nan)
    df["share_ratio"] = ratio.round(3)
    df["status"] = np.select(
        [(a == 0) & (b == 0), a == 0, b == 0],
        ["absent_in_both", f"absent_in_{ref}", f"absent_in_{target}"],
        default="present_in_both",
    )
    df = df[df["status"] != "absent_in_both"]
    return df.reindex(df["share_diff_pp"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


def image_sizes(splits: dict[str, CocoSplit]) -> pd.DataFrame:
    rows = [
        {"split": name, "width": img["width"], "height": img["height"]}
        for name, split in splits.items()
        for img in split.images
    ]
    df = pd.DataFrame(rows)
    out = df.groupby(["split", "width", "height"]).size().rename("images").reset_index()
    order = {name: i for i, name in enumerate(splits)}
    return out.sort_values(
        ["split", "images"], key=lambda c: c.map(order) if c.name == "split" else -c
    ).reset_index(drop=True)


def long_sides(split: CocoSplit) -> np.ndarray:
    return np.array([max(img["width"], img["height"]) for img in split.images], dtype=float)


def long_side_stats(splits: dict[str, CocoSplit]) -> pd.DataFrame:
    rows = []
    for name, split in splits.items():
        ls = long_sides(split)
        row = {
            "split": name,
            "images": len(ls),
            "min": int(ls.min()),
            "p25": float(np.percentile(ls, 25)),
            "median": float(np.median(ls)),
            "p75": float(np.percentile(ls, 75)),
            "max": int(ls.max()),
        }
        for s in INPUT_SIZES:
            row[f"upscaled_pct_at_{s}"] = round(100 * float((ls < s).mean()), 2)
            row[f"downscaled_pct_at_{s}"] = round(100 * float((ls > s).mean()), 2)
        rows.append(row)
    return pd.DataFrame(rows)


def instances_per_image(splits: dict[str, CocoSplit]) -> pd.DataFrame:
    rows = []
    for name, split in splits.items():
        n = np.array([len(split.annotations_by_image.get(img["id"], [])) for img in split.images])
        rows.append(
            {
                "split": name,
                "images": len(n),
                "mean": round(float(n.mean()), 2),
                "median": float(np.median(n)),
                "p90": float(np.percentile(n, 90)),
                "max": int(n.max()),
                "zero": int((n == 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def instance_table(split: CocoSplit) -> pd.DataFrame:
    """One row per annotation with its area and the longest side of its image."""
    rows = []
    for a in split.annotations:
        img = split.images_by_id[a["image_id"]]
        rows.append(
            {
                "image_id": a["image_id"],
                "category_id": a["category_id"],
                "area": float(a["area"]),
                "long_side": max(img["width"], img["height"]),
            }
        )
    return pd.DataFrame(rows)


def object_sizes(splits: dict[str, CocoSplit]) -> pd.DataFrame:
    """Share of instances per COCO size bin at native resolution and after resizing to S.

    Resizing the longest side to S scales every length by ``S / long_side``, so the area scales
    by its square. COCOeval bins (APs/APm/APl) use the area at the evaluated resolution, which
    in this project is the original image resolution ("native").
    """
    rows = []
    for name, split in splits.items():
        inst = instance_table(split)
        scales = {"native": np.ones(len(inst))}
        for s in INPUT_SIZES:
            scales[f"S={s}"] = (s / inst["long_side"].to_numpy()) ** 2
        for label, factor in scales.items():
            bins = pd.Series([size_bin(a) for a in inst["area"].to_numpy() * factor])
            counts = bins.value_counts()
            row = {"split": name, "resolution": label, "instances": len(inst)}
            for b in ("small", "medium", "large"):
                row[f"{b}_pct"] = round(100 * counts.get(b, 0) / len(inst), 2)
            sqrt_area = np.sqrt(inst["area"].to_numpy() * factor)
            row["median_sqrt_area_px"] = round(float(np.median(sqrt_area)), 1)
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)
    ax.set_axisbelow(True)


def _figure(*args, **kwargs) -> tuple[plt.Figure, np.ndarray]:
    fig, axes = plt.subplots(*args, squeeze=False, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        _style(ax)
    return fig, axes


def plot_class_counts(counts: pd.DataFrame, split_names: list[str], out: Path) -> None:
    df = counts[counts["id"] != 0].sort_values(f"{split_names[0]}_instances")
    n = len(df)
    fig, axes = _figure(1, 1, figsize=(8, 0.22 * n + 1.2))
    ax = axes[0, 0]
    height = 0.8 / len(split_names)
    y = np.arange(n)
    for i, split in enumerate(split_names):
        ax.barh(
            y + (i - (len(split_names) - 1) / 2) * height,
            df[f"{split}_instances"],
            height=height * 0.85,
            color=SPLIT_COLORS.get(split),
            label=split,
        )
    ax.set_yticks(y, df["name"], fontsize=7, color=INK)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("instances (symlog scale)")
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_ylim(-0.7, n - 0.3)
    ax.legend(loc="lower right", frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("Instances per class and split (sorted by train count)", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _hist_panels(
    values: dict[str, np.ndarray],
    bins: np.ndarray,
    xlabel: str,
    title: str,
    out: Path,
    vlines: tuple[int, ...] = (),
) -> None:
    fig, axes = _figure(len(values), 1, figsize=(7, 1.9 * len(values) + 0.6), sharex=True)
    for ax, (split, v) in zip(axes[:, 0], values.items(), strict=True):
        ax.hist(v, bins=bins, color=SPLIT_COLORS.get(split), edgecolor=SURFACE, linewidth=1)
        for x in vlines:
            ax.axvline(x, color=INK_2, linewidth=1, linestyle="--")
            ax.text(
                x,
                1.0,
                f" S={x}",
                transform=ax.get_xaxis_transform(),
                color=INK_2,
                fontsize=7,
                va="top",
            )
        ax.set_ylabel(f"{split}\nimages", fontsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
    axes[-1, 0].set_xlabel(xlabel)
    axes[0, 0].set_title(title, fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_object_small_share(sizes: pd.DataFrame, split_names: list[str], out: Path) -> None:
    fig, axes = _figure(1, 1, figsize=(6, 3.6))
    ax = axes[0, 0]
    for split in split_names:
        d = sizes[(sizes["split"] == split) & (sizes["resolution"] != "native")]
        x = [int(r.split("=")[1]) for r in d["resolution"]]
        ax.plot(
            x,
            d["small_pct"],
            color=SPLIT_COLORS.get(split),
            linewidth=2,
            marker="o",
            markersize=6,
            label=split,
        )
        ax.annotate(
            split,
            (x[-1], d["small_pct"].iloc[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=INK_2,
            fontsize=8,
            va="center",
        )
    ax.set_xticks(list(INPUT_SIZES))
    ax.set_xlim(250, 1060)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("input size S (longest side, px)")
    ax.set_ylabel("instances < 32x32 px after resize (%)")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    ax.set_title("Share of small objects as seen by the model", fontsize=10, loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------------


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in df.itertuples(index=False):
        cells = ["" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_summary(
    root: Path,
    counts: pd.DataFrame,
    shift: pd.DataFrame,
    ls_stats: pd.DataFrame,
    ipi: pd.DataFrame,
    sizes: pd.DataFrame,
) -> str:
    rel = root.relative_to(repo_root()).as_posix() if root.is_relative_to(repo_root()) else root
    absent_train = shift[shift["status"] == "absent_in_train"]
    absent_test = shift[shift["status"] == "absent_in_test"]
    present = shift[shift["status"] == "present_in_both"]
    top_cols = [
        "name",
        "train_instances",
        "test_instances",
        "train_share_pct",
        "test_share_pct",
        "share_diff_pp",
    ]
    parts = [
        "# EDA summary",
        "",
        f"- Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"- Dataset root: `{rel}`",
        "- Command: `python -m iiss.data.eda`",
        "- Source: annotation JSON only. Generated file; do not edit by hand.",
        "",
        "## Longest image side (px) and resize direction per input size S",
        "",
        _md_table(ls_stats),
        "",
        "`upscaled_pct_at_S` = share of images whose longest side is smaller than S.",
        "",
        "## Instances per image",
        "",
        _md_table(ipi),
        "",
        "## Object size bins (COCO: small < 32^2, medium < 96^2, large >= 96^2 px^2)",
        "",
        "`native` = original resolution (what COCOeval APs/APm/APl use).",
        "`S=...` = after resizing the longest side to S (what the model sees).",
        "",
        _md_table(sizes),
        "",
        "## Train vs test class shift",
        "",
        f"Classes present in test but absent in train ({len(absent_train)}): "
        + ", ".join(f"`{r.name}` ({r.test_instances})" for r in absent_train.itertuples()),
        "",
        f"Classes present in train but absent in test ({len(absent_test)}): "
        + ", ".join(f"`{r.name}` ({r.train_instances})" for r in absent_test.itertuples()),
        "",
        "Largest share increases from train to test (percentage points):",
        "",
        _md_table(present.nlargest(10, "share_diff_pp")[top_cols]),
        "",
        "Largest share decreases from train to test:",
        "",
        _md_table(present.nsmallest(10, "share_diff_pp")[top_cols]),
        "",
        "Full tables: `class_counts.csv`, `class_shift.csv`.",
        "",
        "## Figures",
        "",
        "- `class_counts.png`",
        "- `long_side_hist.png`",
        "- `instances_per_image_hist.png`",
        "- `small_objects_vs_S.png`",
        "",
        "Total instances: "
        + ", ".join(
            f"{c.removesuffix('_instances')} {int(counts[c].sum())}"
            for c in counts.columns
            if c.endswith("_instances")
        ),
    ]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------------------------


def run_eda(
    dataset_root: str | Path,
    split_names: list[str],
    out_dir: str | Path,
    annotation_file: str = DEFAULT_ANNOTATION_FILE,
) -> dict[str, Path]:
    """Compute all tables and figures and write them to ``out_dir``."""
    root = resolve_path(dataset_root)
    out = resolve_path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    splits = {name: load_split(root, name, annotation_file) for name in split_names}

    counts = class_counts(splits)
    ref, target = split_names[0], split_names[-1]
    shift = class_shift(counts, ref, target)
    sizes_tbl = image_sizes(splits)
    ls_stats = long_side_stats(splits)
    ipi = instances_per_image(splits)
    obj = object_sizes(splits)

    files = {
        "class_counts": out / "class_counts.csv",
        "class_shift": out / "class_shift.csv",
        "image_sizes": out / "image_sizes.csv",
        "long_side_stats": out / "long_side_stats.csv",
        "instances_per_image": out / "instances_per_image.csv",
        "object_sizes": out / "object_sizes.csv",
    }
    counts.to_csv(files["class_counts"], index=False)
    shift.to_csv(files["class_shift"], index=False)
    sizes_tbl.to_csv(files["image_sizes"], index=False)
    ls_stats.to_csv(files["long_side_stats"], index=False)
    ipi.to_csv(files["instances_per_image"], index=False)
    obj.to_csv(files["object_sizes"], index=False)

    files["class_counts_png"] = out / "class_counts.png"
    plot_class_counts(counts, split_names, files["class_counts_png"])

    files["long_side_png"] = out / "long_side_hist.png"
    _hist_panels(
        {n: long_sides(s) for n, s in splits.items()},
        bins=np.arange(0, 4200, 80),
        xlabel="longest image side (px)",
        title="Longest image side per split",
        out=files["long_side_png"],
        vlines=INPUT_SIZES,
    )

    files["ipi_png"] = out / "instances_per_image_hist.png"
    _hist_panels(
        {
            n: np.array([len(s.annotations_by_image.get(i["id"], [])) for i in s.images])
            for n, s in splits.items()
        },
        bins=np.arange(0, 135, 3),
        xlabel="instances per image",
        title="Instances per image",
        out=files["ipi_png"],
    )

    files["small_png"] = out / "small_objects_vs_S.png"
    plot_object_small_share(obj, split_names, files["small_png"])

    files["summary"] = out / "summary.md"
    files["summary"].write_text(
        render_summary(root, counts, shift, ls_stats, ipi, obj), encoding="utf-8"
    )
    logger.info("Wrote %d EDA files to %s", len(files), out)
    return files


def main(argv: list[str] | None = None) -> int:
    base = load_yaml(repo_root() / "configs" / "base.yaml")
    parser = argparse.ArgumentParser(description="Dataset EDA -> results/eda/")
    parser.add_argument("--dataset-root", default=base["paths"]["dataset_root"])
    parser.add_argument("--splits", nargs="+", default=list(base["data"]["splits"].values()))
    parser.add_argument("--annotation-file", default=base["data"]["annotation_file"])
    parser.add_argument("--out-dir", default=str(Path(base["paths"]["results_root"]) / "eda"))
    args = parser.parse_args(argv)
    matplotlib.use("Agg")  # CLI only; the notebook keeps its inline backend
    setup_logging()
    run_eda(args.dataset_root, args.splits, args.out_dir, args.annotation_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
