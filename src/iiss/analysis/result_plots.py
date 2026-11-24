"""Figures of the results report, built from the tables of :mod:`iiss.analysis.domain_shift`.

Each model keeps one color and one marker in every figure. Capped (undertrained) runs are drawn
with a hollow marker and a ``*`` so that they are not read as a normal result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from iiss.analysis.domain_shift import MODEL_NAMES, SETS

# First three slots of the reference categorical palette (validated for all pairs).
MODEL_COLORS = {"yolo": "#2a78d6", "maskrcnn": "#eb6834", "m2f": "#1baf7a"}
MODEL_MARKERS = {"yolo": "o", "maskrcnn": "s", "m2f": "^"}
SHORT_NAMES = {"yolo": "YOLO11s", "maskrcnn": "Mask R-CNN", "m2f": "Mask2Former"}
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
# One hue, light to dark (sequential).
SEQUENTIAL = LinearSegmentedColormap.from_list("seq_blue", ["#f3f6fb", "#2a78d6", "#0b2f5e"])
METRIC_LABELS = {"AP": "Mask AP", "AP_common": "Mask AP (common classes)"}
LINE = 2.0
MARKER = 8


def _style(ax: plt.Axes, title: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, loc="left", fontsize=9, color=INK)


def _figure(rows: int, cols: int, size: tuple[float, float], **kwargs: Any):
    fig, axes = plt.subplots(rows, cols, figsize=size, squeeze=False, **kwargs)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _finish(fig: plt.Figure, save_to: str | Path | None, **kwargs: Any) -> plt.Figure:
    fig.tight_layout(**kwargs)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, facecolor=SURFACE)
    return fig


def _capped(row: Any) -> bool:
    return bool(getattr(row, "cap_hit", False))


def _point(ax: plt.Axes, x: float, y: float, model: str, hollow: bool) -> None:
    ax.plot(
        [x],
        [y],
        marker=MODEL_MARKERS[model],
        ms=MARKER,
        color=MODEL_COLORS[model],
        mfc=SURFACE if hollow else MODEL_COLORS[model],
        mec=MODEL_COLORS[model],
        mew=LINE,
        linestyle="none",
        zorder=3,
    )


def _end(row: Any, metric: str) -> tuple[float, str]:
    value = 100 * row[metric]
    return value, f"{value:.1f}{'*' if _capped(row) else ''}"


def _end_labels(ax: plt.Axes, x: float, ends: Sequence[tuple[float, str]]) -> None:
    """Value labels right of the last points, pushed apart so that they never overlap.

    Call after the y limits are final: the gap is computed in data units.
    """
    low, high = ax.get_ylim()
    gap = 0.052 * (high - low)  # about one line of 8 pt text in a 3.9 inch figure
    placed: list[float] = []
    for value, text in sorted(ends):
        y = max(value, placed[-1] + gap) if placed else value
        placed.append(y)
        ax.annotate(
            text,
            (x, y),
            textcoords="offset points",
            xytext=(9, -3),
            fontsize=8,
            color=INK,
        )


def _legend(fig: plt.Figure, models: Sequence[str], note: str | None = None) -> None:
    handles = [
        plt.Line2D(
            [],
            [],
            color=MODEL_COLORS[m],
            marker=MODEL_MARKERS[m],
            ms=MARKER,
            linewidth=LINE,
            label=MODEL_NAMES[m],
        )
        for m in models
    ]
    if note:
        handles.append(plt.Line2D([], [], linestyle="none", label=note))
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=8,
        labelcolor=INK,
    )


def _models(table: pd.DataFrame) -> list[str]:
    return [m for m in MODEL_COLORS if m in set(table["model"])]


def _cap_note(table: pd.DataFrame) -> str | None:
    if "cap_hit" in table and table["cap_hit"].any():
        return "hollow, *: hit the 2 h training cap"
    return None


def plot_ap_vs_size(
    main: pd.DataFrame, metric: str = "AP_common", save_to: str | Path | None = None
) -> plt.Figure:
    """AP against input size, one panel per eval set, one line per model."""
    fig, axes = _figure(1, len(SETS), (11, 3.9), sharey=True)
    sizes = sorted(main["input_size"].unique())
    ends: dict[str, list[tuple[float, str]]] = {}
    for ax, eval_set in zip(axes[0], SETS, strict=True):
        part = main[main["eval_set"] == eval_set]
        _style(ax, f"{eval_set} ({int(part['n_images'].iloc[0])} images)")
        for model in _models(main):
            g = part[part["model"] == model].sort_values("input_size")
            ax.plot(
                g["input_size"],
                100 * g[metric],
                color=MODEL_COLORS[model],
                linewidth=LINE,
                zorder=2,
            )
            for row in g.itertuples():
                _point(ax, row.input_size, 100 * getattr(row, metric), model, _capped(row))
            ends.setdefault(eval_set, []).append(_end(g.iloc[-1], metric))
        ax.set_xticks(sizes)
        ax.set_xlim(sizes[0] - 60, sizes[-1] + 170)
        ax.set_xlabel("input size S", color=INK_2, fontsize=8)
    axes[0][0].set_ylim(bottom=0)
    for ax, eval_set in zip(axes[0], SETS, strict=True):
        _end_labels(ax, sizes[-1], ends[eval_set])
    axes[0][0].set_ylabel(f"{METRIC_LABELS[metric]} x 100", color=INK_2, fontsize=8)
    _legend(fig, _models(main), _cap_note(main))
    return _finish(fig, save_to, rect=(0, 0.07, 1, 1))


def plot_domain_shift(
    main: pd.DataFrame, metric: str = "AP_common", save_to: str | Path | None = None
) -> plt.Figure:
    """Slope chart valid -> test -> phone, one panel per input size."""
    sizes = sorted(main["input_size"].unique())
    fig, axes = _figure(1, len(sizes), (11, 3.9), sharey=True)
    ends: dict[int, list[tuple[float, str]]] = {}
    for ax, size in zip(axes[0], sizes, strict=True):
        part = main[main["input_size"] == size]
        _style(ax, f"S = {size}")
        for model in _models(main):
            g = part[part["model"] == model].set_index("eval_set").loc[list(SETS)].reset_index()
            x = np.arange(len(SETS))
            ax.plot(x, 100 * g[metric], color=MODEL_COLORS[model], linewidth=LINE, zorder=2)
            for i, row in enumerate(g.itertuples()):
                _point(ax, i, 100 * getattr(row, metric), model, _capped(row))
            ends.setdefault(size, []).append(_end(g.iloc[-1], metric))
        ax.set_xticks(range(len(SETS)), SETS)
        ax.set_xlim(-0.2, len(SETS) - 0.6)
    axes[0][0].set_ylim(bottom=0)
    for ax, size in zip(axes[0], sizes, strict=True):
        _end_labels(ax, len(SETS) - 1, ends[size])
    axes[0][0].set_ylabel(f"{METRIC_LABELS[metric]} x 100", color=INK_2, fontsize=8)
    _legend(fig, _models(main), _cap_note(main))
    return _finish(fig, save_to, rect=(0, 0.07, 1, 1))


def plot_conditions(
    conditions: pd.DataFrame,
    main: pd.DataFrame,
    metric: str = "AP_common",
    base: str = "normal",
    save_to: str | Path | None = None,
) -> plt.Figure:
    """Dumbbell per run: ``base`` condition (filled) against each other condition (hollow)."""
    capped = set(main.loc[main["cap_hit"], "run_id"]) if "cap_hit" in main else set()
    others = [c for c in dict.fromkeys(conditions["condition"]) if c != base]
    runs = list(dict.fromkeys(conditions["run_id"]))
    info = conditions.drop_duplicates("run_id").set_index("run_id")
    by_run = conditions.set_index(["run_id", "condition"])[metric]
    xlabel = METRIC_LABELS[metric]
    if metric == "AP_common":  # not the classes common to the eval sets
        n = int(conditions["n_common_classes"].iloc[0])
        xlabel = f"Mask AP ({n} classes with GT in both conditions)"
    fig, axes = _figure(1, len(others), (6.2 * len(others) + 1, 0.42 * len(runs) + 1.6))
    for ax, other in zip(axes[0], others, strict=True):
        _style(ax, f"{other} (hollow, left value) vs {base} (filled, right value)")
        ax.grid(False)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        for y, run in enumerate(runs):
            model = info.loc[run, "model"]
            a, b = 100 * by_run[run, base], 100 * by_run[run, other]
            ax.plot([a, b], [y, y], color=MODEL_COLORS[model], linewidth=LINE, zorder=2)
            _point(ax, a, y, model, hollow=False)
            _point(ax, b, y, model, hollow=True)
            ax.annotate(
                f"{b:.1f} | {a:.1f}",
                (max(a, b), y),
                textcoords="offset points",
                xytext=(10, -3),
                fontsize=8,
                color=INK,
            )
        labels = [
            f"{SHORT_NAMES[info.loc[r, 'model']]} {info.loc[r, 'input_size']}"
            + ("*" if r in capped else "")
            for r in runs
        ]
        ax.set_yticks(range(len(runs)), labels, color=INK)
        ax.invert_yaxis()
        ax.set_xlim(-0.8, 100 * conditions[metric].max() * 1.25 + 1)
        ax.set_xlabel(f"{xlabel} x 100", color=INK_2, fontsize=8)
    _legend(fig, _models(conditions), "* hit the 2 h cap" if capped else None)
    return _finish(fig, save_to, rect=(0, 0.07, 1, 1))


def heatmap_table(
    per_class: pd.DataFrame, classes: Sequence[str], eval_set: str
) -> tuple[pd.DataFrame, pd.Series]:
    """AP x 100 (rows = classes, columns = (model, size)) and the GT count per class."""
    part = per_class[(per_class["eval_set"] == eval_set) & per_class["name"].isin(list(classes))]
    grid = part.pivot_table(index="name", columns=["model", "input_size"], values="AP")
    columns = [
        (m, s) for m in MODEL_COLORS for s in sorted(part["input_size"].unique()) if (m, s) in grid
    ]
    counts = part.groupby("name")["n_gt"].first()
    return 100 * grid.loc[list(classes), columns], counts.loc[list(classes)]


def plot_per_class_heatmap(
    per_class: pd.DataFrame,
    classes: Sequence[str],
    save_to: str | Path | None = None,
    capped: Sequence[tuple[str, int]] = (),
) -> plt.Figure:
    """Per-class AP of every run, one panel per eval set, the same classes in every panel.

    Classes are ordered by their GT count on the phone set. The GT count of every set is printed
    beside the class name. ``capped`` holds the (model, size) pairs that get a ``*``.
    """
    phone_counts = per_class[per_class["eval_set"] == "phone"].groupby("name")["n_gt"].first()
    order = sorted(classes, key=lambda c: (-int(phone_counts.get(c, 0)), c))
    fig, axes = _figure(1, len(SETS), (15.5, 0.36 * len(order) + 1.9))
    image = None
    counts = {e: heatmap_table(per_class, order, e)[1] for e in SETS}
    names = [f"{c} ({' / '.join(str(counts[e][c]) for e in SETS)})" for c in order]
    for ax, eval_set in zip(axes[0], SETS, strict=True):
        grid, _ = heatmap_table(per_class, order, eval_set)
        image = ax.imshow(grid.to_numpy(), cmap=SEQUENTIAL, vmin=0, vmax=100, aspect="auto")
        for (i, j), v in np.ndenumerate(grid.to_numpy()):
            if not np.isnan(v):
                ax.text(
                    j,
                    i,
                    f"{v:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="white" if v >= 55 else INK,
                )
        if ax is axes[0][0]:
            ax.set_yticks(range(len(order)), names)
        else:
            ax.set_yticks([])
        ax.set_xticks(
            range(grid.shape[1]),
            [f"{SHORT_NAMES[m]} {s}{'*' if (m, s) in capped else ''}" for m, s in grid.columns],
            rotation=60,
            ha="right",
        )
        ax.set_title(eval_set, loc="left", fontsize=9, color=INK)
        ax.tick_params(colors=INK_2, labelsize=7, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        # surface-colored gaps between cells
        ax.set_xticks(np.arange(-0.5, grid.shape[1]), minor=True)
        ax.set_yticks(np.arange(-0.5, len(order)), minor=True)
        ax.grid(which="minor", color=SURFACE, linewidth=2)
        ax.tick_params(which="minor", length=0)
    axes[0][0].set_xlabel(
        f"brackets: GT instances in {' / '.join(SETS)}"
        + ("; * hit the 2 h training cap" if capped else ""),
        loc="left",
        color=INK_2,
        fontsize=8,
    )
    fig.tight_layout()
    bar = fig.colorbar(image, ax=axes[0].tolist(), shrink=0.6, pad=0.01)
    bar.set_label("Mask AP x 100", color=INK_2, fontsize=8)
    bar.ax.tick_params(colors=INK_2, labelsize=7)
    bar.outline.set_visible(False)
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    return fig


def capped_runs(main: pd.DataFrame) -> list[tuple[str, int]]:
    """(model, input size) of the runs that hit the training cap."""
    if "cap_hit" not in main:
        return []
    rows = main.loc[main["cap_hit"].astype(bool), ["model", "input_size"]].drop_duplicates()
    return list(rows.itertuples(index=False, name=None))


def wide_table(main: pd.DataFrame, metric: str) -> pd.DataFrame:
    """``metric`` x 100: one row per run (``*`` = capped), one column per eval set."""
    capped = capped_runs(main)
    grid = 100 * main.pivot(index=["model", "input_size"], columns="eval_set", values=metric)
    order = list(dict.fromkeys(zip(main["model"], main["input_size"], strict=True)))
    grid = grid.loc[order, list(SETS)]
    grid.index = [f"{MODEL_NAMES[m]} {s}{'*' if (m, s) in capped else ''}" for m, s in order]
    grid.columns.name = None
    return grid


def save_all(tables: Mapping[str, Any], out_dir: str | Path) -> dict[str, plt.Figure]:
    """Write every figure of the report to ``out_dir`` and return the figures by file name."""
    out = Path(out_dir)
    main = tables["main_results"]
    common = tables["common_classes"]
    capped = capped_runs(main)
    return {
        "ap_vs_size_common.png": plot_ap_vs_size(main, "AP_common", out / "ap_vs_size_common.png"),
        "ap_vs_size_all.png": plot_ap_vs_size(main, "AP", out / "ap_vs_size_all.png"),
        "domain_shift_common.png": plot_domain_shift(
            main, "AP_common", out / "domain_shift_common.png"
        ),
        "phone_conditions.png": plot_conditions(
            tables["conditions"], main, save_to=out / "phone_conditions.png"
        ),
        "per_class_heatmap.png": plot_per_class_heatmap(
            tables["per_class_ap"], common, out / "per_class_heatmap.png", capped
        ),
    }
