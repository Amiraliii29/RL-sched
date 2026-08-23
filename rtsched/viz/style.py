"""Shared figure styling.

Colours come from a validated categorical palette; only the first three slots are used so
every pairing clears the colour-vision-deficiency separation floor under the all-pairs rule.
Series identity is always carried by a legend *and* a distinct marker, never by hue alone, and
every figure is written alongside the CSV it was drawn from.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#dcdbd6"

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
SEQUENTIAL = "Blues"

STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
MARKERS = ("o", "s", "^", "D")

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.labelsize": 10,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK,
    "legend.frameon": False,
    "legend.fontsize": 9,
    "lines.linewidth": 2.0,
    "lines.markersize": 6.5,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
}


def use_style() -> None:
    plt.rcParams.update(RC)


def series_style(index: int) -> dict:
    return {"color": SERIES[index % len(SERIES)], "marker": MARKERS[index % len(MARKERS)]}


def finish(ax, title: str, xlabel: str, ylabel: str, legend: bool = True) -> None:
    ax.set_title(title, loc="left", pad=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if legend and ax.get_legend_handles_labels()[0]:
        ax.legend(loc="best")
