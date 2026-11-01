"""Shared matplotlib styling. Colors follow the ENTITY (fixed slot per translator / source label), never its rank,
so adding a translator never repaints the existing ones. Palette: the dataviz reference categorical order."""
from __future__ import annotations

import matplotlib.pyplot as plt

from . import corpus, paths

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SOURCE_ORDER = ["Human", "BigScience", "Eleuther-AI", "LLAMA", "OpenAI", "PaLM", "Tsinghua"]


def translator_colors(names) -> dict:
    pref = list(corpus.research_config().get("display_names", {}))
    known = [n for n in names if n in pref]
    extra = sorted(n for n in names if n not in pref)
    slot = {n: pref.index(n) for n in known}
    nxt = len(pref)
    for n in extra:
        slot[n] = nxt
        nxt += 1
    return {n: SERIES[slot[n] % len(SERIES)] for n in names}


def source_colors(labels) -> dict:
    order = SOURCE_ORDER + sorted(set(labels) - set(SOURCE_ORDER))
    return {s: SERIES[order.index(s) % len(SERIES)] for s in labels}


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "bold", "axes.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 8,
        "lines.linewidth": 2, "lines.markersize": 6, "font.size": 9, "figure.dpi": 110})


def save(fig, name: str):
    """Save under results/research/figures/<name>.png (and keep it displayed inline)."""
    p = paths.results("figures") / f"{name}.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    return p
