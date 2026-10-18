"""Grad-CAM figures: one lead on ECG paper with the normalised map as a shaded band behind the trace.

Every function draws real arrays passed in by src/explain/gradcam_val.py; nothing here computes a map. The band
is a highlighter in the frontend's teal accent (PROJECT_SPEC.md section 7): transparent at 0, so the ECG grid
stays readable where the map is low, and band_alpha opaque at 1. The trace is drawn black on top.

Layouts are set in inches (header, plot, footer), so long titles wrap instead of widening the figure.
"""

from __future__ import annotations

import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, no window; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

from src.evaluation.plots import ACCENT, INK, INK_SECONDARY, MUTED, SURFACE, _save  # noqa: E402

GRID_MAJOR, GRID_MINOR = "#E3A3A3", "#F4D9D9"   # ECG paper, as in reports/figures/sample_record.png
TRACE_WIDTH = 0.8
CAPTION = "Model attention (Grad-CAM) — not a clinical rationale."

# Text sizes (points) and the layout bands they need (inches).
TITLE_PT, SUBTITLE_PT, LABEL_PT, CAPTION_PT = 12, 8.5, 8.5, 8.5
CHAR_EM = 0.6                 # average glyph width as a share of the font size, slightly generous, for wrapping
LINE_SPACING = 1.45           # line height as a multiple of the font size
EDGE_IN = 0.15                # blank margin at the figure's top
GAP_IN = 0.25                 # between the header and the plot
X_AXIS_IN = 0.5               # tick labels and the x-axis label under a plot


def band_cmap(alpha: float) -> LinearSegmentedColormap:
    """Teal highlighter: fully transparent at 0, `alpha` opaque at 1."""
    if not 0 < alpha <= 1:
        raise ValueError(f"band alpha must be in (0, 1], got {alpha}")
    r, g, b = to_rgb(ACCENT)
    return LinearSegmentedColormap.from_list("attention", [(r, g, b, 0.0), (r, g, b, alpha)])


def _line_in(pt: float) -> float:
    return pt * LINE_SPACING / 72


def _wrap(text: str, width_in: float, pt: float) -> list[str]:
    return textwrap.wrap(text, width=max(20, int(width_in / (CHAR_EM * pt / 72))))


def _header(fig: plt.Figure, title: str, subtitle: str, width_in: float, height_in: float, left_in: float) -> None:
    """Title and subtitle, wrapped to the figure width, from the top edge down."""
    y = EDGE_IN
    for lines, pt, weight, color in ((_wrap(title, width_in - 2 * left_in, TITLE_PT), TITLE_PT, "bold", INK),
                                     (_wrap(subtitle, width_in - 2 * left_in, SUBTITLE_PT), SUBTITLE_PT, "normal",
                                      INK_SECONDARY)):
        fig.text(left_in / width_in, 1 - y / height_in, "\n".join(lines), ha="left", va="top", fontsize=pt,
                 fontweight=weight, color=color, linespacing=LINE_SPACING)
        y += len(lines) * _line_in(pt)


def header_height(title: str, subtitle: str, width_in: float, left_in: float) -> float:
    text_w = width_in - 2 * left_in
    return (EDGE_IN + len(_wrap(title, text_w, TITLE_PT)) * _line_in(TITLE_PT)
            + len(_wrap(subtitle, text_w, SUBTITLE_PT)) * _line_in(SUBTITLE_PT) + GAP_IN)


def _y_limits(signal: np.ndarray, major_mv: float, min_height_mv: float) -> tuple[float, float]:
    """Whole major grid squares around the signal, at least min_height_mv tall, centred on it."""
    lo, hi = float(np.min(signal)), float(np.max(signal))
    if hi - lo < min_height_mv:
        mid = (lo + hi) / 2
        lo, hi = mid - min_height_mv / 2, mid + min_height_mv / 2
    return np.floor(lo / major_mv) * major_mv, np.ceil(hi / major_mv) * major_mv


def _check(signal: np.ndarray, cam: np.ndarray) -> None:
    if signal.ndim != 1 or cam.shape != signal.shape:
        raise ValueError(f"signal and map must be 1-D of equal length, got {signal.shape} and {cam.shape}")
    if not (np.isfinite(cam).all() and cam.min() >= 0 and cam.max() <= 1):
        raise ValueError("The map must be finite and normalised to [0, 1]")


def draw_band(ax: plt.Axes, cam: np.ndarray, fs: float, ylim: tuple[float, float], cmap: LinearSegmentedColormap) -> None:
    """The map as a full-height shade above the grid and below the trace. Value i is centred on t = i / fs, where the
    trace draws sample i, so it covers [(i - 0.5) / fs, (i + 0.5) / fs)."""
    ax.imshow(cam[None, :], extent=(-0.5 / fs, (len(cam) - 0.5) / fs, *ylim), aspect="auto", cmap=cmap, vmin=0,
              vmax=1, interpolation="nearest", zorder=1)


def _ecg_axes(ax: plt.Axes, n: int, fs: float, ylim: tuple[float, float], grid: Mapping[str, float],
              minor: bool) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_xlim(0, n / fs)
    ax.set_ylim(*ylim)
    ax.xaxis.set_major_locator(MultipleLocator(grid["grid_major_s"]))
    ax.yaxis.set_major_locator(MultipleLocator(grid["grid_major_mv"]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if np.isclose(v, round(v)) else ""))
    ax.grid(which="major", color=GRID_MAJOR, linewidth=0.5)
    if minor:
        ax.xaxis.set_minor_locator(MultipleLocator(grid["grid_minor_s"]))
        ax.yaxis.set_minor_locator(MultipleLocator(grid["grid_minor_mv"]))
        ax.grid(which="minor", color=GRID_MINOR, linewidth=0.3)
    ax.set_axisbelow(True)  # grid under the band, which is transparent where the map is 0
    ax.tick_params(which="both", colors=MUTED, labelcolor=INK_SECONDARY, labelsize=7.5)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_color(MUTED)
        side.set_linewidth(0.6)


def plot_overlay(signal_mv: np.ndarray, cam: np.ndarray, centres: np.ndarray, fs: float, lead: str,
                 grid: Mapping[str, float], path: Path, *, title: str, subtitle: str, footnote: str, band_alpha: float,
                 width_in: float, min_height_mv: float, dpi: int) -> Path:
    """One record on ECG paper (25 mm/s, 10 mm/mV), the band behind it, and a tick per feature position on top."""
    _check(signal_mv, cam)
    cmap = band_cmap(band_alpha)
    ylim = _y_limits(signal_mv, grid["grid_major_mv"], min_height_mv)
    duration = len(signal_mv) / fs
    left, right = 0.75, 1.05           # y-axis labels; colour bar and its labels
    plot_w = width_in - left - right
    # ECG paper proportions: 1 s spans paper_speed mm, 1 mV spans gain mm.
    plot_h = plot_w * (ylim[1] - ylim[0]) * grid["gain_mm_per_mv"] / (duration * grid["paper_speed_mm_per_s"])
    head = header_height(title, subtitle, width_in, left)
    foot_lines = _wrap(f"{CAPTION} {footnote}", plot_w, CAPTION_PT)
    foot = X_AXIS_IN + len(foot_lines) * _line_in(CAPTION_PT) + EDGE_IN
    height = head + plot_h + foot
    fig = plt.figure(figsize=(width_in, height), facecolor=SURFACE)
    ax = fig.add_axes((left / width_in, foot / height, plot_w / width_in, plot_h / height))
    cax = fig.add_axes(((left + plot_w + 0.15) / width_in, foot / height, 0.14 / width_in, plot_h / height))

    _ecg_axes(ax, len(signal_mv), fs, ylim, grid, minor=True)
    draw_band(ax, cam, fs, ylim, cmap)
    ax.plot(np.arange(len(signal_mv)) / fs, signal_mv, color=INK, linewidth=TRACE_WIDTH, zorder=3)
    # The band has one value per tick (feature-position centre) and is interpolated in between.
    ax.plot(centres / fs, np.full(len(centres), ylim[1]), linestyle="none", marker="|", markersize=7,
            markeredgewidth=1.2, color=ACCENT, clip_on=False, zorder=4)
    ax.set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=LABEL_PT)
    ax.set_ylabel(f"Lead {lead} (mV)", color=INK_SECONDARY, fontsize=LABEL_PT)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax)
    cbar.set_label("Grad-CAM, normalised", color=INK_SECONDARY, fontsize=7.5)
    cbar.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_SECONDARY)
    cbar.outline.set_edgecolor(MUTED)
    _header(fig, title, subtitle, width_in, height, left * 0.25)
    fig.text(left / width_in, (EDGE_IN + (len(foot_lines)) * _line_in(CAPTION_PT)) / height, "\n".join(foot_lines),
             ha="left", va="top", fontsize=CAPTION_PT, color=INK_SECONDARY, style="italic", linespacing=LINE_SPACING)
    return _save(fig, path, dpi)


def plot_randomisation(signals_mv: Sequence[np.ndarray], trained: np.ndarray, random: Mapping[str, np.ndarray],
                       row_labels: Sequence[str], annotations: Mapping[str, Sequence[str]], trained_title: str,
                       fs: float, lead: str, grid: Mapping[str, float], path: Path, *, title: str, subtitle: str,
                       footnote: str, band_alpha: float, panel_in: tuple[float, float], min_height_mv: float,
                       dpi: int) -> Path:
    """One row per record: the trained model's map, then each random model's map, behind the same lead."""
    n = len(signals_mv)
    seeds = list(random)
    if trained.shape != (n, len(signals_mv[0])) or any(random[s].shape != trained.shape for s in seeds):
        raise ValueError("Every model needs one map per record, as long as the signals")
    if len(row_labels) != n or any(len(annotations[s]) != n for s in seeds):
        raise ValueError("One row label per record and one annotation per record and random model are needed")
    cmap = band_cmap(band_alpha)
    cols = 1 + len(seeds)
    w, h = panel_in
    width = cols * w
    head = header_height(title, subtitle, width, 0.1)
    col_head = _line_in(10) + 0.05
    cbar_w = 2.4                       # colour bar at the bottom right; the footnote wraps to its left
    foot_lines = _wrap(footnote, width - cbar_w - 0.6, CAPTION_PT)
    foot = max(len(foot_lines) * _line_in(CAPTION_PT), 0.45) + EDGE_IN
    height = head + col_head + n * h + foot
    fig, axes = plt.subplots(n, cols, figsize=(width, height), squeeze=False, facecolor=SURFACE)
    for i, signal in enumerate(signals_mv):
        ylim = _y_limits(signal, grid["grid_major_mv"], min_height_mv)
        t = np.arange(len(signal)) / fs
        for j, cam in enumerate([trained[i], *(random[s][i] for s in seeds)]):
            _check(signal, cam)
            ax = axes[i, j]
            _ecg_axes(ax, len(signal), fs, ylim, grid, minor=False)
            draw_band(ax, cam, fs, ylim, cmap)
            ax.plot(t, signal, color=INK, linewidth=TRACE_WIDTH * 0.7, zorder=3)
            label = row_labels[i] if j == 0 else annotations[seeds[j - 1]][i]
            ax.set_title(label, fontsize=7.5, color=INK, loc="left", pad=3)
            if j == 0:
                ax.set_ylabel(f"{lead} (mV)", color=INK_SECONDARY, fontsize=7.5)
            if i == n - 1:
                ax.set_xlabel("Time (s)", color=INK_SECONDARY, fontsize=7.5)
    fig.tight_layout(rect=(0, foot / height, 1, 1 - (head + col_head) / height), h_pad=0.6, w_pad=0.8)
    for j, text in enumerate([trained_title, *(f"Random weights, init seed {s}" for s in seeds)]):
        fig.text(axes[0, j].get_position().x0, 1 - (head + 0.02) / height, text, ha="left", va="top", fontsize=10,
                 fontweight="bold", color=INK)
    x0 = axes[-1, 0].get_position().x0
    fig.text(x0, (foot - EDGE_IN * 0.5) / height, "\n".join(foot_lines), ha="left", va="top", fontsize=CAPTION_PT,
             color=INK_SECONDARY, style="italic", linespacing=LINE_SPACING)
    cax = fig.add_axes((1 - (cbar_w + 0.3) / width, (foot - 0.25) / height, cbar_w / width, 0.1 / height))
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1)), cax=cax, orientation="horizontal")
    cbar.set_label("Grad-CAM, normalised per panel", color=INK_SECONDARY, fontsize=7.5)
    cbar.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK_SECONDARY)
    cbar.outline.set_edgecolor(MUTED)
    _header(fig, title, subtitle, width, height, 0.1)
    return _save(fig, path, dpi)
