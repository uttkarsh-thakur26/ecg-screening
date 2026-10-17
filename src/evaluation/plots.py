"""ROC, precision-recall, confusion-matrix, training-curve and class-support figures (PROJECT_SPEC.md section 4).

Every function takes real arrays or a run's CSV and writes one PNG; nothing here computes a number that is
not drawn from its inputs. Reliability diagrams come with temperature scaling (src/training/calibrate.py).

Style follows the frontend palette in PROJECT_SPEC.md section 7 (white surface, #1A1A1A ink, muted teal
accent), so the figures sit on the Model Performance page without clashing. Operating points are told
apart by marker shape as well as colour, so they stay distinct in greyscale and for colour-blind readers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # files only, no window; must precede pyplot
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

from src.evaluation.metrics import validate_predictions  # noqa: E402

# Design tokens. Ink and accent are PROJECT_SPEC.md section 7's; the rest are neutral greys and two
# categorical hues checked for colour-blind separation (blue vs orange: CVD delta E 11 against the teal).
SURFACE = "#FFFFFF"
INK = "#1A1A1A"
INK_SECONDARY = "#52514E"
MUTED = "#898781"
GRIDLINE = "#E1E0D9"
AXIS = "#C3C2B7"
ACCENT = "#0F766E"           # the model's curve
SERIES = ("#2A78D6", "#EB6834")  # first and second series: operating points, train/val, AUROC/AUPRC
MARKERS = ("o", "D")
# One-hue blue ramp, light to dark, for confusion-matrix cells (shade = share of the true row).
SEQUENTIAL_BLUE = ("#F3F8FE", "#CDE2FB", "#9EC5F4", "#6DA7EC", "#3987E5", "#256ABF", "#184F95", "#0D366B")
LIGHT_TEXT_FROM = 0.55       # row share above which a cell is dark enough for white text

CURVE_WIDTH = 1.6
REFERENCE_WIDTH = 1.0
MARKER_SIZE = 7
PR_LABEL_GAP = 0.03          # AUPRC annotation sits this far (in precision) above the no-skill line
LABEL_HEADROOM = 0.12        # share of the y range added above the data for an in-axes label


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=8)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)
    ax.grid(color=GRIDLINE, linewidth=0.6)
    ax.set_axisbelow(True)


TITLE_TOP_IN, SUBTITLE_TOP_IN, HEADER_IN = 0.06, 0.34, 0.62  # title, subtitle and the band kept free for them
LEGEND_BAND_IN = 0.40        # band kept free at the bottom for a figure-wide legend


def _finish(fig: plt.Figure, title: str, subtitle: str | None, legend: Sequence[Line2D] = ()) -> None:
    """Title and subtitle in a fixed band at the top, an optional figure legend at the bottom, panels in between.

    Offsets are in inches, so the title and subtitle never overlap whatever the figure's height.
    """
    height = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - TITLE_TOP_IN / height, ha="left", va="top", fontsize=12, fontweight="bold",
                 color=INK)
    if subtitle:
        fig.text(0.01, 1 - SUBTITLE_TOP_IN / height, subtitle, ha="left", va="top", fontsize=8.5, color=INK_SECONDARY)
    bottom = 0.0
    if legend:
        fig.legend(handles=list(legend), loc="lower left", ncol=len(legend), frameon=False, fontsize=8.5,
                   bbox_to_anchor=(0.01, 0.0), labelcolor=INK_SECONDARY)
        bottom = LEGEND_BAND_IN / height
    fig.tight_layout(rect=(0, bottom, 1, 1 - HEADER_IN / height))


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


PANEL_LABELS_IN = 0.55       # per panel: its title above, tick labels and axis label below
PANEL_Y_LABELS_IN = 0.25     # per panel: y tick labels and axis label, taken from its width


def _panels(n: int, panel_in: float, legend: bool) -> tuple[plt.Figure, np.ndarray]:
    """A row of n square panels; the height fits the square plot area, so no band of empty space is left."""
    height = (panel_in - PANEL_Y_LABELS_IN) + PANEL_LABELS_IN + HEADER_IN + (LEGEND_BAND_IN if legend else 0)
    fig, axes = plt.subplots(1, n, figsize=(panel_in * n, height), squeeze=False, facecolor=SURFACE)
    return fig, axes[0]


def _check_points(operating_points: Mapping[str, Mapping[str, Any]]) -> None:
    if len(operating_points) > len(SERIES):
        raise ValueError(f"At most {len(SERIES)} operating points can be drawn, got {list(operating_points)}")
    for key, op in operating_points.items():
        if "label" not in op or "per_class" not in op:
            raise ValueError(f"Operating point {key!r} needs 'label' and 'per_class'")


def _point_handles(operating_points: Mapping[str, Mapping[str, Any]]) -> list[Line2D]:
    return [Line2D([], [], linestyle="none", marker=m, markersize=MARKER_SIZE, markerfacecolor=c,
                   markeredgecolor=SURFACE, label=op["label"])
            for op, c, m in zip(operating_points.values(), SERIES, MARKERS)]


def _class_title(ax: plt.Axes, name: str, descriptions: Mapping[str, str]) -> None:
    ax.set_title(f"{name}  ·  {descriptions[name]}", fontsize=9.5, color=INK, loc="left")


def plot_roc_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str], descriptions: Mapping[str, str],
                    aurocs: Mapping[str, float], operating_points: Mapping[str, Mapping[str, Any]], path: Path, *,
                    title: str, subtitle: str | None, dpi: int, panel_in: float) -> Path:
    """One ROC panel per class, with each operating point marked at (1 - specificity, sensitivity).

    operating_points: {key: {"label": legend text, "per_class": {class: {"sensitivity", "specificity", ...}}}};
    a class missing from an operating point's per_class gets no marker for it.
    """
    y_true, y_prob = validate_predictions(y_true, y_prob, classes)
    _check_points(operating_points)
    fig, axes = _panels(len(classes), panel_in, legend=True)
    for j, (ax, c) in enumerate(zip(axes, classes)):
        fpr, tpr, _ = roc_curve(y_true[:, j], y_prob[:, j], drop_intermediate=False)
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=REFERENCE_WIDTH)
        ax.plot(fpr, tpr, color=ACCENT, linewidth=CURVE_WIDTH, solid_joinstyle="round")
        for op, color, marker in zip(operating_points.values(), SERIES, MARKERS):
            m = op["per_class"].get(c)
            if m is not None:
                ax.plot(1 - m["specificity"], m["sensitivity"], linestyle="none", marker=marker,
                        markersize=MARKER_SIZE, markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5,
                        zorder=3)
        ax.text(0.97, 0.04, f"AUROC {aurocs[c]:.3f}", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=9, color=INK)
        _style(ax)
        ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
        ax.set_xlabel("1 − specificity (false-positive rate)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Sensitivity (true-positive rate)", fontsize=8)
        _class_title(ax, c, descriptions)
    handles = [Line2D([], [], color=ACCENT, linewidth=CURVE_WIDTH, label="ROC curve"),
               Line2D([], [], color=AXIS, linewidth=REFERENCE_WIDTH, label="Chance (AUROC 0.5)"),
               *_point_handles(operating_points)]
    _finish(fig, title, subtitle, handles)
    return _save(fig, path, dpi)


def plot_pr_curves(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str], descriptions: Mapping[str, str],
                   auprcs: Mapping[str, float], operating_points: Mapping[str, Mapping[str, Any]], path: Path, *,
                   title: str, subtitle: str | None, dpi: int, panel_in: float) -> Path:
    """One precision-recall panel per class, the no-skill line at the class's prevalence, operating points marked.

    The curve is drawn as steps, the shape average precision sums over. An operating point whose precision
    is undefined (nothing flagged) gets no marker.
    """
    y_true, y_prob = validate_predictions(y_true, y_prob, classes)
    _check_points(operating_points)
    fig, axes = _panels(len(classes), panel_in, legend=True)
    for j, (ax, c) in enumerate(zip(axes, classes)):
        precision, recall, _ = precision_recall_curve(y_true[:, j], y_prob[:, j], drop_intermediate=False)
        prevalence = y_true[:, j].mean()
        ax.axhline(prevalence, color=AXIS, linewidth=REFERENCE_WIDTH)
        ax.plot(recall, precision, color=ACCENT, linewidth=CURVE_WIDTH, drawstyle="steps-post")
        for op, color, marker in zip(operating_points.values(), SERIES, MARKERS):
            m = op["per_class"].get(c)
            if m is not None and m["precision"] is not None:
                ax.plot(m["sensitivity"], m["precision"], linestyle="none", marker=marker, markersize=MARKER_SIZE,
                        markerfacecolor=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
        # Just above the no-skill line at the left, where precision-recall curves start high: covers nothing.
        ax.text(0.03, prevalence + PR_LABEL_GAP, f"AUPRC {auprcs[c]:.3f}\nprevalence {prevalence:.3f}",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=9, color=INK, linespacing=1.4)
        _style(ax)
        ax.set(xlim=(0, 1), ylim=(0, 1.01), aspect="equal")
        ax.set_xlabel("Sensitivity (recall)", fontsize=8)
        if j == 0:
            ax.set_ylabel("Precision (PPV)", fontsize=8)
        _class_title(ax, c, descriptions)
    handles = [Line2D([], [], color=ACCENT, linewidth=CURVE_WIDTH, label="Precision-recall curve"),
               Line2D([], [], color=AXIS, linewidth=REFERENCE_WIDTH, label="No skill (= prevalence)"),
               *_point_handles(operating_points)]
    _finish(fig, title, subtitle, handles)
    return _save(fig, path, dpi)


def plot_confusion_matrices(per_class: Mapping[str, Mapping[str, Any]], classes: Sequence[str],
                            descriptions: Mapping[str, str], path: Path, *, title: str, subtitle: str | None, dpi: int,
                            panel_in: float) -> Path:
    """One binary confusion matrix per class at its threshold: rows = truth, columns = model decision.

    per_class: {class: {"threshold", "tp", "fp", "fn", "tn", ...}} as from metrics.operating_point_metrics.
    Each cell shows its count and its share of the true row, so the diagonal reads as specificity (top left)
    and sensitivity (bottom right); the shade follows that share.
    """
    missing = [c for c in classes if c not in per_class]
    if missing:
        raise ValueError(f"No confusion counts for {missing}")
    cmap = LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL_BLUE)
    fig, axes = _panels(len(classes), panel_in, legend=False)
    for j, (ax, c) in enumerate(zip(axes, classes)):
        m = per_class[c]
        counts = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        share = counts / counts.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        for r in range(2):
            for k in range(2):
                ax.text(k, r, f"{counts[r, k]:,}\n{share[r, k]:.1%}", ha="center", va="center", fontsize=10,
                        color=SURFACE if share[r, k] > LIGHT_TEXT_FROM else INK, linespacing=1.4)
        ax.set_xticks([0, 1], ["Not flagged", "Flagged"], fontsize=8.5)
        ax.set_yticks([0, 1], ["Absent", "Present"], fontsize=8.5)
        ax.tick_params(length=0, labelcolor=INK_SECONDARY)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("Model", fontsize=8, color=INK_SECONDARY)
        if j == 0:
            ax.set_ylabel("Truth", fontsize=8, color=INK_SECONDARY)
        ax.set_title(f"{c}  ·  threshold {m['threshold']:.3f}", fontsize=9.5, color=INK, loc="left")
    _finish(fig, title, subtitle)
    return _save(fig, path, dpi)


TRAINING_COLUMNS = ("epoch", "train_loss", "val_loss", "val_macro_auroc", "val_macro_auprc", "is_best")


def read_training_log(csv_path: Path) -> pd.DataFrame:
    """A training run's per-epoch CSV, checked for the columns the training-curve figure needs."""
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Training log not found: {csv_path}")
    log = pd.read_csv(csv_path)
    missing = [c for c in TRAINING_COLUMNS if c not in log.columns]
    if missing:
        raise ValueError(f"{csv_path} lacks columns {missing}")
    if log.empty:
        raise ValueError(f"{csv_path} has no epochs")
    if not log["epoch"].is_monotonic_increasing or log["epoch"].duplicated().any():
        raise ValueError(f"{csv_path}: epochs are not strictly increasing")
    flags = log["is_best"].astype(str)
    if not flags.isin(["True", "False"]).all():
        raise ValueError(f"{csv_path}: is_best must be True or False, got {sorted(flags.unique())}")
    log["is_best"] = flags == "True"
    if not log["is_best"].any():
        raise ValueError(f"{csv_path}: no epoch is marked best")
    return log


def plot_training_curves(csv_path: Path, path: Path, *, title: str, subtitle: str | None, dpi: int,
                         panel_in: float) -> Path:
    """Loss (training and validation) and validation macro AUROC and AUPRC per epoch, best epoch marked.

    The best epoch is the last one marked is_best: each new best overwrote the checkpoint, so it is the saved one.
    """
    log = read_training_log(csv_path)
    best = int(log.loc[log["is_best"], "epoch"].iloc[-1])
    fig, (loss_ax, score_ax) = plt.subplots(1, 2, figsize=(panel_in * 3, panel_in + PANEL_LABELS_IN + HEADER_IN + 0.5),
                                            facecolor=SURFACE)
    panels = [
        (loss_ax, [("train_loss", "Training loss (train mode: dropout and augmentation on)"),
                   ("val_loss", "Validation loss")], "BCE loss with pos_weight, per record", "Loss"),
        (score_ax, [("val_macro_auroc", "Validation macro AUROC"), ("val_macro_auprc", "Validation macro AUPRC")],
         "Score", "Validation ranking metrics"),
    ]
    for ax, series, ylabel, panel_title in panels:
        for (column, label), color in zip(series, SERIES):
            ax.plot(log["epoch"], log[column], color=color, linewidth=CURVE_WIDTH, marker="o", markersize=3,
                    label=label)
        low, high = ax.get_ylim()
        ax.set_ylim(low, high + (high - low) * LABEL_HEADROOM)  # a free band for the label, above every point
        ax.axvline(best, color=INK_SECONDARY, linewidth=REFERENCE_WIDTH)
        ax.text(best, 1.0, f" best epoch {best} (saved)", transform=ax.get_xaxis_transform(), ha="left", va="top",
                fontsize=8, color=INK_SECONDARY)
        _style(ax)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_title(panel_title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(log["epoch"].min() - 0.5, log["epoch"].max() + 0.5)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK_SECONDARY, loc="upper left", bbox_to_anchor=(0, -0.16),
                  ncol=1)
    _finish(fig, title, subtitle)
    return _save(fig, path, dpi)


def plot_class_distribution(targets_by_split: Mapping[str, np.ndarray], classes: Sequence[str],
                            descriptions: Mapping[str, str], path: Path, *, title: str, subtitle: str | None, dpi: int,
                            panel_in: float) -> Path:
    """Share of records carrying each label, one column per split, with the count on each column.

    targets_by_split: {legend label: (records, classes) multi-hot}, at most two splits. A record can carry
    several labels, so the shares of one split do not sum to 1.
    """
    if not 1 <= len(targets_by_split) <= len(SERIES):
        raise ValueError(f"Between 1 and {len(SERIES)} splits can be drawn, got {len(targets_by_split)}")
    for label, y in targets_by_split.items():
        y = np.asarray(y)
        if y.ndim != 2 or y.shape[1] != len(classes) or len(y) == 0 or not np.isin(y, (0, 1)).all():
            raise ValueError(f"{label}: targets must be a non-empty (records, {len(classes)}) 0/1 matrix, got {y.shape}")
    fig, ax = plt.subplots(figsize=(panel_in * 3, panel_in + PANEL_LABELS_IN + HEADER_IN), facecolor=SURFACE)
    x = np.arange(len(classes))
    width = 0.8 / len(targets_by_split)
    for i, ((label, y), color) in enumerate(zip(targets_by_split.items(), SERIES)):
        y = np.asarray(y)
        counts, share = y.sum(axis=0), y.mean(axis=0)
        offset = (i - (len(targets_by_split) - 1) / 2) * width
        bars = ax.bar(x + offset, share, width=width * 0.92, color=color, label=f"{label} ({len(y):,} records)")
        for bar, n in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008, f"{int(n):,}", ha="center",
                    va="bottom", fontsize=7.5, color=INK_SECONDARY)
    _style(ax)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x, [f"{c}\n{descriptions[c]}" for c in classes], fontsize=8.5)
    ax.set_ylabel("Share of records with the label", fontsize=8)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(np.asarray(y).mean(axis=0).max() for y in targets_by_split.values()) * 1.15)
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK_SECONDARY, loc="upper right")
    _finish(fig, title, subtitle)
    return _save(fig, path, dpi)
