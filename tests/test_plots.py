"""Every figure function writes a real PNG from its inputs, and refuses inputs it cannot draw honestly.

Random arrays appear here only as test inputs; the figures in reports/figures/ come from evaluate_val.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation import plots
from src.evaluation.metrics import operating_point_metrics, ranking_metrics
from src.evaluation.thresholds import select_thresholds

CLASSES = ["NORM", "MI", "HYP"]
DESCRIPTIONS = {"NORM": "Normal ECG", "MI": "Myocardial infarction", "HYP": "Hypertrophy"}
STYLE = {"dpi": 60, "panel_in": 2.0}
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture(scope="module")
def predictions() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = (rng.random((400, 3)) < [0.45, 0.25, 0.12]).astype(int)
    y[0], y[1] = 1, 0
    return y, np.clip(0.4 * y + rng.random((400, 3)) * 0.6, 0, 1)


@pytest.fixture(scope="module")
def points(predictions) -> dict:
    y, p = predictions
    chosen = select_thresholds(y, p, CLASSES, "NORM", 0.9)
    return {k: {"label": k, "per_class": operating_point_metrics(
        y, p, CLASSES, {c: v["threshold"] for c, v in chosen[k].items()})["per_class"]} for k in chosen}


def _is_png(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[:8] == PNG_SIGNATURE and path.stat().st_size > 1000


def test_roc_and_pr_figures(tmp_path, predictions, points):
    y, p = predictions
    ranking = ranking_metrics(y, p, CLASSES)["per_class"]
    roc = plots.plot_roc_curves(y, p, CLASSES, DESCRIPTIONS, {c: ranking[c]["auroc"] for c in CLASSES}, points,
                                tmp_path / "roc.png", title="ROC", subtitle="test", **STYLE)
    pr = plots.plot_pr_curves(y, p, CLASSES, DESCRIPTIONS, {c: ranking[c]["auprc"] for c in CLASSES}, points,
                              tmp_path / "sub" / "pr.png", title="PR", subtitle=None, **STYLE)
    assert _is_png(roc) and _is_png(pr)


def test_confusion_figure_needs_every_class(tmp_path, points):
    per_class = points["high_sensitivity"]["per_class"]  # no NORM
    out = plots.plot_confusion_matrices(per_class, ["MI", "HYP"], DESCRIPTIONS, tmp_path / "cm.png", title="CM",
                                        subtitle=None, **STYLE)
    assert _is_png(out)
    with pytest.raises(ValueError, match="NORM"):
        plots.plot_confusion_matrices(per_class, CLASSES, DESCRIPTIONS, tmp_path / "x.png", title="CM",
                                      subtitle=None, **STYLE)


def test_too_many_operating_points_fail(tmp_path, predictions, points):
    y, p = predictions
    three = dict(points) | {"third": points["max_f1"]}
    with pytest.raises(ValueError, match="At most"):
        plots.plot_roc_curves(y, p, CLASSES, DESCRIPTIONS, dict.fromkeys(CLASSES, 0.5), three, tmp_path / "r.png",
                              title="ROC", subtitle=None, **STYLE)


def _write_log(path: Path, rows: list[str], header: str = "epoch,train_loss,val_loss,val_macro_auroc,val_macro_auprc,is_best") -> Path:
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def test_training_curves(tmp_path):
    log = _write_log(tmp_path / "run.csv", ["1,0.8,0.7,0.88,0.72,True", "2,0.6,0.65,0.90,0.75,True",
                                            "3,0.5,0.66,0.89,0.74,False"])
    assert int(plots.read_training_log(log).loc[lambda d: d["is_best"], "epoch"].iloc[-1]) == 2
    assert _is_png(plots.plot_training_curves(log, tmp_path / "curves.png", title="T", subtitle=None, **STYLE))


@pytest.mark.parametrize(
    "rows, header, message",
    [
        (["1,0.8,0.7,0.88,0.72,False"], None, "no epoch is marked best"),
        (["1,0.8,0.7,0.88,True"], "epoch,train_loss,val_loss,val_macro_auroc,is_best", "lacks columns"),
        (["2,0.8,0.7,0.88,0.72,True", "1,0.6,0.6,0.9,0.75,False"], None, "strictly increasing"),
        (["1,0.8,0.7,0.88,0.72,yes"], None, "True or False"),
    ],
)
def test_malformed_training_log_fails(tmp_path, rows, header, message):
    log = _write_log(tmp_path / "bad.csv", rows, *([header] if header else []))
    with pytest.raises(ValueError, match=message):
        plots.read_training_log(log)


def test_class_distribution(tmp_path, predictions):
    y, _ = predictions
    out = plots.plot_class_distribution({"Train": y[:300], "Validation": y[300:]}, CLASSES, DESCRIPTIONS,
                                        tmp_path / "dist.png", title="Support", subtitle=None, **STYLE)
    assert _is_png(out)
    with pytest.raises(ValueError, match="0/1 matrix"):
        plots.plot_class_distribution({"Train": y[:, :2]}, CLASSES, DESCRIPTIONS, tmp_path / "d.png", title="S",
                                      subtitle=None, **STYLE)
