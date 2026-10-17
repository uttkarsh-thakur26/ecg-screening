"""Metrics for the five-superclass multi-label task (PROJECT_SPEC.md section 4).

First version: per-class and macro AUROC and AUPRC. Thresholded metrics (sensitivity, specificity,
precision, F1), calibration and reports/metrics.json come later.

Each class is scored as its own binary problem, because the task is multi-label: a record can be
positive for several classes, so there is no single "predicted class" to compare.

- AUROC: the probability that a randomly chosen positive record scores higher than a randomly chosen
  negative one (ties count one half). 0.5 is chance, whatever the class balance.
- AUPRC: average precision, sum over thresholds of (recall_n - recall_n-1) * precision_n, without
  interpolation (scikit-learn's average_precision_score). A model with no skill scores the class's
  prevalence, so it must be read against that number, which is reported next to it.

Macro values are the unweighted mean over classes, so the rare HYP counts as much as the common NORM.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_predictions(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Check labels and probabilities for one split and return them as arrays. Raises on anything malformed.

    y_true: (records, classes) multi-hot 0/1. y_prob: (records, classes) probabilities in [0, 1], one
    independent sigmoid output per class (not a softmax), so a row need not sum to 1.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.ndim != 2 or y_prob.ndim != 2:
        raise ValueError(f"y_true and y_prob must be 2-D (records, classes), got {y_true.shape} and {y_prob.shape}")
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true {y_true.shape} and y_prob {y_prob.shape} differ in shape")
    if y_true.shape[1] != len(classes):
        raise ValueError(f"{y_true.shape[1]} columns but {len(classes)} class names {list(classes)}")
    if y_true.shape[0] == 0:
        raise ValueError("No records to evaluate")
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError(f"y_true must contain only 0 and 1, got values {np.unique(y_true)[:10].tolist()}")
    if not np.isfinite(y_prob).all():
        raise ValueError("y_prob contains NaN or infinite values")
    if y_prob.min() < 0 or y_prob.max() > 1:
        raise ValueError(f"y_prob must be probabilities in [0, 1], got range [{y_prob.min()}, {y_prob.max()}]; "
                         "apply the sigmoid to logits first")
    positives = y_true.sum(axis=0)
    one_sided = [f"{c} ({int(p)} of {len(y_true)} positive)" for c, p in zip(classes, positives) if p in (0, len(y_true))]
    if one_sided:
        raise ValueError(f"AUROC and AUPRC need positives and negatives in every class: {', '.join(one_sided)}")
    return y_true.astype(np.int64), y_prob


def ranking_metrics(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str]) -> dict[str, Any]:
    """Per-class and macro AUROC and AUPRC, with each class's support and prevalence.

    Returns {"per_class": {class: {auroc, auprc, n_positive, n_negative, prevalence}},
             "macro": {auroc, auprc, prevalence}}. Each class's prevalence is the AUPRC of a model with
    no skill on that class, so macro prevalence is the macro AUPRC of such a model.
    """
    y_true, y_prob = validate_predictions(y_true, y_prob, classes)
    per_class: dict[str, dict[str, float | int]] = {}
    for j, name in enumerate(classes):
        t, p = y_true[:, j], y_prob[:, j]
        n_pos = int(t.sum())
        per_class[name] = {
            "auroc": float(roc_auc_score(t, p)),
            "auprc": float(average_precision_score(t, p)),
            "n_positive": n_pos,
            "n_negative": len(t) - n_pos,
            "prevalence": n_pos / len(t),
        }
    macro = {k: float(np.mean([per_class[c][k] for c in classes])) for k in ("auroc", "auprc", "prevalence")}
    return {"per_class": per_class, "macro": macro}
