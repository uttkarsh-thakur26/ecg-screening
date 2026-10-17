"""Metrics for the five-superclass multi-label task (PROJECT_SPEC.md section 4).

Ranking metrics (no threshold), metrics at an operating point (one threshold per class, chosen on the
validation split by src/evaluation/thresholds.py) and calibration metrics (reliability bins, ECE).
reports/metrics.json comes with the final evaluation.

Each class is scored as its own binary problem, because the task is multi-label: a record can be
positive for several classes, so there is no single "predicted class" to compare.

- AUROC: the probability that a randomly chosen positive record scores higher than a randomly chosen
  negative one (ties count one half). 0.5 is chance, whatever the class balance.
- AUPRC: average precision, sum over thresholds of (recall_n - recall_n-1) * precision_n, without
  interpolation (scikit-learn's average_precision_score). A model with no skill scores the class's
  prevalence, so it must be read against that number, which is reported next to it.

At an operating point a record is flagged for a class when its probability is >= the class's threshold:
- sensitivity (recall) = TP / (TP + FN): share of records with the finding that are flagged
- specificity = TN / (TN + FP): share of records without it that are not flagged
- precision (PPV) = TP / (TP + FP): share of flagged records that have it. Undefined (None) when no
  record is flagged; it is reported as undefined, never replaced by 0.
- NPV = TN / (TN + FN): share of records not flagged that truly lack it. Undefined when all are flagged.
- F1 = 2TP / (2TP + FP + FN), the harmonic mean of precision and sensitivity (0 when TP = 0).
Accuracy is not reported: with NORM in 45% of records and HYP in 12%, a model that never flags HYP is
88% accurate on it and useless.

Calibration asks whether a predicted probability means what it says: of the records given about 0.8 for
a class, about 80% should have it. Per class, the probabilities are split into equal-width bins
[0, 0.1), [0.1, 0.2), ..., [0.9, 1.0] (the last bin also holds 1.0):
- reliability bins: per bin, the record count, the mean predicted probability and the observed frequency
  (share of those records that have the class). A calibrated model has the two equal in every bin.
- ECE (expected calibration error) = sum over bins of (records in bin / all records) x |observed frequency
  - mean predicted probability|: the average gap between what the model says and what happens, weighted
  by how many records the model says it about. 0 is perfect; empty bins contribute nothing.
This is the per-class (one-vs-rest) form, the one that fits sigmoid outputs; the "top-label" ECE of
multi-class models scores only the highest probability per record and does not apply to a multi-label task.

Macro values are the unweighted mean over classes, so the rare HYP counts as much as the common NORM.
A macro value is None when the metric is undefined for any class it averages.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_predictions(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Check labels and probabilities for one split and return them as arrays. Raises on anything malformed.

    y_true: (records, classes) multi-hot 0/1. y_prob: (records, classes) probabilities in [0, 1], one
    independent sigmoid output per class (not a softmax), so a row need not sum to 1.
    """
    if isinstance(classes, str) or len(set(classes)) != len(classes):
        raise ValueError(f"classes must be a sequence of distinct class names, got {classes!r}")
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


THRESHOLD_METRICS = ("sensitivity", "specificity", "precision", "npv", "f1")


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def counts_to_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | None]:
    """Sensitivity, specificity, precision, NPV and F1 from one class's confusion counts (None where undefined)."""
    if min(tp, fp, fn, tn) < 0:
        raise ValueError(f"Confusion counts must be non-negative, got tp={tp} fp={fp} fn={fn} tn={tn}")
    return {"sensitivity": _ratio(tp, tp + fn), "specificity": _ratio(tn, tn + fp), "precision": _ratio(tp, tp + fp),
            "npv": _ratio(tn, tn + fn), "f1": _ratio(2 * tp, 2 * tp + fp + fn)}


def macro_mean(per_class: Mapping[str, Mapping[str, Any]], classes: Sequence[str],
               keys: Sequence[str] = THRESHOLD_METRICS) -> dict[str, float | None]:
    """Unweighted mean of each metric over `classes`; None for a metric that is undefined for any of them."""
    if not classes:
        raise ValueError("A macro mean needs at least one class")
    out: dict[str, float | None] = {}
    for k in keys:
        values = [per_class[c][k] for c in classes]
        out[k] = None if any(v is None for v in values) else float(np.mean(values))
    return out


def operating_point_metrics(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str],
                            thresholds: Mapping[str, float]) -> dict[str, Any]:
    """Confusion counts and metrics per class at one threshold per class, and their macro mean.

    thresholds maps class name -> threshold; a record is flagged when its probability is >= the threshold.
    Classes without a threshold (NORM at the high-sensitivity operating point) are left out, and the macro
    mean is over the classes that have one. Returns
    {"classes": [scored classes, in class order],
     "per_class": {class: {threshold, tp, fp, fn, tn, sensitivity, specificity, precision, npv, f1}},
     "macro": {sensitivity, specificity, precision, npv, f1}}.
    """
    y_true, y_prob = validate_predictions(y_true, y_prob, classes)
    unknown = sorted(set(thresholds) - set(classes))
    if unknown:
        raise ValueError(f"Thresholds given for unknown classes {unknown}; classes are {list(classes)}")
    scored = [c for c in classes if c in thresholds]
    if not scored:
        raise ValueError("No thresholds given")
    per_class: dict[str, dict[str, Any]] = {}
    for c in scored:
        threshold = float(thresholds[c])
        if not np.isfinite(threshold):
            raise ValueError(f"Threshold for {c} is {threshold}")
        j = list(classes).index(c)
        t, flagged = y_true[:, j] == 1, y_prob[:, j] >= threshold
        tp, fp = int((flagged & t).sum()), int((flagged & ~t).sum())
        fn, tn = int((~flagged & t).sum()), int((~flagged & ~t).sum())
        per_class[c] = {"threshold": threshold, "tp": tp, "fp": fp, "fn": fn, "tn": tn} | counts_to_metrics(tp, fp, fn, tn)
    return {"classes": scored, "per_class": per_class, "macro": macro_mean(per_class, scored)}


def _binary_scores(t: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One class's 0/1 labels and probabilities, checked. Unlike ranking metrics, calibration needs no negatives."""
    t, p = np.asarray(t), np.asarray(p, dtype=np.float64)
    if t.ndim != 1 or p.ndim != 1 or t.shape != p.shape:
        raise ValueError(f"t and p must be 1-D (records,) of equal length, got {t.shape} and {p.shape}")
    if len(t) == 0:
        raise ValueError("No records to evaluate")
    if not np.isin(t, (0, 1)).all():
        raise ValueError(f"t must contain only 0 and 1, got values {np.unique(t)[:10].tolist()}")
    if not np.isfinite(p).all() or p.min() < 0 or p.max() > 1:
        raise ValueError("p must be finite probabilities in [0, 1]; apply the sigmoid to logits first")
    return t.astype(np.int64), p


def reliability_bins(t: np.ndarray, p: np.ndarray, n_bins: int) -> dict[str, list]:
    """Equal-width reliability bins of one class: bin b holds b/n_bins <= p < (b+1)/n_bins, the last bin also 1.0.

    Returns {"edges": n_bins + 1 floats, "count": records per bin, "mean_predicted": mean p per bin,
    "observed_frequency": share of the bin's records with the class}; the last two are None for an empty bin.
    """
    if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"n_bins must be a positive integer, got {n_bins!r}")
    t, p = _binary_scores(t, p)
    index = np.minimum((p * n_bins).astype(np.int64), n_bins - 1)
    count = np.bincount(index, minlength=n_bins)
    p_sum = np.bincount(index, weights=p, minlength=n_bins)
    t_sum = np.bincount(index, weights=t, minlength=n_bins)
    return {
        "edges": np.linspace(0, 1, n_bins + 1).tolist(),
        "count": count.tolist(),
        "mean_predicted": [float(s / n) if n else None for s, n in zip(p_sum, count)],
        "observed_frequency": [float(s / n) if n else None for s, n in zip(t_sum, count)],
    }


def expected_calibration_error(t: np.ndarray, p: np.ndarray, n_bins: int) -> float:
    """ECE of one class: sum over bins of (bin records / records) x |observed frequency - mean predicted|."""
    bins = reliability_bins(t, p, n_bins)
    total = sum(bins["count"])
    return float(sum(n / total * abs(f - m) for n, f, m in
                     zip(bins["count"], bins["observed_frequency"], bins["mean_predicted"]) if n))
