"""Per-class decision thresholds, chosen on the validation split only (PROJECT_SPEC.md section 4).

A record is flagged for a class when its probability for that class is >= the class's threshold. Each class
gets its own threshold because each is its own binary decision (the task is multi-label), and the classes
differ in prevalence and in how separable they are.

Two operating points:

- max_f1: per class, the threshold with the highest F1 on the validation split. F1 is the harmonic mean of
  sensitivity and precision, so it balances missed findings against false alarms with equal weight.
- high_sensitivity: per abnormal class (every class except labels' normal class, NORM), the HIGHEST threshold
  whose validation sensitivity is >= evaluation.high_sensitivity_target (0.90). Lowering a threshold can only
  raise sensitivity and lower specificity, so the highest threshold that reaches the target is the one that
  keeps the most specificity. NORM gets no such threshold: it is not an abnormality to be caught, and a
  sensitivity target for "normal" has no screening meaning.

Candidate thresholds are the distinct predicted probabilities of the validation records, as in scikit-learn's
roc_curve and precision_recall_curve. Every value in the half-open gap (next lower score, chosen score] flags
exactly the same validation records, so any threshold in it would give identical validation results; the
chosen score is the top of that gap, which for high_sensitivity is by definition the highest threshold.

Ties. When several thresholds share the maximum F1 exactly (compared as integer fractions, not floats), the
lowest one is chosen: equal F1 and more sensitivity, which is the side a screening tool should err on.

Thresholds apply to the probabilities they were chosen on (the model's sigmoid outputs). Temperature scaling
changes the probabilities, so thresholds must be re-derived or mapped after calibration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.evaluation.metrics import validate_predictions


def threshold_curve(t: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Confusion counts at every candidate threshold of one class, highest threshold first.

    t: (records,) 0/1 labels; p: (records,) probabilities. Returns {"thresholds", "tp", "fp", "fn", "tn"},
    where the counts at thresholds[i] are those of the rule p >= thresholds[i]. thresholds are the distinct
    values of p in decreasing order, so every count array is monotone. The class needs positives and negatives.
    """
    t, p = np.asarray(t), np.asarray(p)
    if t.ndim != 1 or p.ndim != 1:
        raise ValueError(f"t and p must be 1-D (records,), got {t.shape} and {p.shape}")
    y, prob = validate_predictions(t[:, None], p[:, None], ["class"])
    t, p = y[:, 0], prob[:, 0]
    order = np.argsort(-p, kind="stable")
    p_sorted, t_sorted = p[order], t[order]
    # Index of the last record of each run of equal scores: flagging at that score flags the whole run.
    last_of_run = np.r_[np.flatnonzero(np.diff(p_sorted)), len(p_sorted) - 1]
    tp = np.cumsum(t_sorted)[last_of_run]
    fp = (last_of_run + 1) - tp
    n_pos = int(t.sum())
    n_neg = len(t) - n_pos
    return {"thresholds": p_sorted[last_of_run], "tp": tp, "fp": fp, "fn": n_pos - tp, "tn": n_neg - fp}


def _point(curve: dict[str, np.ndarray], i: int) -> dict[str, Any]:
    tp, fp, fn, tn = (int(curve[k][i]) for k in ("tp", "fp", "fn", "tn"))
    return {"threshold": float(curve["thresholds"][i]), "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def max_f1_threshold(t: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    """The threshold with the highest F1 = 2TP / (2TP + FP + FN); among exact ties, the lowest threshold.

    Returns {"threshold", "tp", "fp", "fn", "tn", "f1", "tied_thresholds"}, where tied_thresholds counts the
    candidate thresholds that share the maximum F1 (1 when there is no tie).
    """
    curve = threshold_curve(t, p)
    num = 2 * curve["tp"]
    den = 2 * curve["tp"] + curve["fp"] + curve["fn"]  # > 0 whenever the class has a positive
    best = int(np.argmax(num / den))
    # Exact comparison of the fractions num/den: a float comparison could split or merge a tie by one ulp.
    tied = np.flatnonzero(num * den[best] == num[best] * den)
    chosen = int(tied.max())  # thresholds run high to low, so the last tied index is the lowest threshold
    return _point(curve, chosen) | {"f1": float(num[chosen] / den[chosen]), "tied_thresholds": int(len(tied))}


def high_sensitivity_threshold(t: np.ndarray, p: np.ndarray, target: float) -> dict[str, Any]:
    """The highest threshold whose sensitivity TP / (TP + FN) is >= target.

    Returns {"threshold", "tp", "fp", "fn", "tn", "sensitivity"}. Sensitivity only grows as the threshold
    falls, and the lowest candidate (the smallest score) flags every record, so a target in (0, 1] is always
    reachable.
    """
    if not 0 < target <= 1:
        raise ValueError(f"Sensitivity target must be in (0, 1], got {target}")
    curve = threshold_curve(t, p)
    sensitivity = curve["tp"] / (curve["tp"] + curve["fn"])
    # For a target with d decimal places (0.90: d = 2), k / n_pos is either equal to it or at least
    # 1 / (10^d n_pos) away, far more than float64 rounding (about 1e-16), so this float comparison is exact.
    reached = np.flatnonzero(sensitivity >= target)
    chosen = int(reached.min())  # the first index that reaches the target is the highest such threshold
    return _point(curve, chosen) | {"sensitivity": float(sensitivity[chosen])}


def abnormal_classes(classes: Sequence[str], normal_class: str) -> list[str]:
    """Every class except the normal one, in class order."""
    if normal_class not in classes:
        raise ValueError(f"Normal class {normal_class!r} is not one of {list(classes)}")
    return [c for c in classes if c != normal_class]


def select_thresholds(y_true: np.ndarray, y_prob: np.ndarray, classes: Sequence[str], normal_class: str,
                      sensitivity_target: float) -> dict[str, dict[str, dict[str, Any]]]:
    """Both operating points for every class they apply to.

    Returns {"max_f1": {class: max_f1_threshold(...)} for every class,
             "high_sensitivity": {class: high_sensitivity_threshold(...)} for every abnormal class}.
    """
    y_true, y_prob = validate_predictions(y_true, y_prob, classes)
    col = {c: j for j, c in enumerate(classes)}
    return {
        "max_f1": {c: max_f1_threshold(y_true[:, col[c]], y_prob[:, col[c]]) for c in classes},
        "high_sensitivity": {c: high_sensitivity_threshold(y_true[:, col[c]], y_prob[:, col[c]], sensitivity_target)
                             for c in abnormal_classes(classes, normal_class)},
    }
