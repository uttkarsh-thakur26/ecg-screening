"""Threshold selection: hand-worked cases, brute-force reimplementations, exact ties and the 90% boundary."""

from __future__ import annotations

import json

import numpy as np
import pytest
from sklearn.metrics import f1_score, recall_score

from src.config import project_path
from src.evaluation.thresholds import (abnormal_classes, high_sensitivity_threshold, max_f1_threshold,
                                       select_thresholds, threshold_curve)


def _brute_counts(t: np.ndarray, p: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    flagged = p >= threshold
    return (int((flagged & (t == 1)).sum()), int((flagged & (t == 0)).sum()),
            int((~flagged & (t == 1)).sum()), int((~flagged & (t == 0)).sum()))


def _random_class(seed: int, n: int = 400, prevalence: float = 0.2) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    t = (rng.random(n) < prevalence).astype(int)
    t[0], t[1] = 1, 0
    p = np.round(np.clip(rng.normal(0.35 + 0.3 * t, 0.2), 0, 1), 2)  # rounding creates many tied scores
    return t, p


@pytest.mark.parametrize("seed", range(5))
def test_curve_matches_brute_force_counts(seed):
    t, p = _random_class(seed)
    curve = threshold_curve(t, p)
    np.testing.assert_array_equal(curve["thresholds"], np.unique(p)[::-1])
    for i, threshold in enumerate(curve["thresholds"]):
        assert tuple(int(curve[k][i]) for k in ("tp", "fp", "fn", "tn")) == _brute_counts(t, p, threshold)


def test_max_f1_hand_worked():
    t = np.array([1, 1, 0, 1, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.6, 0.3, 0.2])
    # F1 = 2TP / (2TP + FP + FN) at 0.9, 0.8, 0.7, 0.6, 0.3, 0.2: 2/4, 4/5, 4/6, 6/7, 6/8, 6/9. Best: 0.6.
    r = max_f1_threshold(t, p)
    assert r["threshold"] == 0.6
    assert r["f1"] == pytest.approx(6 / 7)
    assert (r["tp"], r["fp"], r["fn"], r["tn"], r["tied_thresholds"]) == (3, 1, 0, 2, 1)


def test_max_f1_tie_goes_to_the_lower_threshold():
    t = np.array([1, 0, 0, 1])
    p = np.array([0.9, 0.8, 0.7, 0.6])
    # At 0.9: TP 1, FP 0, FN 1 -> 2/3. At 0.6: TP 2, FP 2, FN 0 -> 4/6. Same F1; 0.6 has the higher sensitivity.
    r = max_f1_threshold(t, p)
    assert r["threshold"] == 0.6 and r["tied_thresholds"] == 2
    assert r["f1"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("seed", range(5))
def test_max_f1_is_the_brute_force_maximum(seed):
    t, p = _random_class(seed)
    candidates = np.unique(p)
    f1s = np.array([f1_score(t, (p >= c).astype(int)) for c in candidates])
    r = max_f1_threshold(t, p)
    assert r["f1"] == pytest.approx(f1s.max(), abs=1e-12)
    assert r["threshold"] == candidates[np.flatnonzero(np.isclose(f1s, f1s.max(), rtol=0, atol=1e-12)).min()]


def test_high_sensitivity_hand_worked():
    t = np.array([1] * 10 + [0] * 5)
    p = np.array([0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.92, 0.72, 0.58, 0.3, 0.1])
    r = high_sensitivity_threshold(t, p, 0.9)
    # 9 of 10 positives must be flagged; the 9th-highest positive score is 0.55.
    assert r["threshold"] == 0.55
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (9, 1, 3, 2)
    assert r["sensitivity"] == 0.9


def test_high_sensitivity_exactly_at_the_target_needs_no_extra_positive():
    """486 / 540 is exactly 0.9. A float ceil(0.9 * 540) gives 487 and would lower the threshold one step too far."""
    n_pos, n_neg = 540, 1600
    t = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    p = np.r_[np.linspace(0.99, 0.30, n_pos), np.linspace(0.70, 0.01, n_neg)]
    r = high_sensitivity_threshold(t, p, 0.9)
    assert r["tp"] == 486
    assert r["threshold"] == np.sort(p[:n_pos])[::-1][485]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("target", [0.5, 0.8, 0.9, 0.95, 1.0])
def test_high_sensitivity_is_the_highest_threshold_reaching_the_target(seed, target):
    t, p = _random_class(seed)
    candidates = np.unique(p)
    reaching = [c for c in candidates if recall_score(t, (p >= c).astype(int)) >= target]
    r = high_sensitivity_threshold(t, p, target)
    assert r["threshold"] == max(reaching)
    assert r["sensitivity"] >= target
    higher = candidates[candidates > r["threshold"]]
    if len(higher):
        assert recall_score(t, (p >= higher.min()).astype(int)) < target


def test_target_of_one_flags_every_positive():
    t, p = _random_class(0)
    r = high_sensitivity_threshold(t, p, 1.0)
    assert r["threshold"] == p[t == 1].min() and r["fn"] == 0


@pytest.mark.parametrize("target", [0, -0.1, 1.01, 90])
def test_out_of_range_target_fails(target):
    t, p = _random_class(0)
    with pytest.raises(ValueError, match="target"):
        high_sensitivity_threshold(t, p, target)


def test_select_thresholds_gives_norm_no_high_sensitivity_point():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 3)) < [0.45, 0.25, 0.12]).astype(int)
    y[0], y[1] = 1, 0
    p = np.clip(0.5 * y + rng.random((300, 3)) * 0.6, 0, 1)
    chosen = select_thresholds(y, p, ["NORM", "MI", "HYP"], "NORM", 0.9)
    assert list(chosen["max_f1"]) == ["NORM", "MI", "HYP"]
    assert list(chosen["high_sensitivity"]) == ["MI", "HYP"]
    for j, c in ((1, "MI"), (2, "HYP")):
        assert chosen["high_sensitivity"][c] == high_sensitivity_threshold(y[:, j], p[:, j], 0.9)


def test_unknown_normal_class_fails():
    with pytest.raises(ValueError, match="Normal class"):
        abnormal_classes(["NORM", "MI"], "NORMAL")


@pytest.mark.parametrize(
    "t, p, message",
    [
        (np.array([[0], [1]]), np.array([[0.2], [0.7]]), "1-D"),
        (np.array([1, 1, 1]), np.array([0.2, 0.5, 0.7]), "positive"),  # no negatives
        (np.array([0, 1]), np.array([0.2, 1.7]), "sigmoid"),
    ],
)
def test_malformed_input_fails_loudly(t, p, message):
    with pytest.raises(ValueError, match=message):
        threshold_curve(t, p)


def test_saved_thresholds_match_the_config_and_the_metrics_report(cfg):
    """reports/thresholds.json and reports/metrics_val.json come from the same run and agree with each other."""
    e = cfg["evaluation"]
    reports = project_path(cfg["paths"]["reports_dir"])
    saved = json.loads((reports / e["thresholds_file"]).read_text(encoding="utf-8"))
    metrics = json.loads((reports / e["val_metrics_file"]).read_text(encoding="utf-8"))
    assert saved["split"] == metrics["split"] == "val"
    assert saved["model"]["run_id"] == metrics["model"]["run_id"] == e["model_run"]
    assert saved["ecg_ids_sha256"] == metrics["ecg_ids_sha256"]
    points = saved["operating_points"]
    assert points["max_f1"]["classes"] == cfg["labels"]["superclasses"]
    assert points["high_sensitivity"]["classes"] == abnormal_classes(cfg["labels"]["superclasses"], e["normal_class"])
    assert points["high_sensitivity"]["target_sensitivity"] == e["high_sensitivity_target"]
    for name, point in points.items():
        scored = metrics["operating_points"][name]["per_class"]
        assert point["thresholds"] == {c: scored[c]["threshold"] for c in point["classes"]}
    for c, m in metrics["operating_points"]["high_sensitivity"]["per_class"].items():
        assert m["sensitivity"] >= e["high_sensitivity_target"], c
