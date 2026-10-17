"""AUROC, AUPRC and operating-point metrics: hand-worked cases, independent reimplementations, loud failures."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from src.evaluation.metrics import (counts_to_metrics, macro_mean, operating_point_metrics, ranking_metrics,
                                    validate_predictions)


def _auroc_by_ranks(t: np.ndarray, p: np.ndarray) -> float:
    """Mann-Whitney U / (n_pos * n_neg), with tied scores sharing their average rank."""
    ranks = rankdata(p)
    n_pos, n_neg = t.sum(), len(t) - t.sum()
    return (ranks[t == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _average_precision(t: np.ndarray, p: np.ndarray) -> float:
    """Sum over distinct score thresholds, highest first, of (gain in recall) x (precision at that threshold)."""
    ap, previous_recall = 0.0, 0.0
    for threshold in np.unique(p)[::-1]:
        flagged = p >= threshold
        recall = t[flagged].sum() / t.sum()
        ap += (recall - previous_recall) * t[flagged].mean()
        previous_recall = recall
    return ap


def test_hand_worked_example():
    y = np.array([[0], [0], [1], [1]])
    p = np.array([[0.1], [0.4], [0.35], [0.8]])
    m = ranking_metrics(y, p, ["A"])["per_class"]["A"]
    # Positive-negative pairs: 0.35 > 0.1, 0.35 < 0.4, 0.8 > 0.1, 0.8 > 0.4, so 3 of 4 are ordered correctly.
    assert m["auroc"] == pytest.approx(0.75)
    # Ranked by score: 0.8 (positive: recall 1/2, precision 1/1), 0.4 (negative), 0.35 (positive: recall 1, precision 2/3).
    assert m["auprc"] == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert (m["n_positive"], m["n_negative"], m["prevalence"]) == (2, 2, 0.5)


def test_counts_on_unbalanced_data():
    y = np.array([[1, 0], [0, 0], [0, 1], [0, 1], [0, 1]])
    p = np.array([[0.9, 0.1], [0.2, 0.3], [0.4, 0.8], [0.3, 0.7], [0.1, 0.6]])
    m = ranking_metrics(y, p, ["A", "B"])["per_class"]
    assert (m["A"]["n_positive"], m["A"]["n_negative"], m["A"]["prevalence"]) == (1, 4, 0.2)
    assert (m["B"]["n_positive"], m["B"]["n_negative"], m["B"]["prevalence"]) == (3, 2, 0.6)


def test_saturated_probabilities_are_accepted():
    """A float32 sigmoid returns exactly 0 and 1 for large logits; those are valid probabilities."""
    y = np.array([[0], [0], [1], [1]])
    p = np.array([[0.0], [0.3], [0.7], [1.0]], dtype=np.float32)
    m = ranking_metrics(y, p, ["A"])["per_class"]["A"]
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0


def test_constant_scores_rank_at_chance():
    """What the prior baseline does: one score for every record gives AUROC 0.5 and AUPRC = prevalence."""
    y = np.array([[1, 0], [0, 0], [0, 1], [0, 1], [1, 1], [0, 1], [0, 0], [0, 1]])
    p = np.tile([0.3, 0.6], (len(y), 1))
    m = ranking_metrics(y, p, ["A", "B"])
    for j, name in enumerate(["A", "B"]):
        assert m["per_class"][name]["auroc"] == 0.5
        assert m["per_class"][name]["auprc"] == pytest.approx(y[:, j].mean(), abs=1e-12)
        assert m["per_class"][name]["prevalence"] == y[:, j].mean()


def test_perfect_and_reversed_ranking():
    y = np.array([[0], [0], [0], [1], [1]])
    perfect = np.array([[0.1], [0.2], [0.3], [0.8], [0.9]])
    m = ranking_metrics(y, perfect, ["A"])["per_class"]["A"]
    assert m["auroc"] == 1.0 and m["auprc"] == 1.0
    reversed_ = ranking_metrics(y, 1 - perfect, ["A"])["per_class"]["A"]
    assert reversed_["auroc"] == 0.0
    assert reversed_["auprc"] == pytest.approx(0.5 * 1 / 4 + 0.5 * 2 / 5)  # positives found 4th and 5th


def test_macro_is_the_unweighted_mean_over_classes():
    rng = np.random.default_rng(0)
    y = (rng.random((300, 3)) < [0.5, 0.2, 0.05]).astype(int)
    p = rng.random((300, 3))
    m = ranking_metrics(y, p, ["A", "B", "C"])
    for key in ("auroc", "auprc", "prevalence"):
        assert m["macro"][key] == pytest.approx(np.mean([m["per_class"][c][key] for c in "ABC"]), abs=1e-15)


@pytest.mark.parametrize("seed", range(5))
def test_matches_independent_implementations_with_ties(seed):
    rng = np.random.default_rng(seed)
    n = 500
    y = (rng.random((n, 4)) < [0.45, 0.25, 0.12, 0.03]).astype(int)
    y[0], y[1] = 1, 0  # every class has both labels
    p = np.round(rng.random((n, 4)) * 0.5 + 0.4 * y, 2)  # rounding creates many ties
    m = ranking_metrics(y, p, list("ABCD"))
    for j, name in enumerate("ABCD"):
        assert m["per_class"][name]["auroc"] == pytest.approx(_auroc_by_ranks(y[:, j], p[:, j]), abs=1e-12)
        assert m["per_class"][name]["auprc"] == pytest.approx(_average_precision(y[:, j], p[:, j]), abs=1e-12)


@pytest.mark.parametrize(
    "y, p, classes, message",
    [
        ([0, 1], [0.2, 0.7], ["A"], "2-D"),
        ([[0], [1]], [[0.2], [0.7], [0.5]], ["A"], "differ in shape"),
        ([[0], [1]], [[0.2], [0.7]], ["A", "B"], "class names"),
        ([[0], [2]], [[0.2], [0.7]], ["A"], "only 0 and 1"),
        ([[0], [1]], [[0.2], [np.nan]], ["A"], "NaN"),
        ([[0], [1]], [[-1.3], [2.1]], ["A"], "sigmoid"),  # logits passed by mistake
        ([[0], [1]], [[0.2], [1.5]], ["A"], "sigmoid"),  # above 1 only
        ([[0], [1]], [[-0.1], [0.7]], ["A"], "sigmoid"),  # below 0 only
        ([[0, 1], [1, 0]], [[0.2, 0.3], [0.7, 0.4]], ["A", "A"], "distinct"),  # a duplicate would merge two classes
        ([[0, 1], [1, 0]], [[0.2, 0.3], [0.7, 0.4]], "AB", "distinct"),  # a bare string is not a list of names
        ([[0, 1], [0, 1]], [[0.2, 0.3], [0.7, 0.4]], ["A", "B"], r"A \(0 of 2 positive\), B \(2 of 2 positive\)"),
        (np.empty((0, 1)), np.empty((0, 1)), ["A"], "No records"),
    ],
)
def test_malformed_input_fails_loudly(y, p, classes, message):
    with pytest.raises(ValueError, match=message):
        validate_predictions(np.asarray(y), np.asarray(p), classes)


# ------------------------------------------------------------------------------------- operating points

def test_counts_to_metrics_hand_worked():
    m = counts_to_metrics(tp=8, fp=2, fn=2, tn=88)
    assert m == {"sensitivity": 0.8, "specificity": 88 / 90, "precision": 0.8, "npv": 88 / 90, "f1": 0.8}


def test_undefined_rates_are_none_not_zero():
    nothing_flagged = counts_to_metrics(tp=0, fp=0, fn=5, tn=95)
    assert nothing_flagged["precision"] is None
    assert nothing_flagged["f1"] == 0.0 and nothing_flagged["sensitivity"] == 0.0
    everything_flagged = counts_to_metrics(tp=5, fp=95, fn=0, tn=0)
    assert everything_flagged["npv"] is None and everything_flagged["sensitivity"] == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        counts_to_metrics(tp=1, fp=-1, fn=0, tn=0)


def test_threshold_is_inclusive():
    y = np.array([[1], [0], [1], [0]])
    p = np.array([[0.7], [0.5], [0.5], [0.2]])
    m = operating_point_metrics(y, p, ["A"], {"A": 0.5})["per_class"]["A"]
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 0, 1)  # both records at exactly 0.5 are flagged


@pytest.mark.parametrize("seed", range(3))
def test_operating_point_matches_sklearn(seed):
    rng = np.random.default_rng(seed)
    y = (rng.random((500, 3)) < [0.4, 0.2, 0.08]).astype(int)
    y[0], y[1] = 1, 0
    p = np.round(np.clip(0.3 * y + rng.random((500, 3)) * 0.7, 0, 1), 2)
    thresholds = {"A": 0.5, "B": 0.42, "C": 0.61}
    r = operating_point_metrics(y, p, list("ABC"), thresholds)
    for j, c in enumerate("ABC"):
        pred = (p[:, j] >= thresholds[c]).astype(int)
        tn, fp, fn, tp = confusion_matrix(y[:, j], pred, labels=[0, 1]).ravel()
        m = r["per_class"][c]
        assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (tp, fp, fn, tn)
        assert m["sensitivity"] == pytest.approx(recall_score(y[:, j], pred), abs=1e-15)
        assert m["specificity"] == pytest.approx(recall_score(1 - y[:, j], 1 - pred), abs=1e-15)
        assert m["precision"] == pytest.approx(precision_score(y[:, j], pred), abs=1e-15)
        assert m["f1"] == pytest.approx(f1_score(y[:, j], pred), abs=1e-15)
        assert m["threshold"] == thresholds[c]
    for k in ("sensitivity", "specificity", "precision", "npv", "f1"):
        assert r["macro"][k] == pytest.approx(np.mean([r["per_class"][c][k] for c in "ABC"]), abs=1e-15)


def test_classes_without_a_threshold_are_left_out_of_the_macro():
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    p = np.array([[0.9, 0.2, 0.8], [0.3, 0.7, 0.4], [0.6, 0.9, 0.1], [0.2, 0.4, 0.6]])
    r = operating_point_metrics(y, p, ["NORM", "MI", "HYP"], {"HYP": 0.5, "MI": 0.5})
    assert r["classes"] == ["MI", "HYP"]  # class order, not the order of the mapping
    assert set(r["per_class"]) == {"MI", "HYP"}
    assert r["macro"]["sensitivity"] == pytest.approx((r["per_class"]["MI"]["sensitivity"]
                                                       + r["per_class"]["HYP"]["sensitivity"]) / 2)


def test_macro_is_none_when_any_class_is_undefined():
    per_class = {"A": {"precision": 0.5}, "B": {"precision": None}}
    assert macro_mean(per_class, ["A", "B"], ["precision"]) == {"precision": None}
    assert macro_mean(per_class, ["A"], ["precision"]) == {"precision": 0.5}
    with pytest.raises(ValueError, match="at least one class"):
        macro_mean(per_class, [], ["precision"])


@pytest.mark.parametrize(
    "thresholds, message",
    [({}, "No thresholds"), ({"Z": 0.5}, "unknown classes"), ({"A": float("nan")}, "nan")],
)
def test_bad_thresholds_fail_loudly(thresholds, message):
    y = np.array([[0], [1]])
    p = np.array([[0.2], [0.7]])
    with pytest.raises(ValueError, match=message):
        operating_point_metrics(y, p, ["A"], thresholds)
