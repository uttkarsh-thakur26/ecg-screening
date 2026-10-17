"""AUROC and AUPRC: hand-worked cases, independent reimplementations, and loud failures on malformed input."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import rankdata

from src.evaluation.metrics import ranking_metrics, validate_predictions


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
        ([[0, 1], [0, 1]], [[0.2, 0.3], [0.7, 0.4]], ["A", "B"], r"A \(0 of 2 positive\), B \(2 of 2 positive\)"),
        (np.empty((0, 1)), np.empty((0, 1)), ["A"], "No records"),
    ],
)
def test_malformed_input_fails_loudly(y, p, classes, message):
    with pytest.raises(ValueError, match=message):
        validate_predictions(np.asarray(y), np.asarray(p), classes)
