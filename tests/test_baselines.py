"""Baselines: the RR and lead features, the beat detector on real training records, and both models."""

from __future__ import annotations

import copy

import numpy as np
import pytest
from scipy.stats import kurtosis, skew

from src.data.cache import open_cache
from src.evaluation.metrics import ranking_metrics
from src.models.baselines import (
    LEAD_STATS,
    RR_FEATURES,
    FeatureBaseline,
    PriorBaseline,
    check_rpeak_config,
    detect_rpeaks,
    feature_names,
    lead_statistics,
    rr_features,
)

FS = 100

# Training-split records whose lead II beats were counted by eye on a plot. 3151, 1036 and 8457 contain
# large ventricular ectopic beats (PVCs); a threshold taken from the record's maximum found only those.
EYE_COUNTED_BEATS = {1: 11, 3151: 16, 1036: 16, 8457: 18, 20961: 12, 21690: 12}


@pytest.fixture(scope="module")
def cache(cfg):
    return open_cache(cfg, build_if_missing=False)


@pytest.fixture(scope="module")
def rp(cfg):
    return cfg["baselines"]["rpeaks"]


def _lead(cache, ecg_id: int, lead: str) -> np.ndarray:
    row = int(np.flatnonzero(cache.index["ecg_id"].to_numpy() == ecg_id)[0])
    assert row in set(cache.rows("train").tolist())  # only training records were used to check the detector
    return np.asarray(cache.signals[row, cache.meta["leads"].index(lead)], dtype=np.float64)


def test_rr_features_hand_worked():
    f = dict(zip(RR_FEATURES, rr_features(np.array([0, 80, 200]), FS, min_rr_intervals=2, rr_std_ddof=1)))
    # RR intervals 0.8 s and 1.2 s.
    assert f["n_beats"] == 3
    assert f["rr_mean_s"] == pytest.approx(1.0)
    assert f["heart_rate_bpm"] == pytest.approx(60.0)
    assert f["rr_std_s"] == pytest.approx(np.sqrt(0.2**2 + 0.2**2))  # sample std, ddof 1
    assert (f["rr_min_s"], f["rr_max_s"]) == pytest.approx((0.8, 1.2))
    assert f["rr_rmssd_s"] == pytest.approx(0.4)  # one successive difference, 1.2 - 0.8


@pytest.mark.parametrize("beats", [[], [500], [100, 190]])
def test_rr_features_are_missing_with_too_few_intervals(beats):
    f = rr_features(np.array(beats, dtype=np.int64), FS, min_rr_intervals=2, rr_std_ddof=1)
    assert f[0] == len(beats)
    assert np.isnan(f[1:]).all()


def test_lead_statistics_on_a_real_record(cache):
    record = np.asarray(cache.signals[cache.rows("train")[0]], dtype=np.float64)
    stats = lead_statistics(record).reshape(len(record), len(LEAD_STATS))
    expected = np.stack([record.mean(1), record.std(1), record.min(1), record.max(1), skew(record, axis=1),
                         kurtosis(record, axis=1)], axis=1)
    np.testing.assert_allclose(stats, expected, rtol=1e-12)
    assert len(feature_names(cache.meta["leads"])) == len(RR_FEATURES) + 12 * len(LEAD_STATS) == 79


def test_flat_lead_has_undefined_shape_statistics_and_no_beats(rp):
    stats = lead_statistics(np.zeros((12, 1000))).reshape(12, len(LEAD_STATS))
    assert (stats[:, :4] == 0).all()
    assert np.isnan(stats[:, 4:]).all()
    assert len(detect_rpeaks(np.zeros(1000), FS, rp)) == 0


@pytest.mark.parametrize("ecg_id, beats", sorted(EYE_COUNTED_BEATS.items()))
def test_detector_matches_beats_counted_by_eye(cache, rp, ecg_id, beats):
    found = detect_rpeaks(_lead(cache, ecg_id, rp["lead"]), FS, rp)
    assert len(found) == beats
    assert (np.diff(found) >= round(rp["refractory_s"] * FS) - 2 * round(rp["peak_search_s"] * FS)).all()


def test_beat_counts_agree_across_leads(cache, rp):
    """No beat annotations exist, so check consistency: the same heart beats in every lead."""
    rows = np.random.default_rng(0).choice(cache.rows("train"), 500, replace=False)
    leads = [rp["lead"], *rp["agreement_leads"]]
    idx = [cache.meta["leads"].index(name) for name in leads]
    counts = np.array([[len(detect_rpeaks(np.asarray(cache.signals[r, i], dtype=np.float64), FS, rp)) for i in idx] for r in rows])
    within = np.abs(counts[:, :1] - counts[:, 1:]) <= rp["agreement_tolerance_beats"]
    assert within.all(axis=1).mean() >= 0.92  # 95% measured on 3,000 training records
    assert (counts[:, 0] >= 2).mean() >= 0.99


def test_detector_rejects_bad_input_and_settings(rp):
    with pytest.raises(ValueError, match="one lead"):
        detect_rpeaks(np.zeros((12, 1000)), FS, rp)
    with pytest.raises(ValueError, match="NaN"):
        detect_rpeaks(np.full(1000, np.nan), FS, rp)
    with pytest.raises(ValueError, match="half of refractory_s"):
        check_rpeak_config({**rp, "peak_search_s": 0.1}, FS)
    with pytest.raises(ValueError, match="min_rr_intervals"):
        check_rpeak_config({**rp, "min_rr_intervals": 1}, FS)


def test_prior_baseline_is_exactly_at_chance_on_validation(cache, cfg):
    classes = cfg["labels"]["superclasses"]
    y_train, y_val = cache.labels[cache.rows("train")], cache.labels[cache.rows("val")]
    prior = PriorBaseline().fit(y_train)
    np.testing.assert_array_equal(prior.prior_, y_train.mean(axis=0))
    scores = prior.predict_proba(len(y_val))
    assert scores.shape == (len(y_val), len(classes)) and (scores == prior.prior_).all()
    m = ranking_metrics(y_val, scores, classes)
    for j, c in enumerate(classes):
        assert m["per_class"][c]["auroc"] == 0.5
        assert m["per_class"][c]["auprc"] == pytest.approx(y_val[:, j].mean(), abs=1e-12)


@pytest.mark.parametrize("value", [True, "auto"])
def test_feature_baseline_refuses_sklearn_early_stopping(cfg, value):
    with pytest.raises(ValueError, match="early_stopping"):
        FeatureBaseline({**cfg["baselines"]["gradient_boosting"], "early_stopping": value}, seed=0)


def test_feature_baseline_trains_one_configured_classifier_per_class(cfg):
    params = copy.deepcopy(cfg["baselines"]["gradient_boosting"])
    rng = np.random.default_rng(0)  # random arrays: this checks the plumbing, not the data
    X = rng.normal(size=(400, 6))
    X[rng.random(X.shape) < 0.1] = np.nan  # missing features are allowed
    y = (rng.random((400, 5)) < 0.3).astype(np.uint8)
    model = FeatureBaseline(params, seed=7).fit(X, y)
    assert len(model.models_) == 5
    for m in model.models_:
        assert {k: m.get_params()[k] for k in params} == params
        assert m.get_params()["random_state"] == 7
    p = model.predict_proba(X)
    assert p.shape == (400, 5) and ((p >= 0) & (p <= 1)).all()
