"""Baselines: the RR and lead features, the beat detector on real training records, and both models."""

from __future__ import annotations

import copy
import math
import re
import warnings

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
    _features_chunk,
    check_rpeak_config,
    detect_rpeaks,
    extract_features,
    feature_names,
    format_table,
    lead_statistics,
    rr_features,
)
from src.models.rpeak_check import count_beats
from src.provenance import git_state

FS = 100

# Training-split records whose lead II beats were counted by eye on a plot. 3151, 1036 and 8457 contain
# large ventricular ectopic beats (PVCs); with the threshold taken from the record's maximum, far fewer beats are found.
EYE_COUNTED_BEATS = {1: 11, 3151: 16, 1036: 16, 8457: 18, 20961: 12, 21690: 12}

# Training records where one rule of the detector changes the count, checked by eye on leads II, I, V2 and V5.
# 15946 and 4911: without the T-wave rule, the large repolarisation wave ~0.25 s after a PVC is counted as a beat.
# 9545: a PVC that is tiny in lead II (large in I, V2, V5) is missed without the search-back.
# (ecg_id, setting that switches the rule off, beats with the rule, beats without it)
RULE_CASES = [
    (15946, {"twave_slope_ratio": 0.0}, 13, 14),
    (4911, {"twave_slope_ratio": 0.0}, 11, 12),
    (9545, {"searchback_rr_ratio": math.inf}, 12, 11),
]


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


@pytest.mark.parametrize("offset", [0.0, 0.1, -0.85])
def test_constant_lead_has_undefined_shape_statistics_and_no_beats(rp, offset):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no precision-loss warnings from scipy either
        stats = lead_statistics(np.full((12, 1000), offset)).reshape(12, len(LEAD_STATS))
        beats = detect_rpeaks(np.full(1000, offset), FS, rp)
    np.testing.assert_allclose(stats[:, [0, 2, 3]], offset)
    assert np.isnan(stats[:, 4:]).all()
    assert len(beats) == 0  # filter round-off on a constant lead once gave ~40 phantom beats


@pytest.mark.parametrize("ecg_id, beats", sorted(EYE_COUNTED_BEATS.items()))
def test_detector_matches_beats_counted_by_eye(cache, rp, ecg_id, beats):
    found = detect_rpeaks(_lead(cache, ecg_id, rp["lead"]), FS, rp)
    assert len(found) == beats


@pytest.mark.parametrize("ecg_id, rule_off, with_rule, without_rule", RULE_CASES)
def test_each_rule_of_the_detector_matters(cache, rp, ecg_id, rule_off, with_rule, without_rule):
    lead = _lead(cache, ecg_id, rp["lead"])
    assert len(detect_rpeaks(lead, FS, rp)) == with_rule  # the count seen by eye
    assert len(detect_rpeaks(lead, FS, {**rp, **rule_off})) == without_rule


def test_beats_agree_across_leads_and_respect_the_refractory_period(cache, rp):
    """No beat annotations exist, so check consistency: the same heart beats in every lead."""
    rows = np.random.default_rng(0).choice(cache.rows("train"), 500, replace=False)
    leads = [rp["lead"], *rp["agreement_leads"]]
    idx = [cache.meta["leads"].index(name) for name in leads]
    refractory = round(rp["refractory_s"] * FS)
    counts = np.empty((len(rows), len(idx)), dtype=int)
    for k, r in enumerate(rows):
        for j, i in enumerate(idx):
            beats = detect_rpeaks(np.asarray(cache.signals[r, i], dtype=np.float64), FS, rp)
            assert (np.diff(beats) >= refractory).all()
            assert ((beats >= 0) & (beats < cache.signals.shape[-1])).all()
            counts[k, j] = len(beats)
    within = np.abs(counts[:, :1] - counts[:, 1:]) <= rp["agreement_tolerance_beats"]
    # 96% measured on 3,000 training records (src/models/rpeak_check.py); a threshold from the maximum gets 93%.
    assert within.all(axis=1).mean() >= 0.94
    assert (counts[:, 0] >= 2).mean() >= 0.99


def test_threshold_from_the_maximum_misses_beats_in_pvc_records(cache, rp):
    lead = _lead(cache, 3151, rp["lead"])
    assert count_beats("configured", lead, FS, rp) == EYE_COUNTED_BEATS[3151]
    assert count_beats("threshold from max", lead, FS, rp) < EYE_COUNTED_BEATS[3151] // 2
    with pytest.raises(ValueError, match="Unknown variant"):
        count_beats("nonsense", lead, FS, rp)


def test_detector_rejects_bad_input_and_settings(rp):
    with pytest.raises(ValueError, match="one lead"):
        detect_rpeaks(np.zeros((12, 1000)), FS, rp)
    with pytest.raises(ValueError, match="NaN"):
        detect_rpeaks(np.full(1000, np.nan), FS, rp)
    with pytest.raises(ValueError, match="half of refractory_s"):
        check_rpeak_config({**rp, "peak_search_s": 0.1}, FS)
    with pytest.raises(ValueError, match="min_rr_intervals"):
        check_rpeak_config({**rp, "min_rr_intervals": 1}, FS)
    with pytest.raises(ValueError, match="whole number of samples"):  # 7.5 samples would be silently rounded
        detect_rpeaks(np.sin(np.arange(1000)), FS, {**rp, "peak_search_s": 0.075})


def test_parallel_feature_extraction_matches_serial(cache, cfg):
    cfg = copy.deepcopy(cfg)
    cfg["baselines"].update(workers=2, chunk_size=50)  # several chunks over two processes
    rows = cache.rows("train")[:173]
    rp, leads = cfg["baselines"]["rpeaks"], list(cache.meta["leads"])
    features = extract_features(cache, rows, cfg, "test")
    ecg_ids = cache.index["ecg_id"].to_numpy()[rows]
    X, counts = _features_chunk((str(cache.signals_path), rows, ecg_ids, FS, leads, rp))
    np.testing.assert_array_equal(features.X, X)
    np.testing.assert_array_equal(features.beat_counts, counts)
    assert features.names == feature_names(leads)
    for k in (0, 99, 172):  # columns hold what their names say
        record = np.asarray(cache.signals[rows[k]], dtype=np.float64)
        named = dict(zip(features.names, features.X[k]))
        assert named["II_max"] == record[leads.index("II")].max()
        assert named["V6_std"] == record[leads.index("V6")].std()
        assert named["n_beats"] == len(detect_rpeaks(record[leads.index(rp["lead"])], FS, rp)) == counts[k, 0]


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


def test_format_table_lists_every_class_and_the_macro_row(cfg):
    classes = cfg["labels"]["superclasses"]
    y = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]])
    p = np.array([[0.9, 0.2, 0.8, 0.1, 0.7], [0.1, 0.8, 0.3, 0.9, 0.2], [0.6, 0.7, 0.4, 0.2, 0.6], [0.3, 0.1, 0.9, 0.6, 0.4]])
    metrics = ranking_metrics(y, p, classes)
    report = {
        "classes": classes,
        "splits": {"train": {"records": 10}, "val": {"records": 4, "patients": 4, "folds": [9]}},
        "baselines": {"prior": {"metrics": metrics}, "feature_gbdt": {"metrics": metrics}},
    }
    lines = format_table(report).splitlines()
    for c in [*classes, "Macro"]:
        assert sum(line.startswith(c) for line in lines) == 1
    assert "1.000" in next(line for line in lines if line.startswith("NORM"))


def test_git_state_names_a_commit():
    state = git_state()
    assert re.fullmatch(r"[0-9a-f]{40}", state["commit"])
    assert isinstance(state["dirty"], bool)
