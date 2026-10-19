"""The one-time test evaluation's building blocks: patient bootstrap, scoring, nesting and the one-time guard.

Random arrays appear here only as test inputs. These tests never open fold 10: the guard is exercised with
files in tmp_path, and the scoring with small arrays.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.evaluation import evaluate_test
from src.evaluation.evaluate_test import SEP, bootstrap, check_one_time, nest, patient_resamples, score

CLASSES = ["NORM", "MI", "HYP"]
ABNORMAL = ["MI", "HYP"]


def test_patient_resamples_keep_every_record_of_a_drawn_patient():
    patient_ids = np.array([7, 7, 7, 3, 5, 5])
    for idx in patient_resamples(patient_ids, 50, seed=0):
        drawn = {p: np.sum(patient_ids[idx] == p) for p in (7, 3, 5)}
        per_patient = {7: 3, 3: 1, 5: 2}
        for p, n in drawn.items():
            assert n % per_patient[p] == 0  # whole patients only
            for r in np.flatnonzero(patient_ids == p):
                assert np.sum(idx == r) == n // per_patient[p]  # each of its records equally often
        assert sum(n // per_patient[p] for p, n in drawn.items()) == 3  # as many patients as the split has


def test_patient_resamples_are_reproducible_and_seeded():
    ids = np.repeat(np.arange(20), 2)
    a = [x.tolist() for x in patient_resamples(ids, 5, seed=42)]
    b = [x.tolist() for x in patient_resamples(ids, 5, seed=42)]
    c = [x.tolist() for x in patient_resamples(ids, 5, seed=43)]
    assert a == b and a != c
    with pytest.raises(ValueError):
        list(patient_resamples(ids, 0, seed=1))


def test_bootstrap_point_value_and_percentile_interval():
    rng = np.random.default_rng(0)
    values = rng.normal(10.0, 1.0, 400)
    patients = np.arange(400)
    out = bootstrap(lambda idx: {"mean": float(values[idx].mean())}, patients, 300, 0.95, seed=1)["mean"]
    assert out["value"] == pytest.approx(values.mean())
    assert out["ci_low"] < out["value"] < out["ci_high"]
    se = values.std() / np.sqrt(400)  # the interval should be about +/- 1.96 standard errors wide
    assert out["ci_high"] - out["ci_low"] == pytest.approx(2 * 1.96 * se, rel=0.2)
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap(lambda idx: {"m": 0.0}, patients, 5, 95, seed=1)


def test_nest():
    assert nest({"a/b/c": 1, "a/b/d": 2, "a/e": 3}) == {"a": {"b": {"c": 1, "d": 2}, "e": 3}}


def _predictions():
    rng = np.random.default_rng(3)
    y = (rng.random((300, 3)) < [0.45, 0.25, 0.12]).astype(int)
    p_cal = np.clip(0.5 * y + rng.random((300, 3)) * 0.5, 0, 1)
    p_uncal = np.clip(p_cal * 1.1, 0, 1)
    p_gbdt = np.clip(0.3 * y + rng.random((300, 3)) * 0.7, 0, 1)
    thresholds = {"max_f1": {"NORM": 0.5, "MI": 0.5, "HYP": 0.5}, "high_sensitivity": {"MI": 0.3, "HYP": 0.3}}
    return y, p_uncal, p_cal, p_gbdt, thresholds


def test_score_covers_the_plan_and_is_consistent():
    y, p_uncal, p_cal, p_gbdt, thresholds = _predictions()
    s = score(y, p_uncal, p_cal, p_gbdt, CLASSES, ABNORMAL, thresholds, 10)
    for c in [*CLASSES, "macro", "macro_abnormal"]:
        for m in ("auroc", "auprc"):
            assert SEP.join(("cnn", "ranking", c, m)) in s
            diff = s[SEP.join(("cnn", "ranking", c, m))] - s[SEP.join(("feature_gbdt", "ranking", c, m))]
            assert s[SEP.join(("cnn_minus_feature_gbdt", "ranking", c, m))] == pytest.approx(diff, abs=1e-15)
    assert s["cnn/ranking/macro_abnormal/auroc"] == pytest.approx(
        np.mean([s["cnn/ranking/MI/auroc"], s["cnn/ranking/HYP/auroc"]]))
    assert "cnn/max_f1/macro/f1" in s and "cnn/max_f1/macro_abnormal/f1" in s
    assert "cnn/high_sensitivity/macro_abnormal/sensitivity" in s
    assert "cnn/high_sensitivity/macro/sensitivity" not in s  # NORM has no high-sensitivity threshold
    assert "cnn/high_sensitivity/NORM/sensitivity" not in s
    # Thresholds are applied, not re-chosen: sensitivity at 0.3 is the share of positives with p >= 0.3.
    j = CLASSES.index("MI")
    assert s["cnn/high_sensitivity/MI/sensitivity"] == pytest.approx((p_cal[y[:, j] == 1, j] >= 0.3).mean())
    for c in [*CLASSES, "macro"]:
        assert s[f"cnn/ece/calibrated_minus_uncalibrated/{c}"] == pytest.approx(
            s[f"cnn/ece/calibrated/{c}"] - s[f"cnn/ece/uncalibrated/{c}"])


def test_score_stops_on_an_undefined_metric():
    y, p_uncal, p_cal, p_gbdt, thresholds = _predictions()
    unreachable = {**thresholds, "max_f1": {**thresholds["max_f1"], "HYP": 2.0}}  # nothing flagged: no precision
    with pytest.raises(ValueError, match="undefined"):
        score(y, p_uncal, p_cal, p_gbdt, CLASSES, ABNORMAL, unreachable, 10)


@pytest.fixture
def guard_cfg(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_test, "last_commit_of", lambda path: "abc123")
    monkeypatch.setattr(evaluate_test, "uncommitted", lambda paths: [])
    (tmp_path / "TEST_PLAN.md").write_text("plan\n", encoding="utf-8")
    return {"paths": {"reports_dir": tmp_path.as_posix()},
            "final_evaluation": {"plan_file": "TEST_PLAN.md", "metrics_file": "metrics.json",
                                 "access_log": "fold10_access_log.jsonl", "guarded_paths": ["src"]}}


CLEAN = {"commit": "abc123", "dirty": False}


def test_one_time_guard_passes_a_first_clean_run(guard_cfg):
    plan = check_one_time(guard_cfg, CLEAN, None)
    assert plan["commit"] == "abc123" and len(plan["sha256"]) == 64


def test_one_time_guard_refuses_uncommitted_code_or_plan(guard_cfg, monkeypatch):
    with pytest.raises(RuntimeError, match="Commit first"):
        check_one_time(guard_cfg, {"commit": "abc123", "dirty": True}, None)
    monkeypatch.setattr(evaluate_test, "uncommitted", lambda paths: ["?? src/evaluation/evaluate_test.py"])
    with pytest.raises(RuntimeError, match="untracked"):
        check_one_time(guard_cfg, CLEAN, None)
    monkeypatch.setattr(evaluate_test, "uncommitted", lambda paths: [])
    monkeypatch.setattr(evaluate_test, "last_commit_of", lambda path: None)
    with pytest.raises(RuntimeError, match="committed"):
        check_one_time(guard_cfg, CLEAN, None)


def test_access_log_latches_the_first_opening(guard_cfg, monkeypatch):
    plan = check_one_time(guard_cfg, CLEAN, None)
    evaluate_test.record_access(guard_cfg, CLEAN, plan, None)  # as just before fold 10 is opened
    with pytest.raises(FileExistsError):  # a second run started at the same time cannot also be "the first"
        evaluate_test.record_access(guard_cfg, CLEAN, plan, None)
    # A crash after opening leaves no metrics.json, but the log still refuses a plain rerun...
    with pytest.raises(RuntimeError, match="opened already"):
        check_one_time(guard_cfg, CLEAN, None)
    # ...and allows a documented one, which is appended to the log.
    check_one_time(guard_cfg, CLEAN, "fixed a crash in draw_figures")
    evaluate_test.record_access(guard_cfg, CLEAN, plan, "fixed a crash in draw_figures")
    log = evaluate_test.read_access_log(evaluate_test.access_log_path(guard_cfg))
    assert [e["rerun_reason"] for e in log] == [None, "fixed a crash in draw_figures"]


def test_rerun_needs_an_opening_the_same_plan_and_a_new_commit(guard_cfg, tmp_path):
    with pytest.raises(RuntimeError, match="never been opened"):
        check_one_time(guard_cfg, CLEAN, "fixed a bug")
    plan = check_one_time(guard_cfg, CLEAN, None)
    evaluate_test.record_access(guard_cfg, CLEAN, plan, None)
    (tmp_path / "metrics.json").write_text(json.dumps({"git": {"commit": "abc123"}}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="evaluated once"):
        check_one_time(guard_cfg, CLEAN, None)
    with pytest.raises(RuntimeError, match="committed bug fix"):  # no new commit since metrics.json
        check_one_time(guard_cfg, CLEAN, "fixed a bug")
    check_one_time(guard_cfg, {"commit": "def456", "dirty": False}, "fixed a bug")
    (tmp_path / "TEST_PLAN.md").write_text("plan, edited after seeing fold 10\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="same plan"):
        check_one_time(guard_cfg, {"commit": "def456", "dirty": False}, "fixed a bug")


def test_seed_values_keep_means_and_stds_apart():
    report = {"summary": {"macro_auroc": {"mean": 0.93, "std": 0.001}},
              "runs": [{"run_id": "r", "macro": {"auroc": 0.93, "auprc": 0.81}}]}
    values = evaluate_test.seed_values(report)
    assert values["macro_auroc/mean"] == 0.93 and values["macro_auroc/std"] == 0.001
    assert values["r/macro/auprc"] == 0.81
