"""Ranking of the most confident errors, and the seed-stability checks and summary.

Small hand-made arrays appear here only as test inputs; reports/error_analysis_val.json comes from the release
model's real validation predictions.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.evaluation.error_analysis import diagnostic_statements, quality_notes, rank_errors
from src.evaluation.seed_stability import check_runs, spread

CLASSES = ["NORM", "MI", "HYP"]
THRESHOLDS = {"MI": 0.5, "HYP": 0.2}


def _arrays():
    #               NORM  MI  HYP
    y = np.array([[0, 1, 0],    # 0: MI missed at 0.10
                  [0, 1, 1],    # 1: MI missed at 0.30, HYP missed at 0.05 -> ranked by HYP 0.05
                  [1, 0, 0],    # 2: HYP false alarm at 0.90
                  [0, 0, 0],    # 3: MI false alarm at 0.95, HYP false alarm at 0.25
                  [0, 1, 0],    # 4: MI caught
                  [0, 1, 0]])   # 5: MI missed at 0.10, the same probability as record 0
    p = np.array([[0.5, 0.10, 0.10],
                  [0.5, 0.30, 0.05],
                  [0.5, 0.20, 0.90],
                  [0.5, 0.95, 0.25],
                  [0.5, 0.70, 0.10],
                  [0.5, 0.10, 0.10]])
    ecg_ids = np.array([10, 11, 12, 13, 14, 9])
    return y, p, ecg_ids


def test_false_negatives_ranked_by_lowest_probability_one_entry_per_record():
    y, p, ids = _arrays()
    ranked = rank_errors(y, p, CLASSES, THRESHOLDS, ids, "false_negatives", 10)
    assert [r["ecg_id"] for r in ranked] == [11, 9, 10]  # 0.05; then the 0.10 tie goes to the lower ecg_id (9)
    assert [e["class"] for e in ranked[0]["errors"]] == ["HYP", "MI"]  # most confident error first, both listed
    assert ranked[0]["errors"][0] == {"class": "HYP", "probability": 0.05, "threshold": 0.2}


def test_false_positives_ranked_by_highest_probability():
    y, p, ids = _arrays()
    ranked = rank_errors(y, p, CLASSES, THRESHOLDS, ids, "false_positives", 10)
    assert [r["ecg_id"] for r in ranked] == [13, 12]
    assert [(e["class"], e["probability"]) for e in ranked[0]["errors"]] == [("MI", 0.95), ("HYP", 0.25)]


def test_rank_errors_keeps_n_and_never_counts_a_class_without_threshold():
    y, p, ids = _arrays()
    assert len(rank_errors(y, p, CLASSES, THRESHOLDS, ids, "false_negatives", 2)) == 2
    # NORM has no threshold: record 2's NORM label at 0.5 is no error, whatever the probability.
    flat = [e["class"] for r in rank_errors(y, p, CLASSES, THRESHOLDS, ids, "false_negatives", 10) for e in r["errors"]]
    assert "NORM" not in flat
    with pytest.raises(ValueError, match="kind"):
        rank_errors(y, p, CLASSES, THRESHOLDS, ids, "errors", 2)
    with pytest.raises(ValueError, match="unknown"):
        rank_errors(y, p, CLASSES, {"STTC": 0.5}, ids, "false_negatives", 2)


def test_quality_notes_and_statements():
    import pandas as pd

    row = pd.Series({"static_noise": " , I-V1,  ", "burst_noise": float("nan"), "pacemaker": " , "})
    assert quality_notes(row, ["static_noise", "burst_noise", "pacemaker"]) == {
        "static_noise": "I-V1", "burst_noise": None, "pacemaker": None}
    codes = {"IMI": 15.0, "LVH": 100.0, "SR": 0.0}
    assert diagnostic_statements(codes, {"IMI": "MI", "LVH": "HYP"}, CLASSES) == {"MI": {"IMI": 15.0},
                                                                                  "HYP": {"LVH": 100.0}}


# ------------------------------------------------------------------------------------------- seed stability

def _run(run_id: str, seed: int) -> dict:
    return {"run_id": run_id, "seed": seed, "config": {"training": {"seed": seed, "lr": 1e-3}},
            "git": {"commit": "HEAD", "dirty": False}, "classes": CLASSES,
            "data": {"val_ecg_ids_sha256": "v", "signals_sha256": "s"}}


def test_spread_is_mean_and_sample_std():
    s = spread([0.90, 0.92, 0.94], ddof=1)
    assert s["mean"] == pytest.approx(0.92)
    assert s["std"] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="ddof"):
        spread([0.9], ddof=1)


def test_check_runs_accepts_seeds_of_one_config():
    check_runs([_run("a", 42), _run("b", 43), _run("c", 44)], "a")


def test_check_runs_refuses_other_differences():
    runs = [_run("a", 42), _run("b", 43)]
    other = copy.deepcopy(runs)
    other[1]["config"]["training"]["lr"] = 2e-3
    with pytest.raises(ValueError, match="not only"):
        check_runs(other, "a")
    with pytest.raises(ValueError, match="seeds repeat"):
        check_runs([_run("a", 42), _run("b", 42)], "a")
    with pytest.raises(ValueError, match="release run"):
        check_runs(runs, "z")
    dirty = copy.deepcopy(runs)
    dirty[0]["git"]["dirty"] = True
    with pytest.raises(ValueError, match="dirty"):
        check_runs(dirty, "a")
    data = copy.deepcopy(runs)
    data[1]["data"]["val_ecg_ids_sha256"] = "w"
    with pytest.raises(ValueError, match="val_ecg_ids_sha256"):
        check_runs(data, "a")
