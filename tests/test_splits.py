"""Patient-wise splits: the leakage test on real data, plus checks that the assertion can fail."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.splits import SPLITS, assert_no_patient_leakage, assign_splits, fold_to_split


def test_no_patient_leakage(data, cfg):
    patients = {s: set(data.labelled.loc[data.split == s, "patient_id"]) for s in SPLITS}
    assert all(patients.values())
    assert patients["train"].isdisjoint(patients["val"])
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])
    # The same must hold over all 21,799 records, including the ones dropped for having no label.
    assert_no_patient_leakage(data.db, assign_splits(data.db, cfg["splits"]))


def test_splits_use_the_official_folds(data):
    folds = {s: sorted(data.labelled.loc[data.split == s, "strat_fold"].unique().tolist()) for s in SPLITS}
    assert folds == {"train": [1, 2, 3, 4, 5, 6, 7, 8], "val": [9], "test": [10]}
    assert data.split.notna().all()


def test_leakage_assertion_catches_a_shared_patient(cfg):
    db = pd.DataFrame({"patient_id": [1.0, 2.0, 1.0], "strat_fold": [1, 9, 10]}, index=[10, 11, 12])
    with pytest.raises(AssertionError, match="Patient leakage: 1 patient_id"):
        assert_no_patient_leakage(db, assign_splits(db, cfg["splits"]))


def test_a_fold_cannot_be_in_two_splits():
    with pytest.raises(ValueError, match="both"):
        fold_to_split({"train_folds": [1, 9], "val_folds": [9], "test_folds": [10]})


def test_unassigned_fold_fails_loudly(cfg):
    db = pd.DataFrame({"patient_id": [1.0], "strat_fold": [11]}, index=[1])
    with pytest.raises(ValueError, match="11"):
        assign_splits(db, cfg["splits"])
