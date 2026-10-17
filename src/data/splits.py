"""Official PTB-XL splits from strat_fold, and the patient-leakage assertion.

PTB-XL's strat_fold column places every record of a patient in the same fold, so splitting by
fold (1-8 train, 9 validation, 10 test) is splitting by patient. assert_no_patient_leakage checks
that this holds instead of trusting it.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

SPLITS = ("train", "val", "test")


class TestSetLockedError(RuntimeError):
    """Raised when code that has not declared itself the final evaluation asks for the test split."""

    __test__ = False  # the name starts with "Test"; tell pytest this is not a test class


def check_split_access(split: str, allow_test: bool) -> None:
    """The test-split lock. Every API that selects a split's records calls this first.

    Fold 10 is evaluated once, on the frozen model. Only the final evaluation and the frozen-model
    demo may pass allow_test=True; it must be the bool True, so a string such as "False" from a
    command line or environment variable cannot unlock it by being truthy.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if not isinstance(allow_test, bool):
        raise TypeError(f"allow_test must be True or False, got {allow_test!r} ({type(allow_test).__name__})")
    if split == "test" and not allow_test:
        raise TestSetLockedError(
            "The test split (fold 10) is locked. It is evaluated once, on the frozen model, in the final "
            "evaluation. Training and tuning (thresholds, temperature, hyperparameters, early stopping) must "
            "use the validation split (fold 9). Only the final evaluation or the frozen-model demo may pass "
            "allow_test=True."
        )


def fold_to_split(splits_cfg: dict[str, Any]) -> dict[int, str]:
    """{fold: split name} from the config; a fold may belong to one split only."""
    mapping: dict[int, str] = {}
    for split in SPLITS:
        folds = splits_cfg[f"{split}_folds"]
        if not folds:
            raise ValueError(f"splits.{split}_folds is empty")
        for fold in folds:
            if fold in mapping:
                raise ValueError(f"Fold {fold} is assigned to both {mapping[fold]!r} and {split!r}")
            mapping[int(fold)] = split
    return mapping


def assign_splits(db: pd.DataFrame, splits_cfg: dict[str, Any]) -> pd.Series:
    """Split name for each record (same index as db). Every record's strat_fold must be assigned."""
    mapping = fold_to_split(splits_cfg)
    unassigned = sorted(set(db["strat_fold"].unique()) - set(mapping))
    if unassigned:
        raise ValueError(f"strat_fold values {unassigned} are not assigned to any split")
    return db["strat_fold"].map(mapping).rename("split")


def assert_no_patient_leakage(db: pd.DataFrame, split: pd.Series) -> None:
    """Raise if any patient_id appears in more than one split.

    An explicit raise rather than `assert`, so the check still runs under `python -O`.
    """
    if db["patient_id"].isna().any():
        raise AssertionError("Some records have no patient_id, so leakage cannot be ruled out")
    patients = {s: set(db.loc[split == s, "patient_id"]) for s in SPLITS}
    for a, b in combinations(SPLITS, 2):
        shared = patients[a] & patients[b]
        if shared:
            example = sorted(shared)[:5]
            raise AssertionError(f"Patient leakage: {len(shared)} patient_id(s) in both {a} and {b}, e.g. {example}")


def split_summary(db: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    """Folds, records and unique patients per split."""
    rows = []
    for s in SPLITS:
        part = db[split == s]
        rows.append(
            {
                "split": s,
                "folds": ",".join(str(f) for f in sorted(part["strat_fold"].unique())),
                "records": len(part),
                "patients": part["patient_id"].nunique(),
            }
        )
    return pd.DataFrame(rows)
