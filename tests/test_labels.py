"""Superclass labels: real-data counts against PhysioNet, plus the mapping rules on a small table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.labels import build_labels, diagnostic_map

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
# A miniature scp_statements.csv: one diagnostic statement per superclass plus a rhythm statement.
STATEMENTS = pd.DataFrame(
    {"diagnostic": [1, 1, 1, 1, 1, np.nan], "diagnostic_class": ["NORM", "MI", "STTC", "CD", "HYP", np.nan]},
    index=["NORM", "IMI", "NDT", "LAFB", "LVH", "SR"],
)
DB = pd.DataFrame(
    {
        "scp_codes": [
            "{'NORM': 100.0, 'SR': 0.0}",
            "{'IMI': 15.0, 'LVH': 50.0, 'SR': 0.0}",  # two superclasses at once
            "{'SR': 0.0}",  # rhythm only: no superclass
            "{'NDT': 0.0}",  # diagnostic, but likelihood 0
        ]
    },
    index=pd.Index([1, 2, 3, 4], name="ecg_id"),
)


def test_counts_match_physionet(data, cfg):
    y = data.labels.y
    assert y.shape == (21388, 5) and y.dtype == np.uint8 and set(np.unique(y)) == {0, 1}
    assert list(data.labels.classes) == cfg["labels"]["superclasses"]
    counts = dict(zip(data.labels.classes, y.sum(axis=0).tolist()))
    assert counts == cfg["labels"]["physionet_reference_counts"]
    assert data.labels.n_dropped_no_diagnostic == 411 and data.labels.n_dropped_below_likelihood == 0
    assert list(data.labelled.index) == data.labels.ecg_ids.tolist()


def test_multi_hot_mapping_follows_the_official_example():
    labels = build_labels(DB, STATEMENTS, CLASSES, min_likelihood=0)
    assert labels.ecg_ids.tolist() == [1, 2, 4]
    assert labels.y.tolist() == [[1, 0, 0, 0, 0], [0, 1, 0, 0, 1], [0, 0, 1, 0, 0]]
    assert labels.n_dropped_no_diagnostic == 1 and labels.n_dropped_below_likelihood == 0


def test_min_likelihood_drops_weak_statements():
    labels = build_labels(DB, STATEMENTS, CLASSES, min_likelihood=15)
    assert labels.ecg_ids.tolist() == [1, 2]
    assert labels.n_dropped_no_diagnostic == 1 and labels.n_dropped_below_likelihood == 1


def test_unknown_scp_code_fails_loudly():
    db = pd.DataFrame({"scp_codes": ["{'XYZ': 100.0}"]}, index=pd.Index([1], name="ecg_id"))
    with pytest.raises(ValueError, match="XYZ"):
        build_labels(db, STATEMENTS, CLASSES, min_likelihood=0)


def test_unexpected_diagnostic_class_fails_loudly():
    statements = STATEMENTS.copy()
    statements.loc["LVH", "diagnostic_class"] = "OTHER"
    with pytest.raises(ValueError, match="OTHER"):
        diagnostic_map(statements, CLASSES)
