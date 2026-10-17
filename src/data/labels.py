"""Map PTB-XL scp_codes to the 5 diagnostic superclasses as a multi-hot label matrix.

Follows PhysioNet's example_physionet.py: parse each scp_codes string with ast.literal_eval,
keep the statements whose scp_statements.csv row has diagnostic == 1, and map each one to its
diagnostic_class. A record can carry several superclasses at once (multi-label), so the output
is an (N, 5) 0/1 matrix, not a single class per record. Records left with no superclass are
dropped and counted.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Labels:
    ecg_ids: np.ndarray  # (N,) ecg_id of each kept record, in the same order as the rows of y
    y: np.ndarray  # (N, C) uint8 multi-hot; column j is classes[j]
    classes: tuple[str, ...]
    n_dropped_no_diagnostic: int  # scp_codes held only form/rhythm statements
    n_dropped_below_likelihood: int  # had diagnostic statements, all below min_likelihood

    @property
    def n_dropped(self) -> int:
        return self.n_dropped_no_diagnostic + self.n_dropped_below_likelihood


def read_statements(raw_dir: Path, statements_csv: str) -> pd.DataFrame:
    """scp_statements.csv indexed by SCP code."""
    path = raw_dir / statements_csv
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.download` first.")
    return pd.read_csv(path, index_col=0)


def diagnostic_map(statements: pd.DataFrame, classes: list[str]) -> dict[str, str]:
    """SCP code -> superclass for every statement with diagnostic == 1."""
    diag = statements[statements["diagnostic"] == 1]
    if diag.empty:
        raise ValueError("scp_statements.csv has no rows with diagnostic == 1")
    missing = diag.index[diag["diagnostic_class"].isna()].tolist()
    if missing:
        raise ValueError(f"Diagnostic statements without a diagnostic_class: {missing}")
    unexpected = sorted(set(diag["diagnostic_class"]) - set(classes))
    if unexpected:
        raise ValueError(f"diagnostic_class values {unexpected} are not among the configured superclasses {classes}")
    absent = sorted(set(classes) - set(diag["diagnostic_class"]))
    if absent:
        raise ValueError(f"Configured superclasses {absent} have no diagnostic statement in scp_statements.csv")
    return diag["diagnostic_class"].to_dict()


def parse_scp_codes(text: str) -> dict[str, float]:
    """'{"NORM": 100.0, "SR": 0.0}' -> {"NORM": 100.0, "SR": 0.0} (code -> likelihood)."""
    codes = ast.literal_eval(text)
    if not isinstance(codes, dict):
        raise ValueError(f"scp_codes is not a dictionary: {text!r}")
    return codes


def build_labels(db: pd.DataFrame, statements: pd.DataFrame, classes: list[str], min_likelihood: float) -> Labels:
    """Multi-hot superclass labels for every record that has at least one."""
    code_to_class = diagnostic_map(statements, classes)
    column = {c: j for j, c in enumerate(classes)}
    known = set(statements.index)

    rows: list[np.ndarray] = []
    kept: list[int] = []
    no_diagnostic = below_likelihood = 0
    for ecg_id, text in db["scp_codes"].items():
        codes = parse_scp_codes(text)
        unknown = [code for code in codes if code not in known]
        if unknown:
            raise ValueError(f"ecg_id {ecg_id}: SCP codes {unknown} are not in scp_statements.csv")
        diagnostic = {code: lik for code, lik in codes.items() if code in code_to_class}
        hit = np.zeros(len(classes), dtype=np.uint8)
        for code, likelihood in diagnostic.items():
            if likelihood >= min_likelihood:
                hit[column[code_to_class[code]]] = 1
        if hit.any():
            rows.append(hit)
            kept.append(int(ecg_id))
        elif diagnostic:
            below_likelihood += 1
        else:
            no_diagnostic += 1

    if not rows:
        raise ValueError("No record has a diagnostic superclass")
    return Labels(
        ecg_ids=np.asarray(kept, dtype=np.int64),
        y=np.vstack(rows),
        classes=tuple(classes),
        n_dropped_no_diagnostic=no_diagnostic,
        n_dropped_below_likelihood=below_likelihood,
    )


def counts_table(labels: Labels, descriptions: dict[str, str], reference: dict[str, int]) -> pd.DataFrame:
    """Records per superclass, share of labelled records, and the difference from PhysioNet's figures."""
    counts = labels.y.sum(axis=0).astype(int)
    n = len(labels.ecg_ids)
    return pd.DataFrame(
        {
            "superclass": labels.classes,
            "description": [descriptions[c] for c in labels.classes],
            "records": counts,
            "pct_of_labelled": 100.0 * counts / n,
            "physionet": [reference[c] for c in labels.classes],
            "difference": [int(counts[j]) - reference[c] for j, c in enumerate(labels.classes)],
        }
    )
