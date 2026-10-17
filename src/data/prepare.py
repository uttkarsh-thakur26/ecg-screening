"""Labels, official splits and training-only normalisation statistics, in one reproducible run.

    python -m src.data.prepare

1. Map scp_codes to the 5 superclasses; drop records with none and report how many and why.
2. Print per-class counts next to PhysioNet's published figures.
3. Assign folds 1-8 / 9 / 10 to train / val / test, check that no patient is in two splits, and
   print records and patients per split.
4. Build the signal cache of every record once (src/data/cache.py), or check and reuse it.
5. Compute per-lead mean and std on the training split only, after the configured preprocessing,
   check every training record against its cache row, and save them to the preprocessing variant's
   cache folder (data/cache/raw/ with the band-pass off).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.cache import open_cache
from src.data.labels import Labels, build_labels, counts_table, read_statements
from src.data.norm_stats import compute_norm_stats, norm_stats_path, save_norm_stats
from src.data.records import read_database
from src.data.splits import assert_no_patient_leakage, assign_splits, split_summary


@dataclass(frozen=True)
class LabelledSplits:
    db: pd.DataFrame  # all records in ptbxl_database.csv
    labels: Labels
    labelled: pd.DataFrame  # rows of db for labels.ecg_ids, same order as labels.y
    split: pd.Series  # split name per labelled record

    def ids(self, name: str) -> np.ndarray:
        return self.labelled.index[self.split == name].to_numpy()


def load_labelled_splits(cfg: dict[str, Any]) -> LabelledSplits:
    """Labels and splits exactly as training will use them. Raises on patient leakage."""
    ds, lab = cfg["dataset"], cfg["labels"]
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    db = read_database(raw_dir, ds["database_csv"])
    statements = read_statements(raw_dir, ds["statements_csv"])
    labels = build_labels(db, statements, lab["superclasses"], lab["min_likelihood"])
    labelled = db.loc[labels.ecg_ids]
    split = assign_splits(labelled, cfg["splits"])
    assert_no_patient_leakage(db, assign_splits(db, cfg["splits"]))  # every record, labelled or not
    assert_no_patient_leakage(labelled, split)
    return LabelledSplits(db, labels, labelled, split)


def _print_labels(data: LabelledSplits, cfg: dict[str, Any]) -> None:
    labels, lab = data.labels, cfg["labels"]
    n_all, n_kept = len(data.db), len(labels.ecg_ids)
    if lab["min_likelihood"] == 0:
        rule = "every diagnostic statement kept, as in PhysioNet's example_physionet.py"
    else:
        rule = "a likelihood threshold PhysioNet's example_physionet.py does not apply"
    print(f"Labels (min_likelihood = {lab['min_likelihood']}: {rule})")
    print(f"  {n_all:,} records in {cfg['dataset']['database_csv']}")
    print(f"  {n_kept:,} kept with at least one diagnostic superclass")
    print(f"  {labels.n_dropped:,} dropped:")
    print(f"    {labels.n_dropped_no_diagnostic:,} have only form/rhythm statements, no diagnostic statement")
    print(f"    {labels.n_dropped_below_likelihood:,} have diagnostic statements, all below min_likelihood")

    table = counts_table(labels, lab["descriptions"], lab["physionet_reference_counts"])
    print()
    print(f"{'Class':<6} {'Description':<24} {'Records':>8} {'% of kept':>10} {'PhysioNet':>10} {'Diff':>6}")
    for r in table.itertuples():
        print(f"{r.superclass:<6} {r.description:<24} {r.records:>8,} {r.pct_of_labelled:>9.1f}% {r.physionet:>10,} {r.difference:>+6}")
    print(f"{'':<31} {int(table.records.sum()):>8,}  labels on {n_kept:,} records (multi-label, so % sums > 100)")

    per_record = np.bincount(labels.y.sum(axis=1))
    spread = ", ".join(f"{k} class{'es' if k > 1 else ''}: {int(c):,}" for k, c in enumerate(per_record) if k > 0 and c)
    print(f"  records by number of superclasses -> {spread}")


def _print_splits(data: LabelledSplits) -> None:
    print("Splits (official strat_fold, labelled records)")
    summary = split_summary(data.labelled, data.split)
    print(f"  {'Split':<6} {'Folds':<16} {'Records':>8} {'Patients':>9}")
    for r in summary.itertuples():
        print(f"  {r.split:<6} {r.folds:<16} {r.records:>8,} {r.patients:>9,}")
    print(f"  {'total':<6} {'':<16} {int(summary.records.sum()):>8,} {int(summary.patients.sum()):>9,}")
    print("  Patient leakage check passed: no patient_id is in more than one split.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build labels and splits, then compute train-only normalisation stats.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    data = load_labelled_splits(cfg)
    _print_labels(data, cfg)
    print()
    _print_splits(data)
    print()

    cache = open_cache(cfg)
    print(f"Signal cache {cache.dir}: signals {tuple(cache.signals.shape)} {cache.signals.dtype}, "
          f"labels {tuple(cache.labels.shape)}, {cache.meta['n_labelled']:,} of {cache.meta['n_records']:,} labelled")
    print()

    raw_dir = project_path(cfg["paths"]["raw_dir"])
    stats = compute_norm_stats(cfg, raw_dir, data.labelled, data.ids("train"), cache)
    path = norm_stats_path(cfg)
    save_norm_stats(stats, path)
    print(f"Normalisation stats ({stats['preprocessing']} signals) from {stats['n_records']:,} training records (folds {stats['folds']}), "
          f"{stats['n_samples_per_lead']:,} samples per lead")
    print(f"  {'Lead':<5} {'Mean (mV)':>10} {'Std (mV)':>9}")
    for lead, mean, std in zip(stats["leads"], stats["mean"], stats["std"]):
        print(f"  {lead:<5} {mean:>10.4f} {std:>9.4f}")
    print(f"  every training record matches its cache row; saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
