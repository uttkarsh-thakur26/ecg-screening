"""Per-lead mean and standard deviation over the training split only.

The z-score statistics come from the training folds alone. Computing them over validation or test
records would let those records shape the inputs the model trains on, which is leakage. The JSON
written here records the folds and a hash of the exact ecg_ids used, so the claim can be checked;
tests/test_norm_stats.py recomputes the statistics and compares.

Each record's per-lead mean and sum of squared deviations are computed in float64, then merged with
Chan et al.'s parallel update in a fixed record order. The result is therefore identical however
many worker processes read the records.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import project_path
from src.data.records import load_record

REQUIRED_KEYS = {"split", "folds", "n_records", "ecg_ids_sha256", "leads", "mean", "std"}


@dataclass(frozen=True)
class Moments:
    """Count, mean and sum of squared deviations (M2) per lead."""

    n: int
    mean: np.ndarray
    m2: np.ndarray

    def merge(self, other: Moments) -> Moments:
        """Chan et al. (1979): combine two partial results without revisiting the samples."""
        n = self.n + other.n
        delta = other.mean - self.mean
        mean = self.mean + delta * (other.n / n)
        m2 = self.m2 + other.m2 + delta**2 * (self.n * other.n / n)
        return Moments(n, mean, m2)

    def std(self, ddof: int) -> np.ndarray:
        return np.sqrt(self.m2 / (self.n - ddof))


def _record_moments(task: tuple[str, str, dict[str, Any]]) -> tuple[tuple[str, ...], Moments]:
    raw_dir, relative_path, dataset_cfg = task
    signal, fields = load_record(Path(raw_dir), relative_path, dataset_cfg)
    mean = signal.mean(axis=0)
    m2 = ((signal - mean) ** 2).sum(axis=0)
    return tuple(fields["sig_name"]), Moments(signal.shape[0], mean, m2)


def compute_moments(
    raw_dir: Path,
    relative_paths: list[str],
    dataset_cfg: dict[str, Any],
    workers: int,
    chunk_size: int,
    desc: str,
) -> tuple[list[str], Moments]:
    """Per-lead moments over all samples of the given records. Every record must have the same leads."""
    if not relative_paths:
        raise ValueError("No records to compute statistics on")
    tasks = [(str(raw_dir), rel, dataset_cfg) for rel in relative_paths]
    leads: tuple[str, ...] | None = None
    total: Moments | None = None
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_record_moments, tasks, chunksize=chunk_size)  # yields in task order
        for rel, (record_leads, moments) in tqdm(zip(relative_paths, results), total=len(tasks), desc=desc, unit="record"):
            if leads is None:
                leads, total = record_leads, moments
            elif record_leads != leads:
                raise ValueError(f"{rel}: leads {record_leads} differ from {leads}")
            else:
                total = total.merge(moments)
    return list(leads), total


def ecg_id_digest(ecg_ids: Iterable[int]) -> str:
    """SHA-256 of the sorted ecg_ids, so the stats file names exactly which records it came from."""
    return hashlib.sha256(",".join(str(i) for i in sorted(int(x) for x in ecg_ids)).encode()).hexdigest()


def norm_stats_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["cache_dir"]) / cfg["norm_stats"]["file"]


def compute_norm_stats(cfg: dict[str, Any], raw_dir: Path, db: pd.DataFrame, train_ids: np.ndarray) -> dict[str, Any]:
    """Statistics over the given training records, with the metadata needed to audit them."""
    ds, ns = cfg["dataset"], cfg["norm_stats"]
    if cfg["preprocessing"]["bandpass"]["enabled"]:
        raise NotImplementedError("preprocessing.bandpass is enabled but the band-pass filter is not implemented yet")
    folds = sorted(int(f) for f in db.loc[train_ids, "strat_fold"].unique())
    if folds != sorted(cfg["splits"]["train_folds"]):
        raise ValueError(f"Training records come from folds {folds}, expected {sorted(cfg['splits']['train_folds'])}")

    paths = db.loc[train_ids, ds["record_path_column"]].tolist()
    leads, moments = compute_moments(raw_dir, paths, ds, ns["workers"], ns["chunk_size"], "train records")
    std = moments.std(ns["std_ddof"])
    if not (np.isfinite(std).all() and (std > 0).all()):
        raise ValueError(f"Degenerate standard deviation per lead: {dict(zip(leads, std))}")
    return {
        "dataset": ds["name"],
        "version": ds["version"],
        "split": "train",
        "folds": folds,
        "n_records": len(train_ids),
        "n_samples_per_lead": moments.n,
        "ecg_ids_sha256": ecg_id_digest(train_ids),
        "sampling_rate_hz": ds["sampling_rate_hz"],
        "bandpass_enabled": cfg["preprocessing"]["bandpass"]["enabled"],
        "label_min_likelihood": cfg["labels"]["min_likelihood"],
        "std_ddof": ns["std_ddof"],
        "units": "mV",
        "leads": leads,
        "mean": moments.mean.tolist(),
        "std": std.tolist(),
    }


def save_norm_stats(stats: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def load_norm_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data.prepare` first.")
    stats = json.loads(path.read_text(encoding="utf-8"))
    missing = REQUIRED_KEYS - set(stats)
    if missing:
        raise ValueError(f"{path} is missing keys {sorted(missing)}")
    return stats
