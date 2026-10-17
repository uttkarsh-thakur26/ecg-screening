"""Per-lead mean and standard deviation over the training split only.

The z-score statistics come from the training folds alone. Computing them over validation or test
records would let those records shape the inputs the model trains on, which is leakage. The JSON
written here records the folds and a hash of the exact ecg_ids used, so the claim can be checked;
tests/test_norm_stats.py recomputes the statistics and compares.

Each record's per-lead mean and sum of squared deviations are computed in float64, then merged with
Chan et al.'s parallel update in a fixed record order. The result is therefore identical however
many worker processes read the records.

The statistics describe the signals the model actually sees, so each record goes through the same
preprocessing as the cache (src/data/preprocess.py) first. Each preprocessing variant keeps its own
file in its own cache folder. While computing them, every training record's preprocessed float32
bytes are compared with its row in the cache, and the JSON records the cache's SHA-256, so the
Dataset can refuse statistics that do not belong to the arrays it normalises.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import project_path
from src.data.preprocess import bandpass_settings, preprocess, preprocessing_tag, to_cache_layout
from src.data.records import load_record

if TYPE_CHECKING:
    from src.data.cache import Cache

REQUIRED_KEYS = {
    "split", "folds", "n_records", "ecg_ids_sha256", "preprocessing", "cache_signals_sha256", "leads", "mean", "std",
}


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


def record_digest(record: np.ndarray) -> str:
    """SHA-256 of one record in the cache layout (float32, leads x samples, C order)."""
    return hashlib.sha256(np.ascontiguousarray(record, dtype=np.float32).tobytes()).hexdigest()


def _record_moments(task: tuple[str, str, dict[str, Any], dict[str, Any]]) -> tuple[tuple[str, ...], Moments, str]:
    raw_dir, relative_path, dataset_cfg, preprocessing_cfg = task
    signal, fields = load_record(Path(raw_dir), relative_path, dataset_cfg)
    signal = preprocess(signal, dataset_cfg["sampling_rate_hz"], preprocessing_cfg)
    mean = signal.mean(axis=0)
    m2 = ((signal - mean) ** 2).sum(axis=0)
    return tuple(fields["sig_name"]), Moments(signal.shape[0], mean, m2), record_digest(to_cache_layout(signal))


def compute_moments(
    raw_dir: Path,
    relative_paths: list[str],
    dataset_cfg: dict[str, Any],
    preprocessing_cfg: dict[str, Any],
    workers: int,
    chunk_size: int,
    desc: str,
) -> tuple[list[str], Moments, list[str]]:
    """Per-lead moments over all samples of the given preprocessed records, and each record's record_digest.

    Every record must have the same leads.
    """
    if not relative_paths:
        raise ValueError("No records to compute statistics on")
    tasks = [(str(raw_dir), rel, dataset_cfg, preprocessing_cfg) for rel in relative_paths]
    leads: tuple[str, ...] | None = None
    total: Moments | None = None
    digests: list[str] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = pool.map(_record_moments, tasks, chunksize=chunk_size)  # yields in task order
        for rel, (record_leads, moments, digest) in tqdm(zip(relative_paths, results), total=len(tasks), desc=desc, unit="record"):
            if leads is None:
                leads, total = record_leads, moments
            elif record_leads != leads:
                raise ValueError(f"{rel}: leads {record_leads} differ from {leads}")
            else:
                total = total.merge(moments)
            digests.append(digest)
    return list(leads), total, digests


def ecg_id_digest(ecg_ids: Iterable[int]) -> str:
    """SHA-256 of the sorted ecg_ids, so the stats file names exactly which records it came from."""
    return hashlib.sha256(",".join(str(i) for i in sorted(int(x) for x in ecg_ids)).encode()).hexdigest()


def norm_stats_path(cfg: dict[str, Any]) -> Path:
    """Statistics file of the configured preprocessing variant, e.g. data/cache/raw/norm_stats.json."""
    return project_path(cfg["paths"]["cache_dir"]) / preprocessing_tag(cfg) / cfg["norm_stats"]["file"]


def compute_norm_stats(cfg: dict[str, Any], raw_dir: Path, db: pd.DataFrame, train_ids: np.ndarray, cache: Cache) -> dict[str, Any]:
    """Statistics over the given training records, with the metadata needed to audit them.

    Raises if any of those records, preprocessed now, differs from its row in the cache (a stale cache).
    """
    ds, ns = cfg["dataset"], cfg["norm_stats"]
    folds = sorted(int(f) for f in db.loc[train_ids, "strat_fold"].unique())
    if folds != sorted(cfg["splits"]["train_folds"]):
        raise ValueError(f"Training records come from folds {folds}, expected {sorted(cfg['splits']['train_folds'])}")

    paths = db.loc[train_ids, ds["record_path_column"]].tolist()
    leads, moments, digests = compute_moments(raw_dir, paths, ds, cfg["preprocessing"], ns["workers"], ns["chunk_size"], "train records")
    if cache.meta["preprocessing"] != preprocessing_tag(cfg):
        raise ValueError(f"The cache holds {cache.meta['preprocessing']!r} signals, the config wants {preprocessing_tag(cfg)!r}")
    row_of = dict(zip(cache.index["ecg_id"].tolist(), range(len(cache.index))))
    stale = [int(e) for e, d in zip(train_ids, digests) if record_digest(cache.signals[row_of[int(e)]]) != d]
    if stale:
        raise ValueError(f"{len(stale):,} cached training records differ from their WFDB records preprocessed now "
                         f"(e.g. ecg_ids {stale[:5]}). Rebuild the cache with `python -m src.data.cache --rebuild`.")
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
        "preprocessing": preprocessing_tag(cfg),
        "bandpass": bandpass_settings(cfg),
        "cache_signals_sha256": cache.meta["signals_sha256"],
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
