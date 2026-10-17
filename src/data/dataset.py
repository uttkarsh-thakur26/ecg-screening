"""Torch Dataset over the cached PTB-XL arrays: train-only normalisation statistics, train-only augmentation.

One item is (signal, target):
    signal  float32 (12, 1000): (cached record - training mean) / training std, per lead
    target  float32 (5,): multi-hot over labels.superclasses, for BCEWithLogitsLoss

Only labelled records are served (21,388 of 21,799). Rows are in ecg_id order, so for an unshuffled
loader the i-th prediction belongs to dataset.ecg_ids[i].

The test split (fold 10) is locked (src/data/splits.py, check_split_access). ECGDataset, make_loader
and Cache.rows raise TestSetLockedError for "test" unless allow_test=True is passed, and only the
one-time final evaluation and the frozen-model demo may pass it. Training and every tuning step
(thresholds, temperature, hyperparameters, early stopping) use the validation split.
tests/test_dataset.py checks that no module outside src/data/ mentions allow_test or folds, apart
from an explicit allowlist that starts empty.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from src.data.augment import Augmenter
from src.data.cache import Cache, open_cache
from src.data.norm_stats import ecg_id_digest, load_norm_stats, norm_stats_path
from src.data.splits import TestSetLockedError, assert_no_patient_leakage, assign_splits, check_split_access

__all__ = ["ECGDataset", "TestSetLockedError", "check_norm_stats", "make_loader", "make_train_val_loaders"]


def check_norm_stats(stats: dict[str, Any], cache: Cache, train_ecg_ids: np.ndarray) -> None:
    """Raise unless the statistics were computed on exactly this training split, from this cache's signals."""
    problems = []
    if stats["split"] != "train":
        problems.append(f"computed on split {stats['split']!r}, not 'train'")
    if stats["preprocessing"] != cache.meta["preprocessing"]:
        problems.append(f"computed on {stats['preprocessing']!r} signals, the cache holds {cache.meta['preprocessing']!r}")
    if stats["cache_signals_sha256"] != cache.meta["signals_sha256"]:
        problems.append("computed for a different signal cache (its SHA-256 differs from meta.json's)")
    if stats["leads"] != cache.meta["leads"]:
        problems.append(f"leads {stats['leads']} differ from the cache's {cache.meta['leads']}")
    if stats["n_records"] != len(train_ecg_ids) or stats["ecg_ids_sha256"] != ecg_id_digest(train_ecg_ids):
        problems.append(f"computed on {stats['n_records']:,} records that are not this config's "
                        f"{len(train_ecg_ids):,} training records")
    std = np.asarray(stats["std"], dtype=np.float64)
    if not (np.isfinite(std).all() and (std > 0).all()):
        problems.append(f"standard deviations must be finite and positive, got {stats['std']}")
    if problems:
        raise ValueError("Normalisation statistics do not match the data:\n  " + "\n  ".join(problems)
                         + "\nRecompute them with `python -m src.data.prepare`.")


class ECGDataset(Dataset):
    """Labelled records of one split, normalised with the training statistics; augmented only for training."""

    def __init__(self, cfg: dict[str, Any], split: str, *, augment: bool | None = None, allow_test: bool = False) -> None:
        self._setup(cfg, split, augment, allow_test, build_if_missing=True)

    def _setup(self, cfg: dict[str, Any], split: str, augment: bool | None, allow_test: bool, build_if_missing: bool) -> None:
        check_split_access(split, allow_test)
        if augment is None:
            augment = split == "train"
        if augment and split != "train":
            raise ValueError(f"Augmentation is for the training split only; {split!r} is never augmented")
        cfg = copy.deepcopy(cfg)  # a snapshot: later edits to the caller's dict must not reach the workers
        self._args = (cfg, split, augment, allow_test)

        cache = open_cache(cfg, build_if_missing=build_if_missing)
        index = cache.index.set_index("ecg_id")
        assert_no_patient_leakage(index, assign_splits(index, cfg["splits"]))

        stats = load_norm_stats(norm_stats_path(cfg))
        check_norm_stats(stats, cache, cache.index["ecg_id"].to_numpy()[cache.rows("train")])

        rows = cache.rows(split, allow_test=allow_test)
        self.split = split
        self.classes: tuple[str, ...] = tuple(cache.meta["classes"])
        self.leads: tuple[str, ...] = tuple(cache.meta["leads"])
        self.signals_path = cache.signals_path
        self.rows = rows
        self.ecg_ids = cache.index["ecg_id"].to_numpy()[rows]
        self.patient_ids = cache.index["patient_id"].to_numpy()[rows]
        self.targets = cache.labels[rows].astype(np.float32)
        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).unsqueeze(1)  # (leads, 1)
        self.std = torch.tensor(stats["std"], dtype=torch.float32).unsqueeze(1)
        self.augmenter: Augmenter | None = None
        self._generator: torch.Generator | None = None
        if augment:
            ds = cfg["dataset"]
            n_samples = ds["sampling_rate_hz"] * ds["record_duration_s"]
            self.augmenter = Augmenter(cfg["augmentation"], ds["sampling_rate_hz"], n_samples)
            self._generator = torch.Generator().manual_seed(cfg["training"]["seed"])
        # What a worker's copy must agree on with this one: the records, the signals and the statistics.
        self._fingerprint = {
            "ecg_ids_sha256": ecg_id_digest(self.ecg_ids),
            "signals_sha256": cache.meta["signals_sha256"],
            "mean": stats["mean"],
            "std": stats["std"],
        }
        self._signals: np.ndarray | None = None  # opened on first use, in whichever process reads items

    @property
    def augment(self) -> bool:
        return self.augmenter is not None and self.augmenter.enabled

    def __len__(self) -> int:
        return len(self.rows)

    # DataLoader workers receive the dataset pickled. Only the constructor arguments are sent and each
    # worker rebuilds (and re-checks) the dataset from the cache: a pickled np.memmap would carry the
    # whole 1 GB array, and on Windows a pickle larger than the pipe buffer makes workers start one
    # after another (measured: 39 s for 8 workers with the arrays pickled, 7 s without).
    def __getstate__(self) -> dict[str, Any]:
        return {"args": self._args, "fingerprint": self._fingerprint}

    def __setstate__(self, state: dict[str, Any]) -> None:
        cfg, split, augment, allow_test = state["args"]
        self._setup(cfg, split, augment, allow_test, build_if_missing=False)
        differs = sorted(k for k, v in state["fingerprint"].items() if self._fingerprint[k] != v)
        if differs:
            raise RuntimeError(f"The {split} dataset rebuilt in this worker differs from the parent's in {differs}; "
                               "the cache or the statistics changed while the DataLoader was running")

    def normalise(self, record: np.ndarray) -> torch.Tensor:
        """(leads, samples) cached record -> z-scored float32 tensor, with the training mean and std."""
        return (torch.from_numpy(np.array(record, dtype=np.float32)) - self.mean) / self.std

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self._signals is None:
            self._signals = np.load(self.signals_path, mmap_mode="r")
        x = self.normalise(self._signals[self.rows[i]])
        if self.augmenter is not None:
            # In a worker process, torch's RNG was seeded by the DataLoader from the loader's generator.
            # In the main process (num_workers = 0), use this dataset's own generator, seeded from training.seed.
            x = self.augmenter(x, self._generator if get_worker_info() is None else None)
        return x, torch.from_numpy(self.targets[i])


def make_loader(cfg: dict[str, Any], split: str, *, augment: bool | None = None, allow_test: bool = False) -> DataLoader:
    """DataLoader for one split. Training batches are shuffled with a seeded generator; val/test keep ecg_id order.

    With the same training.seed, two loaders give the same batches, augmentation included, for any
    num_workers (each setting gives its own reproducible sequence).
    """
    check_split_access(split, allow_test)  # fail before touching the cache
    dataset = ECGDataset(cfg, split, augment=augment, allow_test=allow_test)
    dl, training = cfg["dataloader"], cfg["training"]
    shuffle = split == "train"
    workers = int(dl["num_workers"][split])
    return DataLoader(
        dataset,
        batch_size=training["batch_size"],
        shuffle=shuffle,
        # Seeds the shuffle order and, in worker processes, the torch RNG that drives the augmentation.
        generator=torch.Generator().manual_seed(training["seed"]) if shuffle else None,
        num_workers=workers,
        pin_memory=bool(dl["pin_memory"]) and torch.cuda.is_available(),  # pinning only speeds up copies to a GPU
        persistent_workers=bool(dl["persistent_workers"]) and workers > 0,
        drop_last=False,
    )


def make_train_val_loaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader]:
    """The two loaders training and tuning code may use."""
    return make_loader(cfg, "train"), make_loader(cfg, "val")
