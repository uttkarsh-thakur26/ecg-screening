"""One-time cache of every PTB-XL 100 Hz record as a single float32 array, plus labels and an index.

    python -m src.data.cache             # build if missing, otherwise check and reuse
    python -m src.data.cache --rebuild   # rebuild from the WFDB files
    python -m src.data.cache --verify    # also re-hash the signal array against meta.json

Reading 21,799 WFDB files takes about a minute; opening the cache takes milliseconds. The cache has
one row per record in ptbxl_database.csv, in ecg_id order, so it holds the 411 records without a
superclass too; the `labelled` column marks the 21,388 that training and evaluation use.

Each preprocessing variant (see src/data/preprocess.py) has its own folder, e.g. data/cache/raw/:

    signals.npy   float32 (records, 12, 1000), mV after preprocessing, opened as a read-only memmap
    labels.npy    uint8 (records, 5) multi-hot over labels.superclasses; all-zero for unlabelled rows
    index.csv     ecg_id, patient_id, strat_fold, labelled, record_path
    meta.json     what the cache was built from; written last, so a crash never leaves a cache that looks whole
    norm_stats.json   written by src.data.prepare, not here

Opening the cache checks meta.json against the config and fails loudly on any difference, so a cache
built with other settings is never used by mistake.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.labels import build_labels, read_statements
from src.data.norm_stats import ecg_id_digest
from src.data.preprocess import bandpass_settings, preprocess, preprocessing_tag, to_cache_layout
from src.data.records import load_record, read_database
from src.data.splits import SPLITS, check_split_access, fold_to_split

INDEX_COLUMNS = ["ecg_id", "patient_id", "strat_fold", "labelled", "record_path"]
SIGNAL_DTYPE = np.float32  # the dtype preprocess.to_cache_layout produces


class CacheMismatchError(RuntimeError):
    """The cache on disk was built from different data or settings than the current config."""


@dataclass(frozen=True)
class Cache:
    """The arrays of every record. Select a split's records with rows(), which applies the test-split lock.

    signals, labels and index hold all folds, fold 10 included. Code outside src/data/ must go through
    rows(), ECGDataset or make_loader rather than indexing these by fold itself; tests/test_dataset.py
    checks that it never mentions folds or allow_test.
    """

    dir: Path
    signals: np.ndarray  # read-only memmap (records, leads, samples), float32
    labels: np.ndarray  # (records, classes) uint8
    index: pd.DataFrame  # one row per array row
    meta: dict[str, Any]
    split_folds: dict[str, tuple[int, ...]]  # from config splits

    @property
    def signals_path(self) -> Path:
        return self.dir / self.meta["files"]["signals"]

    def rows(self, split: str, *, allow_test: bool = False) -> np.ndarray:
        """Array rows of the labelled records of one split, in ecg_id order. "test" needs allow_test=True."""
        check_split_access(split, allow_test)
        folds = list(self.split_folds[split])
        mask = self.index["labelled"].to_numpy() & self.index["strat_fold"].isin(folds).to_numpy()
        return np.flatnonzero(mask)


def cache_dir(cfg: dict[str, Any]) -> Path:
    """Folder of the configured preprocessing variant, e.g. data/cache/raw."""
    return project_path(cfg["paths"]["cache_dir"]) / preprocessing_tag(cfg)


def expected_meta(cfg: dict[str, Any]) -> dict[str, Any]:
    """The meta.json fields that must equal the current config for the cache to be usable."""
    ds, lab = cfg["dataset"], cfg["labels"]
    n_samples = ds["sampling_rate_hz"] * ds["record_duration_s"]
    return {
        "dataset": ds["name"],
        "version": ds["version"],
        "record_path_column": ds["record_path_column"],
        "sampling_rate_hz": ds["sampling_rate_hz"],
        "n_leads": ds["n_leads"],
        "n_samples": n_samples,
        "preprocessing": preprocessing_tag(cfg),
        "bandpass": bandpass_settings(cfg),
        "classes": list(lab["superclasses"]),
        "label_min_likelihood": lab["min_likelihood"],
        "dtype": np.dtype(SIGNAL_DTYPE).name,
    }


def check_meta(meta: dict[str, Any], cfg: dict[str, Any], where: Path) -> None:
    """Raise CacheMismatchError naming every field where the cache and the config disagree."""
    expected = expected_meta(cfg)
    diffs = [f"{k}: cache has {meta.get(k)!r}, config wants {v!r}" for k, v in expected.items() if meta.get(k) != v]
    if diffs:
        raise CacheMismatchError(
            f"The cache in {where} was built with different settings:\n  " + "\n  ".join(diffs)
            + "\nRebuild it with `python -m src.data.cache --rebuild`."
        )


def _load_one(task: tuple[str, str, dict[str, Any], dict[str, Any]]) -> tuple[tuple[str, ...], tuple[str, ...], np.ndarray]:
    raw_dir, relative_path, dataset_cfg, preprocessing_cfg = task
    signal, fields = load_record(Path(raw_dir), relative_path, dataset_cfg)
    x = preprocess(signal, dataset_cfg["sampling_rate_hz"], preprocessing_cfg)
    return tuple(fields["sig_name"]), tuple(fields["units"]), to_cache_layout(x)


def _part(dest: Path) -> Path:
    return dest.with_name(dest.name + ".part")


def _stage_npy(array: np.ndarray, dest: Path) -> Path:
    part = _part(dest)
    with part.open("wb") as f:  # a file object, so np.save does not append ".npy" to the name
        np.save(f, array)
    return part


def _stage_text(text: str, dest: Path) -> Path:
    part = _part(dest)
    part.write_text(text, encoding="utf-8")
    return part


def _commit(staged: list[tuple[Path, Path]], meta_path: Path, meta_text: str) -> None:
    """Swap the staged files into place, then write meta.json.

    The old meta.json is set aside first, so a crash part-way leaves a folder without meta.json, which
    open_cache treats as incomplete and rebuilds. On Windows a file another process has memory-mapped
    cannot be replaced; signals.npy is the only memory-mapped file and is swapped first, so if it is
    locked nothing has changed yet: the old meta.json is restored and the old cache stays usable.
    """
    old_meta = meta_path.with_name(meta_path.name + ".old")
    had_meta = meta_path.exists()
    if had_meta:
        os.replace(meta_path, old_meta)
    for i, (part, dest) in enumerate(staged):
        try:
            os.replace(part, dest)
        except PermissionError as exc:
            for p, _ in staged[i:]:
                p.unlink(missing_ok=True)
            if i == 0 and had_meta:
                os.replace(old_meta, meta_path)
                state = "The existing cache is unchanged and still usable."
            else:
                state = "The cache is now incomplete and will be rebuilt on next use."
            raise PermissionError(
                f"Could not replace {dest}: on Windows this happens while another process (a training run, "
                f"a notebook, pytest) has the cache open. Close it and run again. {state}"
            ) from exc
    os.replace(_stage_text(meta_text, meta_path), meta_path)
    old_meta.unlink(missing_ok=True)


def array_sha256(array: np.ndarray) -> str:
    """SHA-256 of the array's data in C order (not the .npy header)."""
    h = hashlib.sha256()
    for row in array:  # one record at a time, so a memmap is never read into memory whole
        h.update(np.ascontiguousarray(row).tobytes())
    return h.hexdigest()


def build_cache(cfg: dict[str, Any]) -> Path:
    """Read every record listed in ptbxl_database.csv, preprocess it, and write the cache folder."""
    ds, lab, cc = cfg["dataset"], cfg["labels"], cfg["cache"]
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    out = cache_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / cc["meta_file"]

    db = read_database(raw_dir, ds["database_csv"]).sort_index()
    if db.index.has_duplicates:
        raise ValueError(f"{ds['database_csv']} has duplicate ecg_ids")
    if db["patient_id"].isna().any():
        raise ValueError(f"{ds['database_csv']} has records without a patient_id")
    fold_to_split(cfg["splits"])  # fails on a misconfigured split before an expensive build
    if len(db) != ds["expected_records"]:
        raise ValueError(f"{len(db):,} records in {ds['database_csv']}, expected {ds['expected_records']:,}")

    statements = read_statements(raw_dir, ds["statements_csv"])
    labels = build_labels(db, statements, lab["superclasses"], lab["min_likelihood"])
    row_of = {int(e): i for i, e in enumerate(db.index)}
    y = np.zeros((len(db), len(labels.classes)), dtype=np.uint8)
    y[[row_of[int(e)] for e in labels.ecg_ids]] = labels.y
    labelled = np.zeros(len(db), dtype=bool)
    labelled[[row_of[int(e)] for e in labels.ecg_ids]] = True

    n_leads, n_samples = ds["n_leads"], ds["sampling_rate_hz"] * ds["record_duration_s"]
    paths = db[ds["record_path_column"]].tolist()
    signals = np.empty((len(db), n_leads, n_samples), dtype=SIGNAL_DTYPE)
    tasks = [(str(raw_dir), rel, ds, cfg["preprocessing"]) for rel in paths]
    leads: tuple[str, ...] | None = None
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=cc["workers"]) as pool:
        results = pool.map(_load_one, tasks, chunksize=cc["chunk_size"])  # yields in task order
        for i, (rel, (record_leads, units, x)) in enumerate(
            tqdm(zip(paths, results), total=len(tasks), desc=f"cache {preprocessing_tag(cfg)}", unit="record")
        ):
            if leads is None:
                leads = record_leads
            elif record_leads != leads:
                raise ValueError(f"{rel}: leads {record_leads} differ from {leads}")
            if set(units) != {"mV"}:
                raise ValueError(f"{rel}: units {units}, expected mV for every lead")
            signals[i] = x
    build_s = time.perf_counter() - start

    index = pd.DataFrame(
        {
            "ecg_id": db.index.to_numpy(dtype=np.int64),
            "patient_id": db["patient_id"].to_numpy(dtype=np.int64),
            "strat_fold": db["strat_fold"].to_numpy(dtype=np.int64),
            "labelled": labelled,
            "record_path": paths,
        }
    )
    if not np.array_equal(index["patient_id"].to_numpy(), db["patient_id"].to_numpy()):
        raise ValueError("patient_id values are not whole numbers; refusing to round them")

    files = {"signals": cc["signals_file"], "labels": cc["labels_file"], "index": cc["index_file"]}
    staged = [  # signals first: see _commit
        (_stage_npy(signals, out / files["signals"]), out / files["signals"]),
        (_stage_npy(y, out / files["labels"]), out / files["labels"]),
        (_stage_text(index.to_csv(index=False, lineterminator="\n"), out / files["index"]), out / files["index"]),
    ]

    meta = {
        **expected_meta(cfg),
        "layout": "(record, lead, sample)",
        "units": "mV",
        "leads": list(leads),
        "shape": list(signals.shape),
        "n_records": len(db),
        "n_labelled": int(labelled.sum()),
        "ecg_ids_sha256": ecg_id_digest(index["ecg_id"]),
        "signals_sha256": array_sha256(signals),
        "files": files,
        "build_seconds": round(build_s, 1),
    }
    _commit(staged, meta_path, json.dumps(meta, indent=2) + "\n")
    return out


def open_cache(cfg: dict[str, Any], *, build_if_missing: bool = True) -> Cache:
    """The cache of the configured variant, built first if it does not exist. Raises if it is inconsistent."""
    out = cache_dir(cfg)
    meta_path = out / cfg["cache"]["meta_file"]
    if not meta_path.is_file():
        if not build_if_missing:
            raise FileNotFoundError(f"No complete cache in {out}. Run `python -m src.data.cache`.")
        print(f"No cache in {out}; building it once from the WFDB records.")
        build_cache(cfg)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    check_meta(meta, cfg, out)
    files = meta["files"]
    signals = np.load(out / files["signals"], mmap_mode="r")
    labels = np.load(out / files["labels"])
    index = pd.read_csv(out / files["index"])

    n = meta["n_records"]
    problems = []
    if list(signals.shape) != meta["shape"] or signals.shape != (n, meta["n_leads"], meta["n_samples"]):
        problems.append(f"signals shape {signals.shape}, meta says {meta['shape']}")
    if signals.dtype != SIGNAL_DTYPE:
        problems.append(f"signals dtype {signals.dtype}, expected {np.dtype(SIGNAL_DTYPE).name}")
    if labels.shape != (n, len(meta["classes"])) or labels.dtype != np.uint8:
        problems.append(f"labels {labels.shape} {labels.dtype}, expected ({n}, {len(meta['classes'])}) uint8")
    if list(index.columns) != INDEX_COLUMNS or len(index) != n:
        problems.append(f"index has columns {list(index.columns)} and {len(index)} rows, expected {INDEX_COLUMNS} and {n}")
    if not problems:
        lab = index["labelled"].to_numpy()
        if index["labelled"].dtype != bool or int(lab.sum()) != meta["n_labelled"]:
            problems.append(f"index marks {int(lab.sum())} records labelled, meta says {meta['n_labelled']}")
        elif not np.isin(labels, (0, 1)).all() or labels[~lab].any() or not labels[lab].any(axis=1).all():
            problems.append("labels must be 0/1, all-zero for unlabelled records and non-zero for labelled ones")
        if ecg_id_digest(index["ecg_id"]) != meta["ecg_ids_sha256"] or not index["ecg_id"].is_monotonic_increasing:
            problems.append("index ecg_ids do not match meta.json or are not in ascending order")
    if problems:
        raise CacheMismatchError(f"The cache in {out} is inconsistent:\n  " + "\n  ".join(problems)
                                 + "\nRebuild it with `python -m src.data.cache --rebuild`.")

    fold_to_split(cfg["splits"])  # each fold in one split only
    split_folds = {s: tuple(int(f) for f in cfg["splits"][f"{s}_folds"]) for s in SPLITS}
    present = set(index.loc[index["labelled"], "strat_fold"])
    absent = {s: sorted(set(f) - present) for s, f in split_folds.items() if set(f) - present}
    if absent:  # e.g. test_folds pointing at a fold that does not exist would leave fold 10 in another split
        raise ValueError(f"Configured folds have no labelled records: {absent}")
    return Cache(out, signals, labels, index, meta, split_folds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build (once) or check the PTB-XL signal cache.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if a complete cache exists")
    parser.add_argument("--verify", action="store_true", help="re-hash signals.npy against meta.json")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)

    if args.rebuild:
        build_cache(cfg)
    cache = open_cache(cfg)
    m = cache.meta
    size_gb = cache.signals_path.stat().st_size / 1e9
    print(f"Cache {cache.dir} ({m['preprocessing']}), built in {m['build_seconds']} s")
    print(f"  signals {tuple(cache.signals.shape)} {cache.signals.dtype} {m['layout']}, {size_gb:.2f} GB")
    print(f"  labels  {tuple(cache.labels.shape)} {cache.labels.dtype}, classes {m['classes']}")
    print(f"  index   {len(cache.index):,} records, {m['n_labelled']:,} with a superclass")
    if args.verify:
        actual = array_sha256(cache.signals)
        if actual != m["signals_sha256"]:
            raise CacheMismatchError(f"signals SHA-256 {actual} does not match meta.json {m['signals_sha256']}")
        print(f"  signals SHA-256 matches meta.json: {actual}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
