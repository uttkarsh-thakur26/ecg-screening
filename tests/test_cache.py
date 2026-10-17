"""The signal cache: complete, identical to the WFDB records and labels it came from, reused, and checked on open."""

from __future__ import annotations

import copy
import sys

import numpy as np
import pytest

from src.config import project_path
from src.data import cache as cache_module
from src.data.cache import CacheMismatchError, array_sha256, build_cache, cache_dir, check_meta, open_cache
from src.data.records import load_record


@pytest.fixture(scope="module")
def cache(cfg):
    return open_cache(cfg)


def test_cache_layout(cache, data):
    n = len(data.db)
    assert n == 21799
    assert cache.signals.shape == (n, 12, 1000) and cache.signals.dtype == np.float32
    assert cache.labels.shape == (n, 5) and cache.labels.dtype == np.uint8
    assert list(cache.index.columns) == ["ecg_id", "patient_id", "strat_fold", "labelled", "record_path"]
    assert int(cache.index["labelled"].sum()) == len(data.labels.ecg_ids) == 21388
    assert cache.meta["leads"] == ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def test_index_and_labels_match_the_source_files(cache, data):
    db = data.db.sort_index()
    idx = cache.index
    np.testing.assert_array_equal(idx["ecg_id"], db.index)
    np.testing.assert_array_equal(idx["patient_id"], db["patient_id"])
    np.testing.assert_array_equal(idx["strat_fold"], db["strat_fold"])
    np.testing.assert_array_equal(idx["record_path"], db["filename_lr"])
    labelled = idx["labelled"].to_numpy()
    np.testing.assert_array_equal(idx["ecg_id"][labelled], np.sort(data.labels.ecg_ids))
    by_id = dict(zip(data.labels.ecg_ids.tolist(), data.labels.y))
    expected = np.stack([by_id[e] for e in idx["ecg_id"][labelled]])
    np.testing.assert_array_equal(cache.labels[labelled], expected)
    assert not cache.labels[~labelled].any()


def test_cached_signals_equal_the_wfdb_records(cfg, cache):
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    rows = []
    for _, fold in cache.index.groupby("strat_fold"):  # first, middle and last record of every fold
        positions = fold.index.to_numpy()
        rows += [positions[0], positions[len(positions) // 2], positions[-1]]
    rows.append(int(np.flatnonzero(~cache.index["labelled"].to_numpy())[0]))  # and one without a superclass
    assert len(set(cache.index["strat_fold"].iloc[rows])) == 10
    for i in rows:
        signal, _ = load_record(raw_dir, cache.index["record_path"].iloc[i], cfg["dataset"])
        np.testing.assert_array_equal(cache.signals[i], signal.T.astype(np.float32), err_msg=f"row {i}")


def test_signal_hash_matches_meta(cache):
    assert array_sha256(cache.signals) == cache.meta["signals_sha256"]


def test_cache_is_reused_not_rebuilt(cfg, cache):
    files = [cache.dir / name for name in cache.meta["files"].values()] + [cache.dir / cfg["cache"]["meta_file"]]
    before = {f: f.stat().st_mtime_ns for f in files}
    again = open_cache(cfg, build_if_missing=False)
    assert {f: f.stat().st_mtime_ns for f in files} == before
    assert again.meta == cache.meta


def test_a_cache_built_with_other_settings_is_refused(cfg, cache):
    other = copy.deepcopy(cfg)
    other["labels"]["min_likelihood"] = 15
    with pytest.raises(CacheMismatchError, match="label_min_likelihood"):
        check_meta(cache.meta, other, cache.dir)
    other = copy.deepcopy(cfg)
    other["dataset"]["version"] = "1.0.2"
    with pytest.raises(CacheMismatchError, match="version"):
        check_meta(cache.meta, other, cache.dir)


def test_each_preprocessing_variant_has_its_own_folder(cfg):
    on = copy.deepcopy(cfg)
    on["preprocessing"]["bandpass"]["enabled"] = True
    assert cache_dir(cfg).name == "raw"
    assert cache_dir(on) != cache_dir(cfg) and cache_dir(on).parent == cache_dir(cfg).parent


def test_folds_missing_from_the_data_fail_loudly(cfg):
    moved = copy.deepcopy(cfg)
    moved["splits"].update(val_folds=[9, 10], test_folds=[11])  # would quietly put fold 10 in validation
    with pytest.raises(ValueError, match="no labelled records"):
        open_cache(moved)


def test_a_rebuild_that_fails_validation_keeps_the_old_cache(cfg, tmp_path):
    c = copy.deepcopy(cfg)
    c["paths"]["cache_dir"] = str(tmp_path)
    folder = cache_dir(c)
    folder.mkdir()
    (folder / c["cache"]["meta_file"]).write_text("old", encoding="utf-8")
    c["splits"]["val_folds"] = [8]  # also a training fold
    with pytest.raises(ValueError, match="both"):
        build_cache(c)
    assert (folder / c["cache"]["meta_file"]).read_text(encoding="utf-8") == "old"


def _staged(folder):
    signals, labels, meta = folder / "signals.npy", folder / "labels.npy", folder / "meta.json"
    np.save(signals, np.zeros((2, 3), np.float32))
    np.save(labels, np.zeros(2, np.uint8))
    meta.write_text("old", encoding="utf-8")
    staged = [
        (cache_module._stage_npy(np.ones((2, 3), np.float32), signals), signals),
        (cache_module._stage_npy(np.ones(2, np.uint8), labels), labels),
    ]
    return signals, labels, meta, staged


def test_commit_swaps_every_file_then_writes_meta(tmp_path):
    signals, labels, meta, staged = _staged(tmp_path)
    cache_module._commit(staged, meta, "new")
    assert meta.read_text(encoding="utf-8") == "new"
    assert np.load(signals).sum() == 6 and np.load(labels).sum() == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.npy", "meta.json", "signals.npy"]


def test_a_locked_cache_survives_a_failed_rebuild(tmp_path):
    signals, labels, meta, staged = _staged(tmp_path)
    held = np.load(signals, mmap_mode="r")  # as a running training process holds it
    try:
        cache_module._commit(staged, meta, "new")
    except PermissionError as exc:
        assert "unchanged and still usable" in str(exc)
        assert meta.read_text(encoding="utf-8") == "old"
        assert np.load(labels).sum() == 0  # nothing was swapped
        assert sorted(p.name for p in tmp_path.iterdir()) == ["labels.npy", "meta.json", "signals.npy"]
    else:
        assert sys.platform != "win32", "Windows refuses to replace a memory-mapped file"
        assert meta.read_text(encoding="utf-8") == "new"
    finally:
        del held
