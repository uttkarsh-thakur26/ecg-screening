"""Normalisation statistics: saved from the training folds only, and reproducible."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from src.config import project_path
from src.data.norm_stats import Moments, compute_moments, ecg_id_digest, load_norm_stats, norm_stats_path
from src.data.cache import open_cache
from src.data.preprocess import preprocessing_tag


def _moments(cfg, data, ids):
    ds, ns = cfg["dataset"], cfg["norm_stats"]
    paths = data.labelled.loc[ids, ds["record_path_column"]].tolist()
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    leads, moments, _ = compute_moments(raw_dir, paths, ds, cfg["preprocessing"], ns["workers"], ns["chunk_size"], "test")
    return leads, moments


def test_normalisation_stats_from_train_only(cfg, data):
    stats = load_norm_stats(norm_stats_path(cfg))
    train_ids = data.ids("train")

    # The file says it was computed on folds 1-8, and on exactly the training records.
    assert stats["split"] == "train"
    assert stats["folds"] == [1, 2, 3, 4, 5, 6, 7, 8] == sorted(cfg["splits"]["train_folds"])
    assert stats["n_records"] == len(train_ids)
    assert stats["ecg_ids_sha256"] == ecg_id_digest(train_ids)
    assert stats["preprocessing"] == preprocessing_tag(cfg)  # computed on the signals the model will see
    assert stats["cache_signals_sha256"] == open_cache(cfg).meta["signals_sha256"]  # for exactly this cache

    # Recomputing on the training split reproduces the saved values.
    leads, train = _moments(cfg, data, train_ids)
    assert leads == stats["leads"]
    assert stats["n_samples_per_lead"] == train.n
    assert_allclose(train.mean, stats["mean"], rtol=1e-12, atol=0)
    assert_allclose(train.std(stats["std_ddof"]), stats["std"], rtol=1e-12, atol=0)

    # The comparison above can tell the difference: stats over train + val + test are not the saved ones.
    _, held_out = _moments(cfg, data, np.concatenate([data.ids("val"), data.ids("test")]))
    everything = train.merge(held_out)
    assert not np.allclose(everything.std(stats["std_ddof"]), stats["std"], rtol=1e-12, atol=0)


def test_chan_merge_matches_numpy_on_the_concatenation():
    rng = np.random.default_rng(0)
    parts = [rng.normal(loc=0.3, scale=2.0, size=(n, 12)) for n in (1000, 1, 37, 1000)]
    total = None
    for p in parts:
        m = Moments(len(p), p.mean(axis=0), ((p - p.mean(axis=0)) ** 2).sum(axis=0))
        total = m if total is None else total.merge(m)
    everything = np.vstack(parts)
    assert total.n == len(everything)
    assert_allclose(total.mean, everything.mean(axis=0), rtol=1e-12)
    assert_allclose(total.std(0), everything.std(axis=0), rtol=1e-12)
    assert_allclose(total.std(1), everything.std(axis=0, ddof=1), rtol=1e-12)
