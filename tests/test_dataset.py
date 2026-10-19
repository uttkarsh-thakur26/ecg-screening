"""Dataset and DataLoaders on the real cache: shapes, train-only augmentation, normalisation, and the test-split lock."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import PROJECT_ROOT
from src.data.cache import open_cache
from src.data.dataset import ECGDataset, TestSetLockedError, check_norm_stats, make_loader
from src.data.norm_stats import load_norm_stats, norm_stats_path

B = 64  # training.batch_size in config/default.yaml

# Modules outside src/data/ that may unlock fold 10. The one-time evaluation (session 9) was added in the commit
# that also committed reports/TEST_PLAN.md, before fold 10 was opened. Nothing else belongs on this list.
MAY_UNLOCK_TEST: set[str] = {
    "src/evaluation/evaluate_test.py",
    # Added deliberately in session 10, after the one-time evaluation: the demo API serves the held-out test records
    # (ids, signals, one-record predictions with the frozen model, labels only after a prediction) and never computes
    # a metric. The test metrics it serves are reports/metrics.json, verbatim. It is the only API module that opens
    # fold 10; main.py, upload.py and schemas.py stay under the scan.
    "src/api/service.py",
}
SPLIT_TOKENS = ("allow_test", "test_folds", "strat_fold")


def _expected(cfg: dict, ds: ECGDataset, i: int) -> np.ndarray:
    """Item i normalised independently, in float64, straight from the cache and the stats file."""
    stats = load_norm_stats(norm_stats_path(cfg))
    raw = open_cache(cfg).signals[ds.rows[i]].astype(np.float64)
    return (raw - np.asarray(stats["mean"])[:, None]) / np.asarray(stats["std"])[:, None]


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_batch_shapes(cfg, split):
    assert cfg["training"]["batch_size"] == B
    loader = make_loader(cfg, split, allow_test=split == "test")
    x, y = next(iter(loader))
    assert x.shape == (B, 12, 1000) and x.dtype == torch.float32
    assert y.shape == (B, 5) and y.dtype == torch.float32
    assert set(y.unique().tolist()) <= {0.0, 1.0}
    assert torch.isfinite(x).all()


def test_splits_hold_the_labelled_records_of_their_folds(cfg, data):
    sets = {s: ECGDataset(cfg, s, allow_test=s == "test") for s in ("train", "val", "test")}
    for split, ds in sets.items():
        np.testing.assert_array_equal(ds.ecg_ids, data.ids(split))  # same records as the session-2 pipeline
        by_id = dict(zip(data.labels.ecg_ids.tolist(), data.labels.y))
        np.testing.assert_array_equal(ds.targets, np.stack([by_id[e] for e in ds.ecg_ids]).astype(np.float32))
    assert [len(sets[s]) for s in ("train", "val", "test")] == [17084, 2146, 2158]
    patients = {s: set(ds.patient_ids.tolist()) for s, ds in sets.items()}
    assert patients["train"].isdisjoint(patients["val"])
    assert patients["train"].isdisjoint(patients["test"])
    assert patients["val"].isdisjoint(patients["test"])


def test_test_split_is_locked_without_the_explicit_flag(cfg):
    with pytest.raises(TestSetLockedError, match="locked"):
        ECGDataset(cfg, "test")
    with pytest.raises(TestSetLockedError, match="locked"):
        make_loader(cfg, "test")
    with pytest.raises(TestSetLockedError, match="locked"):
        open_cache(cfg).rows("test")  # the raw-array route a baseline would take is locked too
    for truthy in ("False", "0", 1):  # e.g. a flag read from the command line as a string
        with pytest.raises(TypeError, match="True or False"):
            ECGDataset(cfg, "test", allow_test=truthy)


def test_no_code_outside_the_data_layer_touches_the_test_split():
    src = PROJECT_ROOT / "src"
    for folder in ("training", "models", "evaluation", "explain", "api"):
        assert (src / folder).is_dir()  # so the scan cannot pass by finding nothing
    files = [p for p in src.rglob("*.py") if src / "data" not in p.parents]
    assert len(files) >= 10
    offenders = {
        rel: tokens
        for p in files
        if (rel := p.relative_to(PROJECT_ROOT).as_posix()) not in MAY_UNLOCK_TEST
        and (tokens := [t for t in SPLIT_TOKENS if t in p.read_text(encoding="utf-8")])
    }
    assert offenders == {}


@pytest.mark.parametrize("split", ["val", "test"])
def test_augmentation_is_off_for_val_and_test(cfg, split):
    ds = ECGDataset(cfg, split, allow_test=split == "test")
    assert not ds.augment and ds.augmenter is None
    torch.manual_seed(0)
    for i in (0, len(ds) // 2, len(ds) - 1):
        first, second = ds[i][0], ds[i][0]
        assert torch.equal(first, second)  # deterministic: nothing random is applied
        np.testing.assert_allclose(first.numpy(), _expected(cfg, ds, i), rtol=0, atol=1e-5)  # normalisation only
    with pytest.raises(ValueError, match="training split only"):
        ECGDataset(cfg, split, augment=True, allow_test=split == "test")


def test_augmentation_is_on_for_train_only_when_asked(cfg):
    ds = ECGDataset(cfg, "train")
    assert ds.augment
    torch.manual_seed(0)
    assert not torch.equal(ds[0][0], ds[0][0])
    plain = ECGDataset(cfg, "train", augment=False)
    assert not plain.augment
    np.testing.assert_allclose(plain[0][0].numpy(), _expected(cfg, plain, 0), rtol=0, atol=1e-5)


def test_normalised_train_data_has_mean_0_and_std_1_per_lead(cfg):
    ds = ECGDataset(cfg, "train", augment=False)
    total = np.zeros(12)
    total_sq = np.zeros(12)
    n = 0
    for x, _ in DataLoader(ds, batch_size=512, num_workers=0):
        x = x.double()
        total += x.sum(dim=(0, 2)).numpy()
        total_sq += (x**2).sum(dim=(0, 2)).numpy()
        n += x.shape[0] * x.shape[2]
    assert n == 17084 * 1000
    mean = total / n
    std = np.sqrt(total_sq / n - mean**2)
    np.testing.assert_allclose(mean, 0.0, atol=1e-4)
    np.testing.assert_allclose(std, 1.0, atol=1e-4)


def test_stats_from_other_records_or_signals_are_refused(cfg):
    cache = open_cache(cfg)
    stats = load_norm_stats(norm_stats_path(cfg))
    train_ids = cache.index["ecg_id"].to_numpy()[cache.rows("train")]
    check_norm_stats(stats, cache, train_ids)  # the real file passes
    with pytest.raises(ValueError, match="different signal cache"):
        check_norm_stats({**stats, "cache_signals_sha256": "0" * 64}, cache, train_ids)
    with pytest.raises(ValueError, match="training records"):
        check_norm_stats(stats, cache, train_ids[:-1])
    with pytest.raises(ValueError, match="signals"):
        check_norm_stats({**stats, "preprocessing": "bandpass_0.5-40hz_order2"}, cache, train_ids)
    with pytest.raises(ValueError, match="split"):
        check_norm_stats({**stats, "split": "all"}, cache, train_ids)


def test_worker_copies_rebuild_the_same_dataset(cfg):
    ds = ECGDataset(cfg, "val")
    blob = pickle.dumps(ds)
    assert len(blob) < 64 * 1024  # small, so Windows workers start in parallel
    clone = pickle.loads(blob)
    np.testing.assert_array_equal(clone.ecg_ids, ds.ecg_ids)
    np.testing.assert_array_equal(clone.targets, ds.targets)
    assert torch.equal(clone[5][0], ds[5][0])


def test_worker_copies_use_a_snapshot_of_the_config(cfg):
    mine = copy.deepcopy(cfg)
    ds = ECGDataset(mine, "val")
    mine["preprocessing"]["bandpass"]["enabled"] = True  # e.g. an ablation script moving on to the next run
    clone = pickle.loads(pickle.dumps(ds))
    assert clone.signals_path == ds.signals_path
    assert torch.equal(clone[0][0], ds[0][0])


@pytest.mark.parametrize("workers", [0, 2])
def test_training_loader_is_reproducible(cfg, workers):
    c = copy.deepcopy(cfg)
    c["dataloader"]["num_workers"]["train"] = workers
    (xa, ya), (xb, yb) = (next(iter(make_loader(c, "train"))) for _ in range(2))
    assert torch.equal(ya, yb)  # same shuffle order
    assert torch.equal(xa, xb)  # same augmentation
