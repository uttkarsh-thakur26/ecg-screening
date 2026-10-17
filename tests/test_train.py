"""Training-loop unit tests. Random tensors are fine here: these test the loop's parts, not the data or the model's skill."""

from __future__ import annotations

import copy
import csv
import math

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet1d import build_model
from src.training.train import (
    EarlyStopping,
    RunLog,
    apply_overrides,
    check_training_config,
    load_checkpoint,
    make_optimizer,
    make_scheduler,
    pos_weight_from,
    predict,
    probabilities,
    save_checkpoint,
    train_one_epoch,
    warmup_cosine,
)

CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def test_pos_weight_is_negatives_over_positives_per_class():
    y = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0], [0, 0, 0, 1, 0], [1, 0, 0, 0, 1]])
    np.testing.assert_allclose(pos_weight_from(y, CLASSES).numpy(), [2 / 2, 3 / 1, 2 / 2, 3 / 1, 3 / 1])


def test_pos_weight_matches_the_training_split(cfg):
    from src.data.dataset import ECGDataset  # real data: the weights the full run uses

    ds = ECGDataset(cfg, "train", augment=False)
    pw = pos_weight_from(ds.targets, CLASSES).numpy()
    pos = ds.targets.sum(axis=0)
    np.testing.assert_allclose(pw, (len(ds) - pos) / pos, rtol=1e-6)
    assert pw.argmax() == CLASSES.index("HYP") and pw.argmin() == CLASSES.index("NORM")


def test_pos_weight_refuses_a_class_without_positives():
    y = np.array([[1, 0, 1, 0, 0], [0, 1, 1, 0, 0]])
    with pytest.raises(ValueError, match="CD"):
        pos_weight_from(y, CLASSES)


def test_warmup_is_linear_then_cosine_decays_to_the_floor():
    warmup, total, floor = 10, 110, 0.1
    f = [warmup_cosine(s, warmup, total, floor) for s in range(total + 1)]
    np.testing.assert_allclose(f[:warmup], np.arange(1, warmup + 1) / warmup)
    assert f[warmup] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(f[warmup:], f[warmup + 1:]))  # never rises after warmup
    assert f[warmup + (total - warmup) // 2] == pytest.approx(floor + (1 - floor) / 2)  # half-way down the cosine
    assert f[total] == pytest.approx(floor)


def test_scheduler_steps_the_optimizer_learning_rate(cfg):
    t = cfg["training"]
    model = nn.Linear(2, 1)
    opt = make_optimizer(model, t)
    sched = make_scheduler(opt, t, steps_per_epoch=5, epochs=t["max_epochs"])
    lrs = []
    for _ in range(5 * t["max_epochs"]):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    peak = 5 * t["warmup_epochs"] - 1
    assert lrs[0] == pytest.approx(t["lr"] / (5 * t["warmup_epochs"]))
    assert max(lrs) == pytest.approx(t["lr"]) and int(np.argmax(lrs)) == peak
    assert lrs[-1] < 0.01 * t["lr"]


def test_early_stopping_counts_epochs_without_a_new_best():
    stop = EarlyStopping(patience=3, min_delta=0.0)
    history = [0.80, 0.85, 0.84, 0.85, 0.86, 0.86, 0.85, 0.84]
    improved = []
    for epoch, value in enumerate(history, start=1):
        improved.append(stop.step(value, epoch))
        if stop.should_stop:
            break
    assert improved == [True, True, False, False, True, False, False, False]  # equal is not better
    assert (stop.best, stop.best_epoch, epoch) == (0.86, 5, 8)


def test_early_stopping_refuses_nan():
    with pytest.raises(FloatingPointError):
        EarlyStopping(3, 0.0).step(math.nan, 1)


def test_overrides_keep_keys_and_types(cfg):
    before = copy.deepcopy(cfg)
    flipped = not cfg["augmentation"]["amplitude_scale"]["enabled"]
    out = apply_overrides(cfg, [f"augmentation.amplitude_scale.enabled={str(flipped).lower()}",
                                "training.batch_size=32", "training.lr=0.002", "training.lr=1"])
    assert out["augmentation"]["amplitude_scale"]["enabled"] is flipped
    assert out["training"]["batch_size"] == 32
    assert out["training"]["lr"] == 1.0 and type(out["training"]["lr"]) is float
    assert cfg == before  # the original is untouched
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["augmentation.amplitude_scal.enabled=false"])  # typo
    with pytest.raises(TypeError):
        apply_overrides(cfg, ["training.batch_size=big"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["training.batch_size"])


def test_overrides_cannot_reassign_folds(cfg):
    with pytest.raises(ValueError, match="splits"):
        apply_overrides(cfg, ["splits.val_folds=[10]"])


def test_config_checks_reject_unimplemented_settings(cfg):
    check_training_config(cfg["training"])
    for key, value in [("optimizer", "sgd"), ("lr_schedule", "step"), ("early_stopping_metric", "val_loss"),
                       ("warmup_epochs", cfg["training"]["max_epochs"]), ("early_stopping_patience", 0)]:
        t = copy.deepcopy(cfg["training"])
        t[key] = value
        with pytest.raises(ValueError):
            check_training_config(t)


def _toy_loader(seed: int) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(16, 12, 200, generator=g)
    y = (torch.rand(16, 5, generator=g) > 0.7).float()
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=True, generator=torch.Generator().manual_seed(seed))


def _train_steps(cfg, epochs: int) -> tuple[nn.Module, list[float]]:
    torch.manual_seed(cfg["training"]["seed"])
    model = build_model(cfg)
    loader = _toy_loader(0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((5,), 2.0))
    opt = make_optimizer(model, cfg["training"])
    sched = make_scheduler(opt, cfg["training"], len(loader), epochs)
    losses = [train_one_epoch(model, loader, criterion, opt, sched, torch.device("cpu")) for _ in range(epochs)]
    return model, losses


def test_training_lowers_the_loss_and_is_reproducible(cfg):
    model_a, losses_a = _train_steps(cfg, 30)
    model_b, losses_b = _train_steps(cfg, 30)
    assert losses_a[-1] < 0.5 * losses_a[0]
    assert losses_a == losses_b
    for (name, a), (_, b) in zip(model_a.state_dict().items(), model_b.state_dict().items()):
        assert torch.equal(a, b), name


def test_predict_returns_logits_in_loader_order_and_the_mean_loss(cfg):
    torch.manual_seed(0)
    model = build_model(cfg)
    x = torch.randn(10, 12, 200)
    y = (torch.rand(10, 5) > 0.5).float()
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.full((5,), 3.0))
    model.train()
    logits, targets, loss = predict(model, DataLoader(TensorDataset(x, y), batch_size=4), criterion, torch.device("cpu"))
    assert not model.training  # predict switches dropout and BatchNorm to inference behaviour
    with torch.no_grad():
        expected = model(x)
    np.testing.assert_allclose(logits, expected.numpy(), rtol=1e-5, atol=1e-5)
    np.testing.assert_array_equal(targets, y.numpy())
    assert loss == pytest.approx(criterion(expected, y).item(), rel=1e-5)


def test_probabilities_do_not_saturate_in_float32():
    p = probabilities(np.array([[20.0, 25.0, -20.0]], dtype=np.float32))
    assert p.dtype == np.float64
    assert p[0, 0] < p[0, 1] < 1.0 and p[0, 2] > 0.0  # float32 sigmoid would give 1.0, 1.0


def test_checkpoint_round_trip(cfg, tmp_path):
    torch.manual_seed(0)
    model = build_model(cfg).eval()
    path = tmp_path / "run" / "best.pt"
    save_checkpoint(path, model, config=cfg, classes=CLASSES, epoch=3)
    loaded, ckpt = load_checkpoint(path, torch.device("cpu"))
    assert ckpt["epoch"] == 3 and ckpt["classes"] == CLASSES and not loaded.training
    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        torch.testing.assert_close(loaded(x), model(x))
    assert not path.with_suffix(".tmp").exists()


def test_run_log_writes_one_header_and_refuses_to_overwrite(tmp_path):
    log = RunLog(tmp_path, "r1")
    log.write_epoch({"epoch": 1, "train_loss": 0.5})
    log.write_epoch({"epoch": 2, "train_loss": 0.4})
    with pytest.raises(ValueError):
        log.write_epoch({"epoch": 3, "val_loss": 0.4})
    log.write_summary({"status": "running"})
    rows = list(csv.reader(log.csv_path.open(encoding="utf-8")))
    assert rows == [["epoch", "train_loss"], ["1", "0.5"], ["2", "0.4"]]
    with pytest.raises(FileExistsError):
        RunLog(tmp_path, "r1")


def test_paired_bootstrap_of_identical_runs_is_exactly_zero():
    from src.training.compare_runs import paired_bootstrap

    rng = np.random.default_rng(0)
    y = (rng.random((60, 5)) > 0.6).astype(np.int64)
    y[0], y[1] = 1, 0  # every class has positives and negatives
    p = rng.random((60, 5))
    patients = np.repeat(np.arange(20), 3)
    out = paired_bootstrap(y, p, p.copy(), patients, CLASSES, n_resamples=50, ci_level=0.95, seed=0)
    for v in out.values():
        assert v["b_minus_a"] == v["ci_low"] == v["ci_high"] == 0.0 and v["share_b_better"] == 0.0


def test_paired_bootstrap_sees_a_clearly_better_run():
    from src.training.compare_runs import paired_bootstrap

    rng = np.random.default_rng(1)
    y = (rng.random((200, 5)) > 0.6).astype(np.int64)
    y[0], y[1] = 1, 0
    noise = rng.random((200, 5))
    good = 0.7 * y + 0.3 * noise  # ranks almost every positive above every negative
    patients = np.repeat(np.arange(100), 2)
    out = paired_bootstrap(y, noise, good, patients, CLASSES, n_resamples=200, ci_level=0.95, seed=0)
    for k, v in out.items():
        assert v["ci_low"] > 0 and v["share_b_better"] == 1.0, k


def test_config_differences_lists_dotted_keys():
    from src.training.compare_runs import config_differences

    a = {"x": {"y": 1, "z": [1, 2]}, "w": True}
    b = {"x": {"y": 1, "z": [1, 3]}, "w": False, "v": 0}
    assert config_differences(a, b) == {"v": {"a": None, "b": 0}, "w": {"a": True, "b": False},
                                        "x.z": {"a": [1, 2], "b": [1, 3]}}
    assert config_differences(a, a) == {}
