"""Train the 1D ResNet on the training split, early-stopped on the validation split (PROJECT_SPEC.md section 3).

    python -m src.training.train --overfit        # sanity check: memorise 50 training records, augmentation off
    python -m src.training.train --name main      # full run
    python -m src.training.train --name no-amp-scale --set augmentation.amplitude_scale.enabled=false

Loss. BCEWithLogitsLoss: one sigmoid per class, never a softmax, because the task is multi-label (a record
can be MI and STTC at once; a softmax would make the classes compete for one unit of probability).
pos_weight[c] = negatives / positives of class c in the training split, which multiplies the loss on
every positive label of c. Without it the rare classes are dominated by their negatives (HYP: 1 positive
to 7 negatives), and the cheapest way to lower the loss is to predict "absent" for them; with it, the
positives and the negatives of each class carry equal total weight.

Optimisation. AdamW with decoupled weight decay. The learning rate rises linearly over warmup_epochs, then
follows a cosine down to min_lr_ratio x lr at max_epochs, updated after every optimiser step. Warmup keeps
the first updates small while BatchNorm statistics and Adam's moment estimates are still settling.

Model selection. After each epoch the model scores the validation split. The epoch with the highest
validation macro AUROC is saved as checkpoints/<run_id>/best.pt, overwriting the previous best, and no
other checkpoint is kept. Training stops after early_stopping_patience epochs without a new best.

Records. Each run writes reports/runs/<run_id>.csv (one row per epoch: learning rate at the epoch's first
step, train and validation loss, validation macro and per-class AUROC and AUPRC, seconds; with the run id, git commit and seed on every
row) and reports/runs/<run_id>.json (the effective config with --set overrides applied, git commit, seed,
pos_weight, environment, and the best epoch's metrics). The JSON says "running" until the run finishes.

Reproducibility. training.seed seeds the weights, the shuffle order, the augmentation and dropout, and with
training.deterministic cuDNN and cuBLAS use deterministic kernels, so the same commit, config and seed give
the same weights.

Only make_train_val_loaders is used to load data, so this module never sees the test split.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import platform
import random
import re
import sys
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from scipy.special import expit
from torch import nn
from torch.utils.data import DataLoader, Subset

from src.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config, project_path
from src.data.cache import open_cache
from src.data.dataset import ECGDataset, make_train_val_loaders
from src.data.norm_stats import ecg_id_digest
from src.data.preprocess import preprocessing_tag
from src.evaluation.metrics import ranking_metrics
from src.models.resnet1d import ResNet1d, build_model, count_parameters
from src.provenance import git_state

RUN_NAME = re.compile(r"^[A-Za-z0-9_-]+$")
OVERFIT_PRINT_LINES = 10  # progress lines printed during the overfit check
BENCHMARK_REFERENCE = ("Strodthoff et al. (2021), PTB-XL superdiagnostic, best CNNs: about 0.93 macro AUROC "
                       "on fold 10 (test), not fold 9")


# --------------------------------------------------------------------------------------------------- config

def apply_overrides(cfg: dict[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Return a copy of cfg with each "dotted.key=value" applied. The key must exist and keep its type.

    Values are parsed as YAML, so "false" is a boolean and "32" an integer. The fold assignment (config
    `splits`) cannot be overridden: folds 1-8 train, 9 validation, 10 test is fixed for every run.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--set expects dotted.key=value, got {item!r}")
        if key.split(".")[0] == "splits":
            raise ValueError(f"--set {key}: the fold assignment is fixed and cannot be overridden")
        *parents, leaf = key.split(".")
        node: Any = cfg
        for part in parents:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"--set {key}: {part!r} is not in the config")
            node = node[part]
        if not isinstance(node, dict) or leaf not in node:
            raise KeyError(f"--set {key}: {leaf!r} is not in the config")
        value, old = yaml.safe_load(raw), node[leaf]
        same_type = type(value) is type(old) or (type(old) is float and type(value) is int)
        if not same_type:
            raise TypeError(f"--set {key}: {raw!r} is a {type(value).__name__}, the config value {old!r} is a "
                            f"{type(old).__name__}")
        node[leaf] = float(value) if type(old) is float else value
    return cfg


def check_training_config(t: dict[str, Any]) -> None:
    """Raise on settings this module does not implement or that cannot work."""
    supported = {"optimizer": "adamw", "lr_schedule": "warmup_cosine", "early_stopping_metric": "val_macro_auroc"}
    for key, value in supported.items():
        if t[key] != value:
            raise ValueError(f"training.{key} = {t[key]!r}: only {value!r} is implemented")
    if not 0 <= t["min_lr_ratio"] <= 1:
        raise ValueError(f"training.min_lr_ratio = {t['min_lr_ratio']} must be in [0, 1]")
    if not 0 <= t["warmup_epochs"] < t["max_epochs"]:
        raise ValueError(f"training.warmup_epochs = {t['warmup_epochs']} must be >= 0 and below max_epochs "
                         f"= {t['max_epochs']}")
    if t["early_stopping_patience"] < 1:
        raise ValueError(f"training.early_stopping_patience = {t['early_stopping_patience']} must be at least 1")


# ------------------------------------------------------------------------------------- reproducibility

def seed_everything(seed: int, deterministic: bool) -> None:
    """Seed Python, NumPy and torch (CPU and every GPU); optionally force deterministic GPU kernels."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if deterministic:
        # cuBLAS is only deterministic with a fixed workspace; the variable must be set before it initialises.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)  # an operation without a deterministic kernel raises


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training.device is 'cuda' but torch cannot see a CUDA GPU. On the ASUS TUF A15 set "
                           "Armoury Crate GPU Mode to Standard. To train on the CPU on purpose, pass "
                           "--set training.device=cpu.")
    return torch.device(name)


def environment(device: torch.device) -> dict[str, Any]:
    env = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
           "device": str(device)}
    if device.type == "cuda":
        env |= {"cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(device)}
    return env


# ------------------------------------------------------------------------------------- loss and schedule

def pos_weight_from(targets: np.ndarray, classes: Sequence[str]) -> torch.Tensor:
    """negatives / positives per class, from the (records, classes) multi-hot targets of the training split."""
    targets = np.asarray(targets)
    if targets.ndim != 2 or targets.shape[1] != len(classes):
        raise ValueError(f"targets must be (records, {len(classes)}), got {targets.shape}")
    pos = targets.sum(axis=0)
    neg = len(targets) - pos
    empty = [c for c, p, n in zip(classes, pos, neg) if p == 0 or n == 0]
    if empty:
        raise ValueError(f"pos_weight needs positives and negatives in every class; not the case for {empty}")
    return torch.tensor(neg / pos, dtype=torch.float32)


def warmup_cosine(step: int, warmup_steps: int, total_steps: int, min_lr_ratio: float) -> float:
    """Learning-rate multiplier for optimiser step `step` (0-based): linear warmup, then cosine decay."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))


def make_optimizer(model: nn.Module, t: dict[str, Any]) -> torch.optim.Optimizer:
    return torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])


def make_scheduler(optimizer: torch.optim.Optimizer, t: dict[str, Any], steps_per_epoch: int,
                   epochs: int) -> torch.optim.lr_scheduler.LambdaLR:
    """Per-step warmup + cosine over `epochs` epochs of `steps_per_epoch` steps."""
    warmup, total = t["warmup_epochs"] * steps_per_epoch, epochs * steps_per_epoch
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: warmup_cosine(step, warmup, total, t["min_lr_ratio"]))


class EarlyStopping:
    """Tracks the best value of a metric where higher is better; says when `patience` epochs passed without a new best."""

    def __init__(self, patience: int, min_delta: float) -> None:
        self.patience, self.min_delta = patience, min_delta
        self.best = -math.inf
        self.best_epoch: int | None = None
        self.epochs_without_improvement = 0

    def step(self, value: float, epoch: int) -> bool:
        """Record this epoch's value; True if it is a new best."""
        if not math.isfinite(value):
            raise FloatingPointError(f"Early-stopping metric is {value} at epoch {epoch}")
        if value > self.best + self.min_delta:
            self.best, self.best_epoch, self.epochs_without_improvement = value, epoch, 0
            return True
        self.epochs_without_improvement += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.epochs_without_improvement >= self.patience


# --------------------------------------------------------------------------------------- train and predict

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, device: torch.device) -> float:
    """One pass over `loader` in train mode; returns the mean loss per record (dropout and augmentation on)."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        loss = criterion(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Training loss became {loss.item()}; lower training.lr or check the inputs")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total += loss.item() * len(x)
        n += len(x)
    return total / n


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, criterion: nn.Module,
            device: torch.device) -> tuple[np.ndarray, np.ndarray, float]:
    """Eval-mode pass in loader order: (logits (records, classes) float32, targets, mean loss per record)."""
    model.eval()
    logits, targets = [], []
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        total += criterion(out, y).item() * len(x)
        n += len(x)
        logits.append(out.float().cpu().numpy())
        targets.append(y.cpu().numpy())
    return np.concatenate(logits), np.concatenate(targets), total / n


def probabilities(logits: np.ndarray) -> np.ndarray:
    """Sigmoid per class, in float64: float32 rounds logits above about 17 to exactly 1.0, which would create ties."""
    return expit(np.asarray(logits, dtype=np.float64))


# -------------------------------------------------------------------------------------------- checkpoints

def save_checkpoint(path: Path, model: nn.Module, **meta: Any) -> None:
    """Write the weights and metadata atomically: a crash mid-write cannot leave a truncated best.pt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({"model_state": state, **meta}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[ResNet1d, dict[str, Any]]:
    """Rebuild the model from the config stored in the checkpoint and load its weights. Returns (model in eval mode, checkpoint)."""
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model_state"])  # strict: every tensor must match
    return model.to(device).eval(), ckpt


# ---------------------------------------------------------------------------------------------- run records

class RunLog:
    """reports/runs/<run_id>.csv, one row per epoch, flushed as it goes, and reports/runs/<run_id>.json."""

    def __init__(self, runs_dir: Path, run_id: str) -> None:
        runs_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path, self.json_path = runs_dir / f"{run_id}.csv", runs_dir / f"{run_id}.json"
        if self.csv_path.exists() or self.json_path.exists():
            raise FileExistsError(f"Run {run_id} already has records in {runs_dir}")
        self._columns: list[str] | None = None

    def write_epoch(self, row: dict[str, Any]) -> None:
        if self._columns is None:
            self._columns = list(row)
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(self._columns)
        if list(row) != self._columns:
            raise ValueError(f"Epoch row columns {list(row)} differ from the CSV header {self._columns}")
        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([row[c] for c in self._columns])

    def write_summary(self, summary: dict[str, Any]) -> None:
        tmp = self.json_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.json_path)


def repo_path(path: Path) -> str:
    """The path relative to the repository root when it is inside it, else absolute; forward slashes either way."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def new_run_id(name: str) -> str:
    if not RUN_NAME.match(name):
        raise ValueError(f"Run name {name!r} may contain only letters, digits, '-' and '_'")
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{name}"


def epoch_row(run_id: str, git: dict[str, Any], seed: int, epoch: int, lr: float, train_loss: float,
              val_loss: float, metrics: dict[str, Any], classes: Sequence[str], seconds: float,
              is_best: bool) -> dict[str, Any]:
    row = {"run_id": run_id, "git_commit": git["commit"], "git_dirty": git["dirty"], "seed": seed, "epoch": epoch,
           "lr": lr, "train_loss": train_loss, "val_loss": val_loss,
           "val_macro_auroc": metrics["macro"]["auroc"], "val_macro_auprc": metrics["macro"]["auprc"]}
    row |= {f"val_auroc_{c}": metrics["per_class"][c]["auroc"] for c in classes}
    row |= {f"val_auprc_{c}": metrics["per_class"][c]["auprc"] for c in classes}
    return row | {"seconds": round(seconds, 1), "is_best": is_best}


def run_header(cfg: dict[str, Any], run_id: str, mode: str, overrides: Sequence[str], device: torch.device,
               model: nn.Module, pos_weight: torch.Tensor, classes: Sequence[str]) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "mode": mode,
        "status": "running",
        "generated_by": "python -m src.training.train " + " ".join(sys.argv[1:]),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "seed": cfg["training"]["seed"],
        "overrides": list(overrides),
        "config": cfg,
        "environment": environment(device),
        "parameters": count_parameters(model),
        "classes": list(classes),
        "pos_weight": {c: float(w) for c, w in zip(classes, pos_weight)},
    }


# ------------------------------------------------------------------------------------------------- runs

def _setup(cfg: dict[str, Any]) -> tuple[torch.device, dict[str, Any]]:
    t = cfg["training"]
    check_training_config(t)
    seed_everything(t["seed"], t["deterministic"])
    return resolve_device(t["device"]), t


def train(cfg: dict[str, Any], name: str, overrides: Sequence[str] = (),
          log: Callable[[str], None] = print) -> dict[str, Any]:
    """Full run: train split for training, validation split for early stopping. Returns the run summary."""
    device, t = _setup(cfg)
    classes = list(cfg["labels"]["superclasses"])
    train_loader, val_loader = make_train_val_loaders(cfg)
    train_ds, val_ds = train_loader.dataset, val_loader.dataset
    if list(train_ds.classes) != classes:
        raise ValueError(f"Cache classes {list(train_ds.classes)} differ from config {classes}")

    pos_weight = pos_weight_from(train_ds.targets, classes)
    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = make_optimizer(model, t)
    scheduler = make_scheduler(optimizer, t, len(train_loader), t["max_epochs"])
    stopper = EarlyStopping(t["early_stopping_patience"], t["early_stopping_min_delta"])

    run_id = new_run_id(name)
    runs = RunLog(project_path(cfg["paths"]["runs_dir"]), run_id)
    checkpoint = project_path(cfg["paths"]["checkpoints_dir"]) / run_id / "best.pt"
    summary = run_header(cfg, run_id, "train", overrides, device, model, pos_weight, classes)
    summary |= {
        "data": {"preprocessing": preprocessing_tag(cfg), "train_records": len(train_ds), "val_records": len(val_ds),
                 "val_ecg_ids_sha256": ecg_id_digest(val_ds.ecg_ids),
                 "signals_sha256": open_cache(cfg, build_if_missing=False).meta["signals_sha256"],
                 "augmentation": train_ds.augmenter.describe() if train_ds.augmenter else "none"},
        "steps_per_epoch": len(train_loader),
        "checkpoint": repo_path(checkpoint),
    }
    runs.write_summary(summary)
    git = summary["git"]

    log(f"Run {run_id}: {summary['parameters']:,} parameters on {summary['environment'].get('gpu', device)}; "
        f"{len(train_ds):,} training and {len(val_ds):,} validation records; batch {t['batch_size']}, "
        f"{len(train_loader)} steps per epoch")
    log(f"Augmentation: {summary['data']['augmentation']}")
    log("pos_weight (negatives / positives, training split): "
        + ", ".join(f"{c} {w:.2f}" for c, w in summary["pos_weight"].items()))
    if git["dirty"]:
        log(f"WARNING: tracked files differ from commit {git['commit'][:7]}; the run records dirty=true.")
    log(f"{'':>37}{'Validation':>20}  {'Validation AUPRC per class':>{7 * len(classes)}}")
    log(f"{'Epoch':>5}{'LR':>10}{'Train loss':>12}{'Val loss':>10}{'AUROC':>10}{'AUPRC':>10}  "
        + "".join(f"{c:>7}" for c in classes) + f"{'Seconds':>9}")

    best_metrics: dict[str, Any] | None = None
    start = time.perf_counter()
    try:
        for epoch in range(1, t["max_epochs"] + 1):
            epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            logits, targets, val_loss = predict(model, val_loader, criterion, device)
            metrics = ranking_metrics(targets, probabilities(logits), classes)
            is_best = stopper.step(metrics["macro"]["auroc"], epoch)
            if is_best:
                best_metrics = metrics | {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
                save_checkpoint(checkpoint, model, config=cfg, classes=classes, leads=list(val_ds.leads),
                                run_id=run_id, epoch=epoch, git=git, seed=t["seed"],
                                pos_weight=summary["pos_weight"], val_metrics=best_metrics,
                                signals_sha256=summary["data"]["signals_sha256"])
            seconds = time.perf_counter() - epoch_start
            runs.write_epoch(epoch_row(run_id, git, t["seed"], epoch, lr, train_loss, val_loss, metrics, classes,
                                       seconds, is_best))
            log(f"{epoch:>5}{lr:>10.2e}{train_loss:>12.4f}{val_loss:>10.4f}{metrics['macro']['auroc']:>10.4f}"
                f"{metrics['macro']['auprc']:>10.4f}  "
                + "".join(f"{metrics['per_class'][c]['auprc']:>7.3f}" for c in classes)
                + f"{seconds:>9.1f}{'  * best' if is_best else ''}")
            if stopper.should_stop:
                log(f"Early stopping: no improvement in validation macro AUROC for {t['early_stopping_patience']} epochs")
                break
    except torch.OutOfMemoryError as exc:
        summary |= {"status": "failed", "error": f"CUDA out of memory at batch size {t['batch_size']}"}
        runs.write_summary(summary)
        raise RuntimeError(f"The GPU ran out of memory at batch size {t['batch_size']}. Rerun with "
                           "--set training.batch_size=32 and report the change.") from exc
    except BaseException as exc:
        summary |= {"status": "failed", "error": repr(exc)}
        runs.write_summary(summary)
        raise

    summary |= {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "epochs_run": epoch,
        "stopped_early": stopper.should_stop,
        "train_seconds": round(time.perf_counter() - start, 1),
        "best": best_metrics,
        "csv": repo_path(runs.csv_path),
    }
    runs.write_summary(summary)
    return summary


def overfit_check(cfg: dict[str, Any], log: Callable[[str], None] = print) -> dict[str, Any]:
    """Train on a fixed handful of training records, without augmentation, until the loss should be near zero.

    The same model, loss, optimiser, schedule and train_one_epoch as a full run, so a pass means the loop can
    fit data at all. pos_weight is the full training split's, the loss the real run optimises.
    """
    device, t = _setup(cfg)
    o = t["overfit"]
    classes = list(cfg["labels"]["superclasses"])
    full = ECGDataset(cfg, "train", augment=False)
    if o["n_samples"] > len(full):
        raise ValueError(f"training.overfit.n_samples = {o['n_samples']} exceeds the {len(full)} training records")
    picked = torch.randperm(len(full), generator=torch.Generator().manual_seed(t["seed"]))[: o["n_samples"]].tolist()
    subset = Subset(full, picked)
    loader = DataLoader(subset, batch_size=t["batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(t["seed"]), num_workers=0)
    eval_loader = DataLoader(subset, batch_size=t["batch_size"], shuffle=False, num_workers=0)

    pos_weight = pos_weight_from(full.targets, classes)
    model = build_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = make_optimizer(model, t)
    scheduler = make_scheduler(optimizer, t, len(loader), o["epochs"])

    run_id = new_run_id("overfit")
    runs = RunLog(project_path(cfg["paths"]["runs_dir"]), run_id)
    summary = run_header(cfg, run_id, "overfit", [], device, model, pos_weight, classes)
    picked_targets = full.targets[picked]
    summary |= {"overfit": o | {"ecg_ids": [int(i) for i in full.ecg_ids[picked]],
                                "positives": {c: int(n) for c, n in zip(classes, picked_targets.sum(axis=0))}}}
    runs.write_summary(summary)
    log(f"Overfit check: {o['n_samples']} training records, augmentation off, {o['epochs']} epochs of "
        f"{len(loader)} step(s); positives " + ", ".join(f"{c} {n}" for c, n in summary["overfit"]["positives"].items()))

    every = max(1, o["epochs"] // OVERFIT_PRINT_LINES)
    start = time.perf_counter()
    losses = []
    for epoch in range(1, o["epochs"] + 1):
        lr = optimizer.param_groups[0]["lr"]
        loss = train_one_epoch(model, loader, criterion, optimizer, scheduler, device)
        losses.append(loss)
        runs.write_epoch({"run_id": run_id, "git_commit": summary["git"]["commit"], "git_dirty": summary["git"]["dirty"],
                          "seed": t["seed"], "epoch": epoch, "lr": lr, "train_loss": loss})
        if epoch == 1 or epoch % every == 0:
            log(f"  epoch {epoch:>4}  lr {lr:.2e}  training loss (train mode) {loss:.5f}")

    logits, targets, eval_loss = predict(model, eval_loader, criterion, device)
    correct = (logits > 0) == (targets > 0.5)
    passed = eval_loss <= o["max_loss"]
    summary |= {
        "status": "completed",
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": round(time.perf_counter() - start, 1),
        "first_train_loss": losses[0],
        "final_train_loss_train_mode": loss,
        "final_train_loss_eval_mode": eval_loss,
        "labels_correct_at_0_5": float(correct.mean()),
        "records_all_labels_correct_at_0_5": float(correct.all(axis=1).mean()),
        "passed": bool(passed),
        "csv": repo_path(runs.csv_path),
    }
    runs.write_summary(summary)
    return summary


# ------------------------------------------------------------------------------------------------ output

def baseline_table(summary: dict[str, Any], baselines_path: Path) -> str:
    """The run's best validation AUROC and AUPRC per class next to the prior and feature baselines."""
    if not baselines_path.is_file():
        return f"No baseline comparison: {baselines_path} does not exist (python -m src.models.baselines)."
    report = json.loads(baselines_path.read_text(encoding="utf-8"))
    best, classes = summary["best"], summary["classes"]
    problems = []
    if report["classes"] != classes:
        problems.append(f"classes {report['classes']} differ from {classes}")
    if report["splits"]["val"]["records"] != summary["data"]["val_records"]:
        problems.append(f"{report['splits']['val']['records']} validation records, this run has {summary['data']['val_records']}")
    if report["dataset"]["preprocessing"] != summary["data"]["preprocessing"]:
        problems.append(f"{report['dataset']['preprocessing']!r} signals, this run used {summary['data']['preprocessing']!r}")
    if problems:
        return "No baseline comparison, the baselines were scored on different data: " + "; ".join(problems)
    prior, gbdt = (report["baselines"][k]["metrics"] for k in ("prior", "feature_gbdt"))
    head = f"{'':<18}{'Prior':>16}    {'Feature GBDT':>16}    {'ResNet1d (best epoch)':>22}"
    sub = f"{'Class':<6}{'Prevalence':>12}" + f"{'AUROC':>8}{'AUPRC':>8}    " * 2 + f"{'AUROC':>11}{'AUPRC':>11}"
    lines = [head, sub]
    for c in classes:
        p, g, r = prior["per_class"][c], gbdt["per_class"][c], best["per_class"][c]
        lines.append(f"{c:<6}{r['prevalence']:>12.3f}{p['auroc']:>8.3f}{p['auprc']:>8.3f}    {g['auroc']:>8.3f}"
                     f"{g['auprc']:>8.3f}    {r['auroc']:>11.3f}{r['auprc']:>11.3f}")
    pm, gm, rm = prior["macro"], gbdt["macro"], best
    lines.append(f"{'Macro':<6}{rm['macro']['prevalence']:>12.3f}{pm['auroc']:>8.3f}{pm['auprc']:>8.3f}    "
                 f"{gm['auroc']:>8.3f}{gm['auprc']:>8.3f}    {rm['macro']['auroc']:>11.3f}{rm['macro']['auprc']:>11.3f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the 1D ResNet (folds 1-8) with early stopping on the validation split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--name", default="main", help="run name; the run id is <timestamp>_<name>")
    parser.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value for this run, e.g. augmentation.amplitude_scale.enabled=false")
    parser.add_argument("--overfit", action="store_true", help="run the 50-record overfit sanity check instead")
    args = parser.parse_args(argv)
    cfg = apply_overrides(load_config(args.config), args.overrides)

    if args.overfit:
        if args.overrides:
            raise SystemExit("--overfit runs the committed config; it does not take --set")
        s = overfit_check(cfg)
        print()
        print(f"First epoch training loss {s['first_train_loss']:.4f}; final training loss {s['final_train_loss_train_mode']:.5f} "
              f"in train mode (dropout on), {s['final_train_loss_eval_mode']:.5f} in eval mode")
        print(f"Labels correct at probability 0.5: {s['labels_correct_at_0_5']:.1%}; records with all "
              f"{len(s['classes'])} labels correct: {s['records_all_labels_correct_at_0_5']:.1%}")
        verdict = "PASSED" if s["passed"] else "FAILED"
        print(f"Overfit check {verdict}: eval-mode loss {s['final_train_loss_eval_mode']:.5f} "
              f"{'<=' if s['passed'] else '>'} {s['overfit']['max_loss']} (training.overfit.max_loss)")
        print(f"Saved {s['csv']} and its .json")
        return 0 if s["passed"] else 1

    s = train(cfg, args.name, args.overrides)
    best = s["best"]
    print()
    print(f"Best epoch {best['epoch']} of {s['epochs_run']}: validation macro AUROC {best['macro']['auroc']:.4f}, "
          f"macro AUPRC {best['macro']['auprc']:.4f} ({s['train_seconds'] / 60:.1f} min)")
    print()
    print(baseline_table(s, project_path(cfg["paths"]["reports_dir"]) / cfg["baselines"]["report_file"]))
    print()
    print(f"Reference: {BENCHMARK_REFERENCE}.")
    print(f"Saved {s['csv']}, its .json, and {s['checkpoint']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
