"""Compare two training runs on the validation split, with a paired patient-level bootstrap.

    python -m src.training.compare_runs <run_id A> <run_id B>

For each run this loads checkpoints/<run_id>/best.pt, rebuilds the model from the config stored in it,
recomputes its validation predictions and checks they reproduce the validation macro AUROC and AUPRC
logged for that epoch (config comparison.reproduce_tolerance). It then reports per-class and macro AUROC
and AUPRC for both runs, and for each metric the difference B - A with a bootstrap confidence interval.

The bootstrap is paired and resamples patients. Each resample draws validation patients with
replacement, keeps all records of every drawn patient (records of one patient are not independent),
and scores both runs on exactly those records, so the interval reflects how much the difference
depends on which patients happen to be in the validation split. It does not cover training noise:
a second seed could move both runs, which only repeating the runs with other seeds would measure.

The report goes to reports/runs/compare_<A>_vs_<B>.json. Only validation data is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.cache import open_cache
from src.data.dataset import make_loader
from src.evaluation.metrics import ranking_metrics
from src.provenance import git_state
from src.training.train import load_checkpoint, predict, probabilities, repo_path, resolve_device

EVAL_SPLIT = "val"
METRICS = ("auroc", "auprc")


def load_run(runs_dir: Path, run_id: str) -> dict[str, Any]:
    path = runs_dir / f"{run_id}.json"
    if not path.is_file():
        raise FileNotFoundError(f"No run record {path}")
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("mode") != "train" or run.get("status") != "completed":
        raise ValueError(f"Run {run_id} is a {run.get('mode')!r} run with status {run.get('status')!r}; "
                         "only completed training runs can be compared")
    return run


def validation_probabilities(run: dict[str, Any], device: torch.device,
                             tolerance: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(probabilities, targets, ecg_ids, patient_ids) of the run's best checkpoint on the validation split.

    The validation loader is built from the config stored in the checkpoint, so the preprocessing and the
    normalisation statistics are the ones the run used. The signal cache must be the one the run read, and
    the recomputed metrics must match the ones logged for the checkpoint's epoch.
    """
    model, ckpt = load_checkpoint(project_path(run["checkpoint"]), device)
    if ckpt["run_id"] != run["run_id"]:
        raise ValueError(f"{run['checkpoint']} belongs to run {ckpt['run_id']}, not {run['run_id']}")
    run_cfg = ckpt["config"]
    current_signals = open_cache(run_cfg, build_if_missing=False).meta["signals_sha256"]
    if not ckpt["signals_sha256"] == run["data"]["signals_sha256"] == current_signals:
        raise ValueError(f"Run {run['run_id']}: the checkpoint, the run record and the cache on disk do not all "
                         "name the same signal cache")
    loader = make_loader(run_cfg, EVAL_SPLIT)
    ds = loader.dataset
    if list(ds.classes) != ckpt["classes"]:
        raise ValueError(f"Validation classes {list(ds.classes)} differ from the checkpoint's {ckpt['classes']}")
    pos_weight = torch.tensor([ckpt["pos_weight"][c] for c in ckpt["classes"]], device=device)
    logits, targets, _ = predict(model, loader, nn.BCEWithLogitsLoss(pos_weight=pos_weight), device)
    probs = probabilities(logits)
    recomputed = ranking_metrics(targets, probs, ckpt["classes"])
    logged = run["best"]
    for m in METRICS:
        gap = abs(recomputed["macro"][m] - logged["macro"][m])
        if gap > tolerance:
            raise ValueError(f"Run {run['run_id']}: best.pt gives validation macro {m} {recomputed['macro'][m]:.6f}, "
                             f"the run logged {logged['macro'][m]:.6f} at epoch {logged['epoch']} (difference {gap:.2e})")
    return probs, targets, ds.ecg_ids, ds.patient_ids


def config_differences(a: Any, b: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    """{"dotted.key": {"a": value in a, "b": value in b}} for every setting that differs (None where absent)."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: dict[str, dict[str, Any]] = {}
        for k in sorted(set(a) | set(b)):
            key = f"{prefix}{k}"
            if k not in a or k not in b:
                out[key] = {"a": a.get(k), "b": b.get(k)}
            else:
                out |= config_differences(a[k], b[k], f"{key}.")
        return out
    return {} if a == b else {prefix.rstrip("."): {"a": a, "b": b}}


def flat_metrics(y: np.ndarray, p: np.ndarray, classes: list[str]) -> dict[str, float]:
    """{"macro_auroc": ..., "HYP_auprc": ..., ...} for one set of predictions."""
    r = ranking_metrics(y, p, classes)
    out = {f"macro_{m}": r["macro"][m] for m in METRICS}
    out |= {f"{c}_{m}": r["per_class"][c][m] for c in classes for m in METRICS}
    return out


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, patient_ids: np.ndarray, classes: list[str],
                     n_resamples: int, ci_level: float, seed: int) -> dict[str, dict[str, float]]:
    """Per metric: A, B, B - A on the full split, and the bootstrap interval and share of resamples with B > A."""
    point_a, point_b = flat_metrics(y, p_a, classes), flat_metrics(y, p_b, classes)
    patients, record_patient = np.unique(patient_ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    diffs = {k: np.empty(n_resamples) for k in point_a}
    records = np.arange(len(y))
    for i in range(n_resamples):
        drawn = np.bincount(rng.integers(0, len(patients), len(patients)), minlength=len(patients))
        idx = np.repeat(records, drawn[record_patient])  # each record as often as its patient was drawn
        a, b = flat_metrics(y[idx], p_a[idx], classes), flat_metrics(y[idx], p_b[idx], classes)
        for k in diffs:
            diffs[k][i] = b[k] - a[k]
    alpha = 1 - ci_level
    return {k: {"a": point_a[k], "b": point_b[k], "b_minus_a": point_b[k] - point_a[k],
                "ci_low": float(np.quantile(d, alpha / 2)), "ci_high": float(np.quantile(d, 1 - alpha / 2)),
                "share_b_better": float((d > 0).mean())}
            for k, d in diffs.items()}


def compare(cfg: dict[str, Any], run_a: str, run_b: str) -> dict[str, Any]:
    c = cfg["comparison"]
    runs_dir = project_path(cfg["paths"]["runs_dir"])
    runs = {"a": load_run(runs_dir, run_a), "b": load_run(runs_dir, run_b)}
    if runs["a"]["data"]["val_ecg_ids_sha256"] != runs["b"]["data"]["val_ecg_ids_sha256"]:
        raise ValueError("The two runs were validated on different records")
    if runs["a"]["classes"] != runs["b"]["classes"]:
        raise ValueError(f"The runs have different classes: {runs['a']['classes']} and {runs['b']['classes']}")
    classes = runs["a"]["classes"]
    device = resolve_device(cfg["training"]["device"])

    preds = {k: validation_probabilities(r, device, c["reproduce_tolerance"]) for k, r in runs.items()}
    (p_a, y_a, ids_a, pid_a), (p_b, y_b, ids_b, _) = preds["a"], preds["b"]
    if not (np.array_equal(ids_a, ids_b) and np.array_equal(y_a, y_b)):
        raise ValueError("Validation records or labels differ between the two runs' predictions")
    if not np.array_equal(ids_a, np.sort(ids_a)):
        raise ValueError("Validation predictions are not in ecg_id order")

    results = paired_bootstrap(y_a, p_a, p_b, pid_a, classes, c["bootstrap_samples"], c["ci_level"],
                               cfg["training"]["seed"])
    return {
        "description": f"Run B minus run A on the {EVAL_SPLIT} split, best checkpoint of each. Intervals: paired "
                       f"bootstrap over validation patients ({c['bootstrap_samples']} resamples, {c['ci_level']:.0%}). "
                       "They cover the choice of validation patients, not training noise from the seed.",
        "generated_by": f"python -m src.training.compare_runs {run_a} {run_b}",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "runs": {k: {"run_id": r["run_id"], "overrides": r["overrides"], "best_epoch": r["best"]["epoch"],
                     "epochs_run": r["epochs_run"], "git_commit": r["git"]["commit"], "seed": r["seed"]}
                 for k, r in runs.items()},
        # Every setting in which the two runs' effective configs differ: for a clean ablation, only the one studied.
        "config_differences": config_differences(runs["a"]["config"], runs["b"]["config"]),
        "same_commit": runs["a"]["git"]["commit"] == runs["b"]["git"]["commit"]
                       and not runs["a"]["git"]["dirty"] and not runs["b"]["git"]["dirty"],
        "records": len(y_a),
        "patients": int(len(np.unique(pid_a))),
        "bootstrap": {"resamples": c["bootstrap_samples"], "ci_level": c["ci_level"], "unit": "patient",
                      "seed": cfg["training"]["seed"]},
        "metrics": results,
    }


def format_table(report: dict[str, Any]) -> str:
    level = f"{report['bootstrap']['ci_level']:.0%}"
    lines = [f"A = {report['runs']['a']['run_id']}  (best epoch {report['runs']['a']['best_epoch']})",
             f"B = {report['runs']['b']['run_id']}  (best epoch {report['runs']['b']['best_epoch']})",
             f"{report['records']:,} validation records, {report['patients']:,} patients",
             "Config differences (A -> B): " + ("; ".join(f"{k} {v['a']} -> {v['b']}" for k, v
                                                     in report["config_differences"].items()) or "none"),
             f"Same clean commit: {'yes' if report['same_commit'] else 'NO'}",
             "",
             f"{'Metric':<14}{'A':>8}{'B':>8}{'B - A':>9}{level + ' interval':>22}{'B better in':>13}"]
    for k, v in report["metrics"].items():
        interval = f"[{v['ci_low']:+.4f}, {v['ci_high']:+.4f}]"
        lines.append(f"{k:<14}{v['a']:>8.4f}{v['b']:>8.4f}{v['b_minus_a']:>+9.4f}{interval:>22}{v['share_b_better']:>13.1%}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two training runs' best checkpoints on the validation split.")
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    report = compare(cfg, args.run_a, args.run_b)
    out = project_path(cfg["paths"]["runs_dir"]) / f"compare_{args.run_a}_vs_{args.run_b}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(format_table(report))
    print()
    print(f"Saved {repo_path(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
