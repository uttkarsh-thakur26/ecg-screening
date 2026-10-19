"""Seed-to-seed spread of the committed config: validation AUROC and AUPRC of its seed 42, 43 and 44 runs.

    python -m src.evaluation.seed_stability

Nothing is trained. Each run's best checkpoint predicts the validation split (fold 9) again, and its macro
AUROC and AUPRC must reproduce the values the run logged (compare_runs.validation_logits), so every number
here comes from the saved weights. Reported per metric: each seed's value, the mean and the sample standard
deviation over the seeds (seed_stability.std_ddof).

Before scoring it checks, and raises otherwise, that the runs are one config repeated with different seeds:
- every run completed, and their effective configs differ in training.seed only, with distinct seeds;
- every run was trained from a clean tree, from commits whose src/ is identical;
- every run was validated on the same records, from the same signal cache;
- the release model (evaluation.model_run) is one of them.

The spread is training noise: what retraining the same config with another seed changes. It is not a
confidence interval over patients, which the bootstrap of the one-time test evaluation gives. With three
seeds the standard deviation is itself a rough estimate.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.evaluation.evaluate_val import write_json
from src.evaluation.metrics import ranking_metrics
from src.provenance import git_state, git_tree
from src.training.compare_runs import config_differences, load_run, validation_logits
from src.training.train import probabilities, repo_path, resolve_device, seed_everything

COMMAND = "python -m src.evaluation.seed_stability"
SEED_KEY = "training.seed"
CODE_DIR = "src"
METRICS = ("auroc", "auprc")


def check_runs(runs: list[dict[str, Any]], release_run: str) -> None:
    """Raise unless the runs repeat one config with distinct seeds, from the same code and data."""
    if len(runs) < 2:
        raise ValueError(f"Need at least 2 runs to measure seed-to-seed variation, got {len(runs)}")
    problems = []
    first = runs[0]
    for r in runs[1:]:
        extra = sorted(set(config_differences(first["config"], r["config"])) - {SEED_KEY})
        if extra:
            problems.append(f"{r['run_id']} differs from {first['run_id']} in {extra}, not only {SEED_KEY}")
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        problems.append(f"seeds repeat: {seeds}")
    if any(r["config"]["training"]["seed"] != r["seed"] for r in runs):
        problems.append("a run record's seed differs from its config's training.seed")
    dirty = [r["run_id"] for r in runs if r["git"]["dirty"]]
    if dirty:
        problems.append(f"trained from a dirty tree: {dirty}")
    trees = {r["run_id"]: git_tree(r["git"]["commit"], CODE_DIR) for r in runs if not r["git"]["dirty"]}
    if len(set(trees.values())) > 1:
        problems.append(f"{CODE_DIR}/ differs between the training commits: {trees}")
    for key in ("val_ecg_ids_sha256", "signals_sha256"):
        if len({r["data"][key] for r in runs}) != 1:
            problems.append(f"the runs differ in data.{key}")
    if len({tuple(r["classes"]) for r in runs}) != 1:
        problems.append("the runs have different classes")
    if release_run not in [r["run_id"] for r in runs]:
        problems.append(f"the release run {release_run} is not among them")
    if problems:
        raise ValueError("These runs are not one config repeated over seeds:\n  " + "\n  ".join(problems))


def spread(values: list[float], ddof: int) -> dict[str, float]:
    """Mean and standard deviation (with ddof) of one metric over the seeds."""
    v = np.asarray(values, dtype=np.float64)
    if len(v) <= ddof:
        raise ValueError(f"A standard deviation with ddof={ddof} needs more than {ddof} values, got {len(v)}")
    return {"mean": float(v.mean()), "std": float(v.std(ddof=ddof))}


def seed_stability(cfg: dict[str, Any]) -> dict[str, Any]:
    """Score every listed run's best checkpoint on the validation split. Returns the report (not written)."""
    s = cfg["seed_stability"]
    runs_dir = project_path(cfg["paths"]["runs_dir"])
    runs = [load_run(runs_dir, run_id) for run_id in s["runs"]]
    check_runs(runs, cfg["evaluation"]["model_run"])
    classes = runs[0]["classes"]
    device = resolve_device(cfg["training"]["device"])
    per_run = []
    reference = None
    for run in runs:
        seed_everything(run["seed"], run["config"]["training"]["deterministic"])
        logits, y, ecg_ids, patient_ids = validation_logits(run, device, cfg["comparison"]["reproduce_tolerance"])
        if reference is None:
            reference = (ecg_ids, y, patient_ids)
        elif not (np.array_equal(ecg_ids, reference[0]) and np.array_equal(y, reference[1])):
            raise ValueError(f"{run['run_id']} was scored on other validation records or labels than the first run")
        r = ranking_metrics(y, probabilities(logits), classes)
        per_run.append({"run_id": run["run_id"], "seed": run["seed"], "best_epoch": run["best"]["epoch"],
                        "epochs_run": run["epochs_run"], "training_commit": run["git"]["commit"],
                        "macro": {m: r["macro"][m] for m in METRICS},
                        "per_class": {c: {m: r["per_class"][c][m] for m in METRICS} for c in classes}})
    ddof = s["std_ddof"]
    summary = {f"macro_{m}": spread([p["macro"][m] for p in per_run], ddof) for m in METRICS}
    summary |= {f"{c}_{m}": spread([p["per_class"][c][m] for p in per_run], ddof) for c in classes for m in METRICS}
    return {
        "description": "Seed-to-seed spread of the committed config on the validation split (fold 9): each run's best "
                       "checkpoint, uncalibrated sigmoid outputs (calibration is strictly increasing, so AUROC and "
                       "AUPRC are the same either way). Training noise only, not an interval over patients. Each "
                       "run's epoch was chosen on this split, so the values are slightly optimistic.",
        "generated_by": " ".join([COMMAND, *sys.argv[1:]]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "split": "val",
        "records": int(len(reference[0])),
        "patients": int(len(np.unique(reference[2]))),
        "release_run": cfg["evaluation"]["model_run"],
        "std": f"sample standard deviation (ddof={ddof}) over {len(per_run)} seeds",
        "code_tree": git_tree(runs[0]["git"]["commit"], CODE_DIR),
        "classes": classes,
        "runs": per_run,
        "summary": summary,
    }


def report_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / cfg["seed_stability"]["report_file"]


def format_table(report: dict[str, Any]) -> str:
    seeds = [r["seed"] for r in report["runs"]]
    lines = [f"{'Metric':<14}" + "".join(f"{'seed ' + str(s):>12}" for s in seeds) + f"{'mean':>10}{'std':>10}"]
    for key, s in report["summary"].items():
        metric = key.rsplit("_", 1)
        values = [r["macro"][metric[1]] if metric[0] == "macro" else r["per_class"][metric[0]][metric[1]]
                  for r in report["runs"]]
        lines.append(f"{key:<14}" + "".join(f"{v:>12.4f}" for v in values) + f"{s['mean']:>10.4f}{s['std']:>10.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validation AUROC and AUPRC of the committed config over its seeds.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    report = seed_stability(cfg)
    path = report_path(cfg)
    write_json(path, report)
    if report["git"]["dirty"]:
        print(f"WARNING: tracked files differ from commit {report['git']['commit'][:7]}; the report records dirty=true.")
    print(f"Validation split: {report['records']:,} records, {report['patients']:,} patients. Each checkpoint "
          "reproduces its run's logged validation macro AUROC and AUPRC.")
    print(f"Release model: {report['release_run']}. Std: {report['std']}.")
    print()
    print(format_table(report))
    print()
    print(f"Saved {repo_path(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
