"""Evaluate one trained model on the validation split: thresholds, metrics at both operating points, figures.

    python -m src.evaluation.evaluate_val                  # the run named in evaluation.model_run
    python -m src.evaluation.evaluate_val --run <run_id>   # another completed training run

1. Reload checkpoints/<run_id>/best.pt and predict the validation split (fold 9) with deterministic GPU
   kernels. The predictions must reproduce the validation macro AUROC and AUPRC the run logged for that
   epoch (comparison.reproduce_tolerance), so the evaluated model is provably the trained one.
2. Choose per-class thresholds on those predictions (src/evaluation/thresholds.py): max F1 for every
   class, and the highest threshold with sensitivity >= evaluation.high_sensitivity_target for every
   abnormal class.
3. Score AUROC and AUPRC, and sensitivity, specificity, precision, NPV and F1 at both operating points.
4. Write reports/thresholds.json, reports/metrics_val.json and the figures in reports/figures/.

Only the validation loader is ever built, so fold 10 stays unopened. The thresholds are chosen on the same
records they are then scored on, so the thresholded metrics here are optimistic: they show what the
chosen thresholds do on the data they were fitted to. The unbiased estimate is the one-time evaluation on
fold 10 with these thresholds frozen.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.dataset import ECGDataset
from src.data.norm_stats import ecg_id_digest
from src.evaluation import plots
from src.evaluation.metrics import macro_mean, operating_point_metrics, ranking_metrics
from src.evaluation.thresholds import abnormal_classes, select_thresholds
from src.provenance import git_state
from src.training.compare_runs import load_run, validation_probabilities
from src.training.train import repo_path, resolve_device, seed_everything

SPLIT = "val"
POINTS = ("max_f1", "high_sensitivity")
COUNTS = ("tp", "fp", "fn", "tn")
FIGURES = {
    "roc": "roc_curves_val.png",
    "pr": "pr_curves_val.png",
    "confusion_max_f1": "confusion_max_f1_val.png",
    "confusion_high_sensitivity": "confusion_high_sensitivity_val.png",
    "training_curves": "training_curves.png",
    "class_distribution": "class_distribution.png",
}
PROBABILITIES = "sigmoid of the model's logits, uncalibrated (no temperature scaling)"
DECISION_RULE = "flag the class when its probability is >= the class's threshold"


def point_rules(target: float, normal_class: str) -> dict[str, str]:
    return {
        "max_f1": "per class, the threshold with the highest validation F1; among exact ties, the lowest "
                  "threshold (same F1, higher sensitivity)",
        "high_sensitivity": f"per abnormal class, the highest threshold with validation sensitivity >= {target:.2f}; "
                            f"{normal_class} has none, as it is not an abnormality to be caught",
    }


def point_labels(target: float) -> dict[str, str]:
    return {"max_f1": "Max-F1 threshold", "high_sensitivity": f"Sensitivity ≥ {target:.2f} threshold"}


def check_selection_matches_scoring(selection: dict[str, dict[str, Any]], scored: dict[str, Any], target: float) -> None:
    """The counts found while choosing each threshold must equal the counts the metrics recompute from it."""
    for point in POINTS:
        for c, chosen in selection[point].items():
            got = scored[point]["per_class"][c]
            if any(chosen[k] != got[k] for k in COUNTS):
                raise RuntimeError(f"{point} {c}: threshold selection counted {[chosen[k] for k in COUNTS]}, "
                                   f"scoring counted {[got[k] for k in COUNTS]} (tp, fp, fn, tn)")
    for c, m in scored["high_sensitivity"]["per_class"].items():
        if m["sensitivity"] < target:
            raise RuntimeError(f"high_sensitivity {c}: sensitivity {m['sensitivity']} is below the target {target}")


def evaluate(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Predict, choose thresholds, score, write both reports and every figure. Returns the metrics report."""
    e = cfg["evaluation"]
    reports_dir, figures_dir = project_path(cfg["paths"]["reports_dir"]), project_path(cfg["paths"]["figures_dir"])
    run = load_run(project_path(cfg["paths"]["runs_dir"]), run_id)
    classes = run["classes"]
    if classes != cfg["labels"]["superclasses"]:
        raise ValueError(f"Run {run_id} has classes {classes}, the config {cfg['labels']['superclasses']}")
    run_cfg = run["config"]
    if run_cfg["splits"] != cfg["splits"]:
        raise ValueError(f"Run {run_id} was trained with splits {run_cfg['splits']}, the config has {cfg['splits']}")
    seed_everything(run["seed"], run_cfg["training"]["deterministic"])
    device = resolve_device(cfg["training"]["device"])
    probs, y, ecg_ids, patient_ids = validation_probabilities(run, device, cfg["comparison"]["reproduce_tolerance"])
    if ecg_id_digest(ecg_ids) != run["data"]["val_ecg_ids_sha256"]:
        raise ValueError(f"The validation records differ from the ones run {run_id} was validated on")

    target, normal = e["high_sensitivity_target"], e["normal_class"]
    abnormal = abnormal_classes(classes, normal)
    selection = select_thresholds(y, probs, classes, normal, target)
    thresholds = {p: {c: s["threshold"] for c, s in selection[p].items()} for p in POINTS}
    ranking = ranking_metrics(y, probs, classes)
    scored = {p: operating_point_metrics(y, probs, classes, thresholds[p]) for p in POINTS}
    check_selection_matches_scoring(selection, scored, target)
    for p in POINTS:  # the like-for-like average: both operating points over the same four abnormal classes
        scored[p]["macro_abnormal"] = macro_mean(scored[p]["per_class"], abnormal)
    # Only abnormal classes have a high-sensitivity threshold, so its "macro" would equal macro_abnormal while
    # max_f1's "macro" covers all five; keep one name per meaning so the two are never compared by mistake.
    if scored["high_sensitivity"].pop("macro") != scored["high_sensitivity"]["macro_abnormal"]:
        raise RuntimeError("high_sensitivity macro differs from its macro_abnormal; the classes are not the abnormal ones")

    folds = run_cfg["splits"]["val_folds"]  # the folds the validation loader was built from
    common = {
        "generated_by": " ".join(["python -m src.evaluation.evaluate_val", *sys.argv[1:]]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git_state(),
        "split": SPLIT,
        "folds": folds,
        "records": int(len(y)),
        "patients": int(len(np.unique(patient_ids))),
        "ecg_ids_sha256": ecg_id_digest(ecg_ids),
        "model": {"run_id": run_id, "checkpoint": run["checkpoint"], "best_epoch": run["best"]["epoch"],
                  "seed": run["seed"], "training_commit": run["git"]["commit"], "overrides": run["overrides"]},
        "probabilities": PROBABILITIES,
        "decision_rule": DECISION_RULE,
        "classes": classes,
    }
    rules = point_rules(target, normal)
    thresholds_report = {
        "description": "Per-class decision thresholds chosen on the validation split. Apply them unchanged to any "
                       "other data, fold 10 included; never re-choose them there.",
        **common,
        "operating_points": {
            "max_f1": {"rule": rules["max_f1"], "classes": classes, "thresholds": thresholds["max_f1"],
                       "selection": selection["max_f1"]},
            "high_sensitivity": {"rule": rules["high_sensitivity"], "target_sensitivity": target,
                                 "classes": abnormal, "thresholds": thresholds["high_sensitivity"],
                                 "selection": selection["high_sensitivity"]},
        },
    }

    descriptions = cfg["labels"]["descriptions"]
    f = e["figures"]
    style = {"dpi": f["dpi"], "panel_in": f["panel_size_in"]}
    labels = point_labels(target)
    drawn = {p: {"label": labels[p], "per_class": scored[p]["per_class"]} for p in POINTS}
    source = (f"Validation split (fold {', '.join(map(str, folds))}): {len(y):,} records, {common['patients']:,} patients. "
              f"Model {run_id}, best epoch {run['best']['epoch']}. Thresholds chosen on these records.")
    paths = {
        "roc": plots.plot_roc_curves(
            y, probs, classes, descriptions, {c: ranking["per_class"][c]["auroc"] for c in classes}, drawn,
            figures_dir / FIGURES["roc"], title="ROC curves per class", subtitle=source, **style),
        "pr": plots.plot_pr_curves(
            y, probs, classes, descriptions, {c: ranking["per_class"][c]["auprc"] for c in classes}, drawn,
            figures_dir / FIGURES["pr"], title="Precision-recall curves per class", subtitle=source, **style),
        "confusion_max_f1": plots.plot_confusion_matrices(
            scored["max_f1"]["per_class"], classes, descriptions, figures_dir / FIGURES["confusion_max_f1"],
            title="Confusion matrices at the max-F1 threshold", subtitle=source, **style),
        "confusion_high_sensitivity": plots.plot_confusion_matrices(
            scored["high_sensitivity"]["per_class"], abnormal, descriptions,
            figures_dir / FIGURES["confusion_high_sensitivity"],
            title=f"Confusion matrices at the high-sensitivity threshold (sensitivity ≥ {target:.2f})",
            subtitle=source, **style),
        "training_curves": plots.plot_training_curves(
            project_path(run["csv"]), figures_dir / FIGURES["training_curves"], title=f"Training curves, run {run_id}",
            subtitle=f"Seed {run['seed']}; {run['epochs_run']} epochs, early stopping on validation macro AUROC "
                     f"(patience {run_cfg['training']['early_stopping_patience']}).", **style),
        "class_distribution": plots.plot_class_distribution(
            {f"Training, folds {min(run_cfg['splits']['train_folds'])}-{max(run_cfg['splits']['train_folds'])}":
                 ECGDataset(run_cfg, "train", augment=False).targets,
             f"Validation, fold {', '.join(map(str, folds))}": y},
            classes, descriptions, figures_dir / FIGURES["class_distribution"],
            title="Class support: records carrying each label",
            subtitle="Multi-label: a record can carry several labels, so a split's shares add up to more than 100%. "
                     "The test fold is not opened before the final evaluation.", **style),
    }

    metrics_report = {
        "description": "Validation split only. Thresholds were chosen on these same records, so the thresholded "
                       "metrics are optimistic; the unbiased estimate is the one-time fold-10 evaluation. AUROC and "
                       "AUPRC are threshold-free but still optimistic by a little, because the checkpoint epoch was "
                       "also chosen on this split.",
        **common,
        "thresholds_file": repo_path(reports_dir / e["thresholds_file"]),
        "ranking": ranking,
        "operating_points": {
            "max_f1": {"rule": rules["max_f1"], **scored["max_f1"]},
            "high_sensitivity": {"rule": rules["high_sensitivity"], "target_sensitivity": target,
                                 **scored["high_sensitivity"]},
        },
        "figures": {k: repo_path(v) for k, v in paths.items()},
    }
    write_json(reports_dir / e["thresholds_file"], thresholds_report)
    write_json(reports_dir / e["val_metrics_file"], metrics_report)
    return metrics_report


def write_json(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


TABLE_METRICS = ("sensitivity", "specificity", "precision", "f1")


def _fmt(value: float | None, width: int) -> str:
    return f"{'undef.' if value is None else f'{value:.3f}':>{width}}"


def _op_cells(m: dict[str, Any] | None, threshold: bool = True) -> str:
    """Threshold (blank for a macro row) and the four metrics of one operating point; dashes where it does not apply."""
    if m is None:
        return f"{'-':>8}" + f"{'-':>7}" * len(TABLE_METRICS)
    return (_fmt(m["threshold"], 8) if threshold else f"{'':>8}") + "".join(_fmt(m[k], 7) for k in TABLE_METRICS)


def _rank_cells(m: dict[str, float]) -> str:
    return "".join(f"{m[k]:>7.3f}" for k in ("prevalence", "auroc", "auprc"))


def format_table(report: dict[str, Any]) -> str:
    """Per class: prevalence, AUROC, AUPRC, then threshold, sensitivity, specificity, precision and F1 at each point."""
    classes, ranking, points = report["classes"], report["ranking"], report["operating_points"]
    hs, mf = points["high_sensitivity"], points["max_f1"]
    op_head = f"{'Thresh':>8}" + "".join(f"{h:>7}" for h in ("Sens", "Spec", "Prec", "F1"))
    lines = [f"{'':<36}{'Max-F1 threshold':^36}  {'Sensitivity >= ' + format(hs['target_sensitivity'], '.2f') + ' threshold':^36}",
             f"{'Class':<15}{'Prev':>7}{'AUROC':>7}{'AUPRC':>7}" + op_head + "  " + op_head]
    for c in classes:
        lines.append(f"{c:<15}{_rank_cells(ranking['per_class'][c])}{_op_cells(mf['per_class'][c])}  "
                     f"{_op_cells(hs['per_class'].get(c))}")
    abnormal = hs["classes"]
    abnormal_rank = {k: float(np.mean([ranking["per_class"][c][k] for c in abnormal])) for k in ("prevalence", "auroc", "auprc")}
    lines.append(f"{'Macro, all 5':<15}{_rank_cells(ranking['macro'])}{_op_cells(mf['macro'], False)}  {_op_cells(None)}")
    lines.append(f"{'Macro, abnormal':<15}{_rank_cells(abnormal_rank)}{_op_cells(mf['macro_abnormal'], False)}  "
                 f"{_op_cells(hs['macro_abnormal'], False)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Choose thresholds and score one model on the validation split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", help="training run id (default: evaluation.model_run in the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_id = args.run or cfg["evaluation"]["model_run"]
    report = evaluate(cfg, run_id)
    if report["git"]["dirty"]:
        print(f"WARNING: tracked files differ from commit {report['git']['commit'][:7]}; the reports record dirty=true. "
              "Commit, then rerun, so every number traces to a commit.")
    m = report["model"]
    print(f"Validation split (fold {', '.join(map(str, report['folds']))}): {report['records']:,} records, "
          f"{report['patients']:,} patients. Model {m['run_id']}, best epoch {m['best_epoch']}, seed {m['seed']}.")
    print("Predictions reproduce the run's logged validation macro AUROC and AUPRC.")
    print()
    print(format_table(report))
    print()
    print("Macro, abnormal: mean over " + ", ".join(report["operating_points"]["high_sensitivity"]["classes"])
          + ", so the two operating points are compared over the same classes. Accuracy is deliberately not shown.")
    print("Thresholds were chosen on these same records, so the thresholded metrics are optimistic; fold 10 has "
          "not been opened.")
    print()
    metrics_file = repo_path(project_path(cfg["paths"]["reports_dir"]) / cfg["evaluation"]["val_metrics_file"])
    for path in [report["thresholds_file"], metrics_file, *report["figures"].values()]:
        print(f"Saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
