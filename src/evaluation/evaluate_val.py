"""Evaluate one trained model on the validation split: thresholds, metrics at both operating points, figures.

    python -m src.evaluation.evaluate_val                  # the run named in evaluation.model_run
    python -m src.evaluation.evaluate_val --run <run_id>   # another completed training run

src/training/calibrate.py runs this same evaluation right after it refits the calibration, so running this
module on its own is only needed to regenerate the reports from an unchanged reports/calibration.json.

1. Reload checkpoints/<run_id>/best.pt and predict the validation split (fold 9) with deterministic GPU
   kernels. The predictions must reproduce the validation macro AUROC and AUPRC the run logged for that
   epoch (comparison.reproduce_tolerance), so the evaluated model is provably the trained one.
2. Calibrate the logits per class with reports/calibration.json, which must have been fitted for this run's
   checkpoint on these validation records (src/evaluation/calibration.py). Calibration is strictly
   increasing, so AUROC and AUPRC must equal the uncalibrated ones (calibration.ranking_tolerance).
3. Choose per-class thresholds on the calibrated probabilities (src/evaluation/thresholds.py): max F1 for
   every class, and the highest threshold with sensitivity >= evaluation.high_sensitivity_target for every
   abnormal class. They must flag exactly the validation records that the thresholds chosen on the
   uncalibrated probabilities flag: calibration moves the threshold values, not the decisions on these records.
4. Score AUROC and AUPRC, and sensitivity, specificity, precision, NPV and F1 at both operating points.
5. Write reports/thresholds.json, reports/metrics_val.json and the figures in reports/figures/.

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
from typing import Any, NamedTuple

import numpy as np

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.dataset import ECGDataset
from src.data.norm_stats import ecg_id_digest
from src.evaluation import plots
from src.evaluation.calibration import calibrated_probabilities, load_calibration
from src.evaluation.metrics import macro_mean, operating_point_metrics, ranking_metrics
from src.evaluation.thresholds import abnormal_classes, select_thresholds
from src.provenance import git_state
from src.training.compare_runs import load_run, validation_logits
from src.training.train import probabilities, repo_path, resolve_device, seed_everything

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
PROBABILITIES = ("calibrated: per class, sigmoid(a * logit + b) with the calibrator chosen in the calibration file "
                 "(temperature scaling: a = 1/T, b = 0; Platt scaling: fitted a and b)")
DECISION_RULE = "flag the class when its calibrated probability is >= the class's threshold"


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


class ValidationPredictions(NamedTuple):
    run: dict[str, Any]          # the run record, reports/runs/<run_id>.json
    logits: np.ndarray           # (records, classes) float32, the model's outputs before the sigmoid
    targets: np.ndarray          # (records, classes) multi-hot
    ecg_ids: np.ndarray
    patient_ids: np.ndarray


def predict_validation(cfg: dict[str, Any], run_id: str) -> ValidationPredictions:
    """The run's best checkpoint on the validation split, checked against the run's logged metrics and records."""
    run = load_run(project_path(cfg["paths"]["runs_dir"]), run_id)
    classes = run["classes"]
    if classes != cfg["labels"]["superclasses"]:
        raise ValueError(f"Run {run_id} has classes {classes}, the config {cfg['labels']['superclasses']}")
    run_cfg = run["config"]
    if run_cfg["splits"] != cfg["splits"]:
        raise ValueError(f"Run {run_id} was trained with splits {run_cfg['splits']}, the config has {cfg['splits']}")
    seed_everything(run["seed"], run_cfg["training"]["deterministic"])
    device = resolve_device(cfg["training"]["device"])
    logits, y, ecg_ids, patient_ids = validation_logits(run, device, cfg["comparison"]["reproduce_tolerance"])
    if ecg_id_digest(ecg_ids) != run["data"]["val_ecg_ids_sha256"]:
        raise ValueError(f"The validation records differ from the ones run {run_id} was validated on")
    return ValidationPredictions(run, logits, y, ecg_ids, patient_ids)


def calibration_path(cfg: dict[str, Any]) -> Path:
    return project_path(cfg["paths"]["reports_dir"]) / cfg["calibration"]["file"]


def check_ranking_unchanged(before: dict[str, Any], after: dict[str, Any], classes: list[str], tolerance: float) -> None:
    """Per-class AUROC and AUPRC of the calibrated probabilities must equal the uncalibrated ones to within rounding."""
    for c in classes:
        for m in ("auroc", "auprc"):
            gap = abs(after["per_class"][c][m] - before["per_class"][c][m])
            if gap > tolerance:
                raise RuntimeError(f"Calibration changed {c} {m} by {gap:.2e}; a strictly increasing calibrator cannot")


def check_same_records(calibrated: dict[str, dict[str, Any]], uncalibrated: dict[str, dict[str, Any]]) -> None:
    """Thresholds chosen on calibrated and on uncalibrated probabilities must have identical confusion counts."""
    for point in POINTS:
        for c, chosen in calibrated[point].items():
            before = uncalibrated[point][c]
            if any(chosen[k] != before[k] for k in COUNTS):
                raise RuntimeError(f"{point} {c}: on calibrated probabilities the threshold counts "
                                   f"{[chosen[k] for k in COUNTS]}, on uncalibrated ones {[before[k] for k in COUNTS]} "
                                   "(tp, fp, fn, tn); a strictly increasing calibrator must flag the same records")


def evaluate(cfg: dict[str, Any], run_id: str) -> dict[str, Any]:
    """Predict, calibrate with the saved calibration file, choose thresholds, score, write both reports and every figure."""
    git = git_state()  # before anything is written, so the reports' own files do not count as uncommitted changes
    preds = predict_validation(cfg, run_id)
    calibration = load_calibration(calibration_path(cfg), preds.run, preds.run["classes"], ecg_id_digest(preds.ecg_ids))
    return evaluate_predictions(cfg, preds, calibration, git, "python -m src.evaluation.evaluate_val")


def evaluate_predictions(cfg: dict[str, Any], preds: ValidationPredictions, calibration: dict[str, Any],
                         git: dict[str, Any], command: str) -> dict[str, Any]:
    """Thresholds, metrics and figures from validation logits and a loaded calibration. Returns the metrics report.

    git is the repository state taken before the calling CLI wrote anything; command is that CLI's module command.
    """
    e = cfg["evaluation"]
    reports_dir, figures_dir = project_path(cfg["paths"]["reports_dir"]), project_path(cfg["paths"]["figures_dir"])
    run, y, ecg_ids, patient_ids = preds.run, preds.targets, preds.ecg_ids, preds.patient_ids
    run_id, classes, run_cfg = run["run_id"], run["classes"], run["config"]
    uncalibrated = probabilities(preds.logits)
    probs = calibrated_probabilities(preds.logits, calibration["calibrators"], classes)

    target, normal = e["high_sensitivity_target"], e["normal_class"]
    abnormal = abnormal_classes(classes, normal)
    selection = select_thresholds(y, probs, classes, normal, target)
    selection_uncalibrated = select_thresholds(y, uncalibrated, classes, normal, target)
    check_same_records(selection, selection_uncalibrated)
    thresholds = {p: {c: s["threshold"] for c, s in selection[p].items()} for p in POINTS}
    ranking = ranking_metrics(y, probs, classes)
    check_ranking_unchanged(ranking_metrics(y, uncalibrated, classes), ranking, classes,
                            cfg["calibration"]["ranking_tolerance"])
    scored = {p: operating_point_metrics(y, probs, classes, thresholds[p]) for p in POINTS}
    check_selection_matches_scoring(selection, scored, target)
    for p in POINTS:  # the like-for-like average: both operating points over the same four abnormal classes
        scored[p]["macro_abnormal"] = macro_mean(scored[p]["per_class"], abnormal)
    # Only abnormal classes have a high-sensitivity threshold, so its "macro" would equal macro_abnormal while
    # max_f1's "macro" covers all five; keep one name per meaning so the two are never compared by mistake.
    if scored["high_sensitivity"].pop("macro") != scored["high_sensitivity"]["macro_abnormal"]:
        raise RuntimeError("high_sensitivity macro differs from its macro_abnormal; the classes are not the abnormal ones")

    folds = run_cfg["splits"]["val_folds"]  # the folds the validation loader was built from
    calibration_file = repo_path(calibration["path"])
    common = {
        "generated_by": " ".join([command, *sys.argv[1:]]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "split": SPLIT,
        "folds": folds,
        "records": int(len(y)),
        "patients": int(len(np.unique(patient_ids))),
        "ecg_ids_sha256": ecg_id_digest(ecg_ids),
        "model": {"run_id": run_id, "checkpoint": run["checkpoint"], "best_epoch": run["best"]["epoch"],
                  "seed": run["seed"], "training_commit": run["git"]["commit"], "overrides": run["overrides"]},
        "probabilities": PROBABILITIES,
        "post_calibration": True,
        "calibration": {"file": calibration_file, "sha256": calibration["sha256"],
                        "per_class": {c: calibration["calibrators"][c].to_json() for c in classes}},
        "decision_rule": DECISION_RULE,
        "classes": classes,
    }
    rules = point_rules(target, normal)
    thresholds_report = {
        "description": "Per-class decision thresholds chosen on the validation split, on CALIBRATED probabilities "
                       f"(post-calibration). Apply the calibration in {calibration_file} first, then these thresholds, "
                       "both unchanged, to any other data, fold 10 included; never re-choose them there.",
        **common,
        "operating_points": {
            "max_f1": {"rule": rules["max_f1"], "classes": classes, "thresholds": thresholds["max_f1"],
                       "selection": selection["max_f1"]},
            "high_sensitivity": {"rule": rules["high_sensitivity"], "target_sensitivity": target,
                                 "classes": abnormal, "thresholds": thresholds["high_sensitivity"],
                                 "selection": selection["high_sensitivity"]},
        },
        "before_calibration": {
            "description": "For reference only, never to be applied: the same rules on the uncalibrated sigmoid "
                           "outputs. They flag exactly the validation records the thresholds above flag (checked), "
                           "because calibration is strictly increasing; only the threshold values differ.",
            "thresholds": {p: {c: s["threshold"] for c, s in selection_uncalibrated[p].items()} for p in POINTS},
        },
    }

    descriptions = cfg["labels"]["descriptions"]
    f = e["figures"]
    style = {"dpi": f["dpi"], "panel_in": f["panel_size_in"]}
    labels = point_labels(target)
    drawn = {p: {"label": labels[p], "per_class": scored[p]["per_class"]} for p in POINTS}
    source = (f"Validation split (fold {', '.join(map(str, folds))}): {len(y):,} records, {common['patients']:,} patients. "
              f"Model {run_id}, best epoch {run['best']['epoch']}. Calibrated probabilities; calibration and "
              "thresholds chosen on these records.")
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
        "description": "Validation split only. The calibration and the thresholds were chosen on these same records, so "
                       "the thresholded metrics are optimistic; the unbiased estimate is the one-time fold-10 "
                       "evaluation. AUROC and AUPRC are threshold-free and unchanged by calibration, but still "
                       "optimistic by a little, because the checkpoint epoch was also chosen on this split.",
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


def format_threshold_shift(thresholds_report: dict[str, Any]) -> str:
    """Per class and operating point: threshold on uncalibrated probabilities -> threshold on calibrated ones."""
    points, before = thresholds_report["operating_points"], thresholds_report["before_calibration"]["thresholds"]
    methods = {c: m["method"] for c, m in thresholds_report["calibration"]["per_class"].items()}
    target = format(points["high_sensitivity"]["target_sensitivity"], ".2f")
    lines = [f"{'Class':<7}{'Calibrator':<13}{'Max-F1 threshold':>22}{'Sensitivity >= ' + target + ' threshold':>34}",
             f"{'':<20}{'uncal. -> calibrated':>22}{'uncal. -> calibrated':>34}"]
    for c in thresholds_report["classes"]:
        cells = []
        for p, width in (("max_f1", 22), ("high_sensitivity", 34)):
            t = points[p]["thresholds"].get(c)
            cells.append(f"{'-':>{width}}" if t is None else f"{before[p][c]:.3f} -> {t:.3f}".rjust(width))
        lines.append(f"{c:<7}{methods[c]:<13}" + "".join(cells))
    return "\n".join(lines)


def print_header(report: dict[str, Any]) -> None:
    if report["git"]["dirty"]:
        print(f"WARNING: tracked files differ from commit {report['git']['commit'][:7]}; the reports record dirty=true. "
              "Commit, then rerun, so every number traces to a commit.")
    m = report["model"]
    print(f"Validation split (fold {', '.join(map(str, report['folds']))}): {report['records']:,} records, "
          f"{report['patients']:,} patients. Model {m['run_id']}, best epoch {m['best_epoch']}, seed {m['seed']}.")
    print("Predictions reproduce the run's logged validation macro AUROC and AUPRC.")
    print()


def print_report(cfg: dict[str, Any], report: dict[str, Any]) -> None:
    """The evaluation summary both CLIs print: threshold shift, the metrics table and the files written."""
    reports_dir = project_path(cfg["paths"]["reports_dir"])
    thresholds_report = json.loads((reports_dir / cfg["evaluation"]["thresholds_file"]).read_text(encoding="utf-8"))
    print(f"Thresholds chosen on the calibrated probabilities ({report['calibration']['file']}, "
          f"sha256 {report['calibration']['sha256'][:12]}), next to the ones the uncalibrated probabilities gave:")
    print(format_threshold_shift(thresholds_report))
    print("Each pair flags exactly the same validation records (checked): calibration moves the threshold values, "
          "not the decisions on these records.")
    print()
    print(format_table(report))
    print()
    print("Macro, abnormal: mean over " + ", ".join(report["operating_points"]["high_sensitivity"]["classes"])
          + ", so the two operating points are compared over the same classes. Accuracy is deliberately not shown.")
    print("Calibration and thresholds were chosen on these same records, so the thresholded metrics are optimistic; "
          "fold 10 has not been opened.")
    print()
    metrics_file = repo_path(reports_dir / cfg["evaluation"]["val_metrics_file"])
    for path in [report["thresholds_file"], metrics_file, *report["figures"].values()]:
        print(f"Saved {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Choose thresholds and score one model on the validation split.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", help="training run id (default: evaluation.model_run in the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_id = args.run or cfg["evaluation"]["model_run"]
    report = evaluate(cfg, run_id)
    print_header(report)
    print_report(cfg, report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
