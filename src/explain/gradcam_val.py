"""Grad-CAM on the validation split: the randomisation sanity check and one example overlay per class.

    python -m src.explain.gradcam_val                  # the run named in evaluation.model_run
    python -m src.explain.gradcam_val --run <run_id>   # another completed training run

1. Predict the validation split with the run's best checkpoint; the predictions must reproduce the run's logged
   validation metrics (src/evaluation/evaluate_val.py, predict_validation). Calibrate them with
   reports/calibration.json and read the max-F1 thresholds from reports/thresholds.json, both of which must
   belong to this checkpoint and these records.
2. Randomisation check (Adebayo et al., 2018, "Sanity checks for saliency maps"). One record per class, drawn
   at random (explain.randomisation.draw_seed) from that class's true positives at the max-F1 point. For each,
   the Grad-CAM map of that class from the trained model and from randomly initialised models of the same
   architecture (explain.randomisation.init_seeds). If a map barely changes when the weights are replaced by
   random ones, it shows the input, not what the model learned. Similarity: Spearman rank correlation (the
   paper's measure) and Pearson correlation between the two normalised 1000-sample maps. The limit
   (explain.randomisation.max_mean_abs_spearman) was fixed in the config before any map was computed. A random
   map that is all zero has no correlation; explain.randomisation.undefined_correlations says how it enters the
   mean (that rule was added after the first run produced such maps), and both conventions are reported.
3. Calibration check on the same records: the map of the calibrated logit a * y + b equals the raw logit's map.
4. Example overlays: per class in explain.examples.classes, the single-label validation record with every
   decision correct at the max-F1 thresholds and the highest calibrated probability for the class: a best case,
   which the figure says.
5. Write reports/gradcam_val.json and the figures in reports/figures/.

Only the validation split is loaded; fold 10 stays unopened. The process exits with status 1 if the
randomisation criterion fails, after writing the report and figures so the failure can be inspected.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.stats import spearmanr

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.dataset import ECGDataset
from src.data.norm_stats import ecg_id_digest
from src.evaluation.calibration import Calibrator, calibrated_probabilities, load_calibration
from src.evaluation.evaluate_val import calibration_path, predict_validation, write_json
from src.explain import figures
from src.explain.gradcam1d import GradCAM1d, GradCAMResult
from src.models.resnet1d import ResNet1d, build_model
from src.provenance import git_state
from src.training.train import load_checkpoint, repo_path, resolve_device

COMMAND = "python -m src.explain.gradcam_val"
TARGET_LAYER = "model.stages[-1] (output of the last residual stage)"
NORMALISATION = "ReLU, linear interpolation between feature-position centres, then division by the maximum"
WHY_CALIBRATION_CANCELS = (
    "A calibrator maps the logit y to a * y + b with a > 0. The gradient becomes a * dy/dA, so every channel weight "
    "and the weighted sum are multiplied by a; ReLU(a s) = a ReLU(s) because a > 0; interpolation is linear; and the "
    "division by the maximum removes the factor a. b is a constant with no gradient. The normalised map is identical.")


# ----------------------------------------------------------------------------------------------- inputs

def load_thresholds(cfg: dict[str, Any], run: dict[str, Any], calibration: dict[str, Any], ecg_ids_sha256: str,
                    point: str) -> dict[str, float]:
    """The operating point's thresholds, checked to belong to this run, these records and this calibration file."""
    path = project_path(cfg["paths"]["reports_dir"]) / cfg["evaluation"]["thresholds_file"]
    if not path.is_file():
        raise FileNotFoundError(f"No thresholds file {path}. Run python -m src.training.calibrate first")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["model"]["run_id"] != run["run_id"] or report["model"]["best_epoch"] != run["best"]["epoch"]:
        raise ValueError(f"{path} was chosen for run {report['model']['run_id']}, not {run['run_id']}")
    if report["ecg_ids_sha256"] != ecg_ids_sha256:
        raise ValueError(f"{path} was not chosen on these validation records")
    if report["calibration"]["sha256"] != calibration["sha256"]:
        raise ValueError(f"{path} was chosen with a different calibration file than {calibration['path']}")
    return dict(report["operating_points"][point]["thresholds"])


def calibrated_head(model: ResNet1d, class_index: int, cal: Calibrator) -> ResNet1d:
    """A copy whose class_index output is the calibrated logit a * y + b (the row of the linear layer rescaled)."""
    out = copy.deepcopy(model)
    with torch.no_grad():
        out.fc.weight[class_index] *= cal.a
        out.fc.bias[class_index] = cal.a * out.fc.bias[class_index] + cal.b
    return out


# ----------------------------------------------------------------------------------------------- selection

def draw_sanity_records(y: np.ndarray, probs: np.ndarray, classes: list[str], thresholds: dict[str, float],
                        seed: int) -> list[tuple[int, int]]:
    """[(row, class index)]: one distinct record per class, drawn uniformly from its true positives."""
    rng = np.random.default_rng(seed)
    drawn: list[tuple[int, int]] = []
    for j, c in enumerate(classes):
        tp = np.flatnonzero((y[:, j] == 1) & (probs[:, j] >= thresholds[c]))
        tp = np.setdiff1d(tp, [r for r, _ in drawn])
        if len(tp) == 0:
            raise RuntimeError(f"No true positive of {c} left to draw for the randomisation check")
        drawn.append((int(rng.choice(tp)), j))
    return drawn


def pick_example(y: np.ndarray, probs: np.ndarray, classes: list[str], thresholds: dict[str, float],
                 cls: str) -> tuple[int, int]:
    """(row, eligible records): among single-label cls records with every decision correct, the highest P(cls)."""
    j = classes.index(cls)
    flagged = probs >= np.array([thresholds[c] for c in classes])
    only_cls = (y[:, j] == 1) & (y.sum(axis=1) == 1)
    candidates = np.flatnonzero(only_cls & (flagged == (y == 1)).all(axis=1))
    if len(candidates) == 0:
        raise RuntimeError(f"No single-label {cls} record has all five decisions correct at these thresholds")
    return int(candidates[np.argmax(probs[candidates, j])]), int(len(candidates))


# ----------------------------------------------------------------------------------------------- scoring

def correlations(a: np.ndarray, b: np.ndarray) -> dict[str, float | None]:
    """Pearson and Spearman correlation of two maps; None when either map is constant (no correlation exists)."""
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {"pearson": None, "spearman": None}
    return {"pearson": float(np.corrcoef(a, b)[0, 1]), "spearman": float(spearmanr(a, b).statistic)}


UNDEFINED_RULES = ("exclude", "zero")


def summarise_randomisation(per_record: list[dict[str, Any]], seeds: list[str], limit: float,
                            rule: str) -> tuple[dict[str, dict[str, Any]], bool]:
    """Per random seed, mean |Spearman| and |Pearson| under both rules for constant maps; the verdict uses `rule`.

    A seed passes when its mean |Spearman| under `rule` is below `limit`. A seed whose maps are all constant has no
    mean under "exclude" and fails: there is nothing to compare.
    """
    if rule not in UNDEFINED_RULES:
        raise ValueError(f"explain.randomisation.undefined_correlations must be one of {UNDEFINED_RULES}, got {rule!r}")
    summary = {}
    for seed in seeds:
        out: dict[str, Any] = {"records": len(per_record)}
        for key in ("spearman", "pearson"):
            values = [p["random"][seed][key] for p in per_record]
            defined = [abs(v) for v in values if v is not None]
            out[f"mean_abs_{key}"] = {
                "exclude": float(np.mean(defined)) if defined else None,
                "zero": float(np.sum(defined) / len(values)),
            }
            out["defined"] = len(defined)
        verdict = out["mean_abs_spearman"][rule]
        out["passed"] = verdict is not None and verdict < limit
        summary[seed] = out
    return summary, all(s["passed"] for s in summary.values())


def single_map(cam: GradCAM1d, x: torch.Tensor, class_index: int) -> GradCAMResult:
    return cam(x.unsqueeze(0), class_index)


def check_logit(got: float, expected: float, tolerance: float, what: str) -> None:
    if abs(got - expected) > tolerance:
        raise RuntimeError(f"{what}: Grad-CAM's forward pass gives logit {got:.6f}, the evaluated prediction "
                           f"{expected:.6f} (difference {abs(got - expected):.2e} > explain.logit_tolerance {tolerance:.0e})")


# ----------------------------------------------------------------------------------------------- main work

def run(cfg: dict[str, Any], run_id: str) -> tuple[dict[str, Any], bool]:
    git = git_state()  # before anything is written, so this run's own outputs do not count as uncommitted changes
    ex, e = cfg["explain"], cfg["evaluation"]
    preds = predict_validation(cfg, run_id)
    run_rec, classes = preds.run, preds.run["classes"]
    digest = ecg_id_digest(preds.ecg_ids)
    calibration = load_calibration(calibration_path(cfg), run_rec, classes, digest)
    probs = calibrated_probabilities(preds.logits, calibration["calibrators"], classes)
    thresholds = load_thresholds(cfg, run_rec, calibration, digest, ex["examples"]["operating_point"])
    y = np.asarray(preds.targets)

    device = resolve_device(cfg["training"]["device"])
    model, ckpt = load_checkpoint(project_path(run_rec["checkpoint"]), device)
    run_cfg = ckpt["config"]
    ds = ECGDataset(run_cfg, "val")
    if not np.array_equal(ds.ecg_ids, preds.ecg_ids):
        raise RuntimeError("The validation dataset's records differ from the evaluated predictions'")
    signals = np.load(ds.signals_path, mmap_mode="r")  # cached records in mV, for drawing only
    lead = ex["lead"]
    lead_index = ds.leads.index(lead)
    fs = run_cfg["dataset"]["sampling_rate_hz"]
    tol = ex["logit_tolerance"]
    trained_cam = GradCAM1d(model)

    def labels_of(row: int) -> list[str]:
        return [c for c, v in zip(classes, y[row]) if v == 1]

    def record_info(row: int, j: int, result: GradCAMResult) -> dict[str, Any]:
        return {"ecg_id": int(preds.ecg_ids[row]), "patient_id": int(preds.patient_ids[row]), "labels": labels_of(row),
                "target_class": classes[j], "logit": float(result.logits[0]),
                "calibrated_probability": float(probs[row, j]), "max_f1_threshold": thresholds[classes[j]],
                "raw_peak": float(result.raw_peak[0])}

    # 2. Randomisation check
    r = ex["randomisation"]
    drawn = draw_sanity_records(y, probs, classes, thresholds, r["draw_seed"])
    trained = []
    for row, j in drawn:
        res = single_map(trained_cam, ds[row][0].to(device), j)
        check_logit(float(res.logits[0]), float(preds.logits[row, j]), tol, f"ecg_id {preds.ecg_ids[row]}")
        trained.append(res)
    random_maps: dict[int, list[GradCAMResult]] = {}
    for seed in r["init_seeds"]:
        torch.manual_seed(seed)
        random_model = build_model(run_cfg).to(device).eval()
        cam = GradCAM1d(random_model)
        random_maps[seed] = [single_map(cam, ds[row][0].to(device), j) for row, j in drawn]

    per_record = []
    for i, (row, j) in enumerate(drawn):
        entry = record_info(row, j, trained[i])
        entry["random"] = {str(seed): {**correlations(trained[i].maps[0], random_maps[seed][i].maps[0]),
                                       "raw_peak": float(random_maps[seed][i].raw_peak[0]),
                                       "logit": float(random_maps[seed][i].logits[0])}
                           for seed in r["init_seeds"]}
        per_record.append(entry)
    limit, rule = r["max_mean_abs_spearman"], r["undefined_correlations"]
    summary, passed = summarise_randomisation(per_record, [str(s) for s in r["init_seeds"]], limit, rule)
    other = next(u for u in UNDEFINED_RULES if u != rule)
    passed_other = all(s["mean_abs_spearman"][other] is not None and s["mean_abs_spearman"][other] < limit
                       for s in summary.values())

    # 3. Calibration leaves the normalised map unchanged (checked on the same records)
    calibration_check = []
    for i, (row, j) in enumerate(drawn):
        cal = calibration["calibrators"][classes[j]]
        res = single_map(GradCAM1d(calibrated_head(model, j, cal)), ds[row][0].to(device), j)
        expected_logit = float(cal.logits(np.array([trained[i].logits[0]]))[0])
        check_logit(float(res.logits[0]), expected_logit, tol, f"calibrated head, ecg_id {preds.ecg_ids[row]}")
        calibration_check.append({"ecg_id": int(preds.ecg_ids[row]), "class": classes[j], **cal.to_json(),
                                  "max_abs_map_difference": float(np.abs(res.maps[0] - trained[i].maps[0]).max()),
                                  "raw_peak_ratio": float(res.raw_peak[0] / trained[i].raw_peak[0])
                                  if trained[i].raw_peak[0] > 0 else None})

    # 4. Example overlays
    examples = {}
    for cls in ex["examples"]["classes"]:
        (row, eligible), j = pick_example(y, probs, classes, thresholds, cls), classes.index(cls)
        res = single_map(trained_cam, ds[row][0].to(device), j)
        check_logit(float(res.logits[0]), float(preds.logits[row, j]), tol, f"ecg_id {preds.ecg_ids[row]}")
        examples[cls] = {"row": row, "result": res, **record_info(row, j, res), "eligible_records": eligible,
                         "all_probabilities": {c: float(probs[row, k]) for k, c in enumerate(classes)}}

    # 5. Figures and report
    geom = trained[0]
    f, style = ex["figures"], e["figures"]
    figures_dir = project_path(cfg["paths"]["figures_dir"])
    folds = run_cfg["splits"]["val_folds"]
    descriptions = cfg["labels"]["descriptions"]
    resolution = (f"One map value every {geom.centres[1] - geom.centres[0]:.0f} samples "
                  f"({(geom.centres[1] - geom.centres[0]) / fs:.2f} s), linearly interpolated; each value's receptive "
                  f"field spans {geom.receptive_field} samples ({geom.receptive_field / fs:.2f} s).")
    boundary = ("The first and last positions' receptive fields reach past the record into zero padding, so the band "
                "near both ends reflects the record boundary as well as the signal.")

    def annotation(p: dict[str, Any], seed: str) -> str:
        c = p["random"][seed]
        if c["spearman"] is None:
            return "All zero: no position raises this class's logit"
        return f"vs trained: Spearman ρ {c['spearman']:+.2f}, Pearson r {c['pearson']:+.2f}"

    rand_path = figures.plot_randomisation(
        [signals[ds.rows[row], lead_index] for row, _ in drawn],
        np.stack([t.maps[0] for t in trained]),
        {str(s): np.stack([m.maps[0] for m in random_maps[s]]) for s in r["init_seeds"]},
        [f"ecg_id {p['ecg_id']} · map of {p['target_class']} · labels {', '.join(p['labels'])} · "
         f"P cal. {p['calibrated_probability']:.2f}" for p in per_record],
        {str(s): [annotation(p, str(s)) for p in per_record] for s in r["init_seeds"]},
        f"Trained model (epoch {run_rec['best']['epoch']})",
        fs, lead, cfg["sample_plot"], figures_dir / f["randomisation"],
        title="Grad-CAM randomisation check (Adebayo et al., 2018): trained vs randomly initialised weights",
        subtitle=(f"Validation split (fold {', '.join(map(str, folds))}), model {run_id}. One record per class, drawn "
                  "at random from its true positives at the max-F1 threshold; each row shows the map of that class "
                  f"from the trained model and from {len(r['init_seeds'])} randomly initialised models of the same "
                  "architecture. Mean |Spearman ρ| over the defined pairs, per init seed: "
                  + "; ".join(f"seed {k} {'undefined' if v['mean_abs_spearman'][rule] is None else format(v['mean_abs_spearman'][rule], '.2f')}"
                              f" (n = {v['defined']} of {v['records']})" for k, v in summary.items())
                  + f". Criterion: below {limit} for every seed; all-zero random maps have no correlation and are "
                  f"left out of the mean. Verdict: {'PASS' if passed else 'FAIL'}."),
        footnote=(f"{figures.CAPTION} Each map has one value per time position, shared by all 12 leads and drawn "
                  f"behind lead {lead}, and is divided by its own maximum, so shading is not comparable between panels. "
                  f"{resolution} {boundary}"),
        band_alpha=f["band_alpha"], panel_in=tuple(f["randomisation_panel_in"]), min_height_mv=f["min_height_mv"],
        dpi=style["dpi"])

    example_paths = {}
    for cls, info in examples.items():
        others = ", ".join(f"{c} {p:.2f}" for c, p in info["all_probabilities"].items() if c != cls)
        example_paths[cls] = figures.plot_overlay(
            signals[ds.rows[info["row"]], lead_index], info["result"].maps[0], info["result"].centres, fs, lead,
            cfg["sample_plot"], figures_dir / f["example"].format(cls=cls),
            title=f"Grad-CAM for {cls} ({descriptions[cls]}) · validation record ecg_id {info['ecg_id']}, labelled {cls} only",
            subtitle=(f"Calibrated P({cls}) = {info['calibrated_probability']:.3f} (max-F1 threshold "
                      f"{info['max_f1_threshold']:.3f}); other classes {others}, all correctly below their thresholds. "
                      f"Map of the raw {cls} logit from the last residual stage: one value per time position, shared by "
                      f"all 12 leads, drawn behind lead {lead}."),
            footnote=(f"Best case, not a typical one: the highest calibrated P({cls}) among {info['eligible_records']} "
                      f"single-label {cls} validation records with all five decisions correct. Teal ticks along the "
                      f"top: feature-position centres. {resolution} {boundary}"),
            band_alpha=f["band_alpha"], width_in=f["example_width_in"], min_height_mv=f["min_height_mv"],
            dpi=style["dpi"])

    report = {
        "description": "Grad-CAM (PROJECT_SPEC.md section 5) on validation records only: the model parameter "
                       "randomisation check and example overlays. The maps show which time positions raised the "
                       "class's raw logit; they are not a clinical rationale.",
        "generated_by": " ".join([COMMAND, *sys.argv[1:]]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "split": "val",
        "folds": folds,
        "records": int(len(y)),
        "ecg_ids_sha256": digest,
        "model": {"run_id": run_id, "checkpoint": run_rec["checkpoint"], "best_epoch": run_rec["best"]["epoch"],
                  "seed": run_rec["seed"], "training_commit": run_rec["git"]["commit"], "overrides": run_rec["overrides"]},
        "gradcam": {"target_layer": TARGET_LAYER, "target": "raw logit of the class, before sigmoid and calibration",
                    "channel_weights": "mean of d logit / d A over time",
                    "normalisation": NORMALISATION,
                    "feature_positions": len(geom.centres),
                    "position_spacing_samples": float(geom.centres[1] - geom.centres[0]),
                    "first_centre_sample": float(geom.centres[0]),
                    "receptive_field_samples": geom.receptive_field,
                    "sampling_rate_hz": fs},
        "randomisation_check": {
            "reference": "Adebayo et al. (2018), Sanity checks for saliency maps, NeurIPS: model parameter randomisation",
            "record_selection": "one distinct record per class, drawn uniformly (seed "
                                f"{r['draw_seed']}) from that class's true positives at the max-F1 threshold",
            "random_models": f"build_model after torch.manual_seed(seed) for seeds {r['init_seeds']}, eval mode",
            "criterion": f"for every random initialisation, mean |Spearman rho| over the records < {limit} "
                         "(this limit was fixed in config/default.yaml before any map was computed)",
            "undefined_correlations": {
                "rule": rule,
                "meaning": "a random map that is all zero (constant) has no correlation; 'exclude' leaves it out of the "
                           "mean, 'zero' counts it as rho = 0. Added after the first run produced such maps; 'exclude' "
                           "is always the stricter.",
                "passed_under_other_rule": passed_other,
            },
            "passed": passed,
            "per_seed": summary,
            "records": per_record,
        },
        "calibration_check": {"explanation": WHY_CALIBRATION_CANCELS, "records": calibration_check},
        "examples": {
            "selection": "per class: single-label record, all five decisions correct at the max-F1 thresholds, highest "
                         "calibrated probability for the class (a best case, not a typical one)",
            "per_class": {c: {k: v for k, v in info.items() if k not in ("row", "result")}
                          | {"figure": repo_path(example_paths[c])} for c, info in examples.items()},
        },
        "figures": {"randomisation": repo_path(rand_path), **{f"example_{c}": repo_path(p) for c, p in example_paths.items()}},
    }
    write_json(project_path(cfg["paths"]["reports_dir"]) / ex["report_file"], report)
    return report, passed


# ----------------------------------------------------------------------------------------------- output

def print_report(cfg: dict[str, Any], report: dict[str, Any]) -> None:
    g, rc = report["gradcam"], report["randomisation_check"]
    m = report["model"]
    if report["git"]["dirty"]:
        print(f"WARNING: tracked files differ from commit {report['git']['commit'][:7]}; the report records dirty=true. "
              "Commit, then rerun, so every number traces to a commit.")
    print(f"Validation split (fold {', '.join(map(str, report['folds']))}): {report['records']:,} records. Model "
          f"{m['run_id']}, best epoch {m['best_epoch']}, seed {m['seed']}. Predictions reproduce the run's logged "
          "validation macro AUROC and AUPRC; Grad-CAM's forward pass reproduces every logit used below.")
    fs = g["sampling_rate_hz"]
    print(f"Target layer: {g['target_layer']}: {g['feature_positions']} time positions, one every "
          f"{g['position_spacing_samples']:.0f} samples ({g['position_spacing_samples'] / fs:.2f} s), centred on sample "
          f"{g['first_centre_sample']:.0f}, {g['position_spacing_samples']:.0f}, ...; receptive field "
          f"{g['receptive_field_samples']} samples ({g['receptive_field_samples'] / fs:.2f} s).")
    print()
    seeds = list(rc["per_seed"])
    print("Randomisation check (Adebayo et al., 2018): the same records' maps from the trained model and from "
          f"randomly initialised models (init seeds {', '.join(seeds)}).")
    head = f"{'ecg_id':>7}  {'target':<6}{'labels':<15}{'P cal.':>7}"
    print(head + f"{'Spearman rho per init seed':>32}{'Pearson r per init seed':>30}")

    def cells(rec: dict[str, Any], key: str) -> str:
        return "".join(f"{'undef.' if rec['random'][s][key] is None else format(rec['random'][s][key], '+.3f'):>9}"
                       for s in seeds)

    for rec in rc["records"]:
        print(f"{rec['ecg_id']:>7}  {rec['target_class']:<6}{', '.join(rec['labels']):<15}"
              f"{rec['calibrated_probability']:>7.3f}     {cells(rec, 'spearman')}     {cells(rec, 'pearson')}")
    def means(key: str, rule: str) -> str:
        return "".join(f"{'undef.' if v is None else format(v, '.3f'):>9}"
                       for v in (s[f"mean_abs_{key}"][rule] for s in rc["per_seed"].values()))

    u = rc["undefined_correlations"]
    other = next(x for x in UNDEFINED_RULES if x != u["rule"])
    for rule, label in ((u["rule"], f"Mean |corr.|, {u['rule']} (verdict)"), (other, f"Mean |corr.|, {other}")):
        print(f"{label:<35}     {means('spearman', rule)}     {means('pearson', rule)}")
    defined = ", ".join(f"seed {k}: {s['defined']} of {s['records']}" for k, s in rc["per_seed"].items())
    print(f"Defined correlations ({defined}). An undefined one means the random map is all zero: no position raises "
          "the class's logit.")
    print(f"Criterion ({rc['criterion']}; constant maps: {u['rule']}): {'PASS' if rc['passed'] else 'FAIL'}"
          f" (under '{other}': {'PASS' if u['passed_under_other_rule'] else 'FAIL'})")
    print()
    cc = report["calibration_check"]
    worst = max(r["max_abs_map_difference"] for r in cc["records"])
    print("Calibration and the map. Grad-CAM uses the raw logit. " + cc["explanation"])
    print("Checked on the records above: map of the calibrated logit vs map of the raw logit, largest difference "
          f"{worst:.1e}; the un-normalised peaks differ by the factor a: "
          + ", ".join(f"{r['class']} a = {r['a']:.3f}, ratio "
                      f"{'undef.' if r['raw_peak_ratio'] is None else format(r['raw_peak_ratio'], '.3f')}"
                      for r in cc["records"]) + ".")
    print()
    print("Example overlays: per class, the single-label record with all five decisions correct at the max-F1 "
          "thresholds and the highest calibrated probability (a best case, not a typical one).")
    for c, info in report["examples"]["per_class"].items():
        print(f"  {c:<5} ecg_id {info['ecg_id']:>5}  calibrated P = {info['calibrated_probability']:.3f} "
              f"(threshold {info['max_f1_threshold']:.3f}), logit {info['logit']:+.2f}, "
              f"highest of {info['eligible_records']} eligible records")
    print()
    reports_dir = project_path(cfg["paths"]["reports_dir"])
    for path in [repo_path(reports_dir / cfg["explain"]["report_file"]), *report["figures"].values()]:
        print(f"Saved {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grad-CAM randomisation check and example overlays, validation only.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", help="training run id (default: evaluation.model_run in the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    report, passed = run(cfg, args.run or cfg["evaluation"]["model_run"])
    print_report(cfg, report)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
