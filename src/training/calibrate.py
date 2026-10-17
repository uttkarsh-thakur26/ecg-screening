"""Calibrate the model's probabilities per class, on the validation split only (PROJECT_SPEC.md section 4).

    python -m src.training.calibrate                  # the run named in evaluation.model_run
    python -m src.training.calibrate --run <run_id>   # another completed training run

What it does, all on fold 9 (the test fold stays unopened):
1. Predict the validation split with the run's best checkpoint; the logits must reproduce the validation
   metrics the run logged (src/evaluation/evaluate_val.py, predict_validation).
2. Per class, fit two calibrators to the validation logits z (src/evaluation/calibration.py):
   (a) temperature scaling:              p = sigmoid(z / T)
   (b) Platt scaling (temperature + bias): p = sigmoid(a z + b)
   Each minimises the mean binary cross-entropy WITHOUT pos_weight, with L-BFGS from the identity (T = 1;
   a = 1, b = 0). The fit must converge (every gradient component below calibration.max_final_gradient).
3. Per class, the expected calibration error (ECE, evaluation.reliability_bins equal-width bins) of the
   uncalibrated, temperature-scaled and Platt-scaled probabilities. The calibrator with the lower ECE is kept
   (on an exact tie, temperature: fewer parameters).
4. Write reports/calibration.json (every fitted parameter, ECE, cross-entropy and AUROC, the reliability
   bins, the choice) and reports/figures/reliability_val.png (uncalibrated vs the chosen calibrator per class).
5. Run the validation evaluation again with the new calibration (evaluate_val.evaluate_predictions on the
   same logits), which re-chooses both sets of thresholds on the calibrated probabilities and rewrites
   reports/thresholds.json, reports/metrics_val.json and the other validation figures.

Why calibrate. The loss weighted every positive label of class c by pos_weight[c] = negatives / positives in
the training split. That is right for learning to rank rare classes, but it changes what the outputs mean. For
a record whose true chance of having c is q, the weighted loss is lowest at the output w q / (w q + 1 - q),
not q: in logit terms, logit(q) + log(w). Every logit is pushed up by log(pos_weight), most for the rarest
class (HYP: w = 7.06, log w = 1.95; NORM: w = 1.25, log w = 0.22), so the raw probabilities overstate rare
findings. On top of that, a network trained until its validation loss rises becomes overconfident.

Why a bias term may be needed. Temperature scaling divides the logit by T. That fixes over- or
under-confidence, but it leaves z = 0 at p = 0.5, so it cannot undo an additive shift: whatever T is, a record
the model puts at 0.5 stays at 0.5. Platt scaling's bias b moves that point; if the pos_weight shift were the
only distortion, the fit would give a = 1 and b = -log(pos_weight). calibration.json reports b next to
-log(pos_weight) for comparison. Expect agreement in sign and rough trend only: overconfidence also shrinks a
below 1, and the two distortions are fitted together. Temperature scaling is the standard choice for networks
trained with an unweighted loss (Guo et al., 2017); with pos_weight it may not be enough.

Why fit without pos_weight. Calibration asks how often a class is present among records given probability p,
which is a question about the real class balance. Fitting with pos_weight again would re-create the shift.

Optimism. The calibrators are fitted, chosen and scored on the same 2,146 records, so the ECE here is
optimistic, the calibrated ECE most of all. Platt contains temperature scaling (a = 1/T, b = 0), so on the
fold it was fitted on its cross-entropy cannot be worse, and its lower ECE there partly reflects that extra
freedom. The honest ECE comes from fold 10, scored once in the final evaluation with these calibrators frozen.

AUROC. Both calibrators are strictly increasing (T > 0, a > 0), so the order of the records within a class
cannot change, and neither can AUROC or AUPRC. Here this is checked, not assumed: every calibrated score
must keep the order and the ties of the logits, and AUROC and AUPRC must match the uncalibrated values to
within calibration.ranking_tolerance.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.norm_stats import ecg_id_digest
from src.evaluation import plots
from src.evaluation.calibration import METHODS, Calibrator, choose, load_calibration
from src.evaluation.evaluate_val import (ValidationPredictions, calibration_path, evaluate_predictions,
                                         predict_validation, print_header, print_report, write_json)
from src.evaluation.metrics import expected_calibration_error, reliability_bins
from src.provenance import git_state
from src.training.train import repo_path

COMMAND = "python -m src.training.calibrate"
UNCALIBRATED = "uncalibrated"
VARIANTS = (UNCALIBRATED, *METHODS)
METHOD_LABELS = {"temperature": "Temperature", "platt": "Platt"}

WHY_BIAS = ("Training weighted each positive label of class c by pos_weight[c] = negatives / positives, which moves "
            "the loss-minimising logit from logit(q) to logit(q) + log(pos_weight[c]): raw probabilities overstate "
            "every class, the rare ones most. Temperature scaling (z / T) only rescales logits around 0, so z = 0 "
            "stays at p = 0.5 and an additive shift cannot be removed. Platt scaling's bias b can remove it; if the "
            "shift were the only distortion, b would equal -log(pos_weight[c]), which is reported next to b.")
OPTIMISM = ("Optimistic: the calibrators were fitted and chosen on these same fold-9 records, and Platt (two "
            "parameters) contains temperature scaling, so the calibrated ECE here flatters it most. The honest ECE "
            "comes from fold 10, scored once in the final evaluation with these calibrators frozen.")


# ------------------------------------------------------------------------------------------------------ fitting

def _check_class_data(z: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z, t = np.asarray(z), np.asarray(t)
    if z.ndim != 1 or t.shape != z.shape or len(z) == 0:
        raise ValueError(f"logits and labels must be 1-D (records,) of equal, non-zero length, got {z.shape} and {t.shape}")
    if not np.isfinite(z).all():
        raise ValueError("logits contain NaN or infinite values")
    if not np.isin(t, (0, 1)).all():
        raise ValueError(f"labels must be 0 or 1, got values {np.unique(t)[:10].tolist()}")
    if t.min() == t.max():
        raise ValueError("Calibration needs positives and negatives; with one kind the fit runs off to infinity")
    # If no positive scores below any negative, the cross-entropy has no minimum (it keeps falling as the slope
    # grows), yet its gradient fades fast enough to pass the convergence check: refuse instead of saving that.
    if z[t == 1].min() >= z[t == 0].max():
        raise ValueError("Positives and negatives do not overlap in logit (AUROC 1): the cross-entropy has no minimum, "
                         "so there is no calibrator to fit on these records")
    return z.astype(np.float64), t.astype(np.float64)


def _lbfgs(params: list[torch.Tensor], loss_fn: Callable[[], torch.Tensor], s: dict[str, Any], name: str) -> dict[str, Any]:
    """Minimise loss_fn over params in place with L-BFGS; raise unless the result is a converged minimum."""
    opt = torch.optim.LBFGS(params, lr=s["lbfgs"]["lr"], max_iter=s["lbfgs"]["max_iter"],
                            history_size=s["lbfgs"]["history_size"], tolerance_grad=s["lbfgs"]["tolerance_grad"],
                            tolerance_change=s["lbfgs"]["tolerance_change"], line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss

    opt.step(closure)
    loss = loss_fn()
    grads = torch.autograd.grad(loss, params)
    max_gradient = max(float(g.abs().max()) for g in grads)
    if not (math.isfinite(loss.item()) and all(torch.isfinite(p).all() for p in params)):
        raise RuntimeError(f"{name} fit diverged (loss {loss.item()}, parameters {[p.item() for p in params]})")
    if max_gradient > s["max_final_gradient"]:
        raise RuntimeError(f"{name} fit did not converge: largest gradient component {max_gradient:.2e} > "
                           f"calibration.max_final_gradient {s['max_final_gradient']:.0e} after "
                           f"{opt.state[params[0]]['n_iter']} iterations")
    return {"iterations": int(opt.state[params[0]]["n_iter"]), "max_gradient": max_gradient}


def fit_temperature(z: np.ndarray, t: np.ndarray, s: dict[str, Any]) -> tuple[Calibrator, dict[str, Any]]:
    """T minimising mean BCE(sigmoid(z / T), t), without pos_weight. T = exp(log T), so T stays positive."""
    z, t = _check_class_data(z, t)
    # The loss as a function of 1/T is convex; its slope at 1/T = 0 (every p = 0.5) is mean((0.5 - t) z). If that is
    # not negative, no T > 0 beats p = 0.5 for every record and the fit would only run off towards T = infinity.
    # Logits that rank badly do this, and so do logits shifted far above the class rate (a large pos_weight shift).
    if np.mean((0.5 - t) * z) >= 0:
        raise RuntimeError("Temperature fit: no T > 0 gives a lower cross-entropy than p = 0.5 for every record "
                           f"(mean((0.5 - label) * logit) = {np.mean((0.5 - t) * z):.3g} >= 0). The logits either rank "
                           "the class badly or are shifted far above its rate; Platt scaling's bias may still fit, but "
                           "the choice needs both candidates, so this stops here instead of dropping one.")
    z_t, t_t = torch.from_numpy(z), torch.from_numpy(t)
    log_t = torch.zeros((), dtype=torch.float64, requires_grad=True)
    fit = _lbfgs([log_t], lambda: F.binary_cross_entropy_with_logits(z_t / torch.exp(log_t), t_t), s, "Temperature")
    return Calibrator.temperature(math.exp(log_t.item())), fit


def fit_platt(z: np.ndarray, t: np.ndarray, s: dict[str, Any]) -> tuple[Calibrator, dict[str, Any]]:
    """a, b minimising mean BCE(sigmoid(a z + b), t), without pos_weight: logistic regression on the logit (convex)."""
    z, t = _check_class_data(z, t)
    z_t, t_t = torch.from_numpy(z), torch.from_numpy(t)
    a = torch.ones((), dtype=torch.float64, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, requires_grad=True)
    fit = _lbfgs([a, b], lambda: F.binary_cross_entropy_with_logits(a * z_t + b, t_t), s, "Platt")
    if a.item() <= 0:
        raise RuntimeError(f"Platt fit gave slope a = {a.item():.4g} <= 0: higher logits would mean a LOWER chance of "
                           "the class. The logits carry no usable ranking for it; refusing to calibrate.")
    return Calibrator("platt", a.item(), b.item()), fit


# ---------------------------------------------------------------------------------------------- scoring

def mean_bce(t: np.ndarray, logits: np.ndarray) -> float:
    """Mean binary cross-entropy without pos_weight, from logits: mean(log(1 + e^s) - t s), computed stably."""
    s = np.asarray(logits, dtype=np.float64)
    return float(np.mean(np.logaddexp(0.0, s) - np.asarray(t) * s))


def check_order_preserved(z: np.ndarray, p: np.ndarray, name: str) -> None:
    """p must be strictly increasing in z on these records: same order, and tied exactly where z is tied.

    Then every threshold on p flags a set of records that some threshold on z flags, so AUROC, AUPRC and the
    thresholded decisions cannot change. Fails if float rounding merged two distinct logits into one probability.
    """
    order = np.argsort(z, kind="stable")
    dz, dp = np.diff(np.asarray(z, dtype=np.float64)[order]), np.diff(p[order])
    if (dp < 0).any() or ((dp == 0) & (dz != 0)).any() or ((dp != 0) & (dz == 0)).any():
        raise RuntimeError(f"{name}: calibrated probabilities do not keep the order and ties of the logits")


def score(t: np.ndarray, z: np.ndarray, cal: Calibrator | None, n_bins: int) -> dict[str, Any]:
    """ECE, cross-entropy, AUROC, AUPRC and reliability bins of one class's probabilities (cal None: uncalibrated)."""
    s = np.asarray(z, dtype=np.float64) if cal is None else cal.logits(z)
    p = expit(s)  # float64, exactly as train.probabilities and Calibrator.probabilities compute it
    return {"ece": expected_calibration_error(t, p, n_bins), "bce": mean_bce(t, s),
            "auroc": float(roc_auc_score(t, p)), "auprc": float(average_precision_score(t, p)),
            "mean_predicted": float(p.mean()), "reliability": reliability_bins(t, p, n_bins), "probabilities": p}


def calibrate_class(z: np.ndarray, t: np.ndarray, pos_weight: float, cfg: dict[str, Any], name: str) -> dict[str, Any]:
    """Fit both calibrators to one class, score all three variants, choose, and check the ranking is unchanged."""
    s, n_bins, tol = cfg["calibration"], cfg["evaluation"]["reliability_bins"], cfg["calibration"]["ranking_tolerance"]
    temperature, temperature_fit = fit_temperature(z, t, s)
    platt, platt_fit = fit_platt(z, t, s)
    cals = {UNCALIBRATED: None, "temperature": temperature, "platt": platt}
    scored = {k: score(t, z, cal, n_bins) for k, cal in cals.items()}
    for k in METHODS:
        check_order_preserved(z, scored[k]["probabilities"], f"{name} {k}")
        for m in ("auroc", "auprc"):
            gap = abs(scored[k][m] - scored[UNCALIBRATED][m])
            if gap > tol:
                raise RuntimeError(f"{name} {k}: {m} moved by {gap:.2e} > calibration.ranking_tolerance {tol:.0e}")
    chosen = choose({k: scored[k]["ece"] for k in METHODS})
    fits = {"temperature": temperature_fit, "platt": platt_fit}
    candidates = {}
    for k in VARIANTS:
        entry = {m: scored[k][m] for m in ("ece", "bce", "auroc", "auprc", "mean_predicted")}
        if k in METHODS:
            entry = cals[k].to_json() | entry | {"fit": fits[k]}
        candidates[k] = entry
    n_positive = int(np.asarray(t).sum())
    return {
        "n_positive": n_positive,
        "prevalence": n_positive / len(t),  # exact, as ranking_metrics computes it (t may be float32)
        "pos_weight": pos_weight,
        "chosen": chosen,
        "calibrator": cals[chosen].to_json(),
        "candidates": candidates,
        # The pos_weight argument predicts Platt's b near -log(pos_weight) if the shift were the only distortion.
        "platt_b_vs_minus_log_pos_weight": {"b": platt.b, "minus_log_pos_weight": -math.log(pos_weight)},
        "reliability": {k: scored[k]["reliability"] for k in VARIANTS},
    }


# ---------------------------------------------------------------------------------------------- report

def calibration_report(cfg: dict[str, Any], preds: ValidationPredictions, git: dict[str, Any]) -> dict[str, Any]:
    """Fit, score and choose per class; returns the content of reports/calibration.json."""
    run = preds.run
    classes = run["classes"]
    t_all = np.asarray(preds.targets)
    per_class = {c: calibrate_class(preds.logits[:, j], t_all[:, j], float(run["pos_weight"][c]), cfg, c)
                 for j, c in enumerate(classes)}
    macro = {k: float(np.mean([per_class[c]["candidates"][k]["ece"] for c in classes])) for k in VARIANTS}
    macro["chosen"] = float(np.mean([per_class[c]["candidates"][per_class[c]["chosen"]]["ece"] for c in classes]))
    s = cfg["calibration"]
    return {
        "description": "Per-class probability calibration, fitted and chosen on the validation split. Apply "
                       "p = sigmoid(a * logit + b) with each class's 'calibrator' before any threshold in "
                       "thresholds.json. " + OPTIMISM,
        "generated_by": " ".join([COMMAND, *sys.argv[1:]]),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git": git,
        "split": "val",
        "folds": run["config"]["splits"]["val_folds"],
        "records": int(len(t_all)),
        "patients": int(len(np.unique(preds.patient_ids))),
        "ecg_ids_sha256": ecg_id_digest(preds.ecg_ids),
        "model": {"run_id": run["run_id"], "checkpoint": run["checkpoint"], "best_epoch": run["best"]["epoch"],
                  "seed": run["seed"], "training_commit": run["git"]["commit"], "overrides": run["overrides"]},
        "fit": {
            "objective": "mean binary cross-entropy on the validation records, WITHOUT pos_weight",
            "optimizer": "torch.optim.LBFGS, strong Wolfe line search, float64, started at the identity "
                         "(T = 1; a = 1, b = 0)",
            "lbfgs": s["lbfgs"], "max_final_gradient": s["max_final_gradient"],
            "temperature": "p = sigmoid(logit / T), T = exp(fitted log T) > 0",
            "platt": "p = sigmoid(a * logit + b); a must come out > 0",
        },
        "ece": {"bins": cfg["evaluation"]["reliability_bins"],
                "binning": "equal width; bin k holds k/n <= p < (k+1)/n, the last bin also p = 1",
                "definition": "sum over bins of (records in bin / records) x |observed frequency - mean predicted "
                              "probability|, per class (one-vs-rest)",
                "caveat": OPTIMISM},
        "selection_rule": "per class, the calibrator (temperature or Platt) with the lower validation ECE; temperature "
                          "on an exact tie (fewer parameters)",
        "why_bias": WHY_BIAS,
        "ranking_check": f"both calibrators keep the order and ties of the logits on every class, and AUROC and AUPRC "
                         f"match the uncalibrated values to within {s['ranking_tolerance']:.0e}",
        "classes": classes,
        "per_class": per_class,
        "macro_ece": macro,
    }


def plot_reliability(cfg: dict[str, Any], report: dict[str, Any]) -> Path:
    classes, per_class = report["classes"], report["per_class"]
    annotations = {}
    for c in classes:
        chosen = per_class[c]["chosen"]
        cal = per_class[c]["calibrator"]
        params = f"T {cal['temperature']:.2f}" if chosen == "temperature" else f"a {cal['a']:.2f}, b {cal['b']:+.2f}"
        annotations[c] = (f"ECE {per_class[c]['candidates'][UNCALIBRATED]['ece']:.3f} → "
                          f"{per_class[c]['candidates'][chosen]['ece']:.3f}\n{METHOD_LABELS[chosen]}: {params}")
    f = cfg["evaluation"]["figures"]
    folds = ", ".join(map(str, report["folds"]))
    return plots.plot_reliability_diagrams(
        {c: per_class[c]["reliability"][UNCALIBRATED] for c in classes},
        {c: per_class[c]["reliability"][per_class[c]["chosen"]] for c in classes},
        ("Uncalibrated (sigmoid of the logit)", "Calibrated (calibrator chosen per class)"),
        classes, cfg["labels"]["descriptions"], annotations,
        project_path(cfg["paths"]["figures_dir"]) / cfg["calibration"]["figure"],
        title=f"Reliability diagrams per class, before and after calibration ({report['ece']['bins']} bins)",
        subtitle=f"Validation split (fold {folds}): {report['records']:,} records, {report['patients']:,} patients. "
                 f"Model {report['model']['run_id']}, best epoch {report['model']['best_epoch']}. Calibrators fitted "
                 "and chosen on these records, so the calibrated curves are optimistic.",
        dpi=f["dpi"], panel_in=f["panel_size_in"])


def format_table(report: dict[str, Any]) -> str:
    """Per class: prevalence, pos_weight, ECE of the three variants with the fitted parameters, the choice, AUROC."""
    lines = [f"{'':<25}{'Uncal.':>8}{'Temperature':>17}{'Platt scaling (temperature + bias)':>47}",
             f"{'Class':<7}{'Prev':>8}{'pos_w':>10}{'ECE':>8}{'T':>9}{'ECE':>8}{'a':>9}{'b':>9}{'-log pos_w':>13}"
             f"{'ECE':>8}   {'Chosen':<13}{'AUROC (all 3)':>13}"]
    for c in report["classes"]:
        r = report["per_class"][c]
        cand = r["candidates"]
        lines.append(f"{c:<7}{r['prevalence']:>8.3f}{r['pos_weight']:>10.2f}{cand[UNCALIBRATED]['ece']:>8.4f}"
                     f"{cand['temperature']['temperature']:>9.3f}{cand['temperature']['ece']:>8.4f}"
                     f"{cand['platt']['a']:>9.3f}{cand['platt']['b']:>+9.3f}"
                     f"{r['platt_b_vs_minus_log_pos_weight']['minus_log_pos_weight']:>+13.3f}"
                     f"{cand['platt']['ece']:>8.4f}   {METHOD_LABELS[r['chosen']]:<13}{cand[UNCALIBRATED]['auroc']:>13.4f}")
    m = report["macro_ece"]
    lines.append(f"{'Macro ECE':<25}{m[UNCALIBRATED]:>8.4f}{m['temperature']:>17.4f}{m['platt']:>39.4f}"
                 f"   chosen: {m['chosen']:.4f}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit per-class calibrators on the validation split and re-choose the "
                                                 "thresholds on the calibrated probabilities.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--run", help="training run id (default: evaluation.model_run in the config)")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    run_id = args.run or cfg["evaluation"]["model_run"]

    git = git_state()  # before anything is written, so this run's own outputs do not count as uncommitted changes
    preds = predict_validation(cfg, run_id)
    report = calibration_report(cfg, preds, git)
    path = calibration_path(cfg)
    write_json(path, report)
    figure = plot_reliability(cfg, report)
    # Read back from disk, so the thresholds below use exactly what any later reader of the file will apply.
    calibration = load_calibration(path, preds.run, preds.run["classes"], report["ecg_ids_sha256"])
    metrics = evaluate_predictions(cfg, preds, calibration, git, COMMAND)

    print_header(metrics)
    print(f"Calibrators fitted per class to the validation logits by minimising the mean binary cross-entropy "
          f"without pos_weight (L-BFGS); ECE with {report['ece']['bins']} equal-width bins, lower is better.")
    print()
    print(format_table(report))
    print()
    print("Both calibrators are strictly increasing: per-class AUROC and AUPRC are identical for all three variants "
          f"(checked to {cfg['calibration']['ranking_tolerance']:.0e}, and the order of every record is unchanged).")
    worse = [c for c in report["classes"] if report["per_class"][c]["candidates"][UNCALIBRATED]["ece"]
             < report["per_class"][c]["candidates"][report["per_class"][c]["chosen"]]["ece"]]
    if worse:
        print(f"NOTE: for {', '.join(worse)} the uncalibrated ECE is lower than the chosen calibrator's.")
    print("Why a bias term: pos_weight shifts every logit up by about log(pos_weight); temperature scaling keeps "
          "logit 0 at p = 0.5 and cannot undo a shift, Platt's b can (compare b with -log pos_w above).")
    print(f"CAUTION: these ECE values are measured on fold {', '.join(map(str, report['folds']))}, the records the "
          "calibrators were fitted and chosen on, so they are optimistic (the calibrated ones most). The honest ECE "
          "comes from fold 10, scored once in the final evaluation.")
    print()
    print(f"Saved {repo_path(path)}")
    print(f"Saved {repo_path(figure)}")
    print()
    print_report(cfg, metrics)
    return 0


if __name__ == "__main__":
    sys.exit(main())
