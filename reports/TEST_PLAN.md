# Test plan: the one-time evaluation on fold 10

This plan is committed **before fold 10 is opened**, in the same commit that adds the evaluation script
`src/evaluation/evaluate_test.py` and puts it on the fold-10 allowlist (`tests/test_dataset.py`,
`MAY_UNLOCK_TEST`). The script refuses to open fold 10 unless this file is committed, no tracked file is
modified, and nothing under `src/`, `config/`, `scripts/`, `tests/` or this plan is untracked. So the
commit recorded in `reports/metrics.json` holds exactly this plan and the code that ran.

Everything listed here is computed in one run of

```bash
python -m src.evaluation.evaluate_test
```

Nothing else is computed on fold 10. Anything looked at later that this plan does not list would be
exploratory and labelled as such. It would never be used to change the model, the calibrators or the
thresholds.

## 1. What is frozen

Everything below was chosen on fold 9 (validation) or earlier. None of it changes after fold 10 is opened.

| Item | Value | Pinned by |
|---|---|---|
| Model | v1.0: best checkpoint (epoch 19 of 27) of run `20261017-215633_no-amp-scale`, the committed config (amplitude scaling off) with seed 42, trained at commit `46330c1` | `checkpoints/release.pt`, SHA-256 `3dff392569381849041c00b4a72df9d3c3dc817e83c843adf48f6aebeb8cb262`, recorded in `reports/release.json` |
| Calibrators | per class, fitted on fold 9: NORM temperature T = 1.4602; MI Platt a = 0.6720, b = -0.9442; STTC Platt a = 0.7392, b = -1.1782; CD Platt a = 0.7061, b = -1.3781; HYP Platt a = 0.6185, b = -1.4923 | `reports/calibration.json`, SHA-256 `04fbf9eada05186352ea82401ee7b2f3ff0b1d8106da07e0a549ee34ccae38c4` |
| Max-F1 thresholds (calibrated probability) | NORM 0.4153, MI 0.3674, STTC 0.3727, CD 0.3570, HYP 0.3962 | `reports/thresholds.json`, SHA-256 `aaeb36a688b0c6cfc8e8a6d8b0e3a5dfb664a7a615687c54a65221f65e0ccebd` |
| High-sensitivity thresholds (chosen for sensitivity >= 0.90 on fold 9) | MI 0.1813, STTC 0.2187, CD 0.1319, HYP 0.0727; NORM has none | same file |
| Baselines | prior (training-split class frequencies) and feature GBDT (79 features; scikit-learn's default hyperparameters except `early_stopping=False`, so a fixed 100 iterations, because sklearn's early stopping would hold out records, not patients), fitted on folds 1-8 | refitted in the run; must reproduce `reports/baselines_val.json` |

Values in this table are rounded to 4 decimals. The run uses the full-precision values in the files,
and checks the files' SHA-256 values against `reports/release.json`. Before freezing, the calibrators
were refitted and the thresholds re-chosen on this checkpoint's validation logits; both matched the
stored values exactly.

## 2. Data

- PTB-XL 1.0.3, 100 Hz, fold 10 (`strat_fold` = 10) only, labelled records only (records with at least one
  diagnostic superclass). The number of records and of unique patients is computed and reported.
- The patient-leakage check runs on every split before any record is served. Normalisation uses the
  training-fold statistics, as in training.
- No augmentation, batch size 64, deterministic GPU kernels: the settings that reproduced the
  validation logits exactly.

## 3. Checks that must pass before fold 10 is opened (the run stops otherwise)

All of these run, in this order, before the first read of a fold-10 record:

1. `reports/TEST_PLAN.md` is committed; no tracked file is modified; nothing under `src/`, `config/`,
   `scripts/`, `tests/` or this plan is untracked.
2. Fold 10 has never been opened: there is no `reports/metrics.json` and no line in
   `reports/fold10_access_log.jsonl`. The one exception is a documented bug-fix rerun (section 9).
3. `release.pt`, `calibration.json` and `thresholds.json` have the SHA-256 values in `reports/release.json`.
4. On fold 9, in the same run:
   - `release.pt` reproduces the run's logged validation macro AUROC and AUPRC (to 1e-6);
   - re-choosing both operating points gives the frozen thresholds and confusion counts;
   - the seed 42, 43 and 44 checkpoints reproduce every mean, standard deviation and per-run macro value
     in `reports/seed_stability_val.json` (to 1e-12);
   - both baselines, refitted on folds 1-8, reproduce `reports/baselines_val.json` (to 1e-12);
   - the scoring code used for fold 10 reproduces every ranking, operating-point and ECE value in
     `reports/metrics_val.json` and `reports/calibration.json` (to 1e-12).

Then one line (time, commit, command, plan SHA-256) is appended to `reports/fold10_access_log.jsonl`, and
only then is fold 10 opened. The first line creates the file exclusively, so a crash at any later point
is still on record, and two runs started together cannot both count as the first.

Before this plan was committed, the pipeline was rehearsed on fold 9
(`python -m src.evaluation.evaluate_test --rehearse <folder>`, which never opens fold 10). The rehearsal
runs the same prediction, baseline-scoring, bootstrap, figure and report functions as the real run, with
the split set to fold 9. Those functions reproduced the fold-9 logits and baseline scores exactly, and
every check above passed with a largest difference of 0.0. Two things were not rehearsed on data: the
split name itself (the fold-10 loader and cache rows) and the one-time guard, which has unit tests
(`tests/test_evaluate_test.py`).

## 4. Metrics on fold 10

All on the frozen calibrated probabilities. A record is flagged for a class when its calibrated
probability is >= the class's frozen threshold. Thresholds are applied as they are and never re-chosen.

**Ranking (threshold-free), CNN:** AUROC and AUPRC (average precision), per class (NORM, MI, STTC, CD,
HYP), macro over all 5, and macro over the 4 abnormal classes (MI, STTC, CD, HYP). Each class's support
and prevalence (the AUPRC of a no-skill model) are reported next to them.

**Operating points, CNN:**
- Max-F1 point: per class (all 5), confusion counts (TP, FP, FN, TN), sensitivity, specificity,
  precision (PPV), NPV and F1; macro over all 5 and macro over the 4 abnormal classes.
- High-sensitivity point: the same for the 4 abnormal classes, and their macro. There is no all-5 macro,
  because NORM has no threshold here.
- Accuracy is not reported. With NORM in about 45% of records and HYP in about 12%, it misleads.

**Calibration, CNN:** expected calibration error (10 equal-width bins, per class one-vs-rest; macro = mean
over the 5 classes) of the uncalibrated sigmoid outputs and of the calibrated probabilities, their
difference (calibrated minus uncalibrated), and the reliability bins of both.

**Baselines** (ranking metrics only; no thresholds were ever chosen for them):
- Prior: AUROC and AUPRC per class, macro, macro abnormal. These are point values only: by construction
  its AUROC is 0.5 and its AUPRC equals the prevalence.
- Feature GBDT: AUROC and AUPRC per class, macro, macro abnormal.
- Paired difference CNN minus feature GBDT for each of those numbers.

## 5. Uncertainty: patient bootstrap

- Unit: patient. Each resample draws as many patients as fold 10 has, with replacement, and keeps every
  record of each drawn patient, as many times as the patient was drawn. Records of one patient are not
  independent, so resampling records would understate the uncertainty.
- 1000 resamples, `numpy.random.default_rng(42)`.
- 95% percentile interval: the 2.5th and 97.5th percentiles (`numpy.quantile`, linear interpolation) of
  each number over the resamples. The point value is the number on all of fold 10.
- Intervals for the estimated metrics: AUROC and AUPRC, the five operating-point metrics, ECE and its
  calibrated-minus-uncalibrated difference, and the GBDT's ranking metrics and the paired CNN-minus-GBDT
  difference. Support, prevalence, confusion counts, thresholds, reliability bins and the prior baseline
  are point values. The CNN and the GBDT are scored on the same resamples, so their difference is paired.
- If any number is undefined in any resample (for example a class without positives), the run stops with
  an error. No resample is dropped silently.
- Known property, seen in the fold-9 rehearsal: ECE is a binned, non-negative statistic and is biased
  upward under resampling. Its percentile interval, and that of the calibrated-minus-uncalibrated
  difference, therefore describes spread. It is not a confidence interval for the true ECE, and the point
  value can lie below it. This applies on fold 10 as well. On fold 9, where the calibrators were fitted,
  the calibrated macro ECE was 0.016 with interval [0.018, 0.027]. The intervals are reported as computed,
  `metrics.json` carries this note next to them, and the method is not changed after seeing fold 10.
- These intervals cover the choice of test patients only, not training noise (section 6).

## 6. Seed stability (validation)

Recomputed in the run from the saved seed 42, 43 and 44 checkpoints of the committed config: validation
macro AUROC and AUPRC per seed, their mean and sample standard deviation (ddof = 1). Nothing is trained.
Fold 10 is scored with the seed-42 model only.

## 7. Validation column

The same scoring code on fold 9, recomputed in the run (point values only), is reported next to each
fold-10 ranking metric, operating-point metric and ECE of the CNN, next to the GBDT's ranking metrics and
the paired difference, and next to the prior's. It must equal `reports/metrics_val.json` (check 4). The
fold-9 values are optimistic, because the epoch, the calibrators and the thresholds were chosen there.

## 8. Outputs

- `reports/metrics.json`: dataset name and version, fold-10 record and patient counts, model version,
  checkpoint SHA-256, git commit, date, this plan's commit and SHA-256, the fold-10 access log, the frozen
  calibrators and thresholds, every number in sections 4 to 7 (estimated metrics as
  `{value, ci_low, ci_high}`), the bootstrap settings, the precondition checks, and the figure paths. The
  two existing figures that are not test-set figures (training curves, class support of training and
  validation) are listed separately, so the frontend can find every plot from this one file.
- The git commit in `metrics.json` is the commit the code ran from, which holds this plan. The commit that
  adds the outputs, and the `v1.0-model` tag on it, are its children.
- Figures (in `reports/figures/`): `roc_curves_test.png` and `pr_curves_test.png` (both operating points
  marked), `confusion_max_f1_test.png`, `confusion_high_sensitivity_test.png`, `reliability_test.png`
  (uncalibrated vs calibrated).
- `reports/TEST_RUN.md`, written by hand after the run from `metrics.json`: date, commit, and a short
  comparison of fold 10 against fold 9.
- The commit holding these files (and the access log) is tagged `v1.0-model`.

## 9. Rules after fold 10 is opened

- The model, the calibrators and the thresholds never change, whatever fold 10 shows.
- If a bug is found in the evaluation code: fix only the bug, commit, rerun with
  `--rerun-reason "<what and why>"`, and record in `TEST_RUN.md` what changed and why. The script allows a
  rerun only if fold 10 was opened before (access log or `metrics.json`), only with this same plan
  (unchanged SHA-256), and, when a `metrics.json` exists, only from a new commit. The previous
  `metrics.json` and test figures are copied to `reports/superseded_<time>/` with their hashes, and the new
  `metrics.json` lists the rerun under `reruns`. The access log gains a line for every opening.

## 10. Expectations stated in advance

- Sensitivity at the high-sensitivity point is expected near 0.90 on fold 10, on average slightly below
  it. Each threshold is the highest that still reaches 0.90 on fold 9, so it sits at the edge of the
  fold-9 target. That gives a small expected shortfall, plus possibly some optimism from choosing the
  epoch on fold 9. The class-by-class spread from sampling alone is about 0.02 (larger for HYP, which has
  fewer positives), so any single class can land on either side of 0.90. It is reported as measured.
- Ranking metrics are expected near the validation values (fold-9 macro AUROC 0.929, seed spread 0.0009).
  For context, Strodthoff et al. (2021) report about 0.93 macro AUROC on fold 10 for the best CNNs on
  this task.
- Calibrated ECE is expected to be above its fold-9 value, which was measured on the records the
  calibrators were fitted to.
