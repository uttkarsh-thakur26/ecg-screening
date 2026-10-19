# Test run: the one-time evaluation on fold 10

| | |
|---|---|
| Date | 2026-10-19. Fold 10 opened at 09:17:26 UTC; `reports/metrics.json` written at 09:18:33 UTC |
| Code and plan | commit `d1e1b4258da9f7b99073608d8292f4841cd434ce` (clean); `reports/TEST_PLAN.md` committed in that same commit, SHA-256 `de3328c8...` |
| Command | `python -m src.evaluation.evaluate_test` |
| Model | release v1.0: run `20261017-215633_no-amp-scale`, seed 42, epoch 19; `checkpoints/release.pt` SHA-256 `3dff3925...` |
| Test split | PTB-XL 1.0.3, fold 10: 2,158 labelled records from 1,877 patients |
| Openings of fold 10 | 1 (`reports/fold10_access_log.jsonl`). No rerun, no bug fix after opening |
| Outputs | `reports/metrics.json`, `reports/figures/{roc_curves,pr_curves,confusion_max_f1,confusion_high_sensitivity,reliability}_test.png` |

Every number below is copied from `reports/metrics.json`, which this run wrote. Brackets are 95% percentile
intervals from 1000 bootstrap resamples of test patients (seed 42). The "val" columns are the same code
on fold 9, recomputed in the same run. They equal `reports/metrics_val.json` and are optimistic, because
the epoch, the calibrators and the thresholds were chosen on fold 9.

## Checks before fold 10 was opened

All passed, in this order, before the first fold-10 record was read: plan and code committed with no
untracked files; frozen files match `reports/release.json`; on fold 9 the release model reproduced its
logged metrics; re-choosing the thresholds gave the frozen ones; the seed runs reproduced
`seed_stability_val.json`; the refitted baselines reproduced `baselines_val.json`; and the scoring code
reproduced `metrics_val.json` and the ECEs in `calibration.json`. Every reproduction differed by exactly 0.0.

## Test against validation, in short

- **Ranking holds up.** Macro AUROC is 0.926 [0.918, 0.933] on fold 10 against 0.929 on fold 9, and
  macro AUPRC is 0.819 [0.802, 0.833] against 0.816. The fold-9 value lies inside the fold-10 interval
  for every class and both metrics, except STTC AUPRC, which is higher on fold 10 (0.834 against 0.789).
  MI drops the most (AUPRC 0.819 against 0.843). The 0.003 drop in macro AUROC is larger than the
  seed-to-seed spread (0.0009) but well inside the test interval. For context, Strodthoff et al. (2021)
  report about 0.93 macro AUROC on fold 10 for the best CNNs on this task.
- **High-sensitivity point: sensitivity lands just under 0.90, as expected.** Measured: MI 0.887, STTC
  0.902, CD 0.889, HYP 0.889; macro 0.892 [0.876, 0.907]. Every class's interval contains 0.90. Missed
  cases: MI 62 of 550, STTC 51 of 521, CD 55 of 496, HYP 29 of 262. The cost is the same as on fold 9:
  macro specificity 0.775 and macro precision 0.512. HYP precision is 0.299, so about 7 in 10 HYP flags
  are false alarms.
- **Max-F1 point** moves by at most 0.02 per metric from fold 9 (macro F1 0.750 against 0.757). HYP is
  still the weak class: sensitivity 0.584 at its max-F1 threshold.
- **Calibration transfers.** Macro ECE falls from 0.089 uncalibrated to 0.015 calibrated on fold 10, with
  every class between 0.012 and 0.016. The plan expected the calibrated ECE to come out above its fold-9
  value (0.016, measured on the fitting records). It did not: 0.015. As pre-registered, the calibrated
  intervals describe spread and are biased upward. For STTC (0.012 [0.012, 0.029]) and the macro
  (0.015 [0.018, 0.026]) the point value lies below its own interval.
- **The CNN beats both baselines.** The feature GBDT reaches macro AUROC 0.839 [0.827, 0.850] and AUPRC
  0.676 [0.654, 0.696]. The paired CNN-minus-GBDT difference is +0.086 [0.077, 0.097] AUROC and
  +0.142 [0.126, 0.161] AUPRC, so neither interval comes near 0. The prior baseline scores 0.500 and
  0.259 (its prevalence), by construction.
- **Seed stability** (fold 9, seeds 42, 43, 44, sample std): macro AUROC 0.9279 ± 0.0009, macro AUPRC
  0.8148 ± 0.0027. The release model is the seed-42 run.

## Expectations stated in the plan (section 10), against what was measured

| Expectation | Measured |
|---|---|
| High-sensitivity sensitivity near 0.90, on average slightly below; any class can land on either side | 3 of 4 classes slightly below (0.887 to 0.889), STTC 0.902; macro 0.892 |
| Ranking metrics near the validation values | macro AUROC 0.926 against 0.929; macro AUPRC 0.819 against 0.816 |
| Calibrated ECE above its fold-9 value | Not met: 0.015 against 0.016. Reported as measured |

## Full tables

Ranking metrics:

| Class | Prevalence | AUROC Test | AUROC val | AUPRC Test | AUPRC val |
|---|---|---|---|---|---|
| NORM | 0.446 | 0.947 [0.938, 0.955] | 0.947 | 0.929 [0.914, 0.942] | 0.931 |
| MI | 0.255 | 0.922 [0.908, 0.934] | 0.930 | 0.819 [0.788, 0.847] | 0.843 |
| STTC | 0.241 | 0.937 [0.925, 0.947] | 0.930 | 0.834 [0.800, 0.864] | 0.789 |
| CD | 0.230 | 0.922 [0.905, 0.939] | 0.928 | 0.841 [0.808, 0.869] | 0.847 |
| HYP | 0.121 | 0.901 [0.880, 0.921] | 0.909 | 0.670 [0.611, 0.718] | 0.671 |
| macro | 0.259 | 0.926 [0.918, 0.933] | 0.929 | 0.819 [0.802, 0.833] | 0.816 |
| macro_abnormal | 0.212 | 0.921 [0.912, 0.928] | 0.924 | 0.791 [0.771, 0.808] | 0.787 |

Max-F1 operating point:

| Class | Threshold | Sensitivity Test | Sens. val | Specificity Test | Spec. val | Precision Test | Prec. val | F1 Test | F1 val |
|---|---|---|---|---|---|---|---|---|---|
| NORM | 0.415 | 0.916 [0.900, 0.932] | 0.919 | 0.833 [0.809, 0.855] | 0.845 | 0.816 [0.791, 0.840] | 0.826 | 0.863 [0.846, 0.878] | 0.870 |
| MI | 0.367 | 0.776 [0.740, 0.811] | 0.793 | 0.896 [0.880, 0.910] | 0.903 | 0.719 [0.681, 0.752] | 0.734 | 0.747 [0.717, 0.774] | 0.762 |
| STTC | 0.373 | 0.812 [0.777, 0.846] | 0.807 | 0.900 [0.883, 0.915] | 0.897 | 0.721 [0.682, 0.759] | 0.718 | 0.764 [0.732, 0.791] | 0.760 |
| CD | 0.357 | 0.778 [0.741, 0.818] | 0.786 | 0.921 [0.907, 0.935] | 0.924 | 0.747 [0.706, 0.787] | 0.755 | 0.762 [0.731, 0.793] | 0.770 |
| HYP | 0.396 | 0.584 [0.528, 0.645] | 0.578 | 0.957 [0.948, 0.966] | 0.959 | 0.654 [0.588, 0.712] | 0.668 | 0.617 [0.567, 0.664] | 0.620 |
| macro | | 0.773 [0.755, 0.791] | 0.777 | 0.902 [0.895, 0.908] | 0.906 | 0.731 [0.713, 0.748] | 0.740 | 0.750 [0.735, 0.765] | 0.757 |
| macro_abnormal | | 0.738 [0.715, 0.759] | 0.741 | 0.919 [0.911, 0.926] | 0.921 | 0.710 [0.688, 0.730] | 0.719 | 0.722 [0.704, 0.738] | 0.728 |

High-sensitivity operating point (thresholds chosen for sensitivity >= 0.90 on fold 9):

| Class | Threshold | Sensitivity Test | Sens. val | Specificity Test | Spec. val | Precision Test | Prec. val | F1 Test | F1 val |
|---|---|---|---|---|---|---|---|---|---|
| MI | 0.181 | 0.887 [0.859, 0.914] | 0.900 | 0.786 [0.763, 0.806] | 0.801 | 0.587 [0.551, 0.620] | 0.604 | 0.706 [0.676, 0.732] | 0.723 |
| STTC | 0.219 | 0.902 [0.874, 0.927] | 0.902 | 0.811 [0.789, 0.831] | 0.807 | 0.603 [0.565, 0.638] | 0.604 | 0.723 [0.691, 0.749] | 0.723 |
| CD | 0.132 | 0.889 [0.861, 0.916] | 0.901 | 0.792 [0.772, 0.812] | 0.780 | 0.560 [0.524, 0.597] | 0.551 | 0.687 [0.655, 0.717] | 0.684 |
| HYP | 0.073 | 0.889 [0.848, 0.929] | 0.903 | 0.713 [0.688, 0.733] | 0.721 | 0.299 [0.265, 0.331] | 0.316 | 0.448 [0.407, 0.484] | 0.468 |
| macro_abnormal | | 0.892 [0.876, 0.907] | 0.901 | 0.775 [0.763, 0.787] | 0.777 | 0.512 [0.496, 0.528] | 0.519 | 0.641 [0.626, 0.655] | 0.650 |

Expected calibration error (10 bins; intervals describe spread, see section 5 of the plan):

| Class | Uncalibrated Test | Calibrated Test | Uncal. val | Cal. val (fitted there) |
|---|---|---|---|---|
| NORM | 0.036 [0.030, 0.052] | 0.016 [0.015, 0.034] | 0.040 | 0.018 |
| MI | 0.082 [0.071, 0.097] | 0.015 [0.014, 0.034] | 0.078 | 0.017 |
| STTC | 0.109 [0.097, 0.125] | 0.012 [0.012, 0.029] | 0.110 | 0.021 |
| CD | 0.107 [0.096, 0.122] | 0.016 [0.013, 0.031] | 0.110 | 0.014 |
| HYP | 0.113 [0.101, 0.126] | 0.014 [0.012, 0.029] | 0.109 | 0.012 |
| macro | 0.089 [0.086, 0.097] | 0.015 [0.018, 0.026] | 0.089 | 0.016 |

Baselines (ranking only; no thresholds were ever chosen for them):

| | Macro AUROC Test | Macro AUPRC Test | Macro AUROC val | Macro AUPRC val |
|---|---|---|---|---|
| Prior | 0.500 | 0.259 | 0.500 | 0.260 |
| Feature GBDT | 0.839 [0.827, 0.850] | 0.676 [0.654, 0.696] | 0.856 | 0.693 |
| CNN - GBDT (paired) | 0.086 [0.077, 0.097] | 0.142 [0.126, 0.161] | 0.073 | 0.123 |

## Notes

- From here on the model, the calibrators and the thresholds are frozen. Any further look at fold 10 is
  exploratory and must be labelled as such.
- In the ROC and PR figures, the orange marker's legend reads "Sensitivity ≥ 0.90 threshold". That names
  the threshold chosen on fold 9; on fold 10, three of the four classes sit just under 0.90 (table above).
  The label was left as it is: changing evaluation output after fold 10 was opened is allowed for bugs
  only.
