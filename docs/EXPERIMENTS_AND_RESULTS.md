# Experiments and results (validation only)

> Educational course project. Screening aid, not a diagnostic device. Not for clinical use; these
> results must not inform any medical decision.

This document gives the full results behind the summary in the
[README](../README.md#results-validation-only): the two baselines, the 1D ResNet, the
amplitude-scaling experiment and the determinism check.

> **All numbers here are on the validation split (fold 9: 2,146 records, 1,917 patients).** The
> test split (fold 10) was never used to train, tune or select anything. The epoch, the
> augmentation setting and the thresholds were all chosen on fold 9, so fold-9 scores are
> slightly optimistic. The one-time fold-10 result is in [reports/TEST_RUN.md](../reports/TEST_RUN.md).

## How to read AUROC and AUPRC

Each class is scored as its own yes/no problem, because a record can have several classes.

- **AUROC** (area under the receiver operating characteristic curve) is the chance that a randomly chosen record *with* the class
  gets a higher score than a randomly chosen record *without* it. 0.5 is guessing, 1.0 is perfect,
  whatever the class balance.
- **AUPRC** (area under the precision–recall curve, computed as average precision) summarises
  precision across all recall levels. A model with no skill scores the class's **prevalence** (its
  share of positive records), not 0.5. So an AUPRC must always be read next to the prevalence:
  0.63 would be only modestly above no skill (0.445) for a class present in 45% of records, but it
  is about five times no skill for one present in 12%.
- **Macro** values are the unweighted mean over the five classes, so the rare HYP counts as much as
  the common NORM.

## 1. Baselines

Both baselines are fit on the training split (folds 1–8) and scored on fold 9 by
`python -m src.models.baselines`, which writes [`reports/baselines_val.json`](../reports/baselines_val.json).
No setting was tuned on labels or on the validation split.

### Prior baseline

Every record gets each class's training-split frequency as its score. One score for every record
cannot rank anything, so its AUROC is exactly 0.5 and its AUPRC is exactly the class's validation
prevalence. It is the floor any real model must beat.

### Feature baseline

79 hand-crafted features per record, then one gradient-boosted tree classifier per class
(scikit-learn's `HistGradientBoostingClassifier`).

| Feature group | Features | Count |
|---|---|---:|
| Rhythm, from lead II beats | number of beats, heart rate, RR-interval mean, standard deviation, minimum, maximum and RMSSD | 7 |
| Amplitude, per lead | mean, standard deviation, minimum, maximum, skewness and kurtosis of each of the 12 leads, in mV | 72 |

An RR interval is the time between two consecutive heartbeats (R peaks). RMSSD is the root mean
square of successive differences between RR intervals, a short-term measure of heart-rate
variability. Skewness and kurtosis describe the shape of each lead's value distribution.

- The classifier uses scikit-learn's default settings (100 iterations, 31 leaves per tree,
  learning rate 0.1), written out in the `baselines.gradient_boosting` section of the config.
- The one exception is early stopping, which is **off**. scikit-learn's version would hold out a
  random 10% of training *records*, not patients, and all tuning belongs to the validation split.
- No class weighting is used, unlike the CNN's `pos_weight`.
- Features are not z-scored: tree splits do not depend on the scale of a feature.
- Records with fewer than two RR intervals get missing RR features, which the classifier handles
  natively. None of the training or validation records needed this.

### Beat detector

The rhythm features need the time of each heartbeat (the R peak). PTB-XL has no beat annotations,
so the project implements a Pan–Tompkins style QRS detector (Pan & Tompkins, 1985) on lead II:

1. Band-pass filter at 5–15 Hz (the QRS energy band), zero-phase.
2. Differentiate, square, and integrate over a 150 ms moving window.
3. Set the detection threshold to 0.3 × the median height of the highest third of candidate peaks.
   Using a median keeps one or two large ectopic (premature, abnormally shaped) beats from raising
   the threshold above the normal beats.
4. Apply Pan and Tompkins' T-wave rule (a small-slope candidate within 0.36 s of a beat is a
   T wave) and search-back (a long gap is searched again at half the threshold).
5. Enforce a 0.2 s refractory period between beats.

All settings are in `baselines.rpeaks` in the config.

**How the detector was chosen.** Without annotations, detectors were compared by consistency: a
heart beats once whatever the lead, so the lead II count should match independent detections on
leads I and V5. `python -m src.models.rpeak_check` runs this check on 3,000 random training records
(no labels, no validation data) and prints the share of records whose lead II count is within one
beat of **both** lead I and lead V5:

| Detector variant | Agreement |
|---|---:|
| This detector | 96.2% |
| Without the T-wave rule | 95.7% |
| Without the search-back | 95.6% |
| Threshold from the record's maximum instead of the median | 92.9% |
| wfdb GQRS | 83.4% (it misses beats) |
| wfdb XQRS | 56.6% (it marks P and T waves as beats at 100 Hz) |

`rpeak_check` prints this table but does not save it; the values are recorded in the config
comments.

In the saved baseline report, agreement over all training records is 96.1% (95.3% on validation).
Some records are still over-counted: the lead II beat count per record has a median of 12 but a 99th
percentile of 23 and a maximum of 37 in the training split.

`tests/test_baselines.py` pins the detector's counts on training records counted by eye, including
records where the T-wave rule and the search-back each matter.

### Baseline results

| Class | Val. positives | Prevalence | Prior AUROC | Prior AUPRC | Features AUROC | Features AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| NORM | 955 | 0.445 | 0.500 | 0.445 | 0.889 | 0.856 |
| MI | 540 | 0.252 | 0.500 | 0.252 | 0.828 | 0.637 |
| STTC | 528 | 0.246 | 0.500 | 0.246 | 0.828 | 0.602 |
| CD | 495 | 0.231 | 0.500 | 0.231 | 0.850 | 0.742 |
| HYP | 268 | 0.125 | 0.500 | 0.125 | 0.886 | 0.628 |
| **Macro** | | 0.260 | 0.500 | 0.260 | **0.856** | **0.693** |

HYP's feature-baseline AUPRC of 0.628 is five times its prevalence, while NORM's 0.856 is less than
twice its prevalence. This ratio is capped at 1 ÷ prevalence (8.0 for HYP, 2.25 for NORM), so it
always favours rare classes and should not be used to rank classes against each other.

## 2. 1D ResNet results

The 1D ResNet is the committed-config run `20261017-215633_no-amp-scale` (seed 42, best epoch 19 of
27, stopped early).

| Class | Prevalence | Features AUROC | Features AUPRC | ResNet AUROC | ResNet AUPRC |
|---|---:|---:|---:|---:|---:|
| NORM | 0.445 | 0.889 | 0.856 | 0.947 | 0.931 |
| MI | 0.252 | 0.828 | 0.637 | 0.930 | 0.843 |
| STTC | 0.246 | 0.828 | 0.602 | 0.930 | 0.789 |
| CD | 0.231 | 0.850 | 0.742 | 0.928 | 0.847 |
| HYP | 0.125 | 0.886 | 0.628 | 0.909 | 0.671 |
| **Macro** | 0.260 | 0.856 | 0.693 | **0.929** | **0.816** |

- The CNN beats the feature baseline in every class.
- The largest gains are in MI (AUPRC +0.21) and STTC (+0.19).
- The smallest gain is in HYP (AUROC +0.02, AUPRC +0.04). The reason has not been established. One
  untested hypothesis is that per-lead amplitude statistics already capture much of the voltage
  signal of hypertrophy.
- For reference, Strodthoff et al. (2021) report about 0.93 macro AUROC for the best CNNs on this
  task. Their figure is on fold 10 (test), so it is not directly comparable with these fold-9 numbers.

### Three seeds of the committed config

The committed config was trained with seeds 42, 43 and 44 (as the "scaling off" arm of the
experiment below). Mean ± sample standard deviation over the three runs:

| Metric (fold 9) | Mean ± std |
|---|---|
| Macro AUROC | 0.928 ± 0.001 |
| Macro AUPRC | 0.815 ± 0.003 |

Seed 42, the evaluated model, is slightly above the three-seed mean.

## 3. Training curves

See [MODEL_AND_TRAINING.md](MODEL_AND_TRAINING.md#9-training-curves).

## 4. Determinism check

See [MODEL_AND_TRAINING.md](MODEL_AND_TRAINING.md#7-reproducibility): a repeated seed-42 run
reproduced every epoch's metrics and the best checkpoint's weights bit for bit.

## 5. Amplitude-scaling experiment

**Question.** Does random amplitude scaling (multiplying all 12 leads by one factor between 0.8 and
1.2) help the model?

**Setup.** Pairs of runs that differ only in `augmentation.amplitude_scale.enabled`, repeated with
seeds 42, 43 and 44 (six runs).

- `python -m src.training.compare_runs <run A> <run B>` compares one pair on fold 9. It reloads both
  best checkpoints, checks that they reproduce their logged macro AUROC and AUPRC, and computes a
  paired bootstrap: 2,000 resamples of validation *patients* (all of a patient's records together),
  95% intervals.
- `python -m src.training.summarize_comparisons` combines the three pairs. It first checks that all
  six runs share the same `src/` code and a clean git tree, that each arm's runs differ only in the
  seed, and that every pair changes the same single setting.
- Outputs: `reports/runs/compare_*.json` and
  [`reports/runs/summary_amplitude_scale_seeds.json`](../reports/runs/summary_amplitude_scale_seeds.json).

**Decision rule.** Keep scaling off unless scaling on has the higher mean HYP AUPRC over the three
seeds. The rule is recorded in the config (commit `cb20e44`).

**History.** The git history shows the order of events:

1. Commit `3a6eccf`: scaling was switched off after the seed-42 pair alone, keeping the better of
   two same-seed runs on fold 9.
2. Commit `b2a10fe`: the seed-43 and seed-44 pairs were added.
3. Commit `cb20e44`: the decision rule was written into the config.

The repository alone therefore cannot show that the rule was fixed before the results were seen.
The evaluated seed-42 model is the better of two runs on fold 9, which adds a little optimism to its
fold-9 scores.

### Results

Off − on is positive when turning scaling off helped. Mean ± sample standard deviation over seeds.

| Metric (fold 9) | Scaling on | Scaling off | Off − on, seeds 42 / 43 / 44 | Off − on, mean ± std | Off better |
|---|---:|---:|---|---:|---:|
| **HYP AUPRC** | 0.6574 ± 0.0051 | 0.6688 ± 0.0030 | +0.0195 / +0.0037 / +0.0111 | +0.0114 ± 0.0079 | 3 of 3 |
| HYP AUROC | 0.9042 ± 0.0027 | 0.9068 ± 0.0031 | +0.0067 / +0.0011 / −0.0000 | +0.0026 ± 0.0036 | 2 of 3 |
| CD AUPRC | 0.8582 ± 0.0047 | 0.8525 ± 0.0084 | −0.0071 / −0.0088 / −0.0011 | −0.0057 ± 0.0041 | 0 of 3 |
| CD AUROC | 0.9327 ± 0.0018 | 0.9300 ± 0.0014 | −0.0023 / −0.0034 / −0.0025 | −0.0027 ± 0.0006 | 0 of 3 |
| MI AUPRC | 0.8289 ± 0.0060 | 0.8338 ± 0.0087 | +0.0099 / +0.0012 / +0.0035 | +0.0048 ± 0.0045 | 3 of 3 |
| MI AUROC | 0.9235 ± 0.0027 | 0.9262 ± 0.0038 | +0.0038 / +0.0026 / +0.0016 | +0.0027 ± 0.0011 | 3 of 3 |
| STTC AUPRC | 0.7905 ± 0.0032 | 0.7901 ± 0.0020 | −0.0030 / +0.0026 / −0.0007 | −0.0004 ± 0.0028 | 1 of 3 |
| STTC AUROC | 0.9309 ± 0.0016 | 0.9308 ± 0.0014 | −0.0020 / +0.0010 / +0.0008 | −0.0001 ± 0.0017 | 2 of 3 |
| NORM AUPRC | 0.9256 ± 0.0028 | 0.9287 ± 0.0059 | +0.0040 / −0.0005 / +0.0057 | +0.0031 ± 0.0032 | 2 of 3 |
| NORM AUROC | 0.9439 ± 0.0017 | 0.9458 ± 0.0034 | +0.0019 / −0.0002 / +0.0041 | +0.0019 ± 0.0022 | 2 of 3 |
| Macro AUPRC | 0.8121 ± 0.0006 | 0.8148 ± 0.0027 | +0.0047 / −0.0004 / +0.0037 | +0.0027 ± 0.0027 | 2 of 3 |
| Macro AUROC | 0.9270 ± 0.0003 | 0.9279 ± 0.0009 | +0.0016 / +0.0002 / +0.0008 | +0.0009 ± 0.0007 | 3 of 3 |

**Decision.** Scaling on does not have the higher mean HYP AUPRC (0.6574 against 0.6688), so
scaling stays off in the config.

### How strong is the evidence?

- Turning scaling off raised HYP AUPRC in all three seeds, by +0.011 on average. That gap is larger
  than the seed-to-seed spread of either setting (std 0.003–0.005).
- The evidence is still modest. For HYP AUPRC, only seed 42's patient-bootstrap interval excludes
  zero, and only just (+0.0000 to +0.0400). Seed 43 gives −0.0128 to +0.0196 and seed 44 −0.0062 to
  +0.0290.
- Three seeds out of three pointing the same way would happen by chance one time in four even with
  no real effect (two-sided sign test).
- The bootstrap intervals cover the sampling of validation patients, not training noise; the
  spread across seeds covers training noise.
- The setting was chosen on fold 9, so fold-9 numbers are slightly optimistic.

### The trade-off

It is not a free gain. CD is worse without scaling in every seed (AUROC −0.0027, AUPRC −0.0057),
MI improves slightly, and macro AUROC barely moves (+0.0009).

A plausible reading, not tested here:

- Hypertrophy criteria such as Sokolow–Lyon (a voltage criterion for left ventricular hypertrophy)
  are voltage thresholds, so a random ×0.8–1.2 factor blurs the absolute amplitude that HYP depends
  on.
- Conduction disturbances are defined by timing (QRS width, PR interval), which scaling leaves
  intact, so learning to ignore amplitude may help CD.

If CD becomes the priority, the setting is worth revisiting.

## 6. Not yet measured

| Item | Status |
|---|---|
| Band-pass filter (0.5–40 Hz) on vs off | Implemented and cached, no training run reported |

## References

- Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE Transactions on
  Biomedical Engineering*, BME-32(3), 230–236.
- Strodthoff, N., Wagner, P., Schaeffter, T., & Samek, W. (2021). Deep learning for ECG analysis:
  Benchmarks and insights from PTB-XL. *IEEE Journal of Biomedical and Health Informatics*, 25(5),
  1519–1528.
