# CardioScreen: 12-lead ECG abnormality screening with a 1D CNN

Course mini project (CSET343, AI in Healthcare). A screening aid that estimates, for five diagnostic
superclasses (NORM, MI, STTC, CD, HYP), the probability that each is present in a 10 s 12-lead ECG.
**Screening aid. Not a diagnostic device. Not for clinical use.**

The build specification is [PROJECT_SPEC.md](PROJECT_SPEC.md). Every setting lives in
[config/default.yaml](config/default.yaml).

## Dataset

- **PTB-XL, version 1.0.3** (PhysioNet), 100 Hz records only, plus `ptbxl_database.csv` and `scp_statements.csv`.
- 21,799 records from 18,869 patients; each record is 10 s × 12 leads (1000 × 12 samples at 100 Hz).
- Licence: Creative Commons Attribution 4.0 International (CC BY 4.0). The data is not committed to git.
- Files are fetched from PhysioNet's AWS Open Data mirror (`s3://physionet-open/ptb-xl/1.0.3/`, listed on the
  PTB-XL project page), because physionet.org measured about 7 KB/s per connection from the development laptop.
  Every file is checked against PhysioNet's `SHA256SUMS.txt`, whose own SHA-256 is pinned in the config.

### Citation

- Wagner, P., Strodthoff, N., Bousseljot, R., Samek, W., & Schaeffter, T. (2022). *PTB-XL, a large publicly
  available electrocardiography dataset* (version 1.0.3). PhysioNet. https://doi.org/10.13026/kfzx-aw45
- Wagner, P., Strodthoff, N., Bousseljot, R.-D., Kreiseler, D., Lunze, F. I., Samek, W., & Schaeffter, T. (2020).
  PTB-XL, a large publicly available electrocardiography dataset. *Scientific Data*, 7, 154.
  https://doi.org/10.1038/s41597-020-0495-6
- Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource
  for complex physiologic signals. *Circulation*, 101(23), e215–e220.

## Setup (Windows 11, PowerShell)

Requires Python 3.11 and an NVIDIA driver that supports CUDA 12.6 (560 or newer).

```powershell
winget install --id Python.Python.3.11 --exact --scope user   # if Python 3.11 is not installed
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt          # torch comes from PyTorch's cu126 index
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

On an ASUS laptop with the GPU in Eco (iGPU-only) mode the RTX 3050 is powered off and CUDA reports `False`;
switch the GPU mode to Standard or Ultimate first.

## Data

```powershell
python -m src.data.download                # resumable; files already on disk are skipped
python -m src.data.download --verify-only  # counts records and patients, checks every record file exists
python -m src.data.inspect_record          # loads one record and writes reports/figures/sample_record.png
python -m src.data.prepare                 # labels, splits, train-only stats, then the one-time signal cache
python -m src.data.inspect_pipeline        # one batch from each DataLoader and the time to load a training epoch
pytest
```

### Labels and splits

- **Labels.** Each record's `scp_codes` statements with `diagnostic == 1` in `scp_statements.csv` are mapped to their
  `diagnostic_class`, as in PhysioNet's `example_physionet.py`. The result is multi-hot over NORM, MI, STTC, CD and HYP,
  because one ECG can show several. This reproduces PhysioNet's published counts exactly (NORM 9,514, MI 5,469,
  STTC 5,235, CD 4,898, HYP 2,649). 411 records carry only rhythm or form statements (mostly pacemaker, atrial
  fibrillation and flutter) and are dropped, leaving 21,388.
- **Splits.** PTB-XL's `strat_fold` keeps all of a patient's records in one fold: folds 1–8 train (17,084 records,
  14,823 patients), fold 9 validation (2,146 / 1,917), fold 10 test (2,158 / 1,877). `assert_no_patient_leakage`
  checks that no patient is in two splits.
- **Normalisation.** Per-lead mean and std come from the training records only and are saved with their folds and a hash
  of the exact `ecg_id`s used, so `tests/test_norm_stats.py` can recompute and compare them.

### Cache, Dataset and DataLoaders

- **Cache** (`src/data/cache.py`). All 21,799 records are read from WFDB once and saved to `data/cache/raw/`:
  `signals.npy` (float32, 21,799 × 12 × 1000, mV, about 1 GB, opened as a read-only memmap), `labels.npy`
  (21,799 × 5 multi-hot), `index.csv` (`ecg_id`, `patient_id`, `strat_fold`, `labelled`, `record_path`) and
  `meta.json`. The build takes about 30 s; later runs reuse it, and opening it checks `meta.json` against the config
  so a cache built with other settings fails loudly. `python -m src.data.cache --verify` re-hashes the signals.
  `norm_stats.json` records the cache's SHA-256, and `prepare` checks every training record against its cache row,
  so statistics cannot be paired with a different or stale cache.
- **Band-pass filter** (`src/data/preprocess.py`). 0.5–40 Hz zero-phase Butterworth, `preprocessing.bandpass.enabled`,
  off by default. It runs once per record before caching and before the normalisation statistics, and each setting has
  its own folder (`data/cache/bandpass_0.5-40hz_order2/`) with its own `norm_stats.json`, so both can be compared.
- **Dataset** (`src/data/dataset.py`). Serves the 21,388 labelled records of one split as a float32 (12, 1000) signal,
  z-scored per lead with the training statistics, and a float32 (5,) target for `BCEWithLogitsLoss`.
- **Augmentation** (`src/data/augment.py`, training split only, each behind its own flag): Gaussian noise with
  σ = 0.05 lead standard deviations, and a time shift of up to ±2 s with the gap set to zero (at least 8 s of real
  signal remain). Amplitude scaling (one factor in 0.8–1.2 for all 12 leads) is implemented but switched off, because
  it lowered HYP validation AUPRC (see [Training](#training-validation-only)). No time reversal and no sign inversion:
  both change what the ECG means clinically, so the label would no longer be true. Every transform draws its random
  numbers whether or not it is enabled, so a run with one transform off and the same seed sees the same noise and
  shifts as the full run.
- **Test-set lock.** `ECGDataset(cfg, "test")`, `make_loader(cfg, "test")` and `Cache.rows("test")` raise
  `TestSetLockedError` unless `allow_test=True` (the bool, not a truthy string) is passed. Only the one-time final
  evaluation and the demo API may pass it. A test fails if any module outside `src/data/` mentions `allow_test`,
  `test_folds` or `strat_fold`, except an explicit allowlist that starts empty.
- **Loading speed** (measured, batch 64, augmentation on, no model). One training epoch loads in 2.9 s with no worker
  processes, 1.7 s with 2 and 1.0 s with 4, after about 7 s to start the workers once. Each Windows worker
  re-imports CUDA PyTorch and commits about 1.8 GB of memory, so the config uses 2 workers for training and none for
  validation and test.

## Baselines (validation only)

```powershell
python -m src.models.baselines     # about 40 s; writes reports/baselines_val.json
python -m src.models.rpeak_check   # about 1 min; how the beat detector was chosen, on training records only
```

Both baselines are fit on the training split and scored on the validation split (fold 9). No setting is tuned against
labels or the validation split: the gradient boosting uses scikit-learn's defaults, and the beat detector was chosen on
training records with a label-free check. Every setting is in `config/default.yaml` under `baselines`. The test split
is never opened.

- **Prior baseline.** Every record gets each class's training-split frequency. One score for every record cannot rank
  anything, so its AUROC is exactly 0.5 and its AUPRC is exactly the class's validation prevalence: the floor.
- **Feature baseline.** 79 features per record, then one `HistGradientBoostingClassifier` per class (scikit-learn
  defaults, early stopping off because it would hold out random training records rather than patients):
  number of beats, heart rate and RR-interval mean, SD, min, max and RMSSD from lead II, plus mean, SD, min, max,
  skewness and kurtosis of each of the 12 leads (in mV).
- **Beat detector.** Pan–Tompkins style (5–15 Hz band-pass, derivative, squaring, 150 ms integration, T-wave slope rule,
  search-back, 0.2 s refractory period), with the threshold set from the median of the highest third of candidate peaks,
  so that large ventricular ectopic beats do not hide the normal ones. PTB-XL has no beat annotations, so detectors are
  compared by consistency: a heart beats once whatever the lead, so the lead II count should match an independent
  detection on leads I and V5. On 3,000 training records (`python -m src.models.rpeak_check`) it agrees within one beat
  in 96.2% of records, against 92.9% with the threshold taken from the record's maximum, 83.4% for wfdb's GQRS (misses
  beats) and 56.6% for wfdb's XQRS (marks P and T waves at 100 Hz). It still over-counts on very noisy traces. Tests pin
  its counts on training records counted by eye, including ones where the T-wave rule and the search-back each matter.

| Class | Val positives | Prevalence | Prior AUROC | Prior AUPRC | Features AUROC | Features AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| NORM | 955 | 0.445 | 0.500 | 0.445 | 0.889 | 0.856 |
| MI | 540 | 0.252 | 0.500 | 0.252 | 0.828 | 0.637 |
| STTC | 528 | 0.246 | 0.500 | 0.246 | 0.828 | 0.602 |
| CD | 495 | 0.231 | 0.500 | 0.231 | 0.850 | 0.742 |
| HYP | 268 | 0.125 | 0.500 | 0.125 | 0.886 | 0.628 |
| **Macro** | | 0.260 | 0.500 | 0.260 | **0.856** | **0.693** |

AUPRC must be read against prevalence, which is what a model with no skill scores: HYP's 0.628 is five times its
prevalence, NORM's 0.856 under twice.

## Training (validation only)

```powershell
python -m pytest tests/test_cuda.py               # the GPU is visible (Armoury Crate GPU Mode: Standard)
python -m src.models.resnet1d                     # layer shapes and the parameter count
python -m src.training.train --overfit            # sanity check, about 30 s
python -m src.training.train --name main          # full run, about 5 min on the RTX 3050 Laptop
python -m src.training.compare_runs <run A> <run B>   # paired comparison of two runs on the validation split
python -m src.training.summarize_comparisons --name <name> <compare json> ...   # the same comparison over several seeds
```

- **Model** (`src/models/resnet1d.py`). The spec's 1D ResNet: stem Conv1d(12→64, k=7, stride 2) → BN → ReLU →
  MaxPool(3, stride 2); four stages of two residual blocks (64, 128, 256, 512 channels; strides 1, 2, 2, 2); each block
  Conv(k=5) → BN → ReLU → Dropout(0.2) → Conv(k=5) → BN, plus the identity or a 1×1 convolution when the shape changes,
  then ReLU; global average pooling over time → Dropout(0.3) → Linear(512→5). **6,290,821 parameters.** The spec leaves
  two sizes open: the max pool uses ResNet's 3/2 and the in-block dropout Ribeiro et al.'s (2020) 0.2.
- **Loss and optimiser** (`src/training/train.py`). `BCEWithLogitsLoss` with `pos_weight` = negatives / positives per
  class in the training split (NORM 1.25, MI 2.90, STTC 3.08, CD 3.37, HYP 7.06); AdamW (lr 1e-3, weight decay 1e-2);
  linear warmup over 2 epochs, then cosine decay over the 50-epoch budget, updated every step; batch 64 (no
  out-of-memory on the 4 GB GPU).
- **Model selection.** Validation macro AUROC after every epoch; only the best epoch's weights are kept
  (`checkpoints/<run_id>/best.pt`, gitignored); training stops after 8 epochs without a new best.
- **Records.** Every run writes `reports/runs/<run_id>.csv` (per epoch: learning rate, train and validation loss,
  validation macro and per-class AUROC and AUPRC, with the run id, git commit and seed on every row) and
  `reports/runs/<run_id>.json` (effective config, overrides, commit, seed, pos_weight, environment, best epoch).
- **Reproducibility.** One seed drives the weights, shuffling, augmentation and dropout, and cuDNN and cuBLAS run in
  deterministic mode, so the same code, config and seed give the same weights (checked below).
- **Overfit check.** 50 training records, augmentation off, 300 steps: training loss 1.008 → 0.00004 (eval mode),
  all 250 labels right. The loop can fit data, so a poor full run would point at the data or the model, not the loop.

**Results** (fold 9, best epoch of each run; the test fold has not been opened):

| Class | Prevalence | Prior AUPRC | Features AUROC | Features AUPRC | ResNet1d AUROC | ResNet1d AUPRC |
|---|---:|---:|---:|---:|---:|---:|
| NORM | 0.445 | 0.445 | 0.889 | 0.856 | 0.947 | 0.931 |
| MI | 0.252 | 0.252 | 0.828 | 0.637 | 0.930 | 0.843 |
| STTC | 0.246 | 0.246 | 0.828 | 0.602 | 0.930 | 0.789 |
| CD | 0.231 | 0.231 | 0.850 | 0.742 | 0.928 | 0.847 |
| HYP | 0.125 | 0.125 | 0.886 | 0.628 | 0.909 | 0.671 |
| **Macro** | 0.260 | 0.260 | 0.856 | 0.693 | **0.929** | **0.816** |

ResNet1d is the run with the committed config (`20261017-215633_no-amp-scale`: best epoch 19 of 27, early-stopped).
The prior baseline's AUROC is 0.5 for every class. For reference, Strodthoff et al. (2021) report about 0.93 macro
AUROC for the best CNNs on this task, but on fold 10 (test), so the two numbers are not directly comparable.
The CNN beats the feature baseline in every class. Its smallest gain is in HYP (AUROC +0.02, AUPRC +0.04), the
feature baseline's best class, plausibly because per-lead amplitude statistics already capture much of the voltage
signal of hypertrophy.

**Amplitude-scaling ablation.** Pairs of runs that differ only in `augmentation.amplitude_scale.enabled`, repeated
with seeds 42, 43 and 44. Each pair is compared on fold 9 by `compare_runs`, which reloads both best checkpoints and
checks they reproduce their logged metrics; `summarize_comparisons` then checks that all six runs share the same
`src/` code and a clean tree, and that the runs of each arm differ only in the seed.

| Metric (fold 9) | Scaling on, mean ± std | Scaling off, mean ± std | Off − on, seeds 42 / 43 / 44 | Off − on, mean ± std | Off better |
|---|---:|---:|---:|---:|---:|
| HYP AUPRC | 0.6574 ± 0.0051 | 0.6688 ± 0.0030 | +0.0195 / +0.0037 / +0.0111 | +0.0114 ± 0.0079 | 3 of 3 |
| HYP AUROC | 0.9042 ± 0.0027 | 0.9068 ± 0.0031 | +0.0067 / +0.0011 / −0.0000 | +0.0026 ± 0.0036 | 2 of 3 |
| CD AUPRC | 0.8582 ± 0.0047 | 0.8525 ± 0.0084 | −0.0071 / −0.0088 / −0.0011 | −0.0057 ± 0.0041 | 0 of 3 |
| CD AUROC | 0.9327 ± 0.0018 | 0.9300 ± 0.0014 | −0.0023 / −0.0034 / −0.0025 | −0.0027 ± 0.0006 | 0 of 3 |
| MI AUPRC | 0.8289 ± 0.0060 | 0.8338 ± 0.0087 | +0.0099 / +0.0012 / +0.0035 | +0.0048 ± 0.0045 | 3 of 3 |
| Macro AUPRC | 0.8121 ± 0.0006 | 0.8148 ± 0.0027 | +0.0047 / −0.0004 / +0.0037 | +0.0027 ± 0.0027 | 2 of 3 |
| Macro AUROC | 0.9270 ± 0.0003 | 0.9279 ± 0.0009 | +0.0016 / +0.0002 / +0.0008 | +0.0009 ± 0.0007 | 3 of 3 |

**Decision rule**, fixed before it was applied: keep scaling off unless scaling on has the higher mean HYP AUPRC over
the three seeds. Mean ± sample std: HYP AUPRC on 0.6574 ± 0.0051, off 0.6688 ± 0.0030; macro AUROC on 0.9270 ± 0.0003,
off 0.9279 ± 0.0009. Scaling on is not higher, so scaling stays off in the config.

Scaling off raises HYP AUPRC in all three seeds (by +0.011 on average), a gap larger than the
seed-to-seed spread of either setting (std 0.003–0.005). The evidence is still modest. Only seed 42's patient-bootstrap interval excludes zero (seed 43: −0.0128 to
+0.0196; seed 44: −0.0062 to +0.0290), and three seeds out of three pointing the same
way would happen by chance one time in four if there were no effect (two-sided sign test). The setting was also chosen
on fold 9, so fold-9 numbers are slightly optimistic; fold 10 has not been used.

It is a trade-off, not a free gain: CD is worse without scaling in every seed (AUROC −0.0027,
AUPRC −0.0057), while MI improves slightly and macro AUROC barely moves
(+0.0009). A plausible reading: hypertrophy criteria such as Sokolow–Lyon are voltage
thresholds, so a random ×0.8–1.2 factor blurs the absolute amplitude HYP depends on, whereas conduction disturbances
are defined by timing (QRS width, PR interval), which scaling leaves intact, so learning to ignore amplitude may help
CD. If CD becomes the priority, the setting is worth revisiting.

Across the three seeds the committed config reaches validation macro AUROC 0.928 ± 0.001
and macro AUPRC 0.815 ± 0.003.

**Determinism check.** The seed-42 "scaling on" run, repeated from a later commit with the same `src/` code and the same
effective config (`20261017-223546_on-s42-repro`), reproduced all 27 epochs' metrics and the best checkpoint's weights
bit for bit.

## Evaluation (validation only)

```powershell
python -m src.evaluation.evaluate_val     # about 20 s; thresholds, metrics and figures for evaluation.model_run
```

`evaluate_val` reloads the best checkpoint of the committed-config run (`20261017-215633_no-amp-scale`, seed 42),
predicts fold 9 with deterministic GPU kernels and checks that the predictions reproduce the validation metrics the run
logged. It then chooses per-class thresholds on those predictions and writes `reports/thresholds.json`,
`reports/metrics_val.json` and the figures in `reports/figures/`. Fold 10 is not opened.

- **Max-F1 operating point.** Per class, the threshold with the highest validation F1 (exact ties go to the lower
  threshold, which keeps the same F1 with more sensitivity).
- **High-sensitivity operating point.** Per abnormal class (MI, STTC, CD, HYP), the highest threshold whose validation
  sensitivity is at least 0.90: the most specificity that still catches 9 in 10 cases. NORM gets none, because it is
  not an abnormality to be caught.
- A record is flagged when its probability is at least the threshold. Candidate thresholds are the distinct validation
  probabilities. The probabilities are uncalibrated sigmoid outputs, so the thresholds must be re-derived after
  temperature scaling.

| Class | AUROC | AUPRC | Max-F1: threshold | Sens | Spec | Prec | F1 | Sens ≥ 0.90: threshold | Sens | Spec | Prec | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| NORM | 0.947 | 0.931 | 0.378 | 0.919 | 0.845 | 0.826 | 0.870 | – | – | – | – | – |
| MI | 0.930 | 0.843 | 0.645 | 0.793 | 0.903 | 0.734 | 0.762 | 0.302 | 0.900 | 0.801 | 0.604 | 0.723 |
| STTC | 0.930 | 0.789 | 0.709 | 0.807 | 0.897 | 0.718 | 0.760 | 0.468 | 0.902 | 0.807 | 0.604 | 0.723 |
| CD | 0.928 | 0.847 | 0.754 | 0.786 | 0.924 | 0.755 | 0.770 | 0.328 | 0.901 | 0.780 | 0.551 | 0.684 |
| HYP | 0.909 | 0.671 | 0.850 | 0.578 | 0.959 | 0.668 | 0.620 | 0.154 | 0.903 | 0.721 | 0.316 | 0.468 |
| **Macro, abnormal** | 0.924 | 0.787 | | 0.741 | 0.921 | 0.719 | 0.728 | | 0.901 | 0.777 | 0.519 | 0.650 |

The thresholds were chosen on the same records they are scored on, so the thresholded numbers are optimistic; the
unbiased estimate is the one-time fold-10 evaluation with these thresholds frozen. The F1-optimal HYP threshold misses
42% of hypertrophy cases; holding sensitivity at 0.90 costs 524 false alarms among 1,878 records without HYP
(specificity 0.72, precision 0.32). Accuracy is not reported: with HYP in 12% of records, never flagging it is 88%
accurate.

Training curves (`reports/figures/training_curves.png`): validation loss is lowest at epoch 10 (0.547) and back at 0.548
at the saved epoch 19. After epoch 19 it climbs to 0.58–0.87 while training loss keeps falling and validation macro AUROC
slips only from 0.929 to 0.922. The model grows more confident without ranking better, the usual first sign of
overfitting; whether the saved epoch's probabilities are calibrated is measured next (reliability diagrams, ECE).
