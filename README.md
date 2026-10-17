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
- **Augmentation** (`src/data/augment.py`, training split only, each behind its own flag): one amplitude factor in
  0.8–1.2 for all 12 leads, Gaussian noise with σ = 0.05 lead standard deviations, and a time shift of up to ±2 s with
  the gap set to zero (at least 8 s of real signal remain). No time reversal and no sign inversion: both change what
  the ECG means clinically, so the label would no longer be true.
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

AUPRC must be read against prevalence, which is what a model with no skill scores: HYP's 0.629 is five times its
prevalence, NORM's 0.852 under twice.
