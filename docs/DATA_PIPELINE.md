# Data pipeline

> Educational course project. Screening aid, not a diagnostic device. Not for clinical use.

This document covers how PTB-XL is downloaded, labelled, split, cached, normalised and fed to the
model. For a short overview, see the [README](../README.md#dataset). Every setting mentioned here
lives in [`config/default.yaml`](../config/default.yaml).

## At a glance

| Item | Value |
|---|---|
| Dataset | PTB-XL, version 1.0.3 (PhysioNet) |
| Records used | 100 Hz versions only (`records100/`) |
| One record | 10 s × 12 leads at 100 Hz (100 samples per second) = 1000 samples per lead, in millivolts (mV). A lead is one of the 12 directions along which the heart's electrical activity is measured, all recorded at the same time |
| Records in the dataset | 21,799 from 18,869 patients |
| Records with a diagnostic label | 21,388 (411 dropped, see [Labels](#2-labels)) |
| Split | PTB-XL `strat_fold`: folds 1–8 train, 9 validation, 10 test |
| Lead order | I, II, III, aVR, aVL, aVF, V1–V6 |

## 1. Download and integrity checks

**Source.** Files come from PhysioNet's AWS Open Data mirror,
`https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/` (the HTTPS form of
`s3://physionet-open/ptb-xl/1.0.3/`). It is a plain HTTPS download, so no AWS account or tool is
needed. The mirror is the default because physionet.org measured about 7 KB/s per connection during
development, against about 2.5 MB/s for the mirror.

To download from physionet.org instead, set `download.source_url` to the value of
`dataset.base_url` and lower `download.workers` from 32 to 4. Both sources are verified in exactly
the same way.

**What is downloaded.** 43,600 files: `ptbxl_database.csv`, `scp_statements.csv` and the
43,598 files in `records100/`. Records are stored in WFDB, PhysioNet's waveform format: one `.hea`
header file and one `.dat` signal file per record. The checksum list `SHA256SUMS.txt` is downloaded
first. The 500 Hz records are not downloaded.

**How integrity is checked.**

- PhysioNet's `SHA256SUMS.txt` lists a SHA-256 hash for every file. The hash of that list itself is
  pinned in the config (`dataset.checksum_manifest_sha256`), so a mirror cannot supply a different list.
- Each file is written to `<name>.part`, hashed, and only renamed to its real name if the hash
  matches. A file present under `data/raw/` is therefore complete.
- An interrupted `.part` file is resumed with an HTTP Range request. Each file gets up to
  5 attempts with exponential back-off (2 s, 4 s, 8 s, ...).
- A lock file (`data/raw/.download.lock`) stops two downloads from writing the same files.

**Commands.**

| Command | What it does | Network | Hashes files |
|---|---|---|---|
| `python -m src.data.download` | Downloads every missing file, then runs the `--verify-only` checks | yes | new files only |
| `python -m src.data.download --verify-only` | Checks the record and patient counts (21,799 and 18,869) and that every `.hea` and `.dat` exists | no | no |
| `python -m src.data.download --rehash` | Re-hashes every file already on disk and re-downloads any bad copy | only for bad files | all files |

Files already on disk are skipped by presence, not re-hashed. Use `--rehash` if you suspect a
file was corrupted after download.

**Disk space** (measured after a full download and `prepare`):

| Folder | Size |
|---|---|
| `data/raw/` | about 551 MB (more on disk, because each small file occupies whole disk blocks) |
| `data/cache/raw/` | about 1.05 GB |
| `data/cache/bandpass_0.5-40hz_order2/` | about 1.05 GB, only if the band-pass filter is enabled |
| `checkpoints/<run_id>/` | about 25 MB per training run |

## 2. Labels

**Mapping** ([`src/data/labels.py`](../src/data/labels.py)). Each record's `scp_codes` column lists
SCP-ECG statements: standardised codes for ECG findings, from the Standard Communications Protocol
for computer-assisted electrocardiography. A statement counts when `scp_statements.csv` marks
it `diagnostic == 1`; it is then mapped to its `diagnostic_class`, one of the five superclasses.
This follows PhysioNet's own `example_physionet.py`.

**Multi-hot labels.** The result for each record is a vector of five 0/1 values in the order
NORM, MI, STTC, CD, HYP. Several can be 1 at once:

| Superclasses per labelled record | Records |
|---:|---:|
| 1 | 16,244 |
| 2 | 4,068 |
| 3 | 919 |
| 4 | 157 |

**Likelihood threshold.** `labels.min_likelihood` is 0, so every diagnostic statement counts
whatever its stated likelihood. This reproduces PhysioNet's published counts exactly
(NORM 9,514, MI 5,469, STTC 5,235, CD 4,898, HYP 2,649). A threshold of 15 was measured as an
alternative: it would remove 129 labels (127 of them STTC) and drop 13 more records.

**Dropped records.** 411 records carry only rhythm or form statements and no diagnostic one, so
they have no superclass and are dropped. In folds 1–9 they are mostly pacemaker (PACE, 257),
atrial fibrillation (AFIB, 55) and atrial flutter (AFLT, 46) records. They stay in the cache with
`labelled = False` and are never served to the model.

## 3. Patient-level splits

[`src/data/splits.py`](../src/data/splits.py) uses PTB-XL's own `strat_fold` column. PTB-XL
assigns all records of one patient to the same fold, so no patient appears in two splits.

| Split | Folds | Labelled records | Patients |
|---|---|---:|---:|
| Train | 1–8 | 17,084 | 14,823 |
| Validation | 9 | 2,146 | 1,917 |
| Test | 10 | 2,158 | 1,877 |
| Total | | 21,388 | 18,617 |

**Why this matters.** Some patients have several ECGs. If one of them were in training and another
in validation, the model could recognise the patient rather than the disease, and the validation
score would be too high. `assert_no_patient_leakage` checks that the three patient sets are
disjoint, and `tests/test_splits.py` runs it.

`train.py --set` refuses any `splits.*` key. A separate config file passed with `--config` is not
checked against the standard assignment, so keeping the folds fixed also relies on the project's
process.

## 4. Signal cache

Reading 21,799 WFDB files on every epoch would be slow, so
[`src/data/cache.py`](../src/data/cache.py) reads them once and stores them as NumPy arrays in
`data/cache/<variant>/` (`raw` when the band-pass filter is off).

| File | Contents |
|---|---|
| `signals.npy` | float32, shape (21,799, 12, 1000), in mV, rows in ascending `ecg_id` order; opened as a read-only memory map |
| `labels.npy` | uint8, shape (21,799, 5) multi-hot; all-zero rows are the unlabelled records |
| `index.csv` | `ecg_id`, `patient_id`, `strat_fold`, `labelled`, `record_path`, one row per array row |
| `meta.json` | the settings the cache was built with and the SHA-256 of the signal array; written last |
| `norm_stats.json` | the training-split normalisation statistics (section 5) |

**Integrity rules.**

- `meta.json` is written last, so its presence marks a complete build. A crash leaves no
  `meta.json`, and the next run rebuilds.
- Opening the cache compares `meta.json` with the current config. A cache built with other
  settings raises `CacheMismatchError`, which names each differing field.
- `python -m src.data.cache --verify` re-hashes `signals.npy` against `meta.json`.
  `python -m src.data.cache --rebuild` rebuilds it.
- `norm_stats.json` records the cache's SHA-256, and training runs record it too, so statistics,
  checkpoints and caches cannot be mixed up silently.

The WFDB reading step took about 30 s with 8 worker processes (`build_seconds` in `meta.json`);
writing and hashing the 1 GB array comes on top of that.

## 5. Normalisation

Each lead is z-scored: `(signal − mean) / std`, with one mean and one standard deviation per lead.

- The statistics come from the **training split only** (17,084 records). Computing them over all
  records would leak information about the validation and test data into training.
- The standard deviation is the population value (`norm_stats.std_ddof: 0`).
- The same statistics are reused for validation, test and any later inference.
- `norm_stats.json` stores the folds used and a hash of the exact `ecg_id`s, and
  `tests/test_norm_stats.py` recomputes the statistics and compares them.
- `prepare` also re-reads every training record from WFDB and checks it against its cache row.

For the raw signals, the per-lead standard deviations range from about 0.14 mV (aVR, aVL, aVF) to
0.33 mV (V2), and the means are close to 0.

## 6. Optional band-pass filter

[`src/data/preprocess.py`](../src/data/preprocess.py) implements a 0.5–40 Hz zero-phase
Butterworth filter (order 2 per pass; filtering forwards and backwards doubles the effective order).
It removes baseline wander and high-frequency noise.

- It is controlled by `preprocessing.bandpass.enabled` and is **off** by default.
- It runs once per record, before caching and before the normalisation statistics.
- Each setting gets its own cache folder, for example `data/cache/bandpass_0.5-40hz_order2/`,
  with its own `norm_stats.json`, so both variants can exist side by side.
- After switching it on, run `python -m src.data.prepare` again for that variant.

**Status:** no training run with the filter on has been reported yet, so its effect on the model
has not been measured.

## 7. Dataset and DataLoaders

[`src/data/dataset.py`](../src/data/dataset.py) serves the labelled records of one split.

- Each item is a float32 signal of shape (12, 1000), z-scored with the training statistics, and a
  float32 target of shape (5,) for `BCEWithLogitsLoss`.
- Only the training loader shuffles, using a `torch.Generator` seeded with `training.seed`.
- With batch 64 there are 267 training batches, 34 validation batches and 34 test batches per pass.
- Worker processes receive only the dataset's constructor arguments and rebuild it from the cache,
  so starting a worker does not copy the 1 GB array.
- The same seed gives the same batches, augmentation included, for a given
  `dataloader.num_workers`. A different worker count gives a different (but also reproducible)
  augmentation sequence.

## 8. Augmentation

Augmentation creates slightly altered copies of training records, so the model sees more
variety. It is applied to the **training split only**; validation and test are never augmented.
Each transform has its own flag in `augmentation`.

| Transform | Setting | Status | Why |
|---|---|---|---|
| Time shift | Up to ±2 s (±200 samples); vacated samples set to 0, which is the lead's training mean after z-scoring. At least 8 s of real signal remain | on | A finding can occur anywhere in the 10 s window |
| Gaussian noise | σ = 0.05 lead standard deviations (about 7–17 µV), independent per lead and sample | on | Robustness to recording noise; well below the 0.1 mV ST-deviation criterion |
| Amplitude scaling | One factor in 0.8–1.2 for all 12 leads | **off** | Did not improve HYP validation AUPRC under the decision rule; see [the experiment](EXPERIMENTS_AND_RESULTS.md#5-amplitude-scaling-experiment) |
| Time reversal, sign inversion | not implemented | never | Both change the clinical meaning of the waveform, so the label would no longer be true |

Details:

- One shift and one scale factor are shared by all 12 leads, because the leads are recorded at the
  same time and some are derived from others (for example III = II − I).
- The order is scaling, then noise, then shift, so the zero-filled gap stays exactly 0.
- Every transform draws its random numbers whether or not it is enabled. A run with one transform
  switched off therefore sees the same noise and shifts as the full run with the same seed, which
  keeps experiments comparable.

## 9. Test-set lock

Fold 10 must not influence any model decision before the final evaluation. A software lock
enforces this.

- `ECGDataset(cfg, "test")`, `make_loader(cfg, "test")` and `Cache.rows("test")` raise
  `TestSetLockedError` unless `allow_test=True` is passed (the boolean `True`, not a truthy string).
- `tests/test_dataset.py` fails if any module in `src/` outside `src/data/` mentions `allow_test`,
  `test_folds` or `strat_fold`, except for an allowlist (`MAY_UNLOCK_TEST`) that is currently empty.
- The intended callers are the one-time final evaluation and the demo API. **Neither exists yet.**
- Today `allow_test=True` is passed only by the tests and by `python -m src.data.inspect_pipeline`,
  which loads one test batch to print its shape (no model, no metrics).

**What "untouched" means here.** No model has been trained, tuned, selected or scored on fold 10,
and no metric has been computed from it. Its signals are still read when the cache is built and by
a few integrity tests, because the cache holds every record.

The lock is a guard rail, not a security boundary. `Cache.signals` holds all folds, and
`python -m src.data.inspect_record --ecg-id N` accepts any record, so the rule also depends on
following the project's process.

## 10. Loading speed

Measured on the development machine (Windows, batch 64, augmentation on, no model), as recorded in
the `dataloader` comment of the config:

| `num_workers` | Time to load one training epoch |
|---:|---:|
| 0 | 2.9 s |
| 2 | 1.7 s |
| 4 | 1.0 s |

Starting the worker processes adds about 7 s once, on top of the first epoch. On Windows
each worker re-imports PyTorch with CUDA and used about 1.8 GB of memory, so the config uses
2 workers for training and none for validation and test. To tune `dataloader.num_workers` on
another machine, run `python -m src.data.inspect_pipeline --workers 0 2 4`.

## Command summary

| Command | Purpose | Writes |
|---|---|---|
| `python -m src.data.download` | Download and verify PTB-XL | `data/raw/` |
| `python -m src.data.download --verify-only` | Offline completeness check | nothing |
| `python -m src.data.inspect_record [--ecg-id N]` | Plot lead II of one record (default `ecg_id` 1, training fold 3) on ECG-paper scale | `reports/figures/sample_record.png` |
| `python -m src.data.prepare` | Labels, splits and leakage check, then the cache (built if missing), then the training-only statistics | `data/cache/<variant>/` |
| `python -m src.data.cache [--verify] [--rebuild]` | Open, check or rebuild the cache | `data/cache/<variant>/` when building |
| `python -m src.data.inspect_pipeline [--workers N ...] [--epochs N]` | One batch per loader and the loading time per epoch | nothing |

All commands accept `--config PATH` to use another config file.
