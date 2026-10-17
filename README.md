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
python -m src.data.prepare                 # labels, patient-wise splits, train-only stats -> data/cache/norm_stats.json
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
