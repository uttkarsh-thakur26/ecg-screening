"""Uploaded WFDB records (a .hea header and its .dat samples) for POST /api/predict.

An upload is accepted only if the model can read it the way it read its training data:
- one single-segment record whose header names the uploaded .dat as the file of every signal;
- the 12 standard leads (names compared without case, any order; they are put in the training order), all in mV;
- a whole-number sampling rate and exactly dataset.record_duration_s seconds of samples. A record at another rate is
  resampled to dataset.sampling_rate_hz with scipy.signal.resample_poly (api.upload.resample_window and
  resample_padtype), which low-pass filters before decimating, so the result has exactly the model's 1000 samples;
- a .dat of exactly the size the header describes, in a WFDB format whose size can be checked, and whose per-signal
  checksums match the header's when the header has them (PTB-XL's do), so a .dat from another record is caught;
- no missing (NaN) samples, no flat lead (every sample equal: nothing was recorded on it), and no sample beyond the
  largest absolute amplitude of any training record (20 mV in PTB-XL's training folds). Such a record would still get
  probabilities from the model, but they would mean nothing; a value that large usually means a wrong gain in the header.
Anything else raises UploadError, whose message the API returns with a 400.

The accepted signal goes through src/data/preprocess.py at the model's rate, exactly as a cached training record did.
The two files exist on disk only inside a temporary folder that is deleted when read_upload returns or raises.
"""

from __future__ import annotations

import math
import re
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import wfdb
from scipy.signal import resample_poly

from src.data.preprocess import preprocess, to_cache_layout

# Bytes per sample of the WFDB formats whose file size follows from the header alone (WFDB signal(5)): 212 packs two
# 12-bit samples into three bytes. Formats with variable-length or multi-sample frames are not accepted.
BYTES_PER_SAMPLE = {"8": 1.0, "16": 2.0, "61": 2.0, "80": 1.0, "160": 2.0, "212": 1.5, "24": 3.0, "32": 4.0}
FILE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}")


class UploadError(ValueError):
    """The upload is not a record the model can read. The message says why, for the person who uploaded it."""


@dataclass(frozen=True)
class UploadedRecord:
    signal: np.ndarray          # (leads, samples) float32 in mV, training lead order, at the model's rate, preprocessed
    sampling_rate_hz: float     # as recorded
    n_samples: int              # as recorded
    resampled: bool
    checksums_verified: bool    # False only when the header carries no checksums


def check_file_name(name: str | None, suffix: str) -> str:
    """The uploaded file's own name (no folders), if it is a plain file name ending in suffix."""
    base = Path(name or "").name
    if base != (name or "") or not FILE_NAME.fullmatch(base) or not base.lower().endswith(suffix):
        raise UploadError(f"The {suffix} file must have a plain name ending in {suffix} (letters, digits, '_', '-' "
                          f"and '.'), got {name!r}")
    return base


def _same(values: Sequence[Any], what: str) -> Any:
    if len(set(values)) != 1:
        raise UploadError(f"Every signal must have the same {what}, the header has {list(values)}")
    return values[0]


def _check_header(h: Any, dat_name: str, dat_size: int, leads: Sequence[str], ds_cfg: dict[str, Any]) -> list[int]:
    """Raise UploadError unless the header describes a record the model can read. Returns the training-order columns."""
    if isinstance(h, wfdb.MultiRecord):
        raise UploadError("Multi-segment WFDB records are not supported; upload a single-segment record")
    if h.n_sig != len(leads):
        raise UploadError(f"The record has {h.n_sig} signals; the model needs the {len(leads)} standard leads "
                          f"({', '.join(leads)})")
    names = [str(s).strip().upper() for s in (h.sig_name or [])]
    wanted = [lead.upper() for lead in leads]
    if sorted(names) != sorted(wanted):
        missing = [lead for lead in leads if lead.upper() not in names]
        raise UploadError(f"The record's leads are {list(h.sig_name or [])}; the model needs exactly "
                          f"{', '.join(leads)}" + (f" (missing {', '.join(missing)})" if missing else ""))
    units = [str(u).strip() for u in (h.units or [])]
    if any(u.lower() != "mv" for u in units):
        raise UploadError(f"Every lead must be in mV, the header gives {units}")

    fs = h.fs
    if fs is None or not math.isfinite(fs) or fs <= 0 or not float(fs).is_integer():
        raise UploadError(f"The sampling rate must be a whole number of Hz, the header gives {fs}")
    if h.sig_len is None:
        raise UploadError("The header does not state the number of samples")
    duration = ds_cfg["record_duration_s"]
    if h.sig_len != duration * int(fs):
        raise UploadError(f"The record is {h.sig_len / fs:g} s long ({h.sig_len} samples at {fs:g} Hz); the model "
                          f"reads exactly {duration} s")

    file_names = [str(f) for f in (h.file_name or [])]
    if set(file_names) != {dat_name}:
        raise UploadError(f"The header names its samples file {sorted(set(file_names))}, but the uploaded .dat is "
                          f"{dat_name!r}; upload the .dat that belongs to this .hea")
    fmt = str(_same([str(f) for f in h.fmt], "format"))
    if fmt not in BYTES_PER_SAMPLE:
        raise UploadError(f"WFDB format {fmt} is not supported; supported formats: {', '.join(BYTES_PER_SAMPLE)}")
    if any(s not in (None, 1) for s in (h.samps_per_frame or [])):
        raise UploadError("Signals with more than one sample per frame are not supported")
    if any(s not in (None, 0) for s in (h.skew or [])):
        raise UploadError("Skewed signals are not supported")
    offset = _same([o or 0 for o in (h.byte_offset or [0] * h.n_sig)], "byte offset")
    expected = int(offset + math.ceil(h.sig_len * h.n_sig * BYTES_PER_SAMPLE[fmt]))
    if dat_size != expected:
        raise UploadError(f"The .dat file is {dat_size:,} bytes; the header describes {expected:,} "
                          f"({h.sig_len} samples x {h.n_sig} signals in format {fmt}). It is truncated or belongs to "
                          "another record")
    return [names.index(lead) for lead in wanted]


def resample(signal: np.ndarray, fs: int, target_fs: int, upload_cfg: dict[str, Any]) -> np.ndarray:
    """(samples, leads) at fs Hz -> (samples * target_fs / fs, leads) at target_fs Hz; unchanged when the rates agree."""
    if fs == target_fs:
        return signal
    ratio = Fraction(target_fs, fs)
    return resample_poly(signal, ratio.numerator, ratio.denominator, axis=0,
                         window=tuple(upload_cfg["resample_window"]), padtype=upload_cfg["resample_padtype"])


def check_plausible(signal: np.ndarray, leads: Sequence[str], max_abs_mv: float) -> None:
    """(samples, leads) in mV, training lead order: every lead carries a signal, none beyond the training range."""
    flat = [lead for lead, ptp in zip(leads, np.ptp(signal, axis=0)) if ptp == 0]
    if flat:
        raise UploadError(f"Lead(s) {', '.join(flat)} are flat lines (every sample has the same value): nothing was "
                          "recorded on them")
    peak = np.abs(signal).max(axis=0)
    if peak.max() > max_abs_mv:
        lead = leads[int(np.argmax(peak))]
        raise UploadError(f"Lead {lead} reaches {peak.max():,.2f} mV; no training record goes beyond "
                          f"{max_abs_mv:.2f} mV. The gain or units in the header are probably wrong")


def read_upload(hea: tuple[str | None, bytes], dat: tuple[str | None, bytes], leads: Sequence[str],
                ds_cfg: dict[str, Any], preprocessing_cfg: dict[str, Any], upload_cfg: dict[str, Any],
                max_abs_mv: float, tmp_root: Path | None) -> UploadedRecord:
    """Validate an uploaded (name, bytes) .hea and .dat pair and return the signal the model reads."""
    hea_name, dat_name = check_file_name(hea[0], ".hea"), check_file_name(dat[0], ".dat")
    if tmp_root is not None and not Path(tmp_root).is_dir():
        raise FileNotFoundError(f"api.upload.tmp_dir {tmp_root} does not exist")
    with tempfile.TemporaryDirectory(prefix="ecg-upload-", dir=tmp_root) as tmp:
        folder = Path(tmp)
        stem = str(folder / hea_name[: -len(".hea")])
        Path(stem + ".hea").write_bytes(hea[1])  # wfdb appends a lower-case ".hea", whatever the upload's case
        (folder / dat_name).write_bytes(dat[1])  # under the name the header must give for it
        try:
            header = wfdb.rdheader(stem)
        except Exception as exc:  # wfdb raises several types for a malformed header; all mean the same to the user
            raise UploadError(f"The .hea file is not a readable WFDB header: {exc}") from exc
        columns = _check_header(header, dat_name, len(dat[1]), leads, ds_cfg)
        try:
            digital = wfdb.rdrecord(stem, physical=False)
            signal, _ = wfdb.rdsamp(stem)  # physical units, as src/data/records.py read every training record
        except Exception as exc:
            raise UploadError(f"The samples could not be read with this header: {exc}") from exc

    stored = list(header.checksum or [])
    verified = len(stored) == header.n_sig and all(c is not None for c in stored)
    if stored:
        computed = digital.calc_checksum()
        bad = [header.sig_name[i] for i, c in enumerate(stored) if c is not None and int(c) != int(computed[i])]
        if bad:
            raise UploadError(f"The samples do not match the header's checksums for lead(s) {', '.join(bad)}; upload "
                              "the .dat that belongs to this .hea")
    if signal.shape != (header.sig_len, header.n_sig):
        raise UploadError(f"Read {signal.shape} samples, the header describes ({header.sig_len}, {header.n_sig})")
    if not np.isfinite(signal).all():
        raise UploadError("The record has missing samples (NaN); the model needs every sample of every lead")
    signal = signal[:, columns]
    check_plausible(signal, leads, max_abs_mv)

    fs, target_fs = int(header.fs), ds_cfg["sampling_rate_hz"]
    n_samples = target_fs * ds_cfg["record_duration_s"]
    signal = resample(signal, fs, target_fs, upload_cfg)
    if signal.shape != (n_samples, len(leads)):
        raise RuntimeError(f"Resampling gave {signal.shape} samples, expected ({n_samples}, {len(leads)})")
    x = to_cache_layout(preprocess(signal, target_fs, preprocessing_cfg))
    return UploadedRecord(signal=x, sampling_rate_hz=float(header.fs), n_samples=int(header.sig_len),
                          resampled=fs != target_fs, checksums_verified=verified)
