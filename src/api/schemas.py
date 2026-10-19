"""Pydantic models of the API's JSON (PROJECT_SPEC.md section 6).

Field names follow the spec. "class" is a Python keyword, so ClassPrediction stores it as class_ and writes it as
"class". Every number in these responses comes from the release model's forward pass on the record in the request,
from the frozen thresholds and calibrators, or from the dataset's own files; none is computed by the API otherwise.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Health(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    model_version: str
    model_tag: str
    device: str


class RecordSummary(BaseModel):
    """A held-out test record: its id and shape only. No labels and no patient data."""

    id: str
    duration_s: int
    sampling_rate: int


class Signal(BaseModel):
    leads: list[str]
    sampling_rate: int
    units: Literal["mV"]
    data: list[list[float]]    # (leads, samples), before normalisation


class PredictRequest(BaseModel):
    """JSON body of POST /api/predict for a held-out test record. Uploads use a multipart form instead."""

    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str


class ClassPrediction(BaseModel):
    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    class_: str = Field(alias="class")
    label: str
    logit: float               # the model's raw output for the class
    probability: float         # calibrated: sigmoid(a * logit + b) with the class's frozen calibrator
    threshold: float | None    # the operating point's frozen threshold; None for the normal class
    flagged: bool | None       # probability >= threshold; None for the normal class, which is never flagged


class InputInfo(BaseModel):
    source: Literal["test_record", "upload"]
    record_id: str | None      # the ecg_id of a held-out test record; None for an upload
    sampling_rate_hz: float    # as recorded, before any resampling
    n_samples: int             # as recorded
    resampled: bool
    checksums_verified: bool | None   # upload: the .dat matched the header's checksums; None for a test record


class Prediction(BaseModel):
    predictions: list[ClassPrediction]   # every class, in the model's class order
    operating_point: str
    operating_point_rule: str
    flagged_count: int
    abnormal_count: int
    summary: str
    attention: list[float]     # Grad-CAM of attention_class, one value per sample, in [0, 1]
    attention_class: str       # the abnormal class with the highest calibrated probability
    attention_raw_peak: float  # the map's maximum before scaling to [0, 1]; 0: no time position raised the class score
    model_version: str
    inference_ms: float        # forward pass and calibration of this record
    attention_ms: float        # Grad-CAM's forward and backward pass
    input: InputInfo
    signal: Signal | None      # an upload's signal as the model read it (resampled, before normalisation); None for a
                               # test record, whose signal is GET /api/records/{id}/signal


class TruthClass(BaseModel):
    model_config = ConfigDict(validate_by_name=True, serialize_by_alias=True)

    class_: str = Field(alias="class")
    label: str
    present: bool


class Truth(BaseModel):
    id: str
    labels: list[str]          # the superclasses present, from PTB-XL's scp_codes (labels.min_likelihood)
    classes: list[TruthClass]
