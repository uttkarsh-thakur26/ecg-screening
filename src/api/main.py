"""FastAPI backend (PROJECT_SPEC.md section 6): the frozen release model and the held-out test records, on localhost.

    python -m src.api.main              # serves on api.host:api.port from config/default.yaml
    uvicorn src.api.main:app            # the same app

The release model, its calibrators and thresholds are loaded and checked once, at startup (src/api/service.py). If a
SHA-256 in reports/release.json does not match, or any other check fails, startup raises and the server does not run.

    GET  /api/health                 {"status", "model_loaded", "model_version", ...}
    GET  /api/records                the held-out test records: [{"id", "duration_s", "sampling_rate"}], ids only
    GET  /api/records/{id}/signal    the 12 x 1000 signal in mV, before normalisation
    POST /api/predict                {"record_id": "<id>"} as JSON, or a multipart form with files "hea" and "dat"
    GET  /api/records/{id}/truth     the record's labels, only after POST /api/predict for that record (else 409)
    GET  /api/metrics                reports/metrics.json, byte for byte

Each class in a prediction gets its calibrated probability; the abnormal classes also get the threshold of
api.operating_point and a flag (probability >= threshold). The normal class has no threshold and is never flagged.
Grad-CAM is computed for the abnormal class with the highest calibrated probability. Request bodies are capped at
api.upload.max_request_bytes (413 above it). The browser may call the API only from api.cors_origins.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import ValidationError
from starlette.datastructures import Headers, UploadFile
from starlette.formparsers import MultiPartException, MultiPartParser

from src.api.schemas import (ClassPrediction, Health, InputInfo, Prediction, PredictRequest, RecordSummary, Signal,
                             Truth, TruthClass)
from src.api.service import Inference, ReleaseService, TruthLockedError, UnknownRecordError
from src.api.upload import UploadError, read_upload
from src.config import load_config, project_path

log = logging.getLogger("uvicorn.error")
RECORD_ID = re.compile(r"[1-9][0-9]{0,11}")  # an ecg_id in canonical form: no sign, no leading zeros
UPLOAD_FIELDS = ("hea", "dat")

PREDICT_OPENAPI = {"requestBody": {"required": True, "content": {
    "application/json": {"schema": PredictRequest.model_json_schema()},
    "multipart/form-data": {"schema": {"type": "object", "required": list(UPLOAD_FIELDS), "properties": {
        "hea": {"type": "string", "format": "binary", "description": "WFDB header (.hea)"},
        "dat": {"type": "string", "format": "binary", "description": "WFDB samples (.dat) named in the header"}}}},
}}}


def float32_list(a: np.ndarray) -> list[list[float]]:
    """A float32 array as nested lists of the shortest decimals that read back as exactly the same float32 values."""
    return np.asarray(a, dtype=np.float32).astype(str).astype(np.float64).tolist()


async def read_capped_body(request: Request, limit: int) -> bytes:
    """The request body, refusing (413) anything over limit bytes, whether or not it declares its length."""
    declared = request.headers.get("content-length")
    if declared is not None:
        if not declared.isdigit():
            raise HTTPException(400, f"Content-Length {declared!r} is not a number of bytes")
        if int(declared) > limit:
            raise HTTPException(413, f"The request is {int(declared):,} bytes; uploads are limited to {limit:,} bytes "
                                     "(api.upload.max_request_bytes)")
    body = bytearray()
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(413, f"The request is over the {limit:,}-byte limit (api.upload.max_request_bytes)")
    return bytes(body)


def parse_record_id(service: ReleaseService, record_id: str) -> int:
    if not RECORD_ID.fullmatch(record_id) or not service.has_record(int(record_id)):
        raise HTTPException(404, f"Record {record_id!r} is not one of the {len(service.record_ids):,} held-out test "
                                 "records this demo serves; GET /api/records lists them")
    return int(record_id)


async def upload_files(headers: Headers, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    """{"hea": (file name, bytes), "dat": (file name, bytes)} from a multipart body already read into memory."""
    async def stream() -> AsyncIterator[bytes]:
        yield body

    parser = MultiPartParser(headers, stream(), max_files=len(UPLOAD_FIELDS), max_fields=0)
    try:
        form = await parser.parse()
    except MultiPartException as exc:
        raise HTTPException(400, f"Could not read the upload ({exc.message}). Send a multipart form with exactly two "
                                 "files: 'hea' (the .hea header) and 'dat' (its .dat samples).") from exc
    try:
        if sorted(form.keys()) != sorted(UPLOAD_FIELDS) or len(form.multi_items()) != len(UPLOAD_FIELDS):
            raise HTTPException(400, f"Send exactly two files, in form fields 'hea' and 'dat'; got fields "
                                     f"{[k for k, _ in form.multi_items()]}")
        files = {}
        for key in UPLOAD_FIELDS:
            item = form[key]
            if not isinstance(item, UploadFile):
                raise HTTPException(400, f"Form field {key!r} must be a file")
            files[key] = (item.filename, await item.read())
        return files
    finally:
        await form.close()


def prediction(service: ReleaseService, result: Inference, info: InputInfo, signal: Signal | None) -> Prediction:
    rows = []
    for c, logit, p in zip(service.classes, result.logits, result.probabilities):
        threshold = service.thresholds.get(c)
        rows.append(ClassPrediction(class_=c, label=service.descriptions[c], logit=float(logit), probability=float(p),
                                    threshold=threshold, flagged=None if threshold is None else bool(p >= threshold)))
    flagged = sum(bool(r.flagged) for r in rows)
    n_abnormal = len(service.abnormal)
    return Prediction(
        predictions=rows, operating_point=service.operating_point, operating_point_rule=service.operating_point_rule,
        flagged_count=flagged, abnormal_count=n_abnormal, summary=f"{flagged} of {n_abnormal} abnormal categories flagged",
        attention=result.attention.tolist(), attention_class=result.attention_class,
        attention_raw_peak=result.attention_raw_peak, model_version=service.model_version,
        inference_ms=round(result.inference_ms, 2), attention_ms=round(result.attention_ms, 2), input=info,
        signal=signal)


def create_app(cfg: dict[str, Any]) -> FastAPI:
    api = cfg["api"]
    upload_cfg = api["upload"]
    limit = int(upload_cfg["max_request_bytes"])
    tmp_root = project_path(upload_cfg["tmp_dir"]) if upload_cfg["tmp_dir"] is not None else None

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if limit > MultiPartParser.spool_max_size:
            raise ValueError(f"api.upload.max_request_bytes ({limit:,}) is above Starlette's in-memory spool size "
                             f"({MultiPartParser.spool_max_size:,}); larger uploads would be written to disk")
        service = ReleaseService(cfg)  # raises on any mismatch, so the server never starts with other artifacts
        app.state.service = service
        log.info("Release model v%s (%s, sha256 %s) on %s; %s held-out test records", service.model_version,
                 service.model_tag, service.manifest["release_checkpoint"]["sha256"][:12], service.device,
                 f"{len(service.record_ids):,}")
        yield

    app = FastAPI(title="CardioScreen API", lifespan=lifespan,
                  description="Screening aid on PTB-XL with a frozen 1D CNN. Not a diagnostic device. Not for "
                              "clinical use.")
    app.add_middleware(CORSMiddleware, allow_origins=list(api["cors_origins"]), allow_methods=["GET", "POST"],
                       allow_headers=["Content-Type"], allow_credentials=False)

    def service_of(request: Request) -> ReleaseService:
        return request.app.state.service

    @app.get("/api/health", response_model=Health)
    def health(request: Request) -> Health:
        s = service_of(request)
        # True by construction: the app serves requests only after the release model loaded and passed every check.
        return Health(status="ok", model_loaded=s.model is not None, model_version=s.model_version,
                      model_tag=s.model_tag, device=str(s.device))

    @app.get("/api/records", response_model=list[RecordSummary])
    def records(request: Request) -> list[RecordSummary]:
        s = service_of(request)
        return [RecordSummary(id=str(e), duration_s=s.duration_s, sampling_rate=s.sampling_rate_hz)
                for e in s.record_ids]

    @app.get("/api/records/{record_id}/signal", response_model=Signal)
    def signal(record_id: str, request: Request) -> Signal:
        s = service_of(request)
        raw = s.raw_signal(parse_record_id(s, record_id))
        return Signal(leads=s.leads, sampling_rate=s.sampling_rate_hz, units="mV", data=float32_list(raw))

    @app.get("/api/records/{record_id}/truth", response_model=Truth)
    def truth(record_id: str, request: Request) -> Truth:
        s = service_of(request)
        ecg_id = parse_record_id(s, record_id)
        try:
            labels = s.truth(ecg_id)
        except TruthLockedError:
            raise HTTPException(409, f"The labels of record {ecg_id} are revealed only after POST /api/predict has "
                                     "been called for it, so the answer cannot be shown before the prediction") from None
        return Truth(id=str(ecg_id), labels=labels,
                     classes=[TruthClass(class_=c, label=s.descriptions[c], present=c in labels) for c in s.classes])

    @app.post("/api/predict", response_model=Prediction, openapi_extra=PREDICT_OPENAPI)
    async def predict(request: Request) -> Prediction:
        s = service_of(request)
        media = request.headers.get("content-type", "").split(";")[0].strip().lower()
        if media not in ("application/json", "multipart/form-data"):
            raise HTTPException(415, "Send either JSON {\"record_id\": \"<id>\"} for a held-out test record, or a "
                                     "multipart form with files 'hea' and 'dat'")
        body = await read_capped_body(request, limit)

        if media == "application/json":
            try:
                req = PredictRequest.model_validate_json(body)  # malformed or deeply nested JSON: a ValidationError
            except ValidationError as exc:
                raise HTTPException(400, "The JSON body must be {\"record_id\": \"<id>\"}, with the id as a string "
                                         "from GET /api/records") from exc
            ecg_id = parse_record_id(s, req.record_id)
            try:
                result = await run_in_threadpool(s.predict_record, ecg_id)
            except UnknownRecordError:  # has_record was checked; this would be a bug
                raise HTTPException(404, f"Record {ecg_id} is not a held-out test record") from None
            info = InputInfo(source="test_record", record_id=str(ecg_id), sampling_rate_hz=s.sampling_rate_hz,
                             n_samples=s.n_samples, resampled=False, checksums_verified=None)
            return prediction(s, result, info, None)

        files = await upload_files(request.headers, body)
        try:
            record = await run_in_threadpool(read_upload, files["hea"], files["dat"], s.leads, s.run_cfg["dataset"],
                                             s.run_cfg["preprocessing"], upload_cfg, s.max_abs_mv, tmp_root)
        except UploadError as exc:
            raise HTTPException(400, str(exc)) from exc
        result = await run_in_threadpool(s.predict_signal, record.signal)
        info = InputInfo(source="upload", record_id=None, sampling_rate_hz=record.sampling_rate_hz,
                         n_samples=record.n_samples, resampled=record.resampled,
                         checksums_verified=record.checksums_verified)
        sig = Signal(leads=s.leads, sampling_rate=s.sampling_rate_hz, units="mV", data=float32_list(record.signal))
        return prediction(s, result, info, sig)

    @app.get("/api/metrics")
    def metrics(request: Request) -> Response:
        return Response(content=service_of(request).metrics_bytes, media_type="application/json")

    return app


app = create_app(load_config())


def main() -> int:
    api = load_config()["api"]
    uvicorn.run(app, host=api["host"], port=int(api["port"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
