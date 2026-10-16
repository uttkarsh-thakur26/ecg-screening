"""Unit tests for src/data/download.py. No network: a local HTTP server stands in for PhysioNet."""

from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
import requests

import src.data.download as dl_module
from src.data.download import (
    DownloadError,
    RemoteFile,
    download,
    download_file,
    exclusive_lock,
    fetch,
    parse_manifest,
    plan,
    select_files,
)

CHUNK = 1024
H1, H2, H3, H4 = ("a" * 64, "b" * 64, "c" * 64, "d" * 64)


# ---------- manifest parsing and selection ----------

def test_parse_manifest_accepts_text_and_binary_markers():
    text = f"{H1}  ptbxl_database.csv\n{H2} *records100/00000/00001_lr.dat\n\n"
    assert parse_manifest(text) == {"ptbxl_database.csv": H1, "records100/00000/00001_lr.dat": H2}


@pytest.mark.parametrize("line", ["nothash  file.csv", f"{H1}", f"{H1[:-1]}  file.csv"])
def test_parse_manifest_rejects_malformed_lines(line):
    with pytest.raises(ValueError):
        parse_manifest(line)


@pytest.mark.parametrize("path", ["../outside.dat", "/etc/passwd", "records100/../../x", "C:/x.dat", "records100\\x.dat"])
def test_parse_manifest_rejects_paths_that_escape_raw_dir(path):
    with pytest.raises(ValueError, match="Unsafe path"):
        parse_manifest(f"{H1}  {path}")


def test_parse_manifest_rejects_conflicting_duplicates():
    with pytest.raises(ValueError, match="twice"):
        parse_manifest(f"{H1}  a.csv\n{H2}  a.csv")


def test_select_files_keeps_metadata_and_records100_only():
    manifest = {
        "ptbxl_database.csv": H1,
        "scp_statements.csv": H2,
        "records100/00000/00001_lr.dat": H3,
        "records500/00000/00001_hr.dat": H4,
        "example_physionet.py": H4,
        "records1000/fake.dat": H4,  # prefix match must respect the directory boundary
    }
    selected = select_files(manifest, ["ptbxl_database.csv", "scp_statements.csv"], ["records100"])
    assert [f.path for f in selected] == ["ptbxl_database.csv", "records100/00000/00001_lr.dat", "scp_statements.csv"]


def test_select_files_fails_when_metadata_missing():
    with pytest.raises(ValueError, match="scp_statements.csv"):
        select_files({"ptbxl_database.csv": H1, "records100/a.dat": H2}, ["ptbxl_database.csv", "scp_statements.csv"], ["records100"])


def test_select_files_fails_when_record_dir_empty():
    with pytest.raises(ValueError, match="records100/"):
        select_files({"ptbxl_database.csv": H1}, ["ptbxl_database.csv"], ["records100"])


def test_plan_skips_existing_files_and_rehash_catches_bad_copies(tmp_path: Path):
    good, bad = b"good bytes", b"corrupted"
    (tmp_path / "good.dat").write_bytes(good)
    (tmp_path / "bad.dat").write_bytes(bad)
    files = [
        RemoteFile("good.dat", hashlib.sha256(good).hexdigest()),
        RemoteFile("bad.dat", hashlib.sha256(b"what PhysioNet published").hexdigest()),
        RemoteFile("missing.dat", H1),
    ]
    todo, present = plan(files, tmp_path, rehash=False, chunk_size=CHUNK)
    assert [f.path for f in todo] == ["missing.dat"] and present == 2
    todo, present = plan(files, tmp_path, rehash=True, chunk_size=CHUNK)
    assert [f.path for f in todo] == ["bad.dat", "missing.dat"] and present == 1


# ---------- download_file against a real local HTTP server ----------

PAYLOAD = bytes(range(256)) * 40  # 10,240 bytes


class _RangeHandler(BaseHTTPRequestHandler):
    """Serves PAYLOAD at any path, honouring "Range: bytes=N-" like PhysioNet and S3 do."""

    range_headers: list[str | None] = []
    delay_s = 0.0

    def do_GET(self):  # noqa: N802 (http.server API)
        rng = self.headers.get("Range")
        type(self).range_headers.append(rng)
        time.sleep(type(self).delay_s)
        if rng:
            start = int(re.fullmatch(r"bytes=(\d+)-", rng).group(1))
            if start >= len(PAYLOAD):
                self.send_response(416)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            body = PAYLOAD[start:]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{len(PAYLOAD) - 1}/{len(PAYLOAD)}")
        else:
            body = PAYLOAD
            self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


@pytest.fixture()
def server_url():
    _RangeHandler.range_headers = []
    _RangeHandler.delay_s = 0.0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}/file.dat"
    httpd.shutdown()
    httpd.server_close()


SHA = hashlib.sha256(PAYLOAD).hexdigest()


def test_download_file_fetches_and_verifies(tmp_path: Path, server_url: str):
    dest = tmp_path / "sub" / "file.dat"
    received = download_file(requests.Session(), server_url, dest, SHA, timeout_s=5, chunk_size=CHUNK)
    assert dest.read_bytes() == PAYLOAD and received == len(PAYLOAD)
    assert not (tmp_path / "sub" / "file.dat.part").exists()
    assert _RangeHandler.range_headers == [None]


def test_download_file_resumes_a_partial_file(tmp_path: Path, server_url: str):
    dest = tmp_path / "file.dat"
    (tmp_path / "file.dat.part").write_bytes(PAYLOAD[:3000])
    received = download_file(requests.Session(), server_url, dest, SHA, timeout_s=5, chunk_size=CHUNK)
    assert dest.read_bytes() == PAYLOAD
    assert received == len(PAYLOAD) - 3000  # only the missing tail was transferred
    assert _RangeHandler.range_headers == ["bytes=3000-"]


def test_download_file_accepts_a_complete_part_file_on_416(tmp_path: Path, server_url: str):
    dest = tmp_path / "file.dat"
    (tmp_path / "file.dat.part").write_bytes(PAYLOAD)
    assert download_file(requests.Session(), server_url, dest, SHA, timeout_s=5, chunk_size=CHUNK) == 0
    assert dest.read_bytes() == PAYLOAD


def test_download_file_rejects_checksum_mismatch_and_cleans_up(tmp_path: Path, server_url: str):
    dest = tmp_path / "file.dat"
    with pytest.raises(DownloadError, match="SHA-256 mismatch"):
        download_file(requests.Session(), server_url, dest, H1, timeout_s=5, chunk_size=CHUNK)
    assert not dest.exists()
    assert not (tmp_path / "file.dat.part").exists()


# ---------- robustness: transient Windows file locks, Ctrl+C, concurrent runs ----------

DL_CFG = {"timeout_s": 5, "max_attempts": 3, "retry_backoff_s": 0.01, "chunk_size_bytes": CHUNK}


def test_fetch_retries_a_transient_permission_error_on_rename(tmp_path: Path, server_url: str, monkeypatch):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky_replace(src, dst):
        calls["n"] += 1
        if calls["n"] == 1:
            raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_replace(src, dst)

    monkeypatch.setattr(dl_module.os, "replace", flaky_replace)
    dest = tmp_path / "file.dat"
    fetch(server_url, dest, SHA, DL_CFG)
    assert dest.read_bytes() == PAYLOAD and calls["n"] == 2
    assert _RangeHandler.range_headers[1] == f"bytes={len(PAYLOAD)}-"  # second attempt reused the verified .part


def test_fetch_turns_a_persistent_os_error_into_download_error(tmp_path: Path, server_url: str, monkeypatch):
    def locked(src, dst):
        raise PermissionError(13, "locked")

    monkeypatch.setattr(dl_module.os, "replace", locked)
    with pytest.raises(DownloadError, match="failed after 3 attempts"):
        fetch(server_url, tmp_path / "file.dat", SHA, DL_CFG)


def _local_cfg(server_url: str, raw_dir: Path, n_files: int) -> dict:
    """A config whose manifest (written into raw_dir) lists n_files record files served by the test server."""
    lines = [f"{SHA}  db.csv", f"{SHA}  st.csv"] + [f"{SHA}  records100/{i:05d}_lr.dat" for i in range(n_files)]
    manifest = "\n".join(lines) + "\n"
    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "SHA256SUMS.txt").write_bytes(manifest.encode())  # bytes: no CRLF translation on Windows
    return {
        "dataset": {
            "version": "test",
            "checksum_manifest": "SHA256SUMS.txt",
            "checksum_manifest_sha256": hashlib.sha256(manifest.encode()).hexdigest(),
            "database_csv": "db.csv",
            "statements_csv": "st.csv",
            "record_dirs": ["records100"],
        },
        "download": {**DL_CFG, "source_url": server_url.rsplit("/", 1)[0] + "/", "workers": 1, "lock_file": ".download.lock"},
    }


def test_download_fetches_every_file_in_scope(tmp_path: Path, server_url: str):
    cfg = _local_cfg(server_url, tmp_path, n_files=5)
    download(cfg, tmp_path, rehash=False)
    assert sorted(p.name for p in (tmp_path / "records100").iterdir()) == [f"{i:05d}_lr.dat" for i in range(5)]
    assert (tmp_path / "db.csv").read_bytes() == PAYLOAD


class _InterruptOnFirstUpdate:
    """Stands in for tqdm and raises KeyboardInterrupt after the first finished file, like Ctrl+C."""

    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n):
        raise KeyboardInterrupt


def test_ctrl_c_cancels_queued_downloads_instead_of_draining_them(tmp_path: Path, server_url: str, monkeypatch):
    _RangeHandler.delay_s = 0.05
    cfg = _local_cfg(server_url, tmp_path, n_files=30)
    monkeypatch.setattr(dl_module, "tqdm", _InterruptOnFirstUpdate)
    with pytest.raises(KeyboardInterrupt):
        download(cfg, tmp_path, rehash=False)
    assert len(_RangeHandler.range_headers) <= 3  # one finished, at most one in flight; not all 32


def test_second_download_is_refused_while_the_lock_is_held(tmp_path: Path, server_url: str):
    cfg = _local_cfg(server_url, tmp_path, n_files=1)
    with exclusive_lock(tmp_path / ".download.lock"):
        with pytest.raises(DownloadError, match="already running"):
            download(cfg, tmp_path, rehash=False)
    download(cfg, tmp_path, rehash=False)  # lock released: works again
    assert (tmp_path / "records100" / "00000_lr.dat").is_file()
