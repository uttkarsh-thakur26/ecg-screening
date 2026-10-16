"""Download PTB-XL from PhysioNet into data/raw/ and verify it.

    python -m src.data.download                # download whatever is missing, then verify
    python -m src.data.download --verify-only  # no network: only verify what is on disk
    python -m src.data.download --rehash       # also re-check the SHA-256 of files already on disk

The file list comes from PhysioNet's SHA256SUMS.txt for the pinned version, filtered to the
two metadata CSVs and records100/. The manifest itself must match the SHA-256 pinned in the
config (taken from physionet.org), so files fetched from a mirror are still checked against
PhysioNet's own list. Each file is written to "<name>.part" first, checked against its published
SHA-256 and only then renamed into place. A file that exists under data/raw/ is therefore a
complete, verified copy and is skipped on the next run; an interrupted ".part" file is resumed
with an HTTP Range request.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
import threading
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import requests
from tqdm import tqdm

from src.config import DEFAULT_CONFIG_PATH, load_config, project_path
from src.data.records import read_database

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
RECORD_EXTENSIONS = (".hea", ".dat")
# 4xx responses that are worth retrying (rate limiting); every other 4xx fails at once.
RETRYABLE_CLIENT_STATUS = {408, 429}

_thread_state = threading.local()


class DownloadError(RuntimeError):
    """A file could not be downloaded, or its contents failed the checksum."""


class VerificationError(RuntimeError):
    """The data on disk does not match what PhysioNet publishes for this version."""


@dataclass(frozen=True)
class RemoteFile:
    path: str  # POSIX path relative to the dataset root, e.g. "records100/00000/00001_lr.dat"
    sha256: str


def _check_relative(path: str) -> None:
    """Reject manifest paths that could escape data/raw/."""
    p = PurePosixPath(path)
    if p.is_absolute() or ".." in p.parts or "\\" in path or ":" in path or not p.parts:
        raise ValueError(f"Unsafe path in checksum manifest: {path!r}")


def parse_manifest(text: str) -> dict[str, str]:
    """Parse sha256sum-format lines ("<hash>  <path>" or "<hash> *<path>") into {path: hash}."""
    entries: dict[str, str] = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError(f"Malformed checksum manifest line {lineno}: {raw!r}")
        digest, path = parts[0].lower(), parts[1].lstrip("*")
        if not SHA256_RE.match(digest):
            raise ValueError(f"Invalid SHA-256 on checksum manifest line {lineno}: {raw!r}")
        _check_relative(path)
        if path in entries and entries[path] != digest:
            raise ValueError(f"Checksum manifest lists {path!r} twice with different hashes")
        entries[path] = digest
    if not entries:
        raise ValueError("Checksum manifest is empty")
    return entries


def select_files(manifest: dict[str, str], metadata_files: list[str], record_dirs: list[str]) -> list[RemoteFile]:
    """Keep the metadata files and everything under the record directories; fail if any is absent."""
    missing = [name for name in metadata_files if name not in manifest]
    if missing:
        raise ValueError(f"Not listed in the checksum manifest: {missing}")
    prefixes = tuple(f"{d.strip('/')}/" for d in record_dirs)
    selected = [
        RemoteFile(path, digest)
        for path, digest in manifest.items()
        if path in metadata_files or path.startswith(prefixes)
    ]
    for prefix in prefixes:
        if not any(f.path.startswith(prefix) for f in selected):
            raise ValueError(f"Checksum manifest lists no files under {prefix!r}")
    return sorted(selected, key=lambda f: f.path)


def sha256_file(path: Path, chunk_size: int) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def _session() -> requests.Session:
    """One HTTP session per worker thread, so connections are reused."""
    session = getattr(_thread_state, "session", None)
    if session is None:
        session = requests.Session()
        # Raw bytes only: Range offsets and Content-Length must refer to the file on disk.
        session.headers["Accept-Encoding"] = "identity"
        _thread_state.session = session
    return session


def download_file(
    session: requests.Session,
    url: str,
    dest: Path,
    sha256: str | None,
    timeout_s: float,
    chunk_size: int,
) -> int:
    """Fetch url into dest via dest.part, resuming a partial .part. Returns bytes received.

    With sha256 given, the .part file must match it before it is renamed to dest; on a
    mismatch the .part file is deleted so the next attempt starts clean.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    received = 0
    with session.get(url, headers=headers, stream=True, timeout=timeout_s) as resp:
        if resp.status_code == 416 and offset:
            pass  # Range starts at or past the end: .part may already be whole; the checksum decides.
        else:
            resp.raise_for_status()
            resuming = offset > 0 and resp.status_code == 206
            expected = resp.headers.get("Content-Length")
            with part.open("ab" if resuming else "wb") as f:
                for chunk in resp.iter_content(chunk_size):
                    f.write(chunk)
                    received += len(chunk)
            if expected is not None and received != int(expected):
                raise DownloadError(f"{url}: received {received} bytes, server announced {expected}")
    if sha256 is not None:
        actual = sha256_file(part, chunk_size)
        if actual != sha256:
            part.unlink()
            raise DownloadError(f"{url}: SHA-256 mismatch (expected {sha256}, got {actual}); partial file removed")
    os.replace(part, dest)
    return received


def fetch(url: str, dest: Path, sha256: str | None, dl_cfg: dict[str, Any], stop: threading.Event | None = None) -> int:
    """download_file with retries and exponential backoff. Raises DownloadError when attempts run out.

    OSError is retried too: on Windows a freshly written file is often briefly locked by antivirus
    or the search indexer, which makes os.replace fail with WinError 5/32. Setting stop abandons
    the file before its next attempt (used on Ctrl+C).
    """
    attempts = dl_cfg["max_attempts"]
    for attempt in range(1, attempts + 1):
        if stop is not None and stop.is_set():
            raise DownloadError(f"{url}: cancelled")
        try:
            return download_file(_session(), url, dest, sha256, dl_cfg["timeout_s"], dl_cfg["chunk_size_bytes"])
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status is not None and 400 <= status < 500 and status not in RETRYABLE_CLIENT_STATUS:
                raise DownloadError(f"{url}: HTTP {status}") from exc
            error: Exception = exc
        except (DownloadError, OSError) as exc:  # requests.RequestException is an OSError subclass
            error = exc
        if attempt == attempts:
            raise DownloadError(f"{url}: failed after {attempts} attempts: {error}") from error
        wait_s = dl_cfg["retry_backoff_s"] * 2 ** (attempt - 1)
        if stop is not None:
            stop.wait(wait_s)
        else:
            time.sleep(wait_s)
    raise AssertionError("unreachable")


@contextmanager
def exclusive_lock(path: Path) -> Iterator[None]:
    """Hold an OS-level lock on path. The OS releases it if the process dies, so it never goes stale."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        try:
            if os.name == "nt":
                import msvcrt

                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise DownloadError(f"Another download is already running in {path.parent} (it holds {path.name})") from exc
        try:
            yield
        finally:
            if os.name == "nt":
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def plan(files: list[RemoteFile], raw_dir: Path, rehash: bool, chunk_size: int) -> tuple[list[RemoteFile], int]:
    """Split files into (to download, count already present). With rehash, bad copies are re-downloaded."""
    todo: list[RemoteFile] = []
    present = 0
    corrupt: list[str] = []
    for f in tqdm(files, desc="Checking local files", unit="file", disable=not rehash):
        dest = raw_dir / f.path
        if not dest.is_file():
            todo.append(f)
        elif rehash and sha256_file(dest, chunk_size) != f.sha256:
            corrupt.append(f.path)
            todo.append(f)
        else:
            present += 1
    if corrupt:
        print(f"WARNING: {len(corrupt)} file(s) on disk do not match their published SHA-256 "
              f"and will be re-downloaded:", file=sys.stderr)
        for path in corrupt:
            print(f"  {path}", file=sys.stderr)
    return todo, present


def load_manifest(cfg: dict[str, Any], raw_dir: Path, source_url: str) -> dict[str, str]:
    """Fetch SHA256SUMS.txt if absent, check it against the pinned hash and parse it."""
    ds, dl = cfg["dataset"], cfg["download"]
    pinned = ds["checksum_manifest_sha256"]
    if not SHA256_RE.match(str(pinned)):
        raise ValueError(f"dataset.checksum_manifest_sha256 must be a SHA-256 hex digest, got {pinned!r}")
    path = raw_dir / ds["checksum_manifest"]
    if path.is_file():
        actual = sha256_file(path, dl["chunk_size_bytes"])
        if actual != pinned:
            raise VerificationError(f"{path} has SHA-256 {actual}, expected the pinned {pinned}. Delete it and re-run.")
        print(f"Checksum manifest on disk matches the pinned SHA-256: {path}")
    else:
        print(f"Fetching checksum manifest {source_url}{ds['checksum_manifest']}")
        fetch(source_url + ds["checksum_manifest"], path, pinned, dl)
        print("Checksum manifest matches the pinned SHA-256")
    return parse_manifest(path.read_text(encoding="utf-8"))


def download(cfg: dict[str, Any], raw_dir: Path, rehash: bool) -> None:
    with exclusive_lock(raw_dir / cfg["download"]["lock_file"]):
        _download_locked(cfg, raw_dir, rehash)


def _download_locked(cfg: dict[str, Any], raw_dir: Path, rehash: bool) -> None:
    ds, dl = cfg["dataset"], cfg["download"]
    source_url = dl["source_url"].rstrip("/") + "/"
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(cfg, raw_dir, source_url)

    files = select_files(manifest, [ds["database_csv"], ds["statements_csv"]], ds["record_dirs"])
    todo, present = plan(files, raw_dir, rehash, dl["chunk_size_bytes"])
    print(f"PTB-XL {ds['version']}: {len(files)} files in scope, {present} already on disk, {len(todo)} to download")
    if not todo:
        return

    print(f"Source: {source_url}")
    failures: list[str] = []
    total_bytes = 0
    stop = threading.Event()
    pool = ThreadPoolExecutor(max_workers=dl["workers"])
    try:
        futures = [pool.submit(fetch, source_url + f.path, raw_dir / f.path, f.sha256, dl, stop) for f in todo]
        with tqdm(total=len(todo), desc="Downloading", unit="file") as bar:
            for future in as_completed(futures):
                try:
                    total_bytes += future.result()
                except DownloadError as exc:
                    failures.append(str(exc))
                bar.update(1)
    except BaseException:
        # Ctrl+C or an unexpected error: drop the queued files instead of draining the whole queue.
        # In-flight files finish or stop at their next attempt; whatever completed stays on disk.
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    print(f"Received {total_bytes / 1e6:.1f} MB")
    if failures:
        for msg in failures:
            print(f"  {msg}", file=sys.stderr)
        raise DownloadError(f"{len(failures)} of {len(todo)} files failed. Re-run to resume; finished files are kept.")


def verify(cfg: dict[str, Any], raw_dir: Path) -> dict[str, int]:
    """Check record and patient counts against PhysioNet's figures and that every record is on disk."""
    ds = cfg["dataset"]
    db = read_database(raw_dir, ds["database_csv"])
    statements_path = raw_dir / ds["statements_csv"]
    if not statements_path.is_file():
        raise VerificationError(f"Missing {statements_path}")
    if not db.index.is_unique:
        raise VerificationError(f"{ds['database_csv']}: ecg_id values are not unique")
    if db["patient_id"].isna().any():
        raise VerificationError(f"{ds['database_csv']}: {int(db['patient_id'].isna().sum())} rows have no patient_id")

    n_records = len(db)
    n_patients = int(db["patient_id"].nunique())
    print(f"Records listed in {ds['database_csv']}: {n_records:,}")
    print(f"Unique patient_id values:            {n_patients:,}")

    missing = [
        f"{rel}{ext}"
        for rel in db[ds["record_path_column"]]
        for ext in RECORD_EXTENSIONS
        if not (raw_dir / f"{rel}{ext}").is_file()
    ]
    if missing:
        preview = ", ".join(missing[:5])
        raise VerificationError(f"{len(missing)} record files listed in {ds['record_path_column']} are missing, e.g. {preview}")
    print(f"All {n_records:,} records have their {' and '.join(RECORD_EXTENSIONS)} files under {raw_dir}")

    if n_records != ds["expected_records"] or n_patients != ds["expected_patients"]:
        raise VerificationError(
            f"Expected {ds['expected_records']:,} records and {ds['expected_patients']:,} patients "
            f"(PhysioNet, v{ds['version']}), found {n_records:,} and {n_patients:,}"
        )
    print(f"Matches PhysioNet's figures for PTB-XL v{ds['version']}")
    return {"records": n_records, "patients": n_patients}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download and verify PTB-XL (100 Hz records and metadata).")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify-only", action="store_true", help="skip the download, only verify data/raw/")
    mode.add_argument("--rehash", action="store_true", help="re-check the SHA-256 of files already on disk")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    raw_dir = project_path(cfg["paths"]["raw_dir"])
    try:
        if not args.verify_only:
            download(cfg, raw_dir, rehash=args.rehash)
        verify(cfg, raw_dir)
    except KeyboardInterrupt:
        print("\nInterrupted. Finished files are kept; re-run to resume.", file=sys.stderr)
        return 130
    except (DownloadError, VerificationError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
