"""Small shared helpers with no dependencies on any sibling module."""

from __future__ import annotations

import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BYTES_PER_MB = 1024 * 1024


def bytes_to_mb(value, default=None) -> Optional[int]:
    """Best-effort bytes → whole MB; ``default`` on missing/garbage input.

    Callers pick the fallback that matches their contract: ``0`` where a
    number is always expected (Ollama sizes), ``None`` where "unreported"
    is meaningful (NVML per-process sizes on Windows/WDDM).
    """
    try:
        return int(value) // BYTES_PER_MB
    except (TypeError, ValueError, OverflowError):
        return default


def run_capture(cmd: list[str], timeout: int) -> Optional[str]:
    """Run ``cmd`` and return stdout, or ``None`` on ANY failure (missing
    binary, timeout, non-zero exit). For callers whose contract is
    best-effort telemetry — never raises."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError):
        # FileNotFoundError is an OSError subclass — one tuple covers both.
        return None
    return result.stdout


_REPLACE_ATTEMPTS = 10
_REPLACE_RETRY_SLEEP = 0.02


def iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


class LockDisplacedError(OSError):
    """The lock was acquired, but no longer governs the path it was taken on.

    Raised when the lock file is deleted or replaced while this process holds
    its advisory lock — something no vram-mcp client of this generation does.
    It therefore means a foreign writer is present, most likely a pre-0.3
    client, whose lock protocol treats the FILE'S EXISTENCE as ownership and
    unlinks any lock file it judges stale.

    An ``OSError`` on purpose: the ledger's callers already degrade on those, so
    an affected tool reports a structured failure rather than a traceback, and
    the specific type stays available to anyone who wants to say more.
    """


_DISPLACED = (
    "lock {lock} was removed or replaced {when}. Another client is breaking "
    "locks — a pre-0.3 vram-mcp treats the lock file's existence as ownership "
    "and unlinks any it judges stale. Stop it before continuing; sharing one "
    "ledger across generations is unsupported."
)


def _governs(fd: int, lock_path: Path) -> bool:
    """Is the descriptor's file still the file that ``lock_path`` names?

    Holding an advisory lock on an unlinked file is not an error and not
    detectable from the descriptor: the lock stays valid on an inode nobody can
    reach any more, while a second process creates a fresh file at the same
    name and locks that instead. Comparing identities is what turns that
    silence into a fact.
    """
    held = os.fstat(fd)
    try:
        named = os.stat(lock_path)
    except OSError:
        return False
    return (held.st_ino, held.st_dev) == (named.st_ino, named.st_dev)


@contextmanager
def locked(path: Path, timeout: float = 5.0, poll: float = 0.05):
    """Serialize access through an OS-backed sibling lock file.

    The file is deliberately persistent.  Its *advisory lock*, rather than
    its age or existence, represents ownership; the OS releases it when a
    crashed holder exits.  This avoids both stale-file split brain on POSIX and
    sharing violations from deleting an open lock on Windows.

    Identity is checked on both sides of the critical section, and neither check
    is prevention. Mutual exclusion against a client that ignores advisory locks
    cannot be reconstructed from this side; mixing generations on one ledger is
    unsupported (see docs/coordination.md). What the checks buy is that such a
    client cannot pass unnoticed:

    * before the body — the path was already displaced, so this section never
      had exclusion to begin with and must not run;
    * after it — displacement happened *while we worked*, which is the ordinary
      shape of the failure. The work may have raced a concurrent writer, so the
      caller is told its result is unconfirmed rather than being handed a
      success it cannot rely on.

    The second check is why this is worth having. Nothing in this generation
    removes a lock file, so in a single-generation deployment neither check can
    fire; when one does, a foreign writer is a fact rather than an inference.
    """
    lock_path = path.with_suffix(path.suffix + ".lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR)
    acquired = False
    try:
        # Windows locks byte ranges and cannot lock an empty file.
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        while True:
            try:
                _try_lock(fd)
                acquired = True
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"could not acquire lock {lock_path}")
                time.sleep(poll)
        if not _governs(fd, lock_path):
            raise LockDisplacedError(_DISPLACED.format(
                lock=lock_path, when="before this operation began"))
        yield
        # Deliberately not in `finally`: a body that raised has its own story to
        # tell, and masking it with this one would lose the more specific fault.
        if not _governs(fd, lock_path):
            raise LockDisplacedError(_DISPLACED.format(
                lock=lock_path, when="while this operation was running, so it "
                                     "may have raced a concurrent writer"))
    finally:
        if acquired:
            try:
                _unlock(fd)
            except OSError:
                pass
        os.close(fd)


def _try_lock(fd: int) -> None:
    """Take a non-blocking exclusive advisory lock, or raise BlockingIOError."""
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise BlockingIOError from exc
    else:
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise


def _unlock(fd: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _quarantine(path: Path) -> None:
    try:
        os.replace(path, path.with_suffix(path.suffix + ".corrupt"))
    except OSError:
        pass


def load_json(path: Path, default_factory, is_valid):
    """Load JSON; on missing/unparsable/invalid-shape return ``default_factory()``.
    A file that EXISTS but is corrupt/wrong-shape is quarantined to ``.corrupt``
    first, so the next write can't silently destroy it."""
    if not path.exists():
        return default_factory()
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = None
        _quarantine(path)
        return default_factory()
    if not is_valid(doc):
        _quarantine(path)
        return default_factory()
    return doc


def save_json_atomic(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == _REPLACE_ATTEMPTS - 1:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
                raise
            time.sleep(_REPLACE_RETRY_SLEEP)


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def append_jsonl_capped(path: Path, record: dict, cap: int) -> None:
    """Append one JSON line; if the file then exceeds ``cap`` lines, rewrite it
    to the last ``cap``. Bounded growth, no daemon."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
    rows = read_jsonl(path)
    if len(rows) > cap:
        kept = rows[-cap:]
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
        for attempt in range(_REPLACE_ATTEMPTS):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if attempt == _REPLACE_ATTEMPTS - 1:
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass
                    return  # best-effort prune; a failed prune never breaks logging
                time.sleep(_REPLACE_RETRY_SLEEP)
