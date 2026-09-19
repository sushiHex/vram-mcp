"""Tests for vram_mcp._util shared file/lock/json helpers."""
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import pytest

from vram_mcp import _util


def test_iso_roundtrip():
    dt = datetime(2026, 7, 16, 18, 0, 0, tzinfo=timezone.utc)
    assert _util.iso(dt) == "2026-07-16T18:00:00Z"
    assert _util.parse_iso("2026-07-16T18:00:00Z") == dt


def test_save_and_load_json(tmp_path):
    p = tmp_path / "d.json"
    _util.save_json_atomic(p, {"k": [1, 2]})
    assert _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict)) == {"k": [1, 2]}


def test_load_json_missing_returns_default(tmp_path):
    assert _util.load_json(tmp_path / "nope.json", lambda: {"k": []}, lambda d: True) == {"k": []}


def test_load_json_corrupt_quarantines_and_defaults(tmp_path):
    p = tmp_path / "d.json"
    p.write_text("not json {")
    assert _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict)) == {"k": []}
    assert (tmp_path / "d.json.corrupt").exists()


def test_load_json_wrong_shape_quarantines(tmp_path):
    p = tmp_path / "d.json"
    p.write_text(json.dumps([1, 2, 3]))
    assert _util.load_json(p, lambda: {"k": []}, lambda d: isinstance(d, dict) and "k" in d) == {"k": []}
    assert (tmp_path / "d.json.corrupt").exists()


def test_save_json_retries_replace_on_permission_error(tmp_path, monkeypatch):
    real_replace = os.replace
    calls = {"n": 0}

    def flaky(src, dst):
        calls["n"] += 1
        if calls["n"] <= 3:
            raise PermissionError("sharing violation")
        return real_replace(src, dst)

    monkeypatch.setattr(_util.os, "replace", flaky)
    _util.save_json_atomic(tmp_path / "d.json", {"ok": True})
    assert calls["n"] == 4


def _holder_script(path, mode):
    body = "    os._exit(0)" if mode == "crash" else "    import time; time.sleep(2)"
    return (
        "import os, sys\nfrom pathlib import Path\nfrom vram_mcp._util import locked\n"
        "p = Path(sys.argv[1])\nwith locked(p):\n"
        "    os.utime(p.with_suffix(p.suffix + '.lock'), (1, 1))\n"
        "    print('locked', flush=True)\n" + body + "\n"
    )


def test_locked_live_aged_holder_remains_exclusive(tmp_path):
    p = tmp_path / "d.json"
    proc = subprocess.Popen([sys.executable, "-c", _holder_script(p, "hold"), str(p)],
                            stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(TimeoutError):
            with _util.locked(p, timeout=0.1, poll=0.01):
                pass
    finally:
        proc.wait(timeout=5)
    assert p.with_suffix(p.suffix + ".lock").exists()


def test_locked_is_released_when_holder_crashes(tmp_path):
    p = tmp_path / "d.json"
    proc = subprocess.Popen([sys.executable, "-c", _holder_script(p, "crash"), str(p)],
                            stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "locked"
    assert proc.wait(timeout=5) == 0
    with _util.locked(p, timeout=0.5):
        pass


# ---- a lock must still govern the path it was taken on ----------------------
#
# A pre-0.3 client treats the lock file's EXISTENCE as ownership and unlinks any
# lock file it judges stale (30s), and it never takes an advisory lock at all.
# On POSIX that unlink succeeds against a file we hold locked, leaving our lock
# valid on an unreachable inode while it creates a fresh file under the same
# name — two processes, each certain it owns the ledger.


def _legacy_displacer_script(path):
    """A pre-0.3-shaped acquire: unlink the 'stale' lock, then create our own."""
    return (
        "import os, sys\n"
        "from pathlib import Path\n"
        "lock = Path(sys.argv[1]).with_suffix(Path(sys.argv[1]).suffix + '.lock')\n"
        "os.remove(lock)\n"
        "fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)\n"
        "print('displaced', flush=True)\n"
        "os.close(fd)\n"
    )


def test_governs_accepts_the_file_it_locked(tmp_path):
    lock = tmp_path / "d.json.lock"
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR)
    try:
        assert _util._governs(fd, lock) is True
    finally:
        os.close(fd)


def test_governs_rejects_a_replacement_at_the_same_path(tmp_path, monkeypatch):
    """Identity, not existence: a different file under the same name is not ours."""
    lock = tmp_path / "d.json.lock"
    other = tmp_path / "other"
    other.write_text("x", encoding="utf-8")
    real_stat = os.stat
    replacement = real_stat(other)
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR)
    try:
        monkeypatch.setattr(_util.os, "stat", lambda p, *a, **k: replacement)
        assert _util._governs(fd, lock) is False
    finally:
        os.close(fd)


def test_governs_rejects_a_vanished_path(tmp_path):
    """An unlinked lock file still locks fine; there is just nothing at the name."""
    lock = tmp_path / "d.json.lock"
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR)
    try:
        assert _util._governs(fd, tmp_path / "gone.lock") is False
    finally:
        os.close(fd)


def test_locked_refuses_to_enter_an_already_displaced_path(tmp_path, monkeypatch):
    """Displaced before we started: this section never had exclusion at all."""
    p = tmp_path / "d.json"
    monkeypatch.setattr(_util, "_governs", lambda fd, lock_path: False)
    with pytest.raises(_util.LockDisplacedError) as excinfo:
        with _util.locked(p):
            raise AssertionError("must not enter the critical section")
    assert "before this operation began" in str(excinfo.value)


def test_locked_reports_displacement_that_happened_while_it_worked(tmp_path, monkeypatch):
    """The ordinary shape of the failure, and the reason the second check
    exists: the body completed, but a foreign writer was provably active while
    it ran, so its result cannot be reported as confirmed."""
    p = tmp_path / "d.json"
    answers = iter([True, False])           # governs on entry, displaced by exit
    monkeypatch.setattr(_util, "_governs", lambda fd, lock_path: next(answers))
    entered = []
    with pytest.raises(_util.LockDisplacedError) as excinfo:
        with _util.locked(p):
            entered.append(True)
    assert entered == [True]                # the body DID run
    assert "may have raced a concurrent writer" in str(excinfo.value)


def test_a_failing_body_keeps_its_own_exception(tmp_path, monkeypatch):
    """The exit check must not mask a more specific fault from the body."""
    p = tmp_path / "d.json"
    monkeypatch.setattr(_util, "_governs", lambda fd, lock_path: True)
    with pytest.raises(ZeroDivisionError):
        with _util.locked(p):
            1 / 0


def test_lock_displaced_is_an_oserror_so_callers_still_degrade():
    """Ledger callers catch OSError; a new bare exception type would escape as a
    traceback through MCP instead of a structured refusal."""
    assert issubclass(_util.LockDisplacedError, OSError)


@pytest.mark.skipif(sys.platform == "win32",
                    reason="Windows blocks unlinking a file that is held open, "
                           "so the displacement this guards against is POSIX-only")
def test_locked_detects_a_real_legacy_style_unlink(tmp_path):
    """The live failure, reproduced with a real second process: POSIX permits the
    unlink, so without the check both clients would enter."""
    p = tmp_path / "d.json"
    lock = p.with_suffix(p.suffix + ".lock")
    with pytest.raises(_util.LockDisplacedError) as excinfo:
        with _util.locked(p):
            # A real second process does what a pre-0.3 client does to a lock
            # file it judges stale. POSIX permits it against a file we hold.
            proc = subprocess.Popen(
                [sys.executable, "-c", _legacy_displacer_script(p), str(p)],
                stdout=subprocess.PIPE, text=True)
            assert proc.stdout.readline().strip() == "displaced"
            assert proc.wait(timeout=5) == 0
    assert "may have raced a concurrent writer" in str(excinfo.value)
    assert lock.exists()      # the displacer's file now, not ours


def test_append_jsonl_capped_prunes_to_last_n(tmp_path):
    p = tmp_path / "e.jsonl"
    for i in range(10):
        _util.append_jsonl_capped(p, {"i": i}, cap=5)
    assert [r["i"] for r in _util.read_jsonl(p)] == [5, 6, 7, 8, 9]


def test_read_jsonl_skips_bad_lines(tmp_path):
    p = tmp_path / "e.jsonl"
    p.write_text('{"a":1}\nGARBAGE\n{"a":2}\n')
    assert _util.read_jsonl(p) == [{"a": 1}, {"a": 2}]
