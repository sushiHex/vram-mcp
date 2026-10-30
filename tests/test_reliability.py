"""Cross-module regressions for coordination and observation reliability.

Every filesystem path and hardware/network reader is injected. These tests
exercise integration seams without consulting Ollama, a GPU, or user state.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import requests

from vram_mcp import audit, claims, core, server
from vram_mcp.observations import Observation
from vram_mcp.ollama import OllamaClient


_T0 = datetime(2026, 9, 12, 12, 0, 0, tzinfo=timezone.utc)
_MB = 1024 * 1024


def _forbidden(*_args, **_kwargs):
    raise AssertionError("test must inject every external reader and backend")


@pytest.fixture
def isolated_runtime(monkeypatch, tmp_path):
    """Bind server-owned claims and audit calls to this test's temp directory."""
    claim_path = tmp_path / "claims.json"
    event_path = tmp_path / "events.jsonl"
    baseline_path = tmp_path / "last_seen.json"
    sample_path = tmp_path / "last_sample.json"

    monkeypatch.setattr(claims, "_DEFAULT_PATH", claim_path)
    monkeypatch.setattr(server, "_AUDIT_ON", True)

    original_log = audit.log_action
    original_detect = audit.detect_and_log
    original_sample = audit.maybe_log_sample

    def log_action(**kwargs):
        kwargs.setdefault("path", event_path)
        return original_log(**kwargs)

    def detect_and_log(holders, **kwargs):
        kwargs.setdefault("last_seen_path", baseline_path)
        kwargs.setdefault("log_path", event_path)
        return original_detect(holders, **kwargs)

    def maybe_log_sample(sample, **kwargs):
        kwargs.setdefault("state_path", sample_path)
        kwargs.setdefault("log_path", event_path)
        return original_sample(sample, **kwargs)

    # server._audit and audit are the same module; preserve the originals above
    # so these wrappers can safely redirect defaults bound at function creation.
    monkeypatch.setattr(audit, "log_action", log_action)
    monkeypatch.setattr(audit, "detect_and_log", detect_and_log)
    monkeypatch.setattr(audit, "maybe_log_sample", maybe_log_sample)

    monkeypatch.setattr(server, "_gpu_reading", _forbidden)
    monkeypatch.setattr(server, "_procinfo_table", _forbidden)
    monkeypatch.setattr(server, "runner_pid_map", _forbidden)
    monkeypatch.setattr(server._nvml, "nvml_busy_map", _forbidden)
    monkeypatch.setattr(server._ollama, "change_residency", _forbidden)
    monkeypatch.setattr(requests.Session, "request", _forbidden)

    return {
        "claims": claim_path,
        "events": event_path,
        "baseline": baseline_path,
        "sample": sample_path,
    }


class _ResidencyBackend:
    base_url = "http://ollama.test:11434"

    def __init__(self, rows=(), change=None):
        self.rows = list(rows)
        self.calls = []
        self._change = change

    def observe_loaded(self):
        return Observation(list(self.rows), "ollama:/api/ps", scope=self.base_url)

    def change_residency(self, model, keep_alive, *, resident, **_request):
        self.calls.append((model, keep_alive, resident))
        if self._change is not None:
            return self._change(model, keep_alive, resident)
        return {"ok": True, "outcome": "succeeded", "model": model,
                "resident": resident, "detail": "fake residency verified"}


def test_server_bare_unload_is_protected_by_latest_claim(
    isolated_runtime, monkeypatch,
):
    claims.claim("llama3.2:latest", "session-a", "generation")
    backend = _ResidencyBackend()
    monkeypatch.setattr(server, "_ollama", backend)

    result = server._unload_impl("llama3.2", False, "session-b")

    assert result["ok"] is False
    assert result["outcome"] == "refused"
    assert result["reason"] == "model_claimed"
    assert result["model"] == "llama3.2:latest"
    assert backend.calls == []


def test_server_bare_unload_is_protected_by_latest_busy_mapping(
    isolated_runtime, monkeypatch,
):
    backend = _ResidencyBackend()
    monkeypatch.setattr(server, "_ollama", backend)
    monkeypatch.setattr(server, "runner_pid_map", lambda: {"llama3.2:latest": 314})

    def busy_map(pids, *, index):
        assert pids == [314]
        assert index == server._GPU_INDEX
        return {314: True}

    monkeypatch.setattr(server._nvml, "nvml_busy_map", busy_map)

    result = server._unload_impl("llama3.2", False, "session-b")

    assert result["ok"] is False
    assert result["outcome"] == "refused"
    assert result["protected"] is True
    assert result["busy"] is True
    assert result["model"] == "llama3.2:latest"
    assert backend.calls == []


def test_claim_attempted_during_backend_unload_is_refused(
    isolated_runtime, monkeypatch,
):
    claim_error = []
    operation_seen = []

    def during_unload(model, _keep_alive, _resident):
        ledger = json.loads(isolated_runtime["claims"].read_text(encoding="utf-8"))
        operation_seen.extend(ledger["operations"])
        try:
            claims.claim(model, "other-session", "new generation")
        except ValueError as exc:
            claim_error.append(str(exc))
        return {"ok": True, "outcome": "succeeded", "model": model,
                "resident": False, "detail": "fake residency verified"}

    backend = _ResidencyBackend(change=during_unload)
    monkeypatch.setattr(server, "_ollama", backend)

    result = server._unload_impl("qwen3", True, "evictor")

    assert result["outcome"] == "succeeded"
    assert operation_seen[0]["model"] == "qwen3:latest"
    assert claim_error == ["operation pending for qwen3:latest"]
    assert claims.list_claims() == []
    assert json.loads(isolated_runtime["claims"].read_text(encoding="utf-8"))["operations"] == []


def test_ensure_free_observes_claim_added_during_first_candidate_unload(
    isolated_runtime, monkeypatch,
):
    rows = [
        {"name": "first", "size": 8 * 1024 * _MB,
         "size_vram": 8 * 1024 * _MB},
        {"name": "second", "size": 4 * 1024 * _MB,
         "size_vram": 4 * 1024 * _MB},
    ]
    operation_seen = []

    def first_unload_claims_second(model, _keep_alive, _resident):
        assert model == "first:latest"
        ledger = json.loads(isolated_runtime["claims"].read_text(encoding="utf-8"))
        operation_seen.extend(ledger["operations"])
        claims.claim("second:latest", "late-session", "generation")
        return {"ok": True, "outcome": "succeeded", "model": model,
                "resident": False, "detail": "fake residency verified"}

    backend = _ResidencyBackend(rows, first_unload_claims_second)
    monkeypatch.setattr(server, "_ollama", backend)
    monkeypatch.setattr(server, "runner_pid_map", lambda: {})
    monkeypatch.setattr(server._nvml, "nvml_busy_map", lambda *_args, **_kwargs: {})

    readings = iter((1000, 5000, 5000))

    def gpu_reading():
        free = next(readings)
        return [{"index": 0, "free_mb": free, "used_mb": 24000 - free,
                 "total_mb": 24000}]

    result = core.ensure_free(
        12, gpu_reading, backend, settle=0.5, sleep=lambda _seconds: None,
        evict_fn=lambda name: server._unload_impl(
            name, False, "ensure-session", action="ensure_free"
        ),
    )

    assert operation_seen[0]["model"] == "first:latest"
    assert backend.calls == [("first:latest", 0, False)]
    assert result["unloaded"] == ["first"]
    assert result["declined"][0]["name"] == "second"
    assert result["declined"][0]["reason"] == "model_claimed"
    assert claims.list_claims("second")[0]["owner"] == "late-session"


class _TimeoutSession:
    def post(self, *_args, **_kwargs):
        raise requests.Timeout("request timed out")

    def get(self, *_args, **_kwargs):
        raise requests.Timeout("reconciliation timed out")


def test_unknown_timeout_releases_owner_but_retains_lease_and_blocks_claim(
    isolated_runtime, monkeypatch,
):
    client = OllamaClient(
        base_url="http://ollama.test:11434", session=_TimeoutSession(), timeout=0.01,
    )
    monkeypatch.setattr(server, "_ollama", client)

    result = server._unload_impl("timeout-model", True, "session-a")

    assert result["ok"] is False
    assert result["outcome"] == "unknown"
    ledger = json.loads(isolated_runtime["claims"].read_text(encoding="utf-8"))
    assert len(ledger["operations"]) == 1
    operation = ledger["operations"][0]
    assert operation["model"] == "timeout-model:latest"
    assert operation["operation_id"] not in claims._OPERATION_LOCKS

    # Prove the OS owner was released independently of the still-active JSON lease.
    fd = claims._try_operation_lock(isolated_runtime["claims"], "timeout-model:latest")
    assert fd is not None
    claims._release_operation_lock(fd)
    with pytest.raises(ValueError, match="operation pending"):
        claims.claim("timeout-model:latest", "session-b", "generation")


def test_subprocess_live_owner_survives_expired_lease_in_list_coordination(tmp_path):
    """A real second process remains visible after its nominal lease expires."""
    path = tmp_path / "claims.json"
    ready = tmp_path / "ready"
    release = tmp_path / "release"
    script = """
import sys
import time
from pathlib import Path
from vram_mcp import claims

path = Path(sys.argv[1])
ready = Path(sys.argv[2])
release = Path(sys.argv[3])
operation = claims.begin_operation("live-model", "unload", force=True, path=path)
ready.write_text(operation["operation_id"], encoding="utf-8")
while not release.exists():
    time.sleep(0.01)
claims.finish_operation(operation["operation_id"], path=path)
"""
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(path), str(ready), str(release)],
        cwd=source_root, env=environment,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.exists(), process.communicate(timeout=1)
        record = json.loads(path.read_text(encoding="utf-8"))["operations"][0]
        started_at = claims._parse_iso(record["started_at"])
        future = started_at + timedelta(seconds=claims._OPERATION_LEASE_SECONDS + 1)
        state = claims.list_coordination(path=path, now_fn=lambda: future)
        [operation] = state["operations"]
        assert operation["operation_id"] == ready.read_text(encoding="utf-8")
        assert operation["owner_live"] is True
        assert operation["lease_expired"] is True
        assert operation["lifecycle"] == "in_flight"
        assert operation["outcome"] is None
    finally:
        release.write_text("done", encoding="utf-8")
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2)
            raise
    assert process.returncode == 0


def test_failed_readonly_ollama_observation_preserves_audit_baseline(
    isolated_runtime,
):
    scope = "http://ollama.test:11434"

    def status(loaded, observation):
        return {
            "gpus": [], "loaded": loaded, "other_processes": None,
            "observations": {"ollama": observation.metadata()},
            "pressure": {"free_mb": None, "state": "unknown"},
        }

    first = Observation(
        [{"name": "stable-model", "size_vram_mb": 4096}],
        "ollama:/api/ps", observed_at=_T0.isoformat(), scope=scope,
    )
    failed = Observation(
        None, "ollama:/api/ps",
        observed_at=(_T0 + timedelta(seconds=1)).isoformat(),
        error="connection refused", scope=scope,
    )
    recovered_empty = Observation(
        [], "ollama:/api/ps",
        observed_at=(_T0 + timedelta(seconds=2)).isoformat(), scope=scope,
    )

    server._run_detection(status(first.data, first))
    server._run_detection(status(None, failed))
    assert audit.read_events(path=isolated_runtime["events"]) == []

    server._run_detection(status(recovered_empty.data, recovered_empty))
    [event] = audit.read_events(path=isolated_runtime["events"])
    assert event["type"] == "disappeared"
    assert event["target"] == "stable-model:latest"
    assert event["scope"] == scope
