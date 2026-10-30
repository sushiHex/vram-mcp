"""Partial-fit and timing readings: change_residency, warm, and the audit event.

Everything is mocked: a fake Ollama session, an injected clock, and an audit
file under tmp_path. Nothing here touches ``~/.cache/vram-mcp``.
"""
import json
from datetime import datetime, timedelta, timezone

import pytest
import requests

from vram_mcp import audit, core, server
from vram_mcp.observations import Observation
from vram_mcp.ollama import OllamaClient

_REAL_LOG_ACTION = audit.log_action   # captured before any fixture patches it
_MIB = 1024 * 1024
_T0 = datetime(2026, 7, 16, 18, 0, 0, tzinfo=timezone.utc)


class _Response:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}

    def raise_for_status(self):
        if not 200 <= self.status_code < 300:
            raise requests.HTTPError(f"status {self.status_code}")

    def json(self):
        return self._json


class _Session:
    """Scripted /api/ps readings (one per GET, the last repeats) and a POST that
    advances the shared fake clock, so the request duration is exact."""

    def __init__(self, ps_readings, post_error=None, post_takes=2.5):
        self.ps_readings = list(ps_readings)   # each: list of rows, or None for a failed read
        self.post_error = post_error
        self.post_takes = post_takes
        self.now = 1000.0
        self.get_calls = []
        self.post_calls = []

    def clock(self):
        return self.now

    def get(self, url, timeout=None):
        self.get_calls.append(url)
        reading = self.ps_readings.pop(0) if len(self.ps_readings) > 1 else self.ps_readings[0]
        if reading is None:
            raise requests.ConnectionError("refused")
        return _Response(200, {"models": reading})

    def post(self, url, json=None, timeout=None):
        self.post_calls.append({"url": url, "json": json, "timeout": timeout})
        self.now += self.post_takes
        if self.post_error is not None:
            raise self.post_error
        return _Response(200, {"done": True})


def _row(size, size_vram, name="llama3:latest", **extra):
    row = {"name": name}
    if size is not None:
        row["size"] = size
    if size_vram is not None:
        row["size_vram"] = size_vram
    row.update(extra)
    return row


def _warm(session, **kwargs):
    return OllamaClient(session=session).change_residency(
        "llama3", "5m", resident=True, sleep=lambda _: None, clock=session.clock, **kwargs)


# ---- change_residency: timing (astra review) --------------------------------

class _SlowJson(_Response):
    def __init__(self, session, takes):
        super().__init__(200, {"done": True})
        self._session, self._takes = session, takes

    def json(self):
        self._session.now += self._takes
        return super().json()


def test_elapsed_s_excludes_parsing_the_response():
    session = _Session([[_row(_MIB, _MIB)]], post_takes=2.0)
    real_post = session.post
    session.post = lambda *a, **k: (real_post(*a, **k), _SlowJson(session, 7.0))[1]
    assert _warm(session)["elapsed_s"] == 2.0


def test_a_rejected_request_keeps_its_timing():
    session = _Session([[]], post_takes=1.5)
    session.post = lambda *a, **k: (setattr(session, "now", session.now + 1.5), _Response(404))[1]
    result = _warm(session, wall=lambda: _T0)
    assert result["outcome"] == "failed"
    assert result["elapsed_s"] == 1.5 and result["exception"] is None
    assert result["started_at"] == "2026-07-16T18:00:00Z"


def test_started_at_is_the_wall_clock_when_the_request_began():
    result = _warm(_Session([[_row(_MIB, _MIB)]]), wall=lambda: _T0)
    assert result["started_at"] == "2026-07-16T18:00:00Z"


# ---- change_residency: partial fit -----------------------------------------

def test_a_zero_size_row_proves_nothing_about_the_fit():
    result = _warm(_Session([[_row(0, 0)]]))
    assert result["partial_fit"] is None and result["offloaded_mb"] is None

def test_a_model_partly_outside_vram_reports_partial_fit_and_the_offloaded_mb():
    size, vram = 10_000 * _MIB + 12345, 7_000 * _MIB
    result = _warm(_Session([[_row(size, vram)]]))
    assert result["outcome"] == "succeeded"
    assert result["partial_fit"] is True
    assert result["offloaded_mb"] == 3_000          # floor, not round: 3000 MiB + 12345 B


def test_a_model_wholly_in_vram_reports_no_partial_fit_and_zero_offload():
    result = _warm(_Session([[_row(4 * _MIB * 1000, 4 * _MIB * 1000)]]))
    assert result["outcome"] == "succeeded"
    assert result["partial_fit"] is False
    assert result["offloaded_mb"] == 0


def test_a_missing_row_leaves_the_offload_unknown():
    # Unacknowledged warm that is not resident: outcome unknown, no row to read.
    session = _Session([[_row(1000, 500, name="other:latest")]], post_error=requests.ReadTimeout("slow"))
    result = _warm(session)
    assert result["outcome"] == "unknown"
    assert result["offloaded_mb"] is None and result["partial_fit"] is None


def test_a_failed_final_read_does_not_inherit_an_earlier_rows_offload():
    """Read 1 saw a partial fit but could not verify (the request timed out);
    reads 2 and 3 failed. The stale row must not describe the final state."""
    session = _Session([[_row(10_000 * _MIB, 6_000 * _MIB)], None],
                       post_error=requests.ReadTimeout("slow"))
    result = _warm(session)
    assert len(session.get_calls) == 3
    assert result["outcome"] == "unknown"
    assert result["offloaded_mb"] is None and result["partial_fit"] is None
    assert result["resident"] is None               # read 1's "resident" is stale too
    assert "resident" not in server._audit_fields(result)


def test_a_row_without_size_vram_leaves_the_offload_unknown():
    result = _warm(_Session([[_row(10_000 * _MIB, None)]]))
    assert result["outcome"] == "succeeded"
    assert result["offloaded_mb"] is None and result["partial_fit"] is None


def test_size_vram_larger_than_size_leaves_the_offload_unknown():
    result = _warm(_Session([[_row(1_000 * _MIB, 2_000 * _MIB)]]))
    assert result["outcome"] == "succeeded"
    assert result["offloaded_mb"] is None and result["partial_fit"] is None


@pytest.mark.parametrize("size, size_vram", [
    ("10000", "5000"), (1e10, 5e9), (True, False), (-5, -10), (None, 5000),
])
def test_non_integer_or_negative_sizes_leave_the_offload_unknown(size, size_vram):
    result = _warm(_Session([[_row(size, size_vram)]]))
    assert result["offloaded_mb"] is None and result["partial_fit"] is None


# ---- change_residency: elapsed_s and exception ------------------------------

def test_a_successful_residency_change_reports_its_duration_and_no_exception():
    session = _Session([[_row(1000, 1000)]], post_takes=1.23456)
    result = _warm(session)
    assert result["outcome"] == "succeeded"
    assert result["elapsed_s"] == 1.235            # rounded to 3 decimals
    assert result["exception"] is None


def test_a_read_timeout_reports_its_duration_and_exception_name():
    session = _Session([[]], post_error=requests.ReadTimeout("slow"), post_takes=110.0)
    result = _warm(session)
    assert result["outcome"] == "unknown" and result["reason"] == "load_timeout"
    assert result["elapsed_s"] == 110.0
    assert result["exception"] == "ReadTimeout"


def test_a_connection_error_reports_its_duration_and_exception_name():
    session = _Session([[]], post_error=requests.ConnectionError("refused"), post_takes=0.25)
    result = _warm(session)
    assert result["outcome"] == "unknown" and result["reason"] == "request_interrupted"
    assert result["elapsed_s"] == 0.25
    assert result["exception"] == "ConnectionError"


def test_elapsed_s_measures_the_request_not_the_reconciliation_reads():
    session = _Session([None], post_error=requests.ReadTimeout("slow"), post_takes=3.0)
    real_get = session.get

    def slow_get(url, timeout=None):
        session.now += 50.0
        return real_get(url, timeout)

    session.get = slow_get
    assert _warm(session)["elapsed_s"] == 3.0


# ---- log_action fields ------------------------------------------------------

def test_log_action_fields_cannot_override_core_keys(tmp_path):
    p = tmp_path / "events.jsonl"
    audit.log_action(
        action="warm", target="llama3", kind="ollama", actor="tester", force=False,
        outcome="succeeded", detail="real detail",
        fields={"ts": "1999-01-01T00:00:00Z", "type": "sample", "kind": "process",
                "target": "evil", "action": "unload", "actor": "mallory", "force": True,
                "outcome": "failed", "detail": "forged", "partial_fit": True},
        now_fn=lambda: _T0, path=p)
    (event,) = audit.read_events(path=p)
    assert event["ts"] == "2026-07-16T18:00:00Z" and event["type"] == "action"
    assert event["kind"] == "ollama" and event["target"] == "llama3:latest"
    assert event["action"] == "warm" and event["actor"] == "tester"
    assert event["force"] is False and event["outcome"] == "succeeded"
    assert event["detail"] == "real detail"
    assert event["partial_fit"] is True            # ordinary fields still merge


def test_log_action_keeps_numbers_flags_tokens_and_token_lists(tmp_path):
    p = tmp_path / "events.jsonl"
    fields = {"started_at": "2026-07-16T17:58:30Z", "elapsed_s": 90.5, "num_ctx": 64000,
              "partial_fit": False, "exception": "ReadTimeout", "reason": "load_timeout",
              "unavailable": ["gpu", "paging"], "offloaded_mb": None}
    audit.log_action(action="warm", target="llama3", kind="ollama",
                     fields=fields, now_fn=lambda: _T0, path=p)
    (event,) = audit.read_events(path=p)
    assert {k: event[k] for k in fields} == fields


@pytest.mark.parametrize("value", [
    r"C:\Users\someone\secret.txt", "tell me a story about", "x" * 65,
    ["gpu", "C:/private/path"], {"nested": 1}, float("inf"), float("nan")])
def test_log_action_drops_free_text_paths_and_non_finite_values(tmp_path, value):
    """The boundary is enforced here, not trusted to every producer."""
    p = tmp_path / "events.jsonl"
    audit.log_action(action="warm", target="llama3", kind="ollama",
                     fields={"reason": value, "partial_fit": True}, now_fn=lambda: _T0, path=p)
    (event,) = audit.read_events(path=p)
    assert "reason" not in event and event["partial_fit"] is True


def test_log_action_fields_cannot_supply_a_scope(tmp_path):
    p = tmp_path / "events.jsonl"
    audit.log_action(action="warm", target="llama3", kind="ollama",
                     fields={"scope": "forged"}, now_fn=lambda: _T0, path=p)
    (event,) = audit.read_events(path=p)
    assert "scope" not in event


# ---- warm through the server ------------------------------------------------

_SECRET = r"C:\Users\someone\secret\nvidia-smi.exe: driver exploded"


@pytest.fixture
def warmed(monkeypatch, tmp_path):
    """server._warm_impl against a fake Ollama, with a real audit file."""
    events = tmp_path / "events.jsonl"
    session = _Session([[_row(10_000 * _MIB, 7_000 * _MIB, name="m:latest")]], post_takes=2.5)
    monkeypatch.setattr(server._claims, "_DEFAULT_PATH", tmp_path / "claims.json")
    monkeypatch.setattr(server, "_AUDIT_ON", True)
    monkeypatch.setattr(server._ollama, "session", session)
    monkeypatch.setattr(server._audit, "log_action", lambda **kw: _REAL_LOG_ACTION(
        **kw, now_fn=lambda: _T0, path=events))
    monkeypatch.setattr(core, "combined_status", lambda *a, **k: {
        "gpus": [], "loaded": [], "free_mb": 20000, "pressure": {},
        "observations": {"gpu": {"status": "unavailable", "error": _SECRET},
                         "wddm": {"status": "available"}}})
    monkeypatch.setattr(server, "_active_claims", lambda: ([], True))
    monkeypatch.setattr(server._ollama, "observe_tags",
                        lambda: Observation({"m:latest": 4000}, "sizes"))
    unloads = []
    monkeypatch.setattr(server._ollama, "unload", lambda *a, **k: unloads.append((a, k)) or True)
    # change_residency binds time.monotonic as a default argument, so inject the
    # fake clock (and skip the real sleeps) through the call instead.
    real = server._ollama.change_residency
    monkeypatch.setattr(server._ollama, "change_residency", lambda *a, **k: real(
        *a, clock=session.clock, sleep=lambda _: None,
        wall=lambda: _T0 - timedelta(seconds=40), **k))
    return session, events, unloads


def test_warm_with_a_partial_fit_still_succeeds_without_a_retry_or_unload(warmed):
    session, _events, unloads = warmed
    result = server._warm_impl("m", "5m", "tester", False)
    assert result["outcome"] == "succeeded" and result["ok"] is True
    assert result["partial_fit"] is True
    assert result["offloaded_mb"] == 3_000
    assert "partial fit: 3000 MB of the model is outside VRAM" in result["detail"]
    assert len(session.post_calls) == 1
    assert unloads == []
    assert all(call["json"]["keep_alive"] != 0 for call in session.post_calls)


def test_a_sub_megabyte_spill_reads_as_under_1_mb(warmed):
    session, _events, _unloads = warmed
    session.ps_readings = [[_row(10_000 * _MIB, 10_000 * _MIB - 4096, name="m:latest")]]
    result = server._warm_impl("m", "5m", "tester", False)
    assert result["partial_fit"] is True and result["offloaded_mb"] == 0
    assert "partial fit: under 1 MB of the model" in result["detail"]


def test_warm_without_a_partial_fit_adds_no_partial_fit_text(warmed):
    session, _events, _unloads = warmed
    session.ps_readings = [[_row(4_000 * _MIB, 4_000 * _MIB, name="m:latest")]]
    result = server._warm_impl("m", "5m", "tester", False)
    assert result["partial_fit"] is False and result["offloaded_mb"] == 0
    assert "partial fit" not in result["detail"]


def test_the_warm_audit_event_records_the_partial_fit_and_timing(warmed):
    _session, events, _unloads = warmed
    server._warm_impl("m", "5m", "tester", False)
    (event,) = audit.read_events(path=events)
    assert event["type"] == "action" and event["action"] == "warm"
    assert event["outcome"] == "succeeded"
    assert event["partial_fit"] is True
    assert event["offloaded_mb"] == 3_000
    assert event["elapsed_s"] == 2.5
    # Recorded when the POST began, not derived from the write time.
    assert event["started_at"] == "2026-07-16T17:59:20Z"
    assert event["unavailable"] == ["gpu"]          # names only, sorted, no "wddm"


def test_the_warm_audit_event_carries_no_null_fields_and_no_error_text(warmed):
    _session, events, _unloads = warmed
    server._warm_impl("m", "5m", "tester", False)
    raw = events.read_text(encoding="utf-8")
    (event,) = [json.loads(line) for line in raw.splitlines() if line.strip()]
    nulled = [key for key in server._AUDIT_FIELDS if key in event and event[key] is None]
    assert nulled == []
    assert "exception" not in event                # None on success, so omitted
    assert "secret" not in raw and "driver exploded" not in raw
    assert "error" not in event and "observations" not in event


def test_audit_fields_keeps_zero_and_false_but_drops_none_and_error_text():
    fields = server._audit_fields({
        "offloaded_mb": 0, "partial_fit": False, "strict_fit": False, "exception": None,
        "reason": None, "elapsed_s": 0.0, "detail": "not a listed field",
        "observations": {"b": {"status": "unavailable", "error": "leak"},
                         "a": {"status": "stale", "error": "leak"},
                         "c": {"status": "available"}, "d": "not a mapping"}})
    assert fields == {"offloaded_mb": 0, "partial_fit": False, "strict_fit": False,
                      "elapsed_s": 0.0, "unavailable": ["a", "b"]}


def test_audit_fields_omits_unavailable_when_every_source_is_available():
    assert server._audit_fields({"observations": {"gpu": {"status": "available"}}}) == {}
