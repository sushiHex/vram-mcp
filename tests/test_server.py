"""Tests for the thin wiring in vram_mcp.server.

The server module is the only one that imports ``mcp``; it is skipped rather
than failed where that package is absent, so the pure-module suite still runs
anywhere. Every audit call here is monkeypatched: these tests must never touch
the real ``~/.cache/vram-mcp`` files.
"""
import json

import pytest
from pydantic import ValidationError

from vram_mcp import core, server, wddm
from vram_mcp.observations import Observation


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    import requests
    monkeypatch.setattr(server._claims, "_DEFAULT_PATH", tmp_path / "claims.json")
    monkeypatch.setattr(server._audit, "log_action", lambda **kwargs: None)
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must inject network and GPU collectors")
    monkeypatch.setattr(requests.Session, "request", forbidden)
    monkeypatch.setattr(server, "_gpu_reading", forbidden)
    monkeypatch.setattr(server, "_procinfo_table", forbidden)
    # The ledger is isolated above, so only the real process and NVML
    # collectors behind a correlating snapshot need forbidding.
    monkeypatch.setattr(server, "runner_pid_map", forbidden)
    monkeypatch.setattr(server._nvml, "nvml_busy_map", forbidden)



@pytest.fixture
def audit_spy(monkeypatch):
    """Record what _run_detection asks the audit to do, writing nothing."""
    calls = {"detect": [], "sample": []}

    def fake_detect(holders, **kwargs):
        calls["detect"].append(holders)
        return []

    def fake_sample(sample, **kwargs):
        calls["sample"].append(sample)
        return True

    monkeypatch.setattr(server._audit, "detect_and_log", fake_detect)
    monkeypatch.setattr(server._audit, "maybe_log_sample", fake_sample)
    return calls


def _status(gpus, loaded=(), procs=()):
    return {
        "gpus": list(gpus), "loaded": list(loaded),
        "other_processes": list(procs),
        "observations": {"ollama": Observation(list(loaded), "ollama", scope="ollama").metadata()},
        "pressure": {"free_mb": None if not gpus else 500,
                     "non_local_mb": 0, "state": "ok"},
    }


def test_the_server_announces_its_own_version():
    """Left alone, the server reports the MCP SDK's version as its own."""
    import vram_mcp
    options = server.mcp._mcp_server.create_initialization_options()
    assert (options.server_name, options.server_version) == ("vram-mcp", vram_mcp.__version__)


def test_run_detection_skips_sample_when_gpu_data_absent(audit_spy):
    """list_loaded() deliberately skips the nvidia-smi spawn, so its status has
    no GPU rows. Recording {free_mb: None, used_mb: 0, total_mb: 0} would write
    a fiction AND burn the shared throttle slot, dropping the next REAL sample.
    Detection still has to run — only the sample is skipped."""
    server._run_detection(_status(gpus=[], loaded=[{"name": "m", "size_vram_mb": 8000}]))
    assert audit_spy["sample"] == []      # no fake zeros, no throttle slot burned
    assert len(audit_spy["detect"]) == 1  # holder diff still ran


def test_run_detection_samples_when_gpu_data_present(audit_spy):
    server._run_detection(_status(
        gpus=[{"used_mb": 20000, "total_mb": 24576}], loaded=[]))
    assert len(audit_spy["sample"]) == 1
    assert audit_spy["sample"][0]["total_mb"] == 24576


def test_run_detection_samples_the_unexplained_spill_not_the_raw_non_local(audit_spy):
    """A deliberately CPU-offloaded model shows up as its runner's Non Local
    Usage. Recording that as spill would make trend() report the GPU 'spilling
    to system RAM' for the whole life of a normal 32B load."""
    status = _status(gpus=[{"used_mb": 23768, "total_mb": 24576}])
    status["pressure"] = {"free_mb": 559, "non_local_mb": 3950,
                          "explained_offload_mb": 3906,
                          "unexplained_spill_mb": 44, "state": "degraded"}
    server._run_detection(status)
    sample = audit_spy["sample"][0]
    assert sample["spill_mb"] == 44
    # the conflated figure must not be persisted under its old name
    assert "non_local_mb" not in sample


class _StubPaging:
    """A paging monitor whose window is quiet (non-sustained)."""

    def observe(self):
        return Observation(wddm.Paging(((1.0, 0.0),) * 10), "wddm:pdh", scope="gpu:index=0")


def _capture_combined_status(monkeypatch):
    seen = {}

    def fake_combined_status(gpu_fn, ollama, **kwargs):
        seen.update(kwargs)
        return {"gpus": [], "loaded": [], "free_mb": None, "pressure": {}}

    monkeypatch.setattr(core, "combined_status", fake_combined_status)
    return seen


def test_full_status_reads_paging_from_the_monitor(monkeypatch):
    """Without paging_fn the verdict could never be `ok` or `thrashing`."""
    seen = _capture_combined_status(monkeypatch)
    paging = _StubPaging()
    monkeypatch.setattr(server, "_paging", paging)
    server._full_status()
    assert seen["paging_fn"] == paging.observe
    assert "spill_threshold_mb" not in seen


def test_full_status_has_no_paging_reader_where_there_is_no_monitor(monkeypatch):
    """The monitor exists on Windows only; elsewhere paging stays uncovered."""
    seen = _capture_combined_status(monkeypatch)
    monkeypatch.setattr(server, "_paging", None)
    server._full_status()
    assert seen["paging_fn"] is None


def test_the_monitor_exists_on_windows_only():
    import sys
    assert (server._paging is not None) == (sys.platform == "win32")


def test_the_spill_knob_is_gone():
    assert not hasattr(server, "_SPILL_MB")


def test_reserve_tool_surfaces_a_rejected_gb_as_a_structured_error(monkeypatch, tmp_path):
    """A bad argument must degrade like every other ledger failure — an
    {ok: False, summary} payload, never a raw traceback through MCP.

    The ledger path is redirected explicitly: this call reaches the REAL
    ``_claims.reserve``, and it writes nothing today only because validation
    happens to run before the lock is taken. Relying on that ordering would let
    a future reshuffle quietly start editing the user's live claims.json."""
    monkeypatch.setattr(server._claims, "_DEFAULT_PATH", tmp_path / "claims.json")
    result = server._reserve_impl(-8.0, "sneaky", "cancel yours", 60, None)
    assert result["ok"] is False
    assert "gb" in result["summary"]
    assert not (tmp_path / "claims.json").exists()


def test_trend_empty_window_names_missing_gpu_readings_as_a_cause(monkeypatch):
    """Sampling is skipped whenever a status call has no GPU rows — the case for
    anyone without a working nvidia-smi. Blaming only the throttle and
    VRAM_MCP_AUDIT=0 sends that user chasing two causes that aren't theirs."""
    monkeypatch.setattr(server._audit, "read_events", lambda **kwargs: [])
    text = server._trend_impl(1.0)["summary"]
    assert "nvidia-smi" in text
    assert "VRAM_MCP_AUDIT=0" in text


class _StubResidency:
    """An Ollama stand-in that reports a known, empty residency."""

    base_url = "http://ollama.test:11434"

    def __init__(self, rows):
        self._rows = list(rows)

    def observe_loaded(self):
        return Observation(self._rows, "ollama:/api/ps")

    def ps(self):
        return list(self._rows)


def test_status_summary_says_why_pressure_is_unknown(monkeypatch):
    """`summary` is the line an agent reads before the payload. "pressure:
    unknown." alone gives it nothing to act on, and which evidence was missing
    is the entire content of that verdict — so the detail belongs inline."""
    monkeypatch.setattr(server, "_gpu_reading",
                        lambda: [{"index": 0, "total_mb": 24576,
                                  "used_mb": 4000, "free_mb": 20576}])
    monkeypatch.setattr(server, "_procinfo_table",
                        lambda: Observation([], "procs",
                                            coverage={"non_local_memory": True}))
    monkeypatch.setattr(server, "_paging", None)
    monkeypatch.setattr(server, "_ollama", _StubResidency([]))
    monkeypatch.setattr(server, "_run_detection", lambda status: None)

    status = server._vram_status_impl()
    assert status["pressure"]["state"] == "unknown"
    assert "pressure: unknown." in status["summary"]
    assert "Driver paging is unavailable" in status["summary"]


def test_status_summary_stays_terse_when_pressure_is_ok(monkeypatch):
    """"No VRAM pressure detected." adds nothing to "pressure: ok"."""
    monkeypatch.setattr(server, "_gpu_reading",
                        lambda: [{"index": 0, "total_mb": 24576,
                                  "used_mb": 4000, "free_mb": 20576}])
    monkeypatch.setattr(server, "_procinfo_table",
                        lambda: Observation([], "procs",
                                            coverage={"non_local_memory": True}))
    monkeypatch.setattr(server, "_paging", _StubPaging())
    monkeypatch.setattr(server, "_ollama", _StubResidency([]))
    monkeypatch.setattr(server, "_run_detection", lambda status: None)

    status = server._vram_status_impl()
    assert status["pressure"]["state"] == "ok"
    assert status["summary"].endswith("pressure: ok.")


def _idle_status_sources(monkeypatch):
    monkeypatch.setattr(server, "_gpu_reading",
                        lambda: [{"index": 0, "total_mb": 24576,
                                  "used_mb": 4000, "free_mb": 20576}])
    monkeypatch.setattr(server, "_procinfo_table",
                        lambda: Observation([], "procs",
                                            coverage={"non_local_memory": True}))
    monkeypatch.setattr(server, "_ollama", _StubResidency([]))
    monkeypatch.setattr(server, "_run_detection", lambda status: None)


@pytest.mark.parametrize("impl", ["_vram_status_impl", "_list_loaded_impl"])
def test_summary_names_an_unreadable_ledger_with_nothing_loaded(monkeypatch, impl):
    """Without this, an agent cannot tell "no claims" from "claims unreadable"
    until it calls something that refuses."""
    _idle_status_sources(monkeypatch)
    server._claims._DEFAULT_PATH.write_text(
        json.dumps({"protocol": server._claims._PROTOCOL + 1, "claims": []}),
        encoding="utf-8")
    status = getattr(server, impl)()
    assert status["observations"]["coordination"]["status"] == "unavailable"
    assert "Coordination unavailable: claim ledger" in status["summary"]
    assert f"protocol {server._claims._PROTOCOL + 1}" in status["summary"]
    assert ".." not in status["summary"]


# ---- vram_status cmdline_chars ---------------------------------------------

def _stub_full_status(monkeypatch, rows, detection_seen=None):
    """Make _full_status return a fixed payload; record what detection sees."""
    status = {"gpus": [], "loaded": [], "free_mb": 20000, "pressure": {},
              "observations": {}, "other_processes": rows}
    monkeypatch.setattr(server, "_full_status", lambda: status)
    monkeypatch.setattr(
        server, "_run_detection",
        lambda s: detection_seen is not None and detection_seen.append(
            [dict(r) for r in s["other_processes"]]))
    return status


def test_cmdline_default_cap_cuts_long_lines_and_keeps_short_and_null(monkeypatch):
    n = server._CMDLINE_CHARS
    rows = [{"pid": 1, "cmdline": "a" * (n + 1)},
            {"pid": 2, "cmdline": "b" * n},
            {"pid": 3, "cmdline": "short"},
            {"pid": 4, "cmdline": None}]
    _stub_full_status(monkeypatch, rows)
    out = server._vram_status_impl()["other_processes"]
    assert out[0]["cmdline"] == "a" * n + "…"
    assert out[1]["cmdline"] == "b" * n
    assert out[2]["cmdline"] == "short"
    assert out[3]["cmdline"] is None


def test_cmdline_chars_none_returns_lines_whole(monkeypatch):
    long = "x" * 1000
    _stub_full_status(monkeypatch, [{"pid": 1, "cmdline": long}])
    out = server._vram_status_impl(cmdline_chars=None)["other_processes"]
    assert out[0]["cmdline"] == long


def test_cmdline_chars_zero_removes_the_key_rather_than_nulling_it(monkeypatch):
    """`cmdline: null` already means "could not be read"."""
    _stub_full_status(monkeypatch, [{"pid": 1, "cmdline": "abc"},
                                    {"pid": 2, "cmdline": None}])
    out = server._vram_status_impl(cmdline_chars=0)["other_processes"]
    assert all("cmdline" not in row for row in out)
    assert [row["pid"] for row in out] == [1, 2]


def test_cmdline_cap_does_not_mutate_the_source_rows(monkeypatch):
    rows = [{"pid": 1, "cmdline": "y" * 50}]
    _stub_full_status(monkeypatch, rows)
    server._vram_status_impl(cmdline_chars=5)
    server._vram_status_impl(cmdline_chars=0)
    assert rows == [{"pid": 1, "cmdline": "y" * 50}]


@pytest.mark.parametrize("chars", [160, 0, None])
def test_unavailable_process_reading_stays_none_not_empty(monkeypatch, chars):
    _stub_full_status(monkeypatch, None)
    assert server._vram_status_impl(cmdline_chars=chars)["other_processes"] is None


def test_detection_sees_whole_command_lines_before_capping(monkeypatch):
    seen = []
    _stub_full_status(monkeypatch, [{"pid": 1, "cmdline": "z" * 500}], seen)
    out = server._vram_status_impl(cmdline_chars=10)
    assert seen == [[{"pid": 1, "cmdline": "z" * 500}]]
    assert out["other_processes"][0]["cmdline"] == "z" * 10 + "…"


@pytest.mark.parametrize("bad", [-1, True, False, 1.5, 160.0, "160"])
def test_sdk_rejects_cmdline_chars_it_would_otherwise_coerce(bad):
    """Lax validation turns true into 1 and "160" into 160 before the tool runs."""
    metadata = server.mcp._tool_manager.get_tool("vram_status").fn_metadata
    with pytest.raises(ValidationError):
        metadata.arg_model.model_validate({"cmdline_chars": bad})


@pytest.mark.parametrize("good", [0, 160, None])
def test_sdk_accepts_cmdline_chars(good):
    metadata = server.mcp._tool_manager.get_tool("vram_status").fn_metadata
    args = metadata.arg_model.model_validate({"cmdline_chars": good})
    assert args.model_dump_one_level() == {"cmdline_chars": good}


@pytest.mark.parametrize("bad", [-1, True, False, 1.5, "160"])
def test_invalid_cmdline_chars_is_refused_without_collecting_status(monkeypatch, bad):
    def boom():
        raise AssertionError("_full_status must not run")
    monkeypatch.setattr(server, "_full_status", boom)
    result = server._vram_status_impl(cmdline_chars=bad)
    assert result["ok"] is False
    assert result["outcome"] == "refused"
    assert result["summary"]


# ---- warm() admission ------------------------------------------------------

@pytest.fixture
def warm_env(monkeypatch):
    """A GPU with 20 GB free and one 8 GB reservation held by another session."""
    monkeypatch.setattr(server._audit, "log_action", lambda **kwargs: None)
    monkeypatch.setattr(core, "combined_status", lambda *a, **k: {
        "gpus": [], "loaded": [], "free_mb": 20000, "pressure": {}, "observations": {}})
    monkeypatch.setattr(server, "_active_claims", lambda: (
        [{"kind": "reservation", "gb": 8, "owner": "trainer", "purpose": "sd"}], True))
    monkeypatch.setattr(server._ollama, "change_residency", lambda model, keep_alive, **kwargs: {
        "ok": True, "outcome": "succeeded", "detail": "Residency verified"})
    return monkeypatch


def _record_residency(monkeypatch):
    calls = []

    def change_residency(model, keep_alive, **kwargs):
        calls.append({"model": model, "keep_alive": keep_alive, **kwargs})
        return {"ok": True, "outcome": "succeeded", "detail": "Residency verified"}

    monkeypatch.setattr(server._ollama, "change_residency", change_residency)
    return calls


def test_warm_with_unverifiable_size_is_refused_while_a_reservation_needs_headroom(warm_env):
    """tags() returns {} on ANY failure, so a timed-out /api/tags makes EVERY
    model 'size unknown'. A 20 GB model must not sail through an 8 GB
    reservation on a size nobody read."""
    calls = _record_residency(warm_env)
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({}, "sizes"))
    result = server._warm_impl("qwen3:32b", "5m", "tester", False)
    assert result["outcome"] == "refused"
    assert result["reason"] == "size_unknown"
    assert "8192 MB" in result["summary"] and "force=True" in result["summary"]
    assert calls == []


def test_warm_with_unknown_free_memory_is_refused_without_a_request(warm_env):
    """#46: nvidia-smi returned no GPU, so the fit was never checked, yet the
    load was sent. The refusal carries the observation error behind it."""
    calls = _record_residency(warm_env)
    warm_env.setattr(core, "combined_status", lambda *a, **k: {
        "gpus": [], "loaded": [], "free_mb": None, "pressure": {},
        "observations": {"gpu": {"status": "unavailable", "error": "nvidia-smi found no GPU"}}})
    warm_env.setattr(server, "_active_claims", lambda: ([], True))
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"phi4-mini:latest": 2500}, "sizes"))
    result = server._warm_impl("phi4-mini", "5m", "tester", False)
    _validate_tool_output("warm", result)
    assert result["outcome"] == "refused"
    assert result["reason"] == "free_unknown"
    assert "nvidia-smi found no GPU" in result["detail"]
    assert "force=True" in result["detail"]
    assert calls == []


def _show(info):
    return lambda model: Observation(info, "ollama:/api/show")


_KV_INFO = {
    "general.architecture": "llama", "llama.block_count": 32,
    "llama.attention.head_count_kv": 8, "llama.attention.key_length": 128,
    "llama.attention.value_length": 128,
}


def test_warm_counts_the_requested_context_in_admission(warm_env):
    """#44: a 4 GB model fits the 11.8 GB headroom left by an 8 GB
    reservation, but not with the 8 GB KV cache a 64K context adds."""
    calls = _record_residency(warm_env)
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"m:latest": 4000}, "sizes"))
    warm_env.setattr(server._ollama, "observe_show", _show(_KV_INFO), raising=False)

    small = server._warm_impl("m", "5m", "tester", False)
    assert small["reason"] == "fits" and small["kv_cache_mb"] is None
    assert calls[-1]["num_ctx"] is None

    long = server._warm_impl("m", "5m", "tester", False, num_ctx=65536)
    _validate_tool_output("warm", long)
    assert long["outcome"] == "refused"
    assert long["reason"] == "insufficient_headroom"
    assert long["kv_cache_mb"] == 8192
    assert long["model_size_mb"] == 4000 + 8192
    assert long["num_ctx"] == 65536
    assert len(calls) == 1


def test_warm_forwards_num_ctx_and_reports_the_estimate_as_unverified(warm_env):
    calls = _record_residency(warm_env)
    warm_env.setattr(server, "_active_claims", lambda: ([], True))
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"m:latest": 4000}, "sizes"))
    warm_env.setattr(server._ollama, "observe_show", _show(_KV_INFO), raising=False)
    result = server._warm_impl("m", "5m", "tester", False, num_ctx=8192)
    assert result["ok"] is True
    assert calls[0]["num_ctx"] == 8192
    assert result["kv_cache_mb"] == 1024
    assert result["additional_mb"] == 4000 + 1024
    assert "f16" in result["detail"] and "not verified" in result["detail"]


def test_warm_with_num_ctx_and_no_metadata_says_so_instead_of_guessing(warm_env):
    calls = _record_residency(warm_env)
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"m:latest": 4000}, "sizes"))
    warm_env.setattr(server._ollama, "observe_show", lambda model: Observation(
        None, "ollama:/api/show", error="Ollama unavailable or invalid response"), raising=False)
    result = server._warm_impl("m", "5m", "tester", False, num_ctx=65536)
    assert result["outcome"] == "refused"
    assert result["reason"] == "size_unknown"
    assert result["kv_cache_mb"] is None
    assert result["model_size_mb"] is None
    assert "KV cache" in result["detail"] and "Ollama unavailable" in result["detail"]
    assert result["observations"]["architecture"]["status"] == "unavailable"
    assert calls == []


@pytest.mark.parametrize("context_length", [4096, None])
def test_warm_resident_at_another_context_is_a_new_load(warm_env, context_length):
    """Ollama reloads a model whose num_ctx changes, so its residency at
    another (or an unreported) context is no evidence of zero cost."""
    _record_residency(warm_env)
    row = {"name": "m:latest", "size_vram_mb": 4000}
    if context_length is not None:
        row["context_length"] = context_length
    warm_env.setattr(core, "combined_status", lambda *a, **k: {
        "loaded": [row], "free_mb": 20000, "observations": {}})
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"m:latest": 4000}, "sizes"))
    warm_env.setattr(server._ollama, "observe_show", _show(_KV_INFO), raising=False)
    result = server._warm_impl("m", "5m", "tester", False, num_ctx=65536)
    assert result["reason"] == "insufficient_headroom"


def test_warm_resident_at_the_requested_context_needs_no_capacity(warm_env):
    _record_residency(warm_env)
    warm_env.setattr(core, "combined_status", lambda *a, **k: {
        "loaded": [{"name": "m:latest", "size_vram_mb": 12000, "context_length": 65536}],
        "free_mb": 100, "observations": {}})
    result = server._warm_impl("m", "5m", "tester", False, num_ctx=65536)
    assert result["ok"] is True
    assert result["reason"] == "already_resident"


@pytest.mark.parametrize("num_ctx", [0, -1, True, 1.5, "4096", server._MAX_NUM_CTX + 1])
def test_warm_refuses_an_invalid_num_ctx(num_ctx):
    result = server._warm_impl("m", "5m", "tester", True, num_ctx=num_ctx)
    assert result["outcome"] == "refused"
    assert result["reason"] == "invalid_num_ctx"


@pytest.mark.parametrize("timeout", [0, -1, True, "300", float("nan"), float("inf"),
                                     server._MAX_LOAD_TIMEOUT + 1])
def test_warm_refuses_an_invalid_timeout(timeout):
    result = server._warm_impl("m", "5m", "tester", True, timeout_seconds=timeout)
    assert result["outcome"] == "refused"
    assert result["reason"] == "invalid_timeout"


def test_warm_timeout_reaches_the_request_and_its_lease_outlasts_it(warm_env):
    calls = _record_residency(warm_env)
    leases = []
    begin = server._claims.begin_operation

    def spy(*args, **kwargs):
        leases.append(kwargs.get("lease_seconds"))
        return begin(*args, **kwargs)

    warm_env.setattr(server._claims, "begin_operation", spy)
    server._warm_impl("m", "5m", "tester", True, timeout_seconds=600)
    server._warm_impl("m", "5m", "tester", True)
    assert [call["timeout"] for call in calls] == [600, server._LOAD_TIMEOUT]
    assert leases[0] > 600
    assert leases[1] >= server._claims._OPERATION_LEASE_SECONDS


def test_a_warm_timeout_is_not_masked_by_the_admission_reason(warm_env):
    """The admission verdict ("fits") used to overwrite the result's reason, so
    a slow load read as a fit and its operation recorded no cause."""
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"m:latest": 1000}, "sizes"))
    warm_env.setattr(server._ollama, "change_residency", lambda model, keep_alive, **kwargs: {
        "ok": False, "outcome": "unknown", "reason": "load_timeout", "resident": None,
        "detail": "Ollama did not answer within 110 s"})
    result = server._warm_impl("m", "5m", "tester", False)
    _validate_tool_output("warm", result)
    assert result["outcome"] == "unknown"
    assert result["reason"] == "load_timeout"
    assert result["size_verified"] is True
    # Allocation is unknown, so the lease is retained, not released: the
    # same model cannot be mutated again until the pending window ends.
    assert result["pending_until"] is not None
    operation = server._claims.list_coordination()["operations"][0]
    assert operation["reason"] == "load_timeout"
    assert operation["lifecycle"] == "unknown"
    again = server._warm_impl("m", "5m", "tester", True)
    assert again["outcome"] == "refused" and again["reason"] == "operation_pending"


def test_overriding_unknown_free_memory_is_explicit_and_audited(warm_env):
    """force=True is the only way past free_unknown, and the audit says so."""
    audit = []
    warm_env.setattr(server._audit, "log_action", lambda **kwargs: audit.append(kwargs))
    calls = _record_residency(warm_env)
    warm_env.setattr(core, "combined_status", lambda *a, **k: {
        "gpus": [], "loaded": [], "free_mb": None, "pressure": {}, "observations": {}})
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"m:latest": 1000}, "sizes"))
    refused = server._warm_impl("m", "5m", "tester", False)
    forced = server._warm_impl("m", "5m", "tester", True)
    assert refused["reason"] == "free_unknown" and calls[0]["model"] == "m:latest"
    assert len(calls) == 1 and forced["ok"] is True
    assert [(e["outcome"], e["force"]) for e in audit] == [("refused", False), ("succeeded", True)]
    assert "free_unknown" in audit[0]["detail"] or "Free VRAM is unknown" in audit[0]["detail"]


@pytest.mark.parametrize("timeout", [server._LOAD_TIMEOUT, server._MAX_LOAD_TIMEOUT, 0.5])
def test_an_operation_lease_outlasts_its_request(timeout):
    assert server._lease_for(timeout) > timeout
    assert server._lease_for(timeout) >= server._claims._OPERATION_LEASE_SECONDS


@pytest.mark.parametrize("value", ["0", "-5", "nan", "abc", str(10 ** 6)])
def test_an_invalid_load_timeout_setting_is_rejected(monkeypatch, value):
    monkeypatch.setenv("VRAM_MCP_LOAD_TIMEOUT", value)
    with pytest.raises(ValueError, match="VRAM_MCP_LOAD_TIMEOUT"):
        server._load_timeout_setting()


def test_the_load_timeout_setting_defaults_to_110_seconds(monkeypatch):
    monkeypatch.delenv("VRAM_MCP_LOAD_TIMEOUT", raising=False)
    assert server._load_timeout_setting() == 110
    monkeypatch.setenv("VRAM_MCP_LOAD_TIMEOUT", "300")
    assert server._load_timeout_setting() == 300


def test_warm_allowed_with_a_verified_size_is_distinguishable(warm_env):
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({"qwen3:32b": 1900}, "sizes"))
    result = server._warm_impl("qwen3:32b", "5m", "tester", False)
    assert result["ok"] is True
    assert result["reason"] == "fits"
    assert result["size_verified"] is True
    assert "could not be verified" not in result["summary"]


def test_warm_forced_reports_no_admission_verdict(warm_env):
    """force=True skips the check entirely; claiming a size was verified (or
    wasn't) would describe a check that never ran."""
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation({}, "sizes"))
    result = server._warm_impl("qwen3:32b", "5m", "tester", True)
    assert result["ok"] is True
    assert result["size_verified"] is None
    assert result["reason"] is None


class _SchemaBackend:
    base_url = "http://ollama.test:11434"

    def __init__(self, outcome):
        self.outcome = outcome

    def change_residency(self, model, _keep_alive, *, resident, **_request):
        return {
            "ok": self.outcome == "succeeded",
            "outcome": self.outcome,
            "model": model,
            "resident": resident if self.outcome == "succeeded" else None,
            "detail": f"fake {self.outcome} result",
        }


def _validate_tool_output(name, result):
    tool = server.mcp._tool_manager.get_tool(name)
    assert tool is not None
    metadata = tool.fn_metadata
    assert metadata.output_schema is not None
    assert metadata.output_model is not None
    return metadata.output_model.model_validate(result)


def test_coordination_tools_publish_and_validate_stable_outputs(monkeypatch):
    names = ["claim", "reserve", "renew", "release", "list_claims",
             "unload", "warm", "ensure_free"]
    for name in names:
        tool = server.mcp._tool_manager.get_tool(name)
        assert tool is not None
        assert tool.fn_metadata.output_schema is not None

    monkeypatch.setattr(server, "_ollama", _SchemaBackend("succeeded"))
    success = server._unload_impl("success", True, "tester")
    _validate_tool_output("unload", success)
    assert success["outcome"] == "succeeded"

    monkeypatch.setattr(server, "_ollama", _SchemaBackend("failed"))
    failed = server._unload_impl("failed", True, "tester")
    _validate_tool_output("unload", failed)
    assert failed["outcome"] == "failed"

    monkeypatch.setattr(server, "_ollama", _SchemaBackend("unknown"))
    unknown = server._unload_impl("uncertain", True, "tester")
    _validate_tool_output("unload", unknown)
    assert unknown["outcome"] == "unknown"
    assert unknown["pending_until"] is not None

    refused = server._unload_impl("uncertain", True, "tester")
    _validate_tool_output("unload", refused)
    assert refused["outcome"] == "refused"
    assert refused["reason"] == "operation_pending"
    server._claims.finish_operation(unknown["operation_id"])

    claim_result = server._claim_impl("claimed", "owner", "purpose", 60)
    _validate_tool_output("claim", claim_result)
    reserve_result = server._reserve_impl(1, "owner", "purpose", 60, None)
    _validate_tool_output("reserve", reserve_result)
    renew_result = server._renew_impl(claim_result["claim_id"], 60)
    _validate_tool_output("renew", renew_result)
    release_result = server._release_impl(claim_result["claim_id"])
    _validate_tool_output("release", release_result)
    expired_renew_result = server._renew_impl("missing", 60)
    _validate_tool_output("renew", expired_renew_result)
    assert expired_renew_result["ok"] is False
    assert expired_renew_result["expires_at"] is None

    list_result = server._list_claims_impl(None)
    _validate_tool_output("list_claims", list_result)
    assert list_result["operations"] == []

    monkeypatch.setattr(server, "_gpu_reading", lambda: [{"free_mb": 4096}])
    ensure_result = server._ensure_free_impl(1, False, "tester")
    _validate_tool_output("ensure_free", ensure_result)

    attempt_result = server._ensure_free_result({
        "ok": False, "outcome": "refused", "summary": "refused",
        "attempts": [{"model": "attempt", "ok": False, "outcome": "refused",
                       "summary": "refused"}],
    })
    validated_attempt = _validate_tool_output("ensure_free", attempt_result)
    assert validated_attempt.model_dump()["attempts"][0]["claims"] is None


def test_coordination_refusal_fields_survive_structured_output(monkeypatch):
    monkeypatch.setattr(server, "_ollama", _SchemaBackend("succeeded"))
    pending = server._claims.begin_operation("other", "warm", True)
    try:
        result = server._warm_impl("target", "5m", "tester", True)
        _validate_tool_output("warm", result)
        assert result["outcome"] == "refused"
        assert result["reason"] == "capacity_pending"
        assert result["scope"] == f"gpu:index={server._GPU_INDEX}"
    finally:
        server._claims.finish_operation(pending["operation_id"])


def test_malformed_claim_stays_visible_in_structured_protection_refusal():
    path = server._claims._DEFAULT_PATH
    path.write_text(json.dumps({"claims": [{
        "claim_id": "claim", "model": "model:latest", "owner": "owner",
        "purpose": "purpose", "claimed_at": "2099-07-13T18:00:00Z",
        "renewed_at": "2099-07-13T18:00:00Z", "ttl_seconds": 3600.0,
        "expires_at": "2099-07-13T19:00:00Z",
    }]}), encoding="utf-8")
    result = server._unload_impl("model", False, "tester")
    _validate_tool_output("unload", result)
    assert result["outcome"] == "refused"
    assert result["reason"] == "model_claimed"
    assert result["claims"][0]["claim_id"] == "claim"
    assert result["claims"][0]["ttl_seconds"] is None


def test_can_warm_free_mb_survives_structured_output(warm_env):
    warm_env.setattr(core, "combined_status", lambda *a, **k: {
        "gpus": [], "loaded": [], "free_mb": 8000, "pressure": {}, "observations": {},
    })
    warm_env.setattr(server, "_active_claims", lambda: (
        [{"kind": "reservation", "gb": 7, "owner": "trainer", "purpose": "sd"}], True))
    warm_env.setattr(server._ollama, "observe_tags", lambda: Observation(
        {"target:latest": 9000}, "sizes"))
    result = server._warm_impl("target", "5m", "tester", False)
    _validate_tool_output("warm", result)
    assert result["reason"] == "insufficient_headroom"
    assert result["free_mb"] == 8000


def test_cleanup_warning_and_model_survive_structured_output(monkeypatch):
    monkeypatch.setattr(server, "_ollama", _SchemaBackend("succeeded"))
    def fail_cleanup(*args, **kwargs):
        raise OSError("cleanup unavailable")
    monkeypatch.setattr(server._claims, "finish_operation", fail_cleanup)
    result = server._unload_impl("warning", True, "tester")
    try:
        _validate_tool_output("unload", result)
        assert result["model"] == "warning:latest"
        assert "cleanup unavailable" in result["coordination_warning"]
    finally:
        fd = server._claims._OPERATION_LOCKS.pop(result["operation_id"], None)
        if fd is not None:
            server._claims._release_operation_lock(fd)


def test_action_audit_uses_reason_when_detail_is_empty(monkeypatch):
    calls = []
    monkeypatch.setattr(server._audit, "log_action", lambda **kwargs: calls.append(kwargs))
    server._action_result(
        "unload", "model:latest", "tester", False,
        {"ok": False, "outcome": "refused", "reason": "operation_pending", "detail": None},
    )
    assert calls[0]["detail"] == "operation_pending"


def test_warm_resident_refresh_requires_no_additional_capacity(warm_env):
    warm_env.setattr(core, "combined_status", lambda *a, **k: {
        "loaded": [{"name": "llama3:latest", "size_vram_mb": 8192}],
        "free_mb": 2048, "observations": {}})
    warm_env.setattr(server, "_active_claims", lambda: ([{"kind": "reservation", "gb": 1}], True))
    result = server._warm_impl("llama3", "5m", "tester", False)
    assert result["ok"] is True
    assert result["model"] == "llama3:latest"
    assert result["additional_mb"] == 0
    assert result["reason"] == "already_resident"


def test_warm_unknown_ledger_does_not_become_zero_reservations(warm_env):
    warm_env.setattr(server, "_active_claims", lambda: ([], False))
    result = server._warm_impl("llama3", "5m", "tester", False)
    assert result["outcome"] == "refused"
    assert result["reason"] == "observation_unavailable"


@pytest.mark.parametrize("keep_alive", ["0", "0s", "", "nonsense", "0m0s", "-5m"])
def test_warm_cannot_bypass_unload_protection_with_zero_duration(keep_alive):
    result = server._warm_impl("llama3", keep_alive, "tester", True)
    assert result["outcome"] == "refused"


@pytest.mark.parametrize("hours", [-1, 0, float("inf"), float("nan")])
def test_invalid_trend_window_is_structured_error(hours):
    assert server._trend_impl(hours)["outcome"] == "refused"


def test_trend_caps_returned_samples(monkeypatch):
    """trend() output lands in an agent's context window; 10k raw rows would
    flood it. The summary still covers every row."""
    rows = [{"ts": f"2026-07-19T00:00:{i % 60:02d}Z", "type": "sample",
             "free_mb": 1000 + i, "state": "ok"} for i in range(500)]
    # read_events is newest-first; _trend_impl reverses to oldest-first.
    monkeypatch.setattr(server._audit, "read_events",
                        lambda **kwargs: list(reversed(rows)))
    result = server._trend_impl(1.0)
    assert result["count"] == 500                 # summary saw everything
    assert len(result["samples"]) == server._TREND_SAMPLE_CAP
    assert result["samples_truncated"] is True
    assert result["samples"][-1]["free_mb"] == 1499   # the newest rows kept
    assert str(server._TREND_SAMPLE_CAP) in result["summary"]


def test_trend_does_not_flag_truncation_when_everything_fits(monkeypatch):
    rows = [{"ts": "2026-07-19T00:00:00Z", "type": "sample",
             "free_mb": 1000, "state": "ok"}]
    monkeypatch.setattr(server._audit, "read_events", lambda **kwargs: rows)
    result = server._trend_impl(1.0)
    assert result["samples_truncated"] is False
    assert len(result["samples"]) == 1


def test_trend_summary_says_unknown_when_the_newest_sample_has_no_value(monkeypatch):
    rows = [{"ts": "2026-07-19T00:00:01Z", "type": "sample", "free_mb": None,
             "state": "ok"},
            {"ts": "2026-07-19T00:00:00Z", "type": "sample", "free_mb": 900,
             "state": "ok"}]  # newest-first, as read_events returns
    monkeypatch.setattr(server._audit, "read_events", lambda **kwargs: rows)
    result = server._trend_impl(1.0)
    assert result["latest_free_mb"] is None
    assert "now unknown" in result["summary"]
