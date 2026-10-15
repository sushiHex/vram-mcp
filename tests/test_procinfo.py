"""Tests for vram_mcp.procinfo — pure, injected readers."""
from vram_mcp import procinfo, wddm
from vram_mcp.observations import Observation


def test_process_table_uses_nvml_sizes_when_present():
    # Linux/TCC: NVML gives real sizes; names come from the posix reader.
    nvml = lambda: [{"pid": 100, "size_mb": 8000, "kind": "compute"}]
    names = lambda pids: {100: {"name": "python", "cmdline": "python train.py"}}
    out = procinfo.process_table(
        nvml_processes=nvml, posix_name_reader=names, platform="linux",
    )
    assert out == [{"pid": 100, "size_mb": 8000, "shared_mb": None,
                    "non_local_mb": None, "name": "python",
                    "cmdline": "python train.py", "kind": "compute"}]


def test_process_table_windows_counters_only_fill_identity():
    # WDDM counters aggregate adapters, so they cannot size one selected GPU.
    nvml = lambda: [{"pid": 100, "size_mb": None, "kind": "compute"},
                    {"pid": 200, "size_mb": None, "kind": "graphics"}]
    win = lambda pids: [{"pid": 100, "size_mb": None, "name": "python.exe",
                         "cmdline": "python.exe train_lora_kg.py"}]
    out = procinfo.process_table(
        nvml_processes=nvml, win_gpu_reader=win, platform="win32",
    )
    by_pid = {p["pid"]: p for p in out}
    assert by_pid[100]["size_mb"] is None
    assert by_pid[100]["shared_mb"] is None
    assert by_pid[100]["non_local_mb"] is None
    assert by_pid[100]["name"] == "python.exe"
    assert by_pid[100]["kind"] == "compute"          # kind preserved from NVML
    assert by_pid[200]["size_mb"] is None             # win reader didn't see it -> stays null
    assert by_pid[200]["name"] is None


def test_process_table_windows_reader_does_not_add_unattributed_pid():
    # A counter-only PID cannot be attributed to the selected adapter.
    nvml = lambda: []
    win = lambda pids: [{"pid": 300, "size_mb": 2048, "shared_mb": 0,
                         "non_local_mb": 0, "name": "UnrealEditor.exe",
                         "cmdline": "UnrealEditor.exe Project.uproject"}]
    out = procinfo.process_table(
        nvml_processes=nvml, win_gpu_reader=win, platform="win32",
    )
    assert out == []


def test_process_table_no_readers_returns_nvml_only_unnamed():
    nvml = lambda: [{"pid": 100, "size_mb": None, "kind": "compute"}]
    out = procinfo.process_table(nvml_processes=nvml)
    assert out == [{"pid": 100, "size_mb": None, "shared_mb": None,
                    "non_local_mb": None, "name": None,
                    "cmdline": None, "kind": "compute"}]


def test_process_table_does_not_assign_adapter_aggregated_spill_fields():
    rows = procinfo.process_table(
        nvml_processes=lambda: [{"pid": 7, "size_mb": None, "kind": "compute"}],
        win_gpu_reader=lambda pids: [
            {"pid": 7, "size_mb": 500, "shared_mb": 12, "non_local_mb": 300,
             "name": "a.exe", "cmdline": "a"},
        ],
        platform="win32",
    )
    assert rows[0]["non_local_mb"] is None
    assert rows[0]["shared_mb"] is None
    assert rows[0]["size_mb"] is None
    assert rows[0]["name"] == "a.exe"


def test_process_table_defaults_spill_fields_to_none():
    # Linux/TCC has no perf counter — "unreported" must not read as "no spill".
    rows = procinfo.process_table(
        nvml_processes=lambda: [{"pid": 7, "size_mb": 100, "kind": "compute"}],
    )
    assert rows[0]["shared_mb"] is None
    assert rows[0]["non_local_mb"] is None


def test_process_table_dispatches_reader_for_actual_platform():
    calls = {"windows": 0, "posix": 0}

    def windows(pids):
        calls["windows"] += 1
        return [{"pid": 7, "name": "win.exe", "cmdline": "win"}]

    def posix(pids):
        calls["posix"] += 1
        return {7: {"name": "python", "cmdline": "python train.py"}}

    rows = procinfo.process_table(
        nvml_processes=lambda: [{"pid": 7, "size_mb": 100}],
        win_gpu_reader=windows, posix_name_reader=posix, platform="linux",
    )
    assert calls == {"windows": 0, "posix": 1}
    assert rows[0]["name"] == "python"


def test_observe_processes_preserves_failed_vs_empty_and_scope():
    def failed(index, *, nvml=None):
        return Observation(None, "nvml", error="driver unavailable",
                           scope=f"gpu:index={index}")

    def empty(index, *, nvml=None):
        return Observation([], "nvml", scope=f"gpu:index={index}")

    unavailable = procinfo.observe_processes(
        1, platform="linux", nvml_observer=failed,
        posix_reader=lambda pids: {},
    )
    available = procinfo.observe_processes(
        1, platform="linux", nvml_observer=empty,
        posix_reader=lambda pids: {},
    )
    assert unavailable.known is False
    assert unavailable.data is None
    assert available.known is True
    assert available.data == []
    assert available.scope == "gpu:index=1"
    assert available.metadata()["coverage"] == {"non_local_memory": False}


def test_observe_processes_propagates_nvml_query_coverage():
    def partial(index, *, nvml=None):
        return Observation(
            [], "nvml", scope=f"gpu:index={index}",
            coverage={"compute_processes": True, "graphics_processes": False},
        )

    result = procinfo.observe_processes(
        platform="linux", nvml_observer=partial, posix_reader=lambda pids: {},
    )
    assert result.coverage == {
        "compute_processes": True,
        "graphics_processes": False,
        "non_local_memory": False,
    }


def test_win_gpu_procs_queries_only_selected_pids(monkeypatch):
    fake = ("11924|python.exe|python.exe train_lora_kg.py\n"
            "1336|dwm.exe|dwm.exe\n")
    calls = []
    monkeypatch.setattr(procinfo, "_run_powershell",
                        lambda cmd, timeout: calls.append(cmd) or fake)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    out = {p["pid"]: p for p in procinfo.win_gpu_procs([11924, 1336])}
    assert set(out) == {11924, 1336}
    assert out[11924]["size_mb"] is None
    assert out[11924]["name"] == "python.exe"
    assert "Get-Counter" not in calls[0]
    assert "ProcessId=1336 OR ProcessId=11924" in calls[0]


def test_win_gpu_procs_empty_pid_set_skips_reader(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    monkeypatch.setattr(procinfo, "_run_powershell",
                        lambda *args: (_ for _ in ()).throw(AssertionError()))
    assert procinfo.win_gpu_procs([]) == []


def test_win_gpu_procs_keeps_pipe_in_cmdline(monkeypatch):
    # cmdline is last so split("|", 2) leaves its own pipes intact.
    out = "42|sh.exe|sh -c 'a | b | c'\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    (row,) = procinfo.win_gpu_procs([42])
    assert row["cmdline"] == "sh -c 'a | b | c'"


def test_win_gpu_procs_skips_malformed_lines(monkeypatch):
    out = "not-a-pid|x|y\n7|a.exe|a\nshort-line\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    assert [r["pid"] for r in procinfo.win_gpu_procs([7])] == [7]


def test_win_gpu_procs_never_raises_on_non_integer_pids(monkeypatch):
    """The module promises to degrade, not raise. Coercing would break that
    for values NVML cannot produce, so junk PIDs are dropped instead."""
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    monkeypatch.setattr(procinfo, "_run_powershell",
                        lambda *args: (_ for _ in ()).throw(AssertionError(
                            "no query should run when nothing is usable")))
    for junk in (["abc"], [None], [1.5], [{}], [True], [-1]):
        assert procinfo.win_gpu_procs(junk) == []


def test_win_gpu_procs_queries_only_the_usable_pids(monkeypatch):
    """A usable PID alongside junk still gets its identity."""
    calls = []
    monkeypatch.setattr(procinfo, "_run_powershell",
                        lambda cmd, timeout: calls.append(cmd) or "7|a.exe|a\n")
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    (row,) = procinfo.win_gpu_procs([7, "abc", None])
    assert row["pid"] == 7
    assert calls[0].count("ProcessId=") == 1


def test_win_gpu_identity_filter_is_bounded(monkeypatch):
    """Past the cap the query truncates into "identity unavailable" for the
    excess, rather than building a command line Windows will reject."""
    pids = list(range(1000, 1000 + procinfo._MAX_FILTER_PIDS + 50))
    command = procinfo._win_gpu_identity_ps(procinfo._filter_pids(pids))
    assert command.count("ProcessId=") == procinfo._MAX_FILTER_PIDS
    assert len(command) < 32767


def test_win_gpu_procs_empty_off_windows(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "linux")
    assert procinfo.win_gpu_procs([7]) == []


def test_win_gpu_procs_empty_on_reader_failure(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: None)
    assert procinfo.win_gpu_procs([7]) == []


# ---- non-local memory merged into observe_processes -------------------------

class FakeNonLocal:
    """Fake non-local reader: canned Observation, counts observe() calls."""

    def __init__(self, observation):
        self._observation = observation
        self.calls = 0
        self.devices = []

    def observe(self, device=None):
        self.calls += 1
        self.devices.append(device)
        return self._observation


def _usage(data, total=None):
    """A reader whose adapter total defaults to the sum of the readable PIDs."""
    if total is None:
        total = sum(value for value in data.values() if value is not None)
    return FakeNonLocal(Observation(wddm.NonLocal(total, data), "wddm"))


def _nvml_rows(*rows):
    return lambda index, *, nvml=None: Observation(
        list(rows), "nvml", scope=f"gpu:index={index}")


def test_observe_processes_merges_non_local_into_nvml_rows():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 8000, "kind": "compute"},
                                 {"pid": 200, "size_mb": 500, "kind": "graphics"}),
        posix_reader=lambda pids: {},
        non_local=_usage({100: 300, 200: 0}),
    )
    by_pid = {row["pid"]: row for row in result.data}
    assert by_pid[100]["non_local_mb"] == 300
    assert by_pid[100]["size_mb"] == 8000
    assert by_pid[200]["non_local_mb"] == 0     # a measured zero, not unknown
    assert result.coverage["non_local_memory"] is True
    assert result.gaps is None
    assert "gaps" not in result.metadata()


def test_observe_processes_wddm_only_pid_joins_with_identity():
    seen = []

    def win(pids):
        seen.append(sorted(pids))
        return [{"pid": 900, "name": "chrome.exe", "cmdline": "chrome.exe --gpu"}]

    result = procinfo.observe_processes(
        platform="win32",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": None, "kind": "compute"}),
        win_gpu_reader=win,
        non_local=_usage({100: 10, 900: 2048}),
    )
    by_pid = {row["pid"]: row for row in result.data}
    assert by_pid[900]["kind"] == "wddm"
    assert by_pid[900]["size_mb"] is None
    assert by_pid[900]["non_local_mb"] == 2048
    assert by_pid[900]["name"] == "chrome.exe"
    assert by_pid[900]["cmdline"] == "chrome.exe --gpu"
    assert seen == [[100, 900]]                 # identity lookup got the wddm PID
    assert result.coverage["non_local_memory"] is True


def test_observe_processes_zero_mb_wddm_only_pid_joins():
    # Table membership decides which runners are on the selected GPU, so a
    # WDDM-only PID joins even when its non-local memory is a measured zero.
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        posix_reader=lambda pids: {},
        non_local=_usage({100: 5, 777: 0}),
    )
    by_pid = {row["pid"]: row for row in result.data}
    assert sorted(by_pid) == [100, 777]
    assert by_pid[777]["kind"] == "wddm"
    assert by_pid[777]["non_local_mb"] == 0
    assert result.coverage["non_local_memory"] is True
    assert result.gaps is None


def test_observe_processes_passes_nvml_device_to_reader_and_carries_it():
    bus_id = "00000000:2D:00.0"
    reader = _usage({100: 5})

    def nvml(index, *, nvml=None):
        return Observation([{"pid": 100, "size_mb": 1, "kind": "compute"}],
                           "nvml", scope=f"gpu:index={index}", device=bus_id)

    result = procinfo.observe_processes(
        platform="linux", nvml_observer=nvml,
        posix_reader=lambda pids: {}, non_local=reader,
    )
    assert reader.devices == [bus_id]
    assert result.device == bus_id
    assert result.metadata()["device"] == bus_id


def test_observe_processes_carries_device_when_nvml_unavailable():
    bus_id = "00000000:2D:00.0"
    reader = _usage({100: 5})

    def failed(index, *, nvml=None):
        return Observation(None, "nvml", error="process query failed",
                           device=bus_id)

    result = procinfo.observe_processes(
        platform="linux", nvml_observer=failed,
        posix_reader=lambda pids: {}, non_local=reader,
    )
    assert result.known is False
    assert reader.calls == 0
    assert result.device == bus_id
    assert result.metadata()["device"] == bus_id


def test_observe_processes_unreadable_unknown_pid_joins_and_stays_covered():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        posix_reader=lambda pids: {},
        non_local=_usage({100: 5, 777: None}, total=4096),
    )
    by_pid = {row["pid"]: row for row in result.data}
    assert by_pid[777]["kind"] == "wddm"
    assert by_pid[777]["non_local_mb"] is None
    # The adapter total is the measurement; a PID without a reading only
    # lacks attribution.
    assert result.coverage["non_local_memory"] is True
    assert result.gaps is None
    assert result.totals == {"non_local_mb": 4096}


def test_observe_processes_nvml_pid_missing_from_usage_keeps_coverage_and_total():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 9, "size_mb": 1, "kind": "compute"},
                                 {"pid": 5, "size_mb": 1, "kind": "compute"},
                                 {"pid": 7, "size_mb": 1, "kind": "compute"}),
        posix_reader=lambda pids: {},
        non_local=_usage({7: 64}, total=1000),
    )
    by_pid = {row["pid"]: row for row in result.data}
    assert by_pid[5]["non_local_mb"] is None
    assert by_pid[9]["non_local_mb"] is None
    assert by_pid[7]["non_local_mb"] == 64
    assert result.coverage["non_local_memory"] is True
    assert result.gaps is None
    assert result.totals == {"non_local_mb": 1000}
    assert result.metadata()["totals"] == {"non_local_mb": 1000}


def test_observe_processes_total_is_the_measured_adapter_total_not_the_row_sum():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        posix_reader=lambda pids: {},
        non_local=_usage({100: 10}, total=2048),
    )
    assert result.data[0]["non_local_mb"] == 10
    assert result.totals == {"non_local_mb": 2048}


def test_observe_processes_measured_zero_total_is_carried():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        posix_reader=lambda pids: {},
        non_local=_usage({}, total=0),
    )
    assert result.totals == {"non_local_mb": 0}
    assert result.coverage["non_local_memory"] is True


def test_observe_processes_unavailable_reader_gap_is_its_error():
    reader = FakeNonLocal(Observation(None, "wddm", error="PDH query failed"))
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        posix_reader=lambda pids: {},
        non_local=reader,
    )
    assert result.known is True
    assert result.data[0]["non_local_mb"] is None
    assert result.coverage["non_local_memory"] is False
    assert result.gaps == {"non_local_memory": "PDH query failed"}
    assert result.totals is None
    assert "totals" not in result.metadata()


def test_observe_processes_without_reader_gap_names_missing_collector():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        posix_reader=lambda pids: {},
    )
    assert result.data[0]["non_local_mb"] is None
    assert result.coverage["non_local_memory"] is False
    assert result.gaps == {"non_local_memory": procinfo._NO_NON_LOCAL_READER}
    assert result.totals is None
    assert "totals" not in result.metadata()
    assert result.metadata()["gaps"] == {
        "non_local_memory": procinfo._NO_NON_LOCAL_READER}


def test_observe_processes_does_not_call_reader_when_nvml_unavailable():
    reader = _usage({100: 5})

    def failed(index, *, nvml=None):
        return Observation(None, "nvml", error="driver unavailable")

    result = procinfo.observe_processes(
        platform="linux", nvml_observer=failed,
        posix_reader=lambda pids: {}, non_local=reader,
    )
    assert result.known is False
    assert reader.calls == 0
    assert result.coverage["non_local_memory"] is False


def test_observation_metadata_includes_gaps_only_when_present():
    assert "gaps" not in Observation([], "x").metadata()
    assert "gaps" not in Observation([], "x", gaps={}).metadata()
    assert Observation([], "x", gaps={"a": "why"}).metadata()["gaps"] == {"a": "why"}
