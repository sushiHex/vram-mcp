"""Tests for vram_mcp.procinfo — pure, injected readers."""
from vram_mcp import procinfo, wddm
from vram_mcp.observations import Observation


def _fake_winproc(monkeypatch, names, cmdlines=None):
    """Stand in for the ctypes readers; a missing cmdline is a denied process."""
    cmdlines = cmdlines or {}
    calls = []

    def command_line(pid):
        calls.append(pid)
        return cmdlines.get(pid)

    monkeypatch.setattr(procinfo.winproc, "process_names", lambda: dict(names))
    monkeypatch.setattr(procinfo.winproc, "command_line", command_line)
    return calls


def test_windows_name_reader_returns_name_and_cmdline_for_requested_pids(monkeypatch):
    calls = _fake_winproc(
        monkeypatch, {11924: "python.exe", 1336: "dwm.exe", 5: "other.exe"},
        {11924: "python.exe train_lora_kg.py"})
    assert procinfo.windows_name_reader([11924, 1336]) == {
        11924: {"name": "python.exe", "cmdline": "python.exe train_lora_kg.py"},
        1336: {"name": "dwm.exe", "cmdline": None},      # denied: name only
    }
    assert sorted(calls) == [1336, 11924]               # others never opened


def test_windows_name_reader_omits_pids_windows_does_not_list(monkeypatch):
    _fake_winproc(monkeypatch, {7: "a.exe"})
    assert procinfo.windows_name_reader([7, 99]) == {
        7: {"name": "a.exe", "cmdline": None}}


def test_windows_name_reader_ignores_junk_pids(monkeypatch):
    """The module promises to degrade, not raise: junk PIDs are dropped."""
    calls = _fake_winproc(monkeypatch, {7: "a.exe", 1: "b.exe"}, {7: "a"})
    for junk in (["abc"], [None], [1.5], [{}], [True], [-1], []):
        assert procinfo.windows_name_reader(junk) == {}
    assert procinfo.windows_name_reader([7, "abc", None]) == {
        7: {"name": "a.exe", "cmdline": "a"}}
    assert calls == [7]


def test_windows_name_reader_empty_when_snapshot_fails(monkeypatch):
    def fail():
        raise OSError("CreateToolhelp32Snapshot failed")

    monkeypatch.setattr(procinfo.winproc, "process_names", fail)
    assert procinfo.windows_name_reader([7]) == {}


def test_windows_name_reader_empty_when_command_line_apis_are_unavailable(monkeypatch):
    def fail(pid):
        raise OSError("Windows process APIs are unavailable")

    monkeypatch.setattr(procinfo.winproc, "process_names", lambda: {7: "a.exe"})
    monkeypatch.setattr(procinfo.winproc, "command_line", fail)
    assert procinfo.windows_name_reader([7]) == {}


def test_observe_processes_attaches_identity_to_rows():
    names = lambda pids: {100: {"name": "python", "cmdline": "python train.py"}}
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 8000, "kind": "compute"},
                                 {"pid": 200, "size_mb": 5, "kind": "graphics"}),
        name_reader=names,
    )
    by_pid = {row["pid"]: row for row in result.data}
    assert by_pid[100] == {"pid": 100, "size_mb": 8000, "shared_mb": None,
                           "non_local_mb": None, "name": "python",
                           "cmdline": "python train.py", "kind": "compute"}
    assert by_pid[200]["name"] is None                  # reader skipped it
    assert by_pid[200]["cmdline"] is None


def test_observe_processes_ignores_reader_pids_outside_the_table():
    result = procinfo.observe_processes(
        platform="win32",
        nvml_observer=_nvml_rows(),
        name_reader=lambda pids: {300: {"name": "UnrealEditor.exe", "cmdline": "x"}},
    )
    assert result.data == []


def test_observe_processes_names_the_source_by_platform():
    rows = _nvml_rows({"pid": 1, "size_mb": 1, "kind": "compute"})
    win = procinfo.observe_processes(
        platform="win32", nvml_observer=rows, name_reader=lambda pids: {})
    posix = procinfo.observe_processes(
        platform="linux", nvml_observer=rows, name_reader=lambda pids: {})
    assert win.source == "nvml+windows-process-identity"
    assert posix.source == "nvml+ps"


def test_observe_processes_defaults_the_reader_to_the_platform(monkeypatch):
    seen = []
    monkeypatch.setattr(procinfo, "windows_name_reader",
                        lambda pids: seen.append("windows") or {})
    monkeypatch.setattr(procinfo, "posix_name_reader",
                        lambda pids: seen.append("posix") or {})
    rows = _nvml_rows({"pid": 1, "size_mb": 1, "kind": "compute"})
    procinfo.observe_processes(platform="win32", nvml_observer=rows)
    procinfo.observe_processes(platform="linux", nvml_observer=rows)
    assert seen == ["windows", "posix"]


def test_observe_processes_preserves_failed_vs_empty_and_scope():
    def failed(index, *, nvml=None):
        return Observation(None, "nvml", error="driver unavailable",
                           scope=f"gpu:index={index}")

    def empty(index, *, nvml=None):
        return Observation([], "nvml", scope=f"gpu:index={index}")

    unavailable = procinfo.observe_processes(
        1, platform="linux", nvml_observer=failed,
        name_reader=lambda pids: {},
    )
    available = procinfo.observe_processes(
        1, platform="linux", nvml_observer=empty,
        name_reader=lambda pids: {},
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
        platform="linux", nvml_observer=partial, name_reader=lambda pids: {},
    )
    assert result.coverage == {
        "compute_processes": True,
        "graphics_processes": False,
        "non_local_memory": False,
    }


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
        name_reader=lambda pids: {},
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
        return {900: {"name": "chrome.exe", "cmdline": "chrome.exe --gpu"}}

    result = procinfo.observe_processes(
        platform="win32",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": None, "kind": "compute"}),
        name_reader=win,
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
        name_reader=lambda pids: {},
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
        name_reader=lambda pids: {}, non_local=reader,
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
        name_reader=lambda pids: {}, non_local=reader,
    )
    assert result.known is False
    assert reader.calls == 0
    assert result.device == bus_id
    assert result.metadata()["device"] == bus_id


def test_observe_processes_unreadable_unknown_pid_joins_and_stays_covered():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        name_reader=lambda pids: {},
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
        name_reader=lambda pids: {},
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
        name_reader=lambda pids: {},
        non_local=_usage({100: 10}, total=2048),
    )
    assert result.data[0]["non_local_mb"] == 10
    assert result.totals == {"non_local_mb": 2048}


def test_observe_processes_measured_zero_total_is_carried():
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        name_reader=lambda pids: {},
        non_local=_usage({}, total=0),
    )
    assert result.totals == {"non_local_mb": 0}
    assert result.coverage["non_local_memory"] is True


def test_observe_processes_unavailable_reader_gap_is_its_error():
    reader = FakeNonLocal(Observation(None, "wddm", error="PDH query failed"))
    result = procinfo.observe_processes(
        platform="linux",
        nvml_observer=_nvml_rows({"pid": 100, "size_mb": 1, "kind": "compute"}),
        name_reader=lambda pids: {},
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
        name_reader=lambda pids: {},
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
        name_reader=lambda pids: {}, non_local=reader,
    )
    assert result.known is False
    assert reader.calls == 0
    assert result.coverage["non_local_memory"] is False


def test_observation_metadata_includes_gaps_only_when_present():
    assert "gaps" not in Observation([], "x").metadata()
    assert "gaps" not in Observation([], "x", gaps={}).metadata()
    assert Observation([], "x", gaps={"a": "why"}).metadata()["gaps"] == {"a": "why"}
