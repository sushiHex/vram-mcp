"""Tests for vram_mcp.procinfo — pure, injected readers."""
from vram_mcp import procinfo
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
    fake = ("11924||||python.exe|python.exe train_lora_kg.py\n"
            "1336||||dwm.exe|dwm.exe\n")
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
    # cmdline is last so split("|", 5) leaves its own pipes intact.
    out = "42|1048576|0|0|sh.exe|sh -c 'a | b | c'\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    (row,) = procinfo.win_gpu_procs([42])
    assert row["cmdline"] == "sh -c 'a | b | c'"


def test_win_gpu_procs_skips_malformed_lines(monkeypatch):
    out = "not-a-pid|1|2|3|x|y\n7|1048576|0|0|a.exe|a\nshort|line\n"
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: out)
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    assert [r["pid"] for r in procinfo.win_gpu_procs([7])] == [7]


def test_win_gpu_procs_empty_off_windows(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "linux")
    assert procinfo.win_gpu_procs([7]) == []


def test_win_gpu_procs_empty_on_reader_failure(monkeypatch):
    monkeypatch.setattr(procinfo.sys, "platform", "win32")
    monkeypatch.setattr(procinfo, "_run_powershell", lambda cmd, timeout: None)
    assert procinfo.win_gpu_procs([7]) == []
