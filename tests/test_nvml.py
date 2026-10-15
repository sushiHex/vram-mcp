"""Tests for vram_mcp.nvml — pure, no real GPU/driver needed."""

import time

from vram_mcp import nvml as nvml_mod


class FakeNVMLError(Exception):
    pass


class _Mem:
    def __init__(self, total, used, free, reserved):
        self.total, self.used, self.free, self.reserved = total, used, free, reserved


class _Proc:
    def __init__(self, pid, used_gpu_memory):
        self.pid = pid
        self.usedGpuMemory = used_gpu_memory


class _UtilSample:
    def __init__(self, pid, sm_util, timestamp=None):
        self.pid = pid
        self.smUtil = sm_util
        self.timeStamp = timestamp if timestamp is not None else int(time.time() * 1_000_000)


class FakeNvml:
    """Stand-in for pynvml: canned responses, or raise NVMLError on demand."""
    NVMLError = FakeNVMLError

    def __init__(self, *, fail_init=False, fail_handle=False,
                 memory=None, compute_procs=None, graphics_procs=None,
                 util_samples=None, fail_memory=False, fail_compute=False,
                 fail_graphics=False, fail_utilization=False):
        self.fail_init = fail_init
        self.fail_handle = fail_handle
        self._memory = memory
        self._compute = compute_procs or []
        self._graphics = graphics_procs or []
        self._util = util_samples or []
        self.fail_memory = fail_memory
        self.fail_compute = fail_compute
        self.fail_graphics = fail_graphics
        self.fail_utilization = fail_utilization
        self.shutdown_called = False
        self.utilization_calls = 0
        self.utilization_timestamp = None
        self.nvmlMemory_v2 = "v2-marker"

    def nvmlInit(self):
        if self.fail_init:
            raise FakeNVMLError("no driver")

    def nvmlShutdown(self):
        self.shutdown_called = True

    def nvmlDeviceGetHandleByIndex(self, index):
        if self.fail_handle:
            raise FakeNVMLError("no such device")
        return f"handle-{index}"

    def nvmlDeviceGetMemoryInfo(self, handle, version):
        if self.fail_memory:
            raise FakeNVMLError("memory unavailable")
        assert version == self.nvmlMemory_v2
        return self._memory

    def nvmlDeviceGetUUID(self, handle):
        return "GPU-fake-uuid"

    def nvmlDeviceGetPciInfo(self, handle):
        class _Pci:
            busId = self.bus_id
        return _Pci()

    bus_id = "00000000:2D:00.0"

    def nvmlDeviceGetComputeRunningProcesses_v3(self, handle):
        if self.fail_compute:
            raise FakeNVMLError("not supported")
        return self._compute

    def nvmlDeviceGetGraphicsRunningProcesses_v3(self, handle):
        if self.fail_graphics:
            raise FakeNVMLError("not supported")
        return self._graphics

    def nvmlDeviceGetProcessUtilization(self, handle, timestamp):
        self.utilization_calls += 1
        self.utilization_timestamp = timestamp
        if self.fail_utilization:
            raise FakeNVMLError("not supported")
        return self._util


class LegacyNvml:
    """Simulates the old `pynvml` PyPI package: init/handle work, but modern
    symbols (nvmlMemory_v2, *RunningProcesses_v3, ...) raise AttributeError."""

    def nvmlInit(self):
        pass

    def nvmlShutdown(self):
        pass

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"handle-{index}"

    def __getattr__(self, name):
        raise AttributeError(name)


class ExplodingNvml:
    """Fails the test if ANY attribute is touched — proves NVML wasn't contacted."""

    def __getattribute__(self, name):
        raise AssertionError(f"NVML was touched: {name}")


def test_nvml_memory_reports_v2_fields():
    fake = FakeNvml(memory=_Mem(total=24 * 1024**3, used=10 * 1024**3,
                                free=14 * 1024**3, reserved=250 * 1024**2))
    result = nvml_mod.nvml_memory(nvml=fake)
    assert result == {
        "total_mb": 24576, "used_mb": 10240, "free_mb": 14336,
        "reserved_mb": 250, "uuid": "GPU-fake-uuid",
    }
    assert fake.shutdown_called is True


def test_nvml_memory_none_when_init_fails():
    assert nvml_mod.nvml_memory(nvml=FakeNvml(fail_init=True)) is None


def test_nvml_memory_none_when_handle_fails():
    assert nvml_mod.nvml_memory(nvml=FakeNvml(fail_handle=True)) is None


def test_nvml_memory_none_when_query_fails_but_still_shuts_down():
    fake = FakeNvml(fail_memory=True)
    assert nvml_mod.nvml_memory(nvml=fake) is None
    assert fake.shutdown_called is True


def test_nvml_processes_merges_compute_and_graphics_dedup():
    # PID 200: compute entry has size None (WDDM), graphics entry has a real
    # size — the merge keeps kind "compute" but takes the first non-None size.
    fake = FakeNvml(
        compute_procs=[_Proc(100, 500 * 1024**2), _Proc(200, None)],
        graphics_procs=[_Proc(200, 300 * 1024**2), _Proc(300, 50 * 1024**2)],
    )
    result = nvml_mod.nvml_processes(nvml=fake)
    assert result == [
        {"pid": 100, "size_mb": 500, "kind": "compute"},
        {"pid": 200, "size_mb": 300, "kind": "compute"},
        {"pid": 300, "size_mb": 50, "kind": "graphics"},
    ]


def test_nvml_processes_partial_failure_still_returns_other_list():
    fake = FakeNvml(fail_compute=True, graphics_procs=[_Proc(300, 50 * 1024**2)])
    result = nvml_mod.nvml_processes(nvml=fake)
    assert result == [{"pid": 300, "size_mb": 50, "kind": "graphics"}]


def test_observe_processes_reports_partial_query_coverage():
    fake = FakeNvml(fail_compute=True, graphics_procs=[_Proc(300, 50 * 1024**2)])
    result = nvml_mod.observe_processes(nvml=fake)
    assert result.known is True
    assert result.coverage == {
        "compute_processes": False,
        "graphics_processes": True,
    }


def test_nvml_processes_empty_when_unavailable():
    assert nvml_mod.nvml_processes(nvml=FakeNvml(fail_init=True)) == []


def test_observe_processes_distinguishes_unavailable_from_empty():
    unavailable = nvml_mod.observe_processes(nvml=FakeNvml(fail_init=True))
    empty = nvml_mod.observe_processes(nvml=FakeNvml())
    assert unavailable.known is False
    assert unavailable.data is None
    assert empty.known is True
    assert empty.data == []
    assert empty.scope == "gpu:index=0"
    assert empty.coverage == {
        "compute_processes": True,
        "graphics_processes": True,
    }


def test_observe_processes_uses_selected_device_index():
    fake = FakeNvml()
    result = nvml_mod.observe_processes(1, nvml=fake)
    assert result.known is True
    assert result.scope == "gpu:index=1"


def test_nvml_busy_true_when_sample_has_positive_sm_util():
    fake = FakeNvml(util_samples=[_UtilSample(111, 0), _UtilSample(222, 40)])
    assert nvml_mod.nvml_busy(222, nvml=fake) is True


def test_nvml_busy_false_when_sample_present_but_zero():
    fake = FakeNvml(util_samples=[_UtilSample(111, 0)])
    assert nvml_mod.nvml_busy(111, nvml=fake) is False


def test_nvml_busy_none_when_pid_not_in_samples():
    fake = FakeNvml(util_samples=[_UtilSample(111, 40)])
    assert nvml_mod.nvml_busy(999, nvml=fake) is None


def test_nvml_busy_none_when_unavailable():
    assert nvml_mod.nvml_busy(111, nvml=FakeNvml(fail_init=True)) is None


def test_nvml_busy_true_when_any_sample_positive_not_just_first():
    # Zero sample first, positive later: ALL samples are considered.
    fake = FakeNvml(util_samples=[_UtilSample(111, 0), _UtilSample(111, 40)])
    assert nvml_mod.nvml_busy(111, nvml=fake) is True


def test_nvml_busy_map_mixed_verdicts_in_one_fetch():
    fake = FakeNvml(util_samples=[
        _UtilSample(222, 0),   # zero sample first...
        _UtilSample(111, 0),   # idle pid: sample exists, all zero
        _UtilSample(222, 40),  # ...but a later positive sample makes 222 busy
    ])
    result = nvml_mod.nvml_busy_map([111, 222, 999], nvml=fake)
    assert result == {111: False, 222: True, 999: None}
    assert fake.utilization_calls == 1
    assert fake.shutdown_called is True


def test_nvml_busy_map_ignores_stale_samples():
    now = 1_000.0
    fake = FakeNvml(util_samples=[
        _UtilSample(111, 80, int((now - 30) * 1_000_000)),
        _UtilSample(222, 0, int((now - 1) * 1_000_000)),
    ])
    result = nvml_mod.nvml_busy_map(
        [111, 222], nvml=fake, now_fn=lambda: now, recent_seconds=5,
    )
    assert result == {111: None, 222: False}
    assert fake.utilization_timestamp == int((now - 5) * 1_000_000)


def test_nvml_busy_map_ignores_sample_without_timestamp():
    undated = type("UndatedSample", (), {"pid": 111, "smUtil": 80})()
    fake = FakeNvml(util_samples=[undated])
    assert nvml_mod.nvml_busy_map([111], nvml=fake) == {111: None}


def test_nvml_busy_map_empty_pids_never_touches_nvml():
    assert nvml_mod.nvml_busy_map([], nvml=ExplodingNvml()) == {}


def test_nvml_busy_map_all_none_when_unavailable():
    fake = FakeNvml(fail_init=True)
    assert nvml_mod.nvml_busy_map([111, 222], nvml=fake) == {111: None, 222: None}


def test_nvml_busy_map_all_none_when_nvml_import_is_unavailable(monkeypatch):
    def missing():
        raise ImportError("missing")

    monkeypatch.setattr(nvml_mod, "_default_nvml", missing)
    assert nvml_mod.nvml_busy_map([111, 222]) == {111: None, 222: None}


def test_legacy_nvml_attribute_errors_degrade_to_defaults():
    # Old `pynvml` PyPI package lacks nvmlMemory_v2 / *_v3 symbols: every
    # public function must swallow the AttributeError and return its default.
    assert nvml_mod.nvml_memory(nvml=LegacyNvml()) is None
    assert nvml_mod.nvml_processes(nvml=LegacyNvml()) == []
    assert nvml_mod.nvml_busy(111, nvml=LegacyNvml()) is None
    assert nvml_mod.nvml_busy_map([111, 222], nvml=LegacyNvml()) == {111: None, 222: None}


def test_b2mb_non_convertible_returns_none():
    assert nvml_mod._b2mb("garbage") is None
    assert nvml_mod._b2mb(None) is None
    assert nvml_mod._b2mb(300 * 1024**2) == 300


def test_process_observation_names_its_device_in_either_binding_form():
    fake = FakeNvml()
    assert nvml_mod.observe_processes(0, nvml=fake).device == "00000000:2D:00.0"
    fake.bus_id = b"00000000:41:00.0"   # older bindings return bytes
    assert nvml_mod.observe_processes(0, nvml=fake).device == "00000000:41:00.0"


def test_an_unreadable_bus_id_leaves_the_process_reading_intact():
    fake = FakeNvml(compute_procs=[_Proc(7, None)])
    fake.nvmlDeviceGetPciInfo = lambda handle: (_ for _ in ()).throw(FakeNVMLError("no"))

    observation = nvml_mod.observe_processes(0, nvml=fake)

    assert observation.known
    assert observation.device is None
    assert observation.data == [{"pid": 7, "size_mb": None, "kind": "compute"}]
