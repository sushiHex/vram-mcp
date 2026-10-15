"""Tests for vram_mcp.core — pure orchestration with injected fakes."""

from vram_mcp import core, wddm
from vram_mcp.observations import Observation


_MB = 1024 * 1024

# What a collector that reads driver paging reports: a quiet window, and one
# where the System process kept an engine busy in most seconds.
QUIET_PAGING = {"sustained": False, "active_seconds": 0, "seconds": 10,
                "peak_percent": 0.4}
SUSTAINED_PAGING = {"sustained": True, "active_seconds": 8, "seconds": 10,
                    "peak_percent": 41.5}


def paging_fn(*percents):
    """A paging collector whose window holds ``percents`` (quiet by default)."""
    window = wddm.Paging(tuple(percents) or (0.0,) * 10)
    return lambda: Observation(window, "wddm:pdh", scope="gpu:index=0")


def paging_gap(error):
    """A paging collector that could not read the driver."""
    return lambda: Observation(None, "wddm:pdh", error=error, scope="gpu:index=0")


def gb_bytes(gb):
    return int(gb * 1024 * _MB)


class FakeOllama:
    """Fake OllamaClient: canned ps() list + records unload() calls."""

    def __init__(self, models=None):
        self._models = models or []
        self.unloaded = []

    def ps(self):
        return list(self._models)

    def unload(self, model):
        self.unloaded.append(model)
        return True


def gpu_fn_const(free_mb):
    """gpu_status_fn returning a single GPU with fixed free VRAM."""
    def _fn():
        if free_mb is None:
            return []
        return [{"index": 0, "name": "GPU", "total_mb": 24000,
                 "used_mb": 24000 - free_mb, "free_mb": free_mb}]
    return _fn


def gpu_fn_sequence(free_values):
    """gpu_status_fn yielding a new free value on each successive call."""
    calls = {"i": 0}

    def _fn():
        i = min(calls["i"], len(free_values) - 1)
        calls["i"] += 1
        val = free_values[i]
        if val is None:
            return []
        return [{"index": 0, "name": "GPU", "total_mb": 24000,
                 "used_mb": 24000 - val, "free_mb": val}]
    return _fn


# ---- combined_status --------------------------------------------------------

def test_combined_status():
    models = [
        {"name": "big", "size": gb_bytes(8), "size_vram": gb_bytes(8),
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    status = core.combined_status(gpu_fn_const(6000), FakeOllama(models))
    assert status["free_mb"] == 6000
    assert len(status["gpus"]) == 1
    assert status["loaded"] == [
        {"name": "big", "size_vram_mb": 8192, "total_size_mb": 8192,
         "offloaded_to_cpu": False, "expires_at": "2026-07-11T10:00:00Z"},
    ]


def test_combined_status_no_gpu():
    status = core.combined_status(gpu_fn_const(None), FakeOllama([]))
    assert status["gpus"] == []
    assert status["free_mb"] is None
    assert status["loaded"] == []


def test_failed_model_read_is_unknown_not_empty_or_healthy():
    class Unavailable(FakeOllama):
        def observe_loaded(self):
            return Observation(None, "ollama", error="offline")
    status = core.combined_status(gpu_fn_const(6000), Unavailable())
    assert status["loaded"] is None
    assert status["pressure"]["state"] == "unknown"
    assert status["observations"]["ollama"]["status"] == "unavailable"


def test_unrelated_gpu_does_not_hide_selected_device_pressure():
    reading = Observation([{"index": 0, "free_mb": 128}, {"index": 1, "free_mb": 24000}],
                          "gpu", scope="gpu:index=0")
    status = core.combined_status(lambda: reading, FakeOllama())
    assert status["free_mb"] == 128
    assert status["pressure"]["state"] == "tight"


def test_other_device_process_reading_cannot_invent_local_spill():
    gpu = Observation([{"index": 0, "free_mb": 2048}], "gpu", scope="gpu:index=0")
    processes = Observation([{"pid": 10, "non_local_mb": 8000}], "nvml", scope="gpu:index=1")
    status = core.combined_status(lambda: gpu, FakeOllama(), procinfo_fn=lambda: processes)
    assert status["other_processes"] is None
    assert status["pressure"]["non_local_mb"] is None
    assert status["pressure"]["unexplained_spill_mb"] is None
    assert status["pressure"]["state"] != "thrashing"


def test_missing_model_memory_is_not_reported_as_zero():
    status = core.combined_status(gpu_fn_const(6000), FakeOllama([{"name": "m"}]))
    assert status["loaded"][0]["size_vram_mb"] is None
    assert status["loaded"][0]["offloaded_to_cpu"] is None


# ---- offload detection -------------------------------------------------------

def test_loaded_models_detects_cpu_offload():
    models = [
        {"name": "partial", "size": gb_bytes(10), "size_vram": gb_bytes(6),
         "expires_at": None},
    ]
    status = core.combined_status(gpu_fn_const(4000), FakeOllama(models))
    entry = status["loaded"][0]
    assert entry["total_size_mb"] == 10240
    assert entry["size_vram_mb"] == 6144
    assert entry["offloaded_to_cpu"] is True


# ---- Snapshot ------------------------------------------------------------------

def _snap(all_claims=None, pid_map=None, busy_map=None):
    return core.Snapshot(all_claims or [], pid_map or {}, busy_map or {})


def test_snapshot_capture_runs_each_collector_once():
    calls = {"claims": 0, "pids": 0, "busy": 0}

    def all_claims_fn():
        calls["claims"] += 1
        return [{"model": "m1", "owner": "x"}]

    def pid_map_fn():
        calls["pids"] += 1
        return {"m1": 123, "m2": 456}

    def busy_map_fn(pids):
        calls["busy"] += 1
        assert pids == [123, 456]        # exactly the pids the map surfaced
        return {123: True, 456: False}

    snap = core.Snapshot.capture(all_claims_fn, pid_map_fn, busy_map_fn)
    assert calls == {"claims": 1, "pids": 1, "busy": 1}
    assert snap.busy_for("m1") is True
    assert snap.busy_for("m2") is False


def test_snapshot_capture_skips_busy_fetch_when_no_pids():
    def busy_map_fn(pids):
        raise AssertionError("must not be called with no pids")

    snap = core.Snapshot.capture(lambda: [], lambda: {}, busy_map_fn)
    assert snap.busy_for("anything") is None


def test_snapshot_none_name_gets_no_claims_and_no_busy():
    # A nameless ps() row must NEVER be attributed everyone's claims or a pid.
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 123}, busy_map={123: True})
    assert snap.claims_for(None) == []
    assert snap.pid_for(None) is None
    assert snap.busy_for(None) is None


# ---- attach_coordination ---------------------------------------------------------

def test_attach_coordination_adds_claims_and_busy():
    loaded = [{"name": "m1"}, {"name": "m2"}]
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 123}, busy_map={123: True})
    result, resolved = core.attach_coordination(loaded, snap)
    assert result[0]["claims"] == [{"model": "m1", "owner": "x"}]
    assert result[0]["busy"] is True
    assert result[1]["claims"] == []
    assert result[1]["busy"] is None   # no pid -> undetermined, never guessed
    assert resolved == {123}


def test_attach_coordination_nameless_row_stays_unattributed():
    loaded = [{"name": None}]
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 123}, busy_map={123: True})
    result, resolved = core.attach_coordination(loaded, snap)
    assert result[0]["claims"] == []
    assert result[0]["busy"] is None
    assert resolved == set()


# ---- other_processes ------------------------------------------------------------

def test_other_processes_excludes_known_ollama_pids():
    procs = [
        {"pid": 100, "size_mb": 500, "kind": "compute"},
        {"pid": 200, "size_mb": 300, "kind": "graphics"},
    ]
    result = core.other_processes(procs, exclude_pids={100})
    assert result == [{"pid": 200, "size_mb": 300, "kind": "graphics"}]


# ---- pressure ---------------------------------------------------------------

GPUS_OK = [{"index": 0, "total_mb": 24576, "used_mb": 4000, "free_mb": 20576}]
GPUS_TIGHT = [{"index": 0, "total_mb": 24576, "used_mb": 24400, "free_mb": 176}]


def test_pressure_ok():
    p = core.pressure(GPUS_OK, [], [])
    assert p["state"] == "ok"
    assert p["paging"] is None
    assert "spilling" not in p
    assert p["non_local_mb"] == 0


def test_pressure_tight_when_free_low():
    assert core.pressure(GPUS_TIGHT, [], [])["state"] == "tight"


def test_pressure_degraded_on_cpu_offload():
    loaded = [{"name": "qwen3:32b", "offloaded_to_cpu": True}]
    p = core.pressure(GPUS_OK, loaded, [])
    assert p["state"] == "degraded"
    assert p["offloaded_models"] == ["qwen3:32b"]


def test_pressure_thrashing_beats_degraded():
    loaded = [{"name": "qwen3:32b", "offloaded_to_cpu": True}]
    p = core.pressure(GPUS_OK, loaded, [], paging=SUSTAINED_PAGING)
    assert p["state"] == "thrashing"
    assert p["paging"] == SUSTAINED_PAGING
    assert "8 of the last 10 seconds" in p["detail"]
    assert "41.5%" in p["detail"]


def test_pressure_thrashing_beats_tight():
    p = core.pressure(GPUS_TIGHT, [], [], paging=SUSTAINED_PAGING)
    assert p["state"] == "thrashing"


def test_pressure_quiet_paging_does_not_hide_other_states():
    assert core.pressure(GPUS_TIGHT, [], [], paging=QUIET_PAGING)["state"] == "tight"
    loaded = [{"name": "qwen3:32b", "offloaded_to_cpu": True}]
    assert core.pressure(GPUS_OK, loaded, [], paging=QUIET_PAGING)["state"] == "degraded"
    p = core.pressure(GPUS_OK, [], [], paging=QUIET_PAGING)
    assert p["state"] == "ok"
    assert p["paging"] == QUIET_PAGING


def test_pressure_sustained_paging_is_thrashing_with_or_without_non_local_memory():
    """The verdict follows the driver's measured paging, not the non-local
    figure: none of it, or 4 GiB of it, changes nothing."""
    bare = core.pressure(GPUS_OK, [], [], paging=SUSTAINED_PAGING)
    heavy = core.pressure(GPUS_OK, [], [{"pid": 1, "non_local_mb": 4096}],
                          paging=SUSTAINED_PAGING)
    assert bare["state"] == heavy["state"] == "thrashing"


def test_pressure_sustained_paging_is_thrashing_when_free_vram_is_unreadable():
    p = core.pressure([{"index": 0, "free_mb": None}], [], [], paging=SUSTAINED_PAGING)
    assert p["free_mb"] is None
    assert p["state"] == "thrashing"


# The live shape this machine runs: a 32B model deliberately part-offloaded on
# a 24 GB card. On Windows/WDDM the runner's CPU-side layers are reported as
# that PROCESS's "Non Local Usage", so its non-local MB is explained memory,
# not driver paging.
RUNNER_PID = 41508
QWEN_TOTAL_MB = 27802
QWEN_VRAM_MB = 22281
QWEN_OFFLOAD_MB = QWEN_TOTAL_MB - QWEN_VRAM_MB   # 5521


def _offloaded_model(name="qwen3:32b"):
    return {"name": name, "total_size_mb": QWEN_TOTAL_MB,
            "size_vram_mb": QWEN_VRAM_MB, "offloaded_to_cpu": True}


def test_pressure_runner_non_local_within_its_offload_is_not_paging():
    """The steady state of a 32B on a 24 GB card: the runner's non-local MB is
    the deliberate CPU offload seen from the driver's side. Calling that
    'the driver is paging' is a false statement of fact."""
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 3906}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB})
    assert p["state"] == "degraded"
    assert p["paging"] is None
    assert p["explained_offload_mb"] == 3906
    assert p["unexplained_spill_mb"] == 0
    assert p["non_local_mb"] == 3906          # the total is still reported
    assert "not paging" in p["detail"]


def test_pressure_runner_non_local_beyond_its_offload_is_reported_not_graded():
    """Non-local memory is a quantity, not a symptom: a runner holding more than
    its offload explains is reported as unexplained, but only measured paging
    reads as thrashing."""
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 9000}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB})
    assert p["state"] == "degraded"
    assert p["explained_offload_mb"] == QWEN_OFFLOAD_MB
    assert p["unexplained_spill_mb"] == 9000 - QWEN_OFFLOAD_MB
    paged = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                          runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB},
                          paging=SUSTAINED_PAGING)
    assert paged["state"] == "thrashing"


def test_pressure_non_runner_non_local_is_unexplained_not_thrashing():
    """Nothing explains another process's non-local memory, so it is reported
    as unexplained; without measured paging it does not make a verdict."""
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 3906},
             {"pid": 3220, "size_mb": 4000, "non_local_mb": 4096}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB})
    assert p["state"] == "degraded"
    assert p["unexplained_spill_mb"] == 4096
    assert p["explained_offload_mb"] == 3906
    assert p["non_local_mb"] == 3906 + 4096


def test_pressure_sums_runner_excess_and_other_processes():
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 9000},
             {"pid": 3220, "size_mb": 4000, "non_local_mb": 4096}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB})
    assert p["state"] == "degraded"
    assert p["unexplained_spill_mb"] == (9000 - QWEN_OFFLOAD_MB) + 4096
    assert p["explained_offload_mb"] == QWEN_OFFLOAD_MB


def test_pressure_unknown_runner_pids_attribute_nothing():
    """Without a runner map (no snapshot) every row is 'other' — the honest
    fallback: we cannot prove anything explains the non-local memory."""
    procs = [{"pid": RUNNER_PID, "non_local_mb": 3906}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs)
    assert p["unexplained_spill_mb"] == 3906
    assert p["explained_offload_mb"] == 0
    assert p["state"] == "degraded"


# A full GPU that is not overcommitted already holds more desktop non-local
# memory than it has free VRAM, so neither the amount nor its ratio to free VRAM
# says anything about paging.
GPUS_ROOMY = [{"index": 0, "total_mb": 24576, "used_mb": 21894, "free_mb": 2682}]


def test_pressure_small_non_local_with_room_is_ok():
    """The live false alarm: 386 MB non-local spread across ordinary desktop
    apps while 2682 MB is free."""
    procs = [{"pid": 32220, "non_local_mb": 244},   # UnrealEditor
             {"pid": 19808, "non_local_mb": 82},    # a CUDA python job
             {"pid": 26344, "non_local_mb": 30},
             {"pid": 27004, "non_local_mb": 12},
             {"pid": 29572, "non_local_mb": 10},
             {"pid": 1500, "non_local_mb": 8}]
    p = core.pressure(GPUS_ROOMY, [], procs)
    assert p["unexplained_spill_mb"] == 386   # still reported, just not graded
    assert p["state"] == "ok"


def test_pressure_large_non_local_without_paging_is_not_thrashing():
    """More non-local memory than free VRAM used to read as paging. A full,
    healthy desktop GPU does exactly that, so it is only `tight` (free < 1024)."""
    p = core.pressure(GPUS_TIGHT, [], [{"pid": 1, "non_local_mb": 3949}])
    assert p["unexplained_spill_mb"] == 3949
    assert p["state"] == "tight"
    quiet = core.pressure(GPUS_TIGHT, [], [{"pid": 1, "non_local_mb": 3949}],
                          paging=QUIET_PAGING)
    assert quiet["state"] == "tight"


def test_pressure_large_non_local_with_free_vram_unreadable_is_unknown_not_thrashing():
    p = core.pressure([{"index": 0, "free_mb": None}], [],
                      [{"pid": 1, "non_local_mb": 4096}])
    assert p["free_mb"] is None
    assert p["state"] == "unknown"


def test_runner_offloads_maps_pid_to_deliberate_offload():
    loaded = [_offloaded_model(), {"name": "small", "total_size_mb": 4096,
                                   "size_vram_mb": 4096}]
    snap = _snap(pid_map={"qwen3:32b": RUNNER_PID, "small": 777})
    assert core.runner_offloads(loaded, snap) == {RUNNER_PID: QWEN_OFFLOAD_MB,
                                                  777: 0}


def test_runner_offloads_skips_uncorrelated_models():
    """A model whose runner PID could not be resolved entitles nobody."""
    assert core.runner_offloads([_offloaded_model()], _snap(pid_map={})) == {}


def test_combined_status_live_offload_shape_is_degraded_not_thrashing():
    """End-to-end wiring of the shipped defect: full process table + resolved
    runner PID + a deliberately offloaded model must read 'degraded'."""
    models = [{"name": "qwen3:32b", "size": 29153380267,
               "size_vram": 23363762257, "expires_at": None}]
    status = core.combined_status(
        lambda: [{"index": 0, "total_mb": 24576, "used_mb": 24017, "free_mb": 559}],
        FakeOllama(models),
        snapshot_fn=lambda correlate: _snap(pid_map={"qwen3:32b": RUNNER_PID}),
        procinfo_fn=lambda: [
            {"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 3906},
            {"pid": 9968, "size_mb": 671, "non_local_mb": 3},
        ],
    )
    p = status["pressure"]
    assert p["state"] == "degraded"
    assert p["paging"] is None
    assert p["unexplained_spill_mb"] == 3
    assert p["explained_offload_mb"] == 3906


def test_pressure_reports_small_non_local_without_grading_it():
    procs = [{"pid": 1, "non_local_mb": 10}, {"pid": 2, "non_local_mb": 20}]
    p = core.pressure(GPUS_OK, [], procs)
    assert p["non_local_mb"] == 30
    assert p["state"] == "ok"


def test_pressure_tolerates_none_values():
    procs = [{"pid": 1, "non_local_mb": None}, {"pid": 2}]
    p = core.pressure([{"index": 0, "free_mb": None}], [{"name": None}], procs)
    assert p["non_local_mb"] == 0
    assert p["free_mb"] is None
    assert p["state"] == "unknown"


def test_combined_status_includes_pressure():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=lambda: [],
        paging_fn=paging_fn(),
    )
    assert status["pressure"]["state"] == "ok"


# ---- `ok` is earned, not defaulted to ---------------------------------------
#
# `ok` asserts that nothing is wrong, so it needs the Ollama residency and the
# driver's paging to have been read. Where no collector reads paging (anywhere
# but Windows/WDDM) or it is not yet readable, a healthy verdict would be an
# assertion about something never measured. Non-local memory is not required:
# it is reported, never graded.

def _blind_to_non_local(rows=()):
    """A process reading from a platform that cannot see non-local memory."""
    return lambda: Observation(list(rows), "procs",
                               coverage={"non_local_memory": False})


def test_ok_is_withheld_when_paging_could_not_be_ruled_out():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=lambda: [],
    )
    pressure = status["pressure"]
    assert pressure["state"] == "unknown"
    assert pressure["detail"] == (
        "Driver paging is unavailable, so it cannot be ruled out "
        "(no collector reads driver paging on this platform).")
    assert pressure["coverage"]["paging"] is False
    assert pressure["paging"] is None
    assert "paging" not in status["observations"]


def test_ok_withheld_detail_carries_the_paging_reading_error():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=lambda: [],
        paging_fn=paging_gap("paging history is still being collected (2 of 5 s)"),
    )
    assert status["pressure"]["state"] == "unknown"
    assert status["pressure"]["detail"] == (
        "Driver paging is unavailable, so it cannot be ruled out "
        "(paging history is still being collected (2 of 5 s)).")
    assert status["pressure"]["coverage"]["paging"] is False
    assert status["observations"]["paging"]["status"] == "unavailable"


def test_missing_residency_is_reported_before_missing_paging_coverage():
    """Both requirements unmet: the reason names the nearer one rather than
    whichever branch happens to run first."""
    class _Blind:
        def observe_loaded(self):
            return Observation(None, "ollama:/api/ps", error="transport down")

    status = core.combined_status(
        lambda: GPUS_OK, _Blind(), procinfo_fn=lambda: [],
    )
    assert status["pressure"]["state"] == "unknown"
    assert status["pressure"]["detail"] == "Ollama residency is unavailable."


def test_missing_paging_coverage_does_not_suppress_tight():
    """`tight` rests on a capacity reading it did observe. Downgrading it would
    discard a fact, not withhold a guess."""
    status = core.combined_status(
        lambda: GPUS_TIGHT, FakeOllama([]), procinfo_fn=lambda: [],
    )
    assert status["pressure"]["state"] == "tight"


def test_missing_paging_coverage_does_not_suppress_degraded():
    models = [{"name": "m1", "size": gb_bytes(20), "size_vram": gb_bytes(16),
               "expires_at": None}]
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama(models),
        snapshot_fn=lambda correlate: _snap(pid_map={"m1": 555}),
        procinfo_fn=_blind_to_non_local([{"pid": 555, "size_mb": 16384}]),
    )
    assert status["pressure"]["state"] == "degraded"
    assert status["pressure"]["offloaded_models"] == ["m1"]


def test_ok_survives_when_every_requirement_is_covered():
    """The downgrade must not fire on a platform that CAN see paging and
    reports none — that is a measurement, not an absence."""
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]),
        procinfo_fn=lambda: Observation(
            [{"pid": 999, "size_mb": 100, "non_local_mb": 0}], "procs",
            coverage={"non_local_memory": True}),
        paging_fn=paging_fn(),
    )
    assert status["pressure"]["state"] == "ok"
    assert status["pressure"]["coverage"]["paging"] is True
    assert status["pressure"]["paging"]["sustained"] is False
    assert status["observations"]["paging"]["status"] == "available"


def test_ok_does_not_need_non_local_coverage_when_paging_is_read():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=_blind_to_non_local(),
        paging_fn=paging_fn(),
    )
    pressure = status["pressure"]
    assert pressure["state"] == "ok"
    assert pressure["coverage"]["non_local_memory"] is False
    # The figures stay null rather than reading as a measured zero.
    assert pressure["unexplained_spill_mb"] is None
    assert pressure["non_local_mb"] is None


def test_ok_with_full_coverage_and_integer_non_local_readings():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]),
        procinfo_fn=lambda: Observation(
            [{"pid": 100, "size_mb": None, "non_local_mb": 40, "kind": "compute"},
             {"pid": 900, "size_mb": None, "non_local_mb": 0, "kind": "wddm"}],
            "procs", coverage={"non_local_memory": True}),
        paging_fn=paging_fn(),
    )
    assert status["pressure"]["state"] == "ok"


def test_sustained_paging_reads_thrashing_through_combined_status():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=lambda: [],
        paging_fn=paging_fn(*[20.0] * 8, 0.0, 0.0),
    )
    pressure = status["pressure"]
    assert pressure["state"] == "thrashing"
    assert pressure["paging"] == {"sustained": True, "active_seconds": 8,
                                  "seconds": 10, "peak_percent": 20.0}
    assert pressure["coverage"]["paging"] is True


def test_sustained_paging_is_thrashing_even_without_other_coverage():
    """`thrashing` rests on the paging it measured; missing residency does not
    downgrade it."""
    class _Blind:
        def observe_loaded(self):
            return Observation(None, "ollama:/api/ps", error="transport down")

    status = core.combined_status(
        lambda: GPUS_OK, _Blind(), procinfo_fn=_blind_to_non_local(),
        paging_fn=paging_fn(*[20.0] * 10),
    )
    assert status["pressure"]["state"] == "thrashing"


def test_unexplained_wddm_row_exceeding_free_vram_is_not_thrashing():
    """The false positive this verdict used to give: a full GPU already holds
    more non-local memory than it has free, with no paging."""
    status = core.combined_status(
        lambda: GPUS_TIGHT, FakeOllama([]),
        procinfo_fn=lambda: Observation(
            [{"pid": 900, "size_mb": None, "non_local_mb": 2048, "kind": "wddm"}],
            "procs", coverage={"non_local_memory": True}),
        paging_fn=paging_fn(),
    )
    pressure = status["pressure"]
    assert pressure["state"] == "tight"
    assert pressure["unexplained_spill_mb"] == 2048
    assert "spilling" not in pressure


def test_wddm_row_below_free_vram_is_ok_without_paging():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]),
        procinfo_fn=lambda: Observation(
            [{"pid": 900, "size_mb": None, "non_local_mb": 2048, "kind": "wddm"}],
            "procs", coverage={"non_local_memory": True}),
        paging_fn=paging_fn(),
    )
    assert status["pressure"]["state"] == "ok"


# ---- a measured adapter total: rows only attribute it ------------------------

def _with_total(rows, total_mb, *, coverage=None):
    return lambda: Observation(
        list(rows), "procs", coverage=coverage or {"non_local_memory": True},
        totals={"non_local_mb": total_mb})


def test_memory_no_row_accounts_for_is_unexplained_but_not_thrashing():
    status = core.combined_status(
        gpu_fn_const(1000), FakeOllama([]),
        procinfo_fn=_with_total(
            [{"pid": 900, "size_mb": None, "non_local_mb": 10, "kind": "wddm"}],
            2048),
        paging_fn=paging_fn(),
    )
    p = status["pressure"]
    assert p["unexplained_spill_mb"] == 2048      # the total, not the 10 MB rows
    assert p["non_local_mb"] == 2048
    assert p["state"] == "tight"                  # free < 1024, no paging


def test_hidden_memory_with_room_to_spare_is_ok_and_covered():
    status = core.combined_status(
        gpu_fn_const(20000), FakeOllama([]),
        procinfo_fn=_with_total(
            [{"pid": 900, "size_mb": None, "non_local_mb": 10, "kind": "wddm"}],
            2048),
        paging_fn=paging_fn(),
    )
    p = status["pressure"]
    assert p["state"] == "ok"
    assert p["unexplained_spill_mb"] == 2048       # reported, just not graded
    assert p["coverage"]["non_local_memory"] is True


def test_row_without_reading_is_covered_when_the_adapter_total_is_known():
    status = core.combined_status(
        gpu_fn_const(20000), FakeOllama([]),
        procinfo_fn=_with_total(
            [{"pid": 900, "size_mb": None, "non_local_mb": None, "kind": "wddm"}],
            0),
        paging_fn=paging_fn(),
    )
    assert status["pressure"]["coverage"]["non_local_memory"] is True
    assert status["pressure"]["state"] == "ok"


def test_row_without_reading_and_no_total_leaves_non_local_figures_null():
    """Without a measured total, a row with no reading leaves the figures
    unknown. `ok` no longer depends on them, only on paging."""
    status = core.combined_status(
        gpu_fn_const(20000), FakeOllama([]),
        procinfo_fn=lambda: Observation(
            [{"pid": 900, "size_mb": None, "non_local_mb": None, "kind": "wddm"}],
            "procs", coverage={"non_local_memory": True}),
        paging_fn=paging_fn(),
    )
    p = status["pressure"]
    assert p["coverage"]["non_local_memory"] is False
    assert p["non_local_mb"] is None
    assert p["unexplained_spill_mb"] is None
    assert p["state"] == "ok"


def test_runner_offload_is_still_explained_under_a_total():
    models = [{"name": "m", "size": 13000 * _MB, "size_vram": 10000 * _MB}]
    status = core.combined_status(
        gpu_fn_const(500), FakeOllama(models),
        snapshot_fn=lambda correlate: _snap(pid_map={"m": 999}),
        procinfo_fn=_with_total(
            [{"pid": 999, "size_mb": 10000, "non_local_mb": 3000,
              "kind": "compute"}], 3010),
        paging_fn=paging_fn(),
    )
    p = status["pressure"]
    assert p["explained_offload_mb"] == 3000
    assert p["unexplained_spill_mb"] == 10
    assert p["non_local_mb"] == 3010
    assert p["state"] == "degraded"


def test_pressure_total_below_explained_never_goes_negative():
    """Rows that double-count (or a stale total) must not yield negative spill."""
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 3906}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB},
                      non_local_total_mb=1000)
    assert p["explained_offload_mb"] == 1000      # clamped to the measurement
    assert p["unexplained_spill_mb"] == 0
    assert p["non_local_mb"] == 1000


def test_double_counted_rows_cannot_push_unexplained_above_the_total():
    """Two processes each report one shared 2 GiB allocation, and a third
    process another 2 GiB, while the adapter measures 3 GiB. The rows add up to
    6 GiB, but what is unexplained cannot exceed the 3 GiB the adapter holds."""
    procs = [{"pid": 1, "non_local_mb": 2048}, {"pid": 2, "non_local_mb": 2048},
             {"pid": 3, "non_local_mb": 2048}]
    p = core.pressure([{"free_mb": 512}], [], procs,
                      runner_offloads={1: 1024}, non_local_total_mb=3072)
    assert p["non_local_mb"] == 3072
    assert p["unexplained_spill_mb"] <= p["non_local_mb"]
    assert p["explained_offload_mb"] == 1024
    assert p["unexplained_spill_mb"] == 3072 - 1024
    assert p["explained_offload_mb"] + p["unexplained_spill_mb"] == 3072


def test_explained_and_unexplained_always_add_up_to_a_measured_total():
    procs = [{"pid": 1, "non_local_mb": 2048}, {"pid": 2, "non_local_mb": 2048},
             {"pid": 3, "non_local_mb": 1024}]
    for total in (0, 500, 3072, 5120, 9000):
        p = core.pressure([{"free_mb": 512}], [], procs,
                          runner_offloads={1: 2048, 2: 2048}, non_local_total_mb=total)
        assert p["non_local_mb"] == total
        assert 0 <= p["unexplained_spill_mb"] <= total
        assert p["explained_offload_mb"] + p["unexplained_spill_mb"] == total


# ---- combined_status full wiring -------------------------------------------------

def test_combined_status_full_wiring():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4),
              "expires_at": None}]
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 555}, busy_map={555: True})
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        snapshot_fn=lambda correlate: snap,
        nvml_processes_fn=lambda: [
            {"pid": 555, "size_mb": 4096, "kind": "compute"},
            {"pid": 999, "size_mb": 100, "kind": "graphics"},
        ],
    )
    entry = status["loaded"][0]
    assert entry["claims"] == [{"model": "m1", "owner": "x"}]
    assert entry["busy"] is True
    # pid 555 IS the "m1" runner -> excluded from other_processes; pid 999 stays.
    assert status["other_processes"] == [{"pid": 999, "size_mb": 100, "kind": "graphics"}]


def test_combined_status_other_processes_from_procinfo():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4), "expires_at": None}]
    snap = _snap(pid_map={"m1": 555})
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        snapshot_fn=lambda correlate: snap,
        procinfo_fn=lambda: [
            {"pid": 555, "size_mb": 4096, "name": "llama-server.exe", "cmdline": "...", "kind": "compute"},
            {"pid": 999, "size_mb": 14492, "name": "python.exe", "cmdline": "python train.py", "kind": "compute"},
        ],
    )
    # pid 555 is the m1 runner -> excluded; 999 stays, WITH its size + name.
    assert status["other_processes"] == [
        {"pid": 999, "size_mb": 14492, "name": "python.exe",
         "cmdline": "python train.py", "kind": "compute"},
    ]


def test_pressure_sees_ollama_runner_spill():
    """The runner is filtered out of other_processes but MUST still count
    toward the non-local figures: it is the biggest holder.

    The model here is DELIBERATELY OFFLOADED (4 GB of a 20 GB model on the CPU)
    and its runner holds 8 GB non-local — double what that placement explains."""
    calls = []

    def source():
        calls.append(1)
        return [{"pid": 999, "size_mb": 20000, "non_local_mb": 8192},
                {"pid": 111, "size_mb": 100, "non_local_mb": 0}]

    status = core.combined_status(
        lambda: [{"index": 0, "total_mb": 24576, "used_mb": 24000, "free_mb": 576}],
        FakeOllama([{"name": "m", "size": gb_bytes(20), "size_vram": gb_bytes(16)}]),
        snapshot_fn=lambda correlate: _snap(pid_map={"m": 999}),
        procinfo_fn=source,
    )
    p = status["pressure"]
    assert p["non_local_mb"] == 8192
    assert p["explained_offload_mb"] == 4096      # the deliberate placement
    assert p["unexplained_spill_mb"] == 4096      # the excess, reported
    assert p["state"] == "degraded"               # the model is part-offloaded
    # the runner is still excluded from the "other processes" view
    assert [p["pid"] for p in status["other_processes"]] == [111]
    # and the expensive source was sampled exactly once
    assert len(calls) == 1


def test_combined_status_reads_paging_once_and_forwards_its_summary():
    calls = []

    def reading():
        calls.append(1)
        return paging_fn(*[10.0] * 10)()

    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=lambda: [], paging_fn=reading,
    )
    assert len(calls) == 1
    assert status["pressure"]["paging"] == {
        "sustained": True, "active_seconds": 10, "seconds": 10, "peak_percent": 10.0}
    assert status["observations"]["paging"]["source"] == "wddm:pdh"


def test_combined_status_has_no_spill_threshold_parameter():
    """The non-local threshold was retired with the verdict it gated."""
    assert not hasattr(core, "SPILL_THRESHOLD_MB")
    try:
        core.combined_status(lambda: GPUS_OK, FakeOllama([]), spill_threshold_mb=128)
    except TypeError:
        return
    raise AssertionError("spill_threshold_mb must no longer be accepted")


def test_combined_status_reads_only_the_ledger_when_nothing_is_loaded():
    """Reservations exist without any loaded model, so the ledger is read on
    every call; the runner listing and NVML session it would correlate are not."""
    calls = []

    def snapshot(correlate):
        calls.append(correlate)
        return _snap()

    status = core.combined_status(gpu_fn_const(8000), FakeOllama([]), snapshot_fn=snapshot)
    assert calls == [False]
    assert status["loaded"] == []
    coordination = status["observations"]["coordination"]
    assert (coordination["status"], coordination["source"]) == ("available", "claims")


def test_combined_status_reports_an_unreadable_ledger_with_nothing_loaded():
    """An unreadable ledger must never look like an empty one."""
    def unreadable(correlate):
        raise ValueError("claim ledger uses coordination protocol 2")

    status = core.combined_status(gpu_fn_const(8000), FakeOllama([]), snapshot_fn=unreadable)
    coordination = status["observations"]["coordination"]
    assert coordination["status"] == "unavailable"
    assert coordination["error"] == "claim ledger uses coordination protocol 2"


def test_combined_status_correlates_when_models_are_loaded():
    calls = []

    def snapshot(correlate):
        calls.append(correlate)
        return _snap(pid_map={"m": 999})

    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama([{"name": "m", "size_vram": gb_bytes(4)}]),
        snapshot_fn=snapshot)
    assert calls == [True]
    assert status["observations"]["coordination"]["source"] == "claims+nvml"


# ---- ensure_free ------------------------------------------------------------

def test_ensure_free_already_free_short_circuits():
    ollama = FakeOllama([
        {"name": "m", "size_vram": gb_bytes(4), "expires_at": None},
    ])
    result = core.ensure_free(
        4, gpu_fn_const(8192), ollama, sleep=lambda *_: None
    )
    assert result["ok"] is True
    assert result["already_free"] is True
    assert result["unloaded"] == []
    assert ollama.unloaded == []  # nothing evicted


def test_ensure_free_unloads_largest_first_and_stops():
    models = [
        {"name": "small", "size_vram": gb_bytes(2), "expires_at": None},
        {"name": "huge", "size_vram": gb_bytes(10), "expires_at": None},
        {"name": "medium", "size_vram": gb_bytes(5), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    # free: 1000 (initial check) -> 11000 after first unload meets 8GB target.
    gpu_fn = gpu_fn_sequence([1000, 11000])
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)

    assert result["ok"] is True
    assert result["already_free"] is False
    # Largest-first: "huge" unloaded first; target met -> stop before others.
    assert ollama.unloaded == ["huge"]
    assert result["unloaded"] == ["huge"]
    assert result["free_mb"] == 11000
    assert result["target_mb"] == 8 * 1024


def test_ensure_free_unloads_multiple_until_target():
    models = [
        {"name": "a", "size_vram": gb_bytes(4), "expires_at": None},
        {"name": "b", "size_vram": gb_bytes(6), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    # 1000 initial, 5000 after first unload (still < 8GB), 9000 after second.
    gpu_fn = gpu_fn_sequence([1000, 5000, 9000])
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)
    assert result["ok"] is True
    assert ollama.unloaded == ["b", "a"]  # largest first
    assert result["free_mb"] == 9000


def test_ensure_free_cannot_reach_target():
    models = [
        {"name": "a", "size_vram": gb_bytes(2), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 3000])  # never reaches 8GB
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)
    assert result["ok"] is False
    assert ollama.unloaded == ["a"]
    assert result["free_mb"] == 3000


def test_ensure_free_unknown_vram_is_not_ok():
    ollama = FakeOllama([
        {"name": "a", "size_vram": gb_bytes(2), "expires_at": None},
    ])
    result = core.ensure_free(
        8, gpu_fn_const(None), ollama, sleep=lambda *_: None
    )
    assert result["ok"] is False
    assert result["free_mb"] is None


def test_ensure_free_settle_calls_sleep():
    calls = []
    models = [{"name": "a", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])
    core.ensure_free(
        8, gpu_fn, ollama, settle=0.5, sleep=lambda s: calls.append(s)
    )
    assert calls == [0.5]


# ---- is_protected -----------------------------------------------------------

def test_is_protected_true_when_active_claim_exists():
    snap = _snap(all_claims=[{"model": "m", "owner": "x"}])
    protected, detail = core.is_protected("m", snap)
    assert protected is True
    assert detail["claims"] == [{"model": "m", "owner": "x"}]


def test_is_protected_true_when_busy():
    snap = _snap(pid_map={"m": 123}, busy_map={123: True})
    protected, detail = core.is_protected("m", snap)
    assert protected is True
    assert detail["busy"] is True


def test_is_protected_false_when_unclaimed_and_idle():
    snap = _snap(pid_map={"m": 123}, busy_map={123: False})
    protected, _ = core.is_protected("m", snap)
    assert protected is False


def test_is_protected_false_when_no_pid_and_no_claim():
    snap = _snap(busy_map={123: True})   # a busy pid exists, but not for "m"
    protected, detail = core.is_protected("m", snap)
    assert protected is False
    assert detail["busy"] is None


# ---- reserved_mb -------------------------------------------------------------

def test_reserved_mb_sums_active_reservations():
    recs = [{"kind": "reservation", "gb": 8.0},
            {"kind": "reservation", "gb": 2.5},
            {"kind": "model", "model": "llama3"}]
    assert core.reserved_mb(recs) == int(round(10.5 * 1024))


def test_reserved_mb_skips_malformed():
    recs = [{"kind": "reservation", "gb": "eight"},
            {"kind": "reservation"},
            "not-a-dict",
            {"kind": "reservation", "gb": 1.0}]
    assert core.reserved_mb(recs) == 1024


def test_reserved_mb_empty():
    assert core.reserved_mb([]) == 0


def test_reserved_mb_ignores_non_positive_gb():
    """A hand-edited (or hostile) ledger must not let one record cancel
    another's reservation — the total is only ever a floor, never negative."""
    recs = [{"kind": "reservation", "gb": 8.0},
            {"kind": "reservation", "gb": -8.0},
            {"kind": "reservation", "gb": 0}]
    assert core.reserved_mb(recs) == 8192


# ---- can_warm ----------------------------------------------------------------

def test_can_warm_allows_when_it_fits():
    ok, detail = core.can_warm("llama3", free_mb=20000, reserved_mb=8192,
                               model_size_mb=4096)
    assert ok is True
    assert detail["headroom_mb"] == 20000 - 8192


def test_can_warm_refuses_when_reservations_consume_headroom():
    ok, detail = core.can_warm("qwen3:32b", free_mb=10000, reserved_mb=8192,
                               model_size_mb=20480)
    assert ok is False
    assert detail["reason"] == "insufficient_headroom"
    assert detail["model_size_mb"] == 20480


def test_can_warm_refuses_when_headroom_exhausted_and_size_unknown():
    ok, detail = core.can_warm("mystery", free_mb=4000, reserved_mb=8192,
                               model_size_mb=None)
    assert ok is False
    assert detail["reason"] == "no_headroom"


def test_can_warm_allows_unknown_size_with_headroom():
    # Allowed, but the reason must NOT claim a size check happened — a caller
    # has to be able to tell a verified fit from an unverified one.
    ok, detail = core.can_warm("mystery", free_mb=20000, reserved_mb=1024,
                               model_size_mb=None)
    assert ok is True
    assert detail["reason"] == "size_unknown"


def test_can_warm_reason_fits_only_when_size_was_checked():
    _, detail = core.can_warm("llama3", free_mb=20000, reserved_mb=8192,
                              model_size_mb=4096)
    assert detail["reason"] == "fits"


def test_can_warm_allows_when_free_unknown():
    # VRAM unreadable -> we cannot prove it will not fit; never block on a guess.
    ok, detail = core.can_warm("m", free_mb=None, reserved_mb=8192,
                               model_size_mb=4096)
    assert ok is True
    assert detail["reason"] == "free_unknown"


def test_can_warm_allows_when_nothing_reserved():
    ok, _ = core.can_warm("m", free_mb=100, reserved_mb=0, model_size_mb=99999)
    assert ok is True


def test_reservations_never_protect_a_model():
    snap = core.Snapshot(
        all_claims=[{"kind": "reservation", "model": None, "gb": 8.0}],
        pid_map={}, busy_map={},
    )
    protected, detail = core.is_protected("llama3", snap)
    assert protected is False
    assert detail["claims"] == []


# ---- ensure_free protection ---------------------------------------------------

def test_ensure_free_skips_protected_model_and_reports_declined():
    models = [
        {"name": "protected", "size_vram": gb_bytes(10), "expires_at": None},
        {"name": "free-game", "size_vram": gb_bytes(6), "expires_at": None},
    ]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 7000])

    result = core.ensure_free(
        6, gpu_fn, ollama, sleep=lambda *_: None,
        snapshot_fn=lambda: _snap(
            all_claims=[{"model": "protected", "owner": "other"}]),
    )
    assert ollama.unloaded == ["free-game"]  # "protected" skipped despite being largest
    assert result["ok"] is True
    assert len(result["declined"]) == 1
    assert result["declined"][0]["name"] == "protected"


def test_ensure_free_rechecks_claims_after_each_eviction():
    claims = []

    class ClaimingOllama(FakeOllama):
        def unload(self, name):
            self.unloaded.append(name)
            claims.append({"model": "b:latest", "owner": "new session"})
            return True

    ollama = ClaimingOllama([{"name": "a", "size_vram": gb_bytes(8)},
                            {"name": "b", "size_vram": gb_bytes(4)}])
    result = core.ensure_free(12, gpu_fn_sequence([1000, 9000]), ollama,
                              snapshot_fn=lambda: _snap(all_claims=list(claims)))
    assert ollama.unloaded == ["a"]
    assert result["declined"][0]["name"] == "b"


def test_ensure_free_force_bypasses_protection_without_snapshotting():
    models = [{"name": "protected", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])

    def snapshot_fn():
        raise AssertionError("force=True must not pay for a snapshot")

    result = core.ensure_free(
        8, gpu_fn, ollama, sleep=lambda *_: None, force=True,
        snapshot_fn=snapshot_fn,
    )
    assert ollama.unloaded == ["protected"]
    assert result["declined"] == []


def test_ensure_free_protection_noop_when_snapshot_not_provided():
    """Existing callers that don't wire a snapshot see unchanged behavior."""
    models = [{"name": "a", "size_vram": gb_bytes(10), "expires_at": None}]
    ollama = FakeOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 11000])
    result = core.ensure_free(8, gpu_fn, ollama, sleep=lambda *_: None)
    assert ollama.unloaded == ["a"]
    assert result["declined"] == []


def test_ensure_free_skips_snapshot_when_no_models_to_evict():
    """Target unreachable with zero loaded models: no snapshot is captured."""
    def exploding_snapshot():
        raise AssertionError("snapshot must not be captured for zero models")

    result = core.ensure_free(
        8, gpu_fn_const(1000), FakeOllama([]), sleep=lambda *_: None,
        snapshot_fn=exploding_snapshot,
    )
    assert result["ok"] is False
    assert result["unloaded"] == []


def test_ensure_free_no_settle_sleep_when_unload_fails():
    """The settle sleep exists to let the driver release memory after an
    eviction — a FAILED unload released nothing, so sleeping is pure waste."""
    class RefusingOllama(FakeOllama):
        def unload(self, model):
            self.unloaded.append(model)
            return False

    models = [{"name": "a", "size_vram": gb_bytes(4), "expires_at": None}]
    ollama = RefusingOllama(models)
    gpu_fn = gpu_fn_sequence([1000, 1000])
    sleeps = []
    core.ensure_free(8, gpu_fn, ollama, settle=0.5,
                     sleep=lambda s: sleeps.append(s))
    assert ollama.unloaded == ["a"]   # attempt made
    assert sleeps == []               # but no pointless settle wait


# ---- advise -----------------------------------------------------------------

def test_advise_recommends_max_loaded_when_many_and_low_free():
    models = [
        {"name": "a", "size_vram": gb_bytes(4), "expires_at": None},
        {"name": "b", "size_vram": gb_bytes(4), "expires_at": None},
    ]
    result = core.advise(gpu_fn_const(500), FakeOllama(models))
    joined = " ".join(result["suggestions"])
    assert "OLLAMA_MAX_LOADED_MODELS=1" in joined


def test_advise_recommends_keep_alive_when_pinned_forever():
    models = [
        {"name": "pinned", "size_vram": gb_bytes(4),
         "expires_at": "0001-01-01T00:00:00Z"},
    ]
    result = core.advise(gpu_fn_const(20000), FakeOllama(models))
    joined = " ".join(result["suggestions"])
    assert "OLLAMA_KEEP_ALIVE" in joined
    assert "pinned" in joined


def test_advise_quiet_when_healthy():
    models = [
        {"name": "a", "size_vram": gb_bytes(4),
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    result = core.advise(gpu_fn_const(20000), FakeOllama(models))
    assert result["suggestions"] == []


def test_advise_no_max_loaded_when_free_is_high():
    models = [
        {"name": "a", "size_vram": gb_bytes(4),
         "expires_at": "2026-07-11T10:00:00Z"},
        {"name": "b", "size_vram": gb_bytes(4),
         "expires_at": "2026-07-11T10:00:00Z"},
    ]
    # Two models but plenty free -> no OLLAMA_MAX_LOADED_MODELS suggestion.
    result = core.advise(gpu_fn_const(20000), FakeOllama(models))
    joined = " ".join(result["suggestions"])
    assert "OLLAMA_MAX_LOADED_MODELS" not in joined
