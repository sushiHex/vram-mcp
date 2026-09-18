"""Tests for vram_mcp.core — pure orchestration with injected fakes."""

from vram_mcp import core
from vram_mcp.observations import Observation


_MB = 1024 * 1024


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
    assert status["pressure"]["spilling"] is None


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
    assert p["spilling"] is False
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
    procs = [{"pid": 1, "non_local_mb": 4096}]
    p = core.pressure(GPUS_TIGHT, loaded, procs)
    assert p["state"] == "thrashing"
    assert p["spilling"] is True
    assert p["non_local_mb"] == 4096


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
    assert p["spilling"] is False
    assert p["explained_offload_mb"] == 3906
    assert p["unexplained_spill_mb"] == 0
    assert p["non_local_mb"] == 3906          # the total is still reported
    assert "not paging" in p["detail"]


def test_pressure_runner_non_local_beyond_its_offload_is_real_spill():
    """'My model is being paged because something else ballooned' — the case
    neither the filtered nor the unfiltered table ever caught.

    The GPU must be CONSTRAINED for this to be paging at all: spill is only
    credible when the card lacked the room to keep it resident."""
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 9000}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB})
    assert p["state"] == "thrashing"
    assert p["spilling"] is True
    assert p["explained_offload_mb"] == QWEN_OFFLOAD_MB
    assert p["unexplained_spill_mb"] == 9000 - QWEN_OFFLOAD_MB


def test_pressure_non_runner_non_local_is_real_spill():
    """Nothing explains another process's non-local memory -> genuine paging,
    even while a model is legitimately offloaded."""
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 3906},
             {"pid": 3220, "size_mb": 4000, "non_local_mb": 4096}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB})
    assert p["state"] == "thrashing"
    assert p["unexplained_spill_mb"] == 4096
    assert p["explained_offload_mb"] == 3906
    assert p["non_local_mb"] == 3906 + 4096
    assert "4096 MB" in p["detail"]


def test_pressure_sums_runner_excess_and_other_processes():
    procs = [{"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 9000},
             {"pid": 3220, "size_mb": 4000, "non_local_mb": 4096}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs,
                      runner_offloads={RUNNER_PID: QWEN_OFFLOAD_MB})
    assert p["state"] == "thrashing"
    assert p["unexplained_spill_mb"] == (9000 - QWEN_OFFLOAD_MB) + 4096
    assert p["explained_offload_mb"] == QWEN_OFFLOAD_MB


def test_pressure_unknown_runner_pids_attribute_nothing():
    """Without a runner map (no snapshot) every row is 'other' — the honest
    fallback: we cannot prove anything explains the non-local memory."""
    procs = [{"pid": RUNNER_PID, "non_local_mb": 3906}]
    p = core.pressure(GPUS_TIGHT, [_offloaded_model()], procs)
    assert p["unexplained_spill_mb"] == 3906
    assert p["state"] == "thrashing"


# The driver only evicts when it runs out of room. Non-local memory while the
# card has plenty free is routine allocation (staging buffers, shared surfaces),
# not paging — and "free VRAM or reduce load" is nonsense advice with 2.6 GB free.
GPUS_ROOMY = [{"index": 0, "total_mb": 24576, "used_mb": 21894, "free_mb": 2682}]


def test_pressure_spill_below_free_vram_is_not_paging():
    """The live false alarm: 386 MB non-local spread across ordinary desktop
    apps while 2682 MB is free read as 'the driver is paging'. It cannot be —
    with that much free the driver had no reason to evict anything."""
    procs = [{"pid": 32220, "non_local_mb": 244},   # UnrealEditor
             {"pid": 19808, "non_local_mb": 82},    # a CUDA python job
             {"pid": 26344, "non_local_mb": 30},
             {"pid": 27004, "non_local_mb": 12},
             {"pid": 29572, "non_local_mb": 10},
             {"pid": 1500, "non_local_mb": 8}]
    p = core.pressure(GPUS_ROOMY, [], procs)
    assert p["unexplained_spill_mb"] == 386   # still reported, just not alarming
    assert p["spilling"] is False
    assert p["state"] == "ok"


def test_pressure_spill_exceeding_free_vram_is_paging():
    """More has been pushed to system RAM than the card has free — the driver
    would have kept it resident if there were room."""
    p = core.pressure(GPUS_TIGHT, [], [{"pid": 1, "non_local_mb": 3949}])
    assert p["spilling"] is True
    assert p["state"] == "thrashing"


def test_pressure_spill_gate_needs_both_the_floor_and_the_free_comparison():
    """A spill larger than free VRAM but below the absolute floor is still
    noise: 200 MB of non-local memory is not a multi-x slowdown."""
    p = core.pressure([{"index": 0, "free_mb": 100}], [],
                      [{"pid": 1, "non_local_mb": 200}])
    assert p["unexplained_spill_mb"] == 200
    assert p["spilling"] is False


def test_pressure_spill_alarms_when_free_vram_is_unreadable():
    """No nvidia-smi -> no free figure to compare against. We cannot prove the
    card had room, so a large unexplained spill still warrants the warning."""
    p = core.pressure([{"index": 0, "free_mb": None}], [],
                      [{"pid": 1, "non_local_mb": 4096}])
    assert p["free_mb"] is None
    assert p["spilling"] is True
    assert p["state"] == "thrashing"


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
        snapshot_fn=lambda: _snap(pid_map={"qwen3:32b": RUNNER_PID}),
        procinfo_fn=lambda: [
            {"pid": RUNNER_PID, "size_mb": 22371, "non_local_mb": 3906},
            {"pid": 9968, "size_mb": 671, "non_local_mb": 3},
        ],
    )
    p = status["pressure"]
    assert p["state"] == "degraded"
    assert p["spilling"] is False
    assert p["unexplained_spill_mb"] == 3
    assert p["explained_offload_mb"] == 3906


def test_pressure_ignores_noise_below_threshold():
    procs = [{"pid": 1, "non_local_mb": 10}, {"pid": 2, "non_local_mb": 20}]
    p = core.pressure(GPUS_OK, [], procs)
    assert p["non_local_mb"] == 30
    assert p["spilling"] is False
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
    )
    assert status["pressure"]["state"] == "ok"


# ---- `ok` is earned, not defaulted to ---------------------------------------
#
# Windows/WDDM is the live case: NVML cannot report per-process memory and the
# GPU counters aggregate adapters, so non-local coverage is always false there
# and driver spill can never be observed. A healthy verdict on that evidence
# would be an assertion about something never measured.

def _blind_to_non_local(rows=()):
    """A process reading from a platform that cannot see non-local memory."""
    return lambda: Observation(list(rows), "procs",
                               coverage={"non_local_memory": False})


def test_ok_is_withheld_when_spill_could_not_be_ruled_out():
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]), procinfo_fn=_blind_to_non_local(),
    )
    pressure = status["pressure"]
    assert pressure["state"] == "unknown"
    assert pressure["detail"] == (
        "Non-local memory is unavailable; driver spill cannot be ruled out.")
    assert pressure["coverage"]["non_local_memory"] is False
    # The figures stay null rather than reading as a measured zero.
    assert pressure["unexplained_spill_mb"] is None
    assert pressure["spilling"] is None


def test_missing_residency_is_reported_before_missing_spill_coverage():
    """Both requirements unmet: the reason names the nearer one rather than
    whichever branch happens to run first."""
    class _Blind:
        def observe_loaded(self):
            return Observation(None, "ollama:/api/ps", error="transport down")

    status = core.combined_status(
        lambda: GPUS_OK, _Blind(), procinfo_fn=_blind_to_non_local(),
    )
    assert status["pressure"]["state"] == "unknown"
    assert status["pressure"]["detail"] == "Ollama residency is unavailable."


def test_missing_spill_coverage_does_not_suppress_tight():
    """`tight` rests on a capacity reading it did observe. Downgrading it would
    discard a fact, not withhold a guess."""
    status = core.combined_status(
        lambda: GPUS_TIGHT, FakeOllama([]), procinfo_fn=_blind_to_non_local(),
    )
    assert status["pressure"]["state"] == "tight"


def test_missing_spill_coverage_does_not_suppress_degraded():
    models = [{"name": "m1", "size": gb_bytes(20), "size_vram": gb_bytes(16),
               "expires_at": None}]
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama(models),
        snapshot_fn=lambda: _snap(pid_map={"m1": 555}),
        procinfo_fn=_blind_to_non_local([{"pid": 555, "size_mb": 16384}]),
    )
    assert status["pressure"]["state"] == "degraded"
    assert status["pressure"]["offloaded_models"] == ["m1"]


def test_ok_survives_when_every_requirement_is_covered():
    """The downgrade must not fire on a platform that CAN see non-local
    memory and reports none — that is a measurement, not an absence."""
    status = core.combined_status(
        lambda: GPUS_OK, FakeOllama([]),
        procinfo_fn=lambda: Observation(
            [{"pid": 999, "size_mb": 100, "non_local_mb": 0}], "procs",
            coverage={"non_local_memory": True}),
    )
    assert status["pressure"]["state"] == "ok"
    assert status["pressure"]["coverage"]["non_local_memory"] is True


# ---- combined_status full wiring -------------------------------------------------

def test_combined_status_full_wiring():
    models = [{"name": "m1", "size": gb_bytes(4), "size_vram": gb_bytes(4),
              "expires_at": None}]
    snap = _snap(all_claims=[{"model": "m1", "owner": "x"}],
                 pid_map={"m1": 555}, busy_map={555: True})
    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama(models),
        snapshot_fn=lambda: snap,
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
        snapshot_fn=lambda: snap,
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
    toward spill: it is the biggest holder and the thing that actually spills.

    The model here is DELIBERATELY OFFLOADED (4 GB of a 20 GB model on the CPU)
    and its runner holds 8 GB non-local — double what that placement explains.
    An earlier version of this test used size == size_vram, the one shape where
    'any runner non-local means thrashing' happens to be right, so it could not
    see the offloaded case regressing."""
    calls = []

    def source():
        calls.append(1)
        return [{"pid": 999, "size_mb": 20000, "non_local_mb": 8192},
                {"pid": 111, "size_mb": 100, "non_local_mb": 0}]

    status = core.combined_status(
        lambda: [{"index": 0, "total_mb": 24576, "used_mb": 24000, "free_mb": 576}],
        FakeOllama([{"name": "m", "size": gb_bytes(20), "size_vram": gb_bytes(16)}]),
        snapshot_fn=lambda: _snap(pid_map={"m": 999}),
        procinfo_fn=source,
    )
    p = status["pressure"]
    assert p["non_local_mb"] == 8192
    assert p["explained_offload_mb"] == 4096      # the deliberate placement
    assert p["unexplained_spill_mb"] == 4096      # the excess: real paging
    assert p["state"] == "thrashing"
    # the runner is still excluded from the "other processes" view
    assert [p["pid"] for p in status["other_processes"]] == [111]
    # and the expensive source was sampled exactly once
    assert len(calls) == 1


def test_combined_status_forwards_spill_threshold():
    """A caller-supplied spill threshold must reach pressure(); the same
    non-local MB is a verdict either way depending on the knob.

    Constrained GPU: the threshold is only the floor half of the gate, so the
    free-VRAM comparison has to pass before the knob decides anything."""
    procs = [{"pid": 7, "size_mb": 900, "non_local_mb": 300}]
    loud = core.combined_status(
        lambda: GPUS_TIGHT, FakeOllama([]), procinfo_fn=lambda: list(procs),
        spill_threshold_mb=4096,
    )
    assert loud["pressure"]["spilling"] is False
    quiet = core.combined_status(
        lambda: GPUS_TIGHT, FakeOllama([]), procinfo_fn=lambda: list(procs),
        spill_threshold_mb=128,
    )
    assert quiet["pressure"]["spilling"] is True
    assert quiet["pressure"]["state"] == "thrashing"


def test_combined_status_skips_snapshot_when_nothing_loaded():
    """An idle-status call must not pay the snapshot's subprocess/IO cost."""
    def exploding_snapshot():
        raise AssertionError("snapshot must not be captured for zero models")

    status = core.combined_status(
        gpu_fn_const(8000), FakeOllama([]), snapshot_fn=exploding_snapshot,
    )
    assert status["loaded"] == []


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
