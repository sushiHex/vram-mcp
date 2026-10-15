"""NVML-based VRAM/process telemetry via nvidia-ml-py.

Pure module: the real ``pynvml`` (nvidia-ml-py) is injected as the ``nvml``
parameter, defaulting to the real package, so tests never need a real GPU or
driver. Every function degrades to ``None``/``[]``/``{}`` on ANY failure —
not just ``NVMLError`` but also e.g. ``AttributeError`` from the legacy
``pynvml`` PyPI package shadowing nvidia-ml-py (missing ``nvmlMemory_v2`` /
``*RunningProcesses_v3``) — never raises to the caller.
"""

from __future__ import annotations

import time
from typing import Optional

from ._util import bytes_to_mb as _shared_bytes_to_mb
from .observations import Observation


def _default_nvml():
    """Resolve the default NVML implementation (the real ``pynvml`` package)."""
    import pynvml

    return pynvml


def _b2mb(value) -> Optional[int]:
    """Bytes -> whole MB; ``None`` where "unreported" is meaningful (WDDM)."""
    return _shared_bytes_to_mb(value, default=None)


def _with_device(nvml, device_index, fn, default):
    """Init NVML, get the device handle, run ``fn(nvml, handle)``, always shut down.

    Returns ``default`` on ANY exception from init, handle lookup, or ``fn``
    itself. Swallowing broadly is deliberate and IS the module contract: this
    is best-effort telemetry, and an injected implementation may raise
    anything (e.g. ``AttributeError`` from the legacy ``pynvml`` package) —
    nothing may escape to the caller.
    """
    try:
        nvml.nvmlInit()
    except Exception:
        return default
    try:
        handle = nvml.nvmlDeviceGetHandleByIndex(device_index)
        return fn(nvml, handle)
    except Exception:
        return default
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def nvml_memory(device_index: int = 0, *, nvml=None) -> Optional[dict]:
    """Accurate VRAM total/used/free/reserved + stable UUID for one GPU.

    Returns ``{"total_mb","used_mb","free_mb","reserved_mb","uuid"}``, or
    ``None`` if NVML is unavailable / the device index doesn't exist. Uses
    ``nvmlMemory_v2`` for the ``reserved`` (driver-reserved) field a plain
    ``nvidia-smi`` query can't give — v1's ``used`` conflates allocated and
    reserved memory.
    """
    if nvml is None:
        nvml = _default_nvml()

    def _query(nvml, handle):
        mem = nvml.nvmlDeviceGetMemoryInfo(handle, nvml.nvmlMemory_v2)
        return {
            "total_mb": _b2mb(mem.total),
            "used_mb": _b2mb(mem.used),
            "free_mb": _b2mb(mem.free),
            "reserved_mb": _b2mb(mem.reserved),
            "uuid": nvml.nvmlDeviceGetUUID(handle),
        }

    return _with_device(nvml, device_index, _query, None)


def nvml_pci_bus_id(device_index: int = 0, *, nvml=None) -> Optional[str]:
    """The device's PCI bus ID, such as ``"00000000:2D:00.0"``; ``None`` if unknown."""
    if nvml is None:
        try:
            nvml = _default_nvml()
        except Exception:
            return None

    def _query(nvml, handle):
        bus_id = nvml.nvmlDeviceGetPciInfo(handle).busId
        return bus_id.decode() if isinstance(bus_id, bytes) else bus_id

    return _with_device(nvml, device_index, _query, None)


def nvml_processes(device_index: int = 0, *, nvml=None) -> list[dict]:
    """Every process holding VRAM on this GPU: compute AND graphics contexts.

    Returns ``[{"pid","size_mb","kind"}, ...]``; ``kind`` is ``"compute"`` or
    ``"graphics"``. ``size_mb`` is ``None`` when the driver doesn't report
    per-process memory (observed on Windows/WDDM — the PID/kind are still
    useful even without a size). A PID present in both lists is reported
    once, tagged ``"compute"``, with the first non-``None`` size from either
    entry (a WDDM ``None`` in the compute list must not shadow a real size
    in the graphics list). The compute and graphics queries fail
    independently — if one is unsupported, the other's results still come
    back. ``[]`` if NVML is unavailable entirely.
    """
    observation = observe_processes(device_index, nvml=nvml)
    return observation.data or []


def observe_processes(
    index: int = 0, *, nvml=None,
) -> Observation[list[dict]]:
    """Observe processes attached to exactly one NVML device.

    Successful empty results remain distinguishable from NVML initialization,
    device lookup, or query failures. Compute and graphics process queries are
    independent; either supported query is enough for a known observation.
    """
    source = "nvml"
    scope = f"gpu:index={index}"
    no_coverage = {"compute_processes": False, "graphics_processes": False}
    if nvml is None:
        try:
            nvml = _default_nvml()
        except Exception as exc:
            return Observation(
                None, source, error=str(exc), scope=scope, coverage=no_coverage,
            )

    try:
        nvml.nvmlInit()
    except Exception as exc:
        return Observation(
            None, source, error=str(exc), scope=scope, coverage=no_coverage,
        )
    try:
        try:
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
        except Exception as exc:
            return Observation(
                None, source, error=str(exc), scope=scope, coverage=no_coverage,
            )

        out: list[dict] = []
        by_pid: dict[int, dict] = {}
        query_successes = 0
        errors: list[str] = []
        compute_ok = False
        graphics_ok = False
        try:
            compute = nvml.nvmlDeviceGetComputeRunningProcesses_v3(handle)
            query_successes += 1
            compute_ok = True
        except Exception as exc:
            compute = []
            errors.append(str(exc))
        for p in compute:
            entry = {"pid": p.pid, "size_mb": _b2mb(p.usedGpuMemory), "kind": "compute"}
            out.append(entry)
            by_pid[p.pid] = entry
        try:
            graphics = nvml.nvmlDeviceGetGraphicsRunningProcesses_v3(handle)
            query_successes += 1
            graphics_ok = True
        except Exception as exc:
            graphics = []
            errors.append(str(exc))
        for p in graphics:
            size = _b2mb(p.usedGpuMemory)
            existing = by_pid.get(p.pid)
            if existing is not None:
                if existing["size_mb"] is None:
                    existing["size_mb"] = size
                continue
            out.append({"pid": p.pid, "size_mb": size, "kind": "graphics"})
        if query_successes == 0:
            detail = "; ".join(error for error in errors if error)
            return Observation(
                None, source, error=detail or "NVML process queries failed",
                scope=scope, coverage=no_coverage,
            )
        return Observation(
            out, source, scope=scope,
            coverage={
                "compute_processes": compute_ok,
                "graphics_processes": graphics_ok,
            },
        )
    except Exception as exc:
        return Observation(
            None, source, error=str(exc) or "Invalid NVML process response",
            scope=scope, coverage=no_coverage,
        )
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass


def nvml_busy_map(
    pids,
    index: int = 0,
    *,
    nvml=None,
    now_fn=time.time,
    recent_seconds: float = 5.0,
) -> dict:
    """Busy signal for MANY pids in ONE NVML session + ONE utilization-buffer fetch.

    Returns ``{pid: True|False|None}`` for every pid in ``pids``: ``True`` if
    ANY sample in NVML's internal buffer for that pid has ``smUtil > 0``,
    ``False`` if at least one sample exists but all are zero, ``None`` if the
    pid has no sample at all or NVML is unavailable — never guesses ``False``
    for a PID we simply have no data on. Empty ``pids`` returns ``{}``
    without touching NVML.

    Requests and independently filters samples newer than ``recent_seconds``.
    NVML timestamps are epoch microseconds. A PID represented only by stale
    samples remains ``None`` rather than being mislabeled busy or idle.
    """
    pids = list(pids)
    if not pids:
        return {}
    if nvml is None:
        try:
            nvml = _default_nvml()
        except Exception:
            return {pid: None for pid in pids}

    cutoff_us = max(0, int((float(now_fn()) - recent_seconds) * 1_000_000))

    def _query(nvml, handle):
        verdicts: dict = {pid: None for pid in pids}
        wanted = set(pids)
        for s in nvml.nvmlDeviceGetProcessUtilization(handle, cutoff_us):
            if s.pid not in wanted:
                continue
            timestamp = getattr(s, "timeStamp", None)
            if timestamp is None or timestamp < cutoff_us:
                continue
            if s.smUtil > 0:
                verdicts[s.pid] = True
            elif verdicts[s.pid] is None:
                verdicts[s.pid] = False
        return verdicts

    return _with_device(nvml, index, _query, {pid: None for pid in pids})


def nvml_busy(
    pid: int,
    index: int = 0,
    *,
    nvml=None,
    now_fn=time.time,
    recent_seconds: float = 5.0,
) -> Optional[bool]:
    """Was ``pid`` doing GPU compute recently (windowed, not point-in-time)?

    Thin wrapper over :func:`nvml_busy_map` for a single pid — same semantics:
    ``True`` if any buffered sample has ``smUtil > 0``, ``False`` if samples
    exist but all zero, ``None`` if undetermined.

    Kept as public convenience API for external callers checking one pid;
    vram-mcp's own server path uses :func:`nvml_busy_map` (one NVML session
    for all pids).
    """
    return nvml_busy_map(
        [pid], index, nvml=nvml, now_fn=now_fn, recent_seconds=recent_seconds,
    ).get(pid)
