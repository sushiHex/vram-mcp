"""Sized + named process table of GPU VRAM holders.

NVML gives per-process VRAM directly on Linux/TCC but returns ``None`` on
Windows/WDDM. There, NVML's selected-GPU PID set is enriched with process
identity read through ctypes (:mod:`.winproc`), and with each process's
non-local memory on the selected adapter (:mod:`.wddm`). Pure module: every
external reader is injected; each degrades to ``[]``/``{}`` on any failure and
never raises.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

from ._util import run_capture
from .observations import Observation
from . import nvml as _nvml
from . import winproc


def windows_name_reader(pids) -> dict:
    """``{pid: {"name","cmdline"}}`` via :mod:`.winproc` for the given pids.

    Only PIDs Windows lists get an entry; one it denies keeps its name with
    ``cmdline`` None. Non-integers are DROPPED, not coerced: this module never
    raises, so a caller's junk gets identity-less rows. ``bool`` is excluded
    because ``isinstance(True, int)`` is True. ``{}`` where the APIs fail.
    """
    wanted = {p for p in pids
              if isinstance(p, int) and not isinstance(p, bool) and p >= 0}
    if not wanted:
        return {}
    try:
        names = winproc.process_names()
        return {pid: {"name": names[pid], "cmdline": winproc.command_line(pid)}
                for pid in sorted(wanted) if pid in names}
    except OSError:
        return {}


_PS_NAME = re.compile(r"^\s*(\d+)\s+(\S+)\s+(.*)$")


def posix_name_reader(pids, timeout: int = 5) -> dict:
    """``{pid: {"name","cmdline"}}`` via ``ps`` for the given pids (POSIX)."""
    wanted = set(pids)
    if not wanted:
        return {}
    stdout = run_capture(["ps", "-eo", "pid,comm,args"], timeout)
    if stdout is None:
        return {}
    names = {}
    for line in stdout.splitlines()[1:]:
        m = _PS_NAME.match(line)
        if not m:
            continue
        pid = int(m.group(1))
        if pid in wanted:
            names[pid] = {"name": m.group(2), "cmdline": m.group(3).strip()}
    return names


def _base_table(rows: list[dict]) -> dict[int, dict]:
    """Normalize selected-device NVML rows into the public process shape."""
    return {
        p["pid"]: {
            "pid": p["pid"], "size_mb": p.get("size_mb"),
            "shared_mb": None, "non_local_mb": None,
            "name": None, "cmdline": None, "kind": p.get("kind", "compute"),
        }
        for p in rows
    }


def _enrich_processes(
    rows: list[dict], *, name_reader, non_local: Optional[dict] = None,
) -> list[dict]:
    """Attach non-local memory and process identity to selected-GPU rows.

    ``non_local`` maps PID to non-local MB on the selected adapter. It may name
    processes NVML did not report: WDDM accounts for every process using the
    adapter, and each joins the table as ``kind="wddm"``. Their non-local
    memory counts toward driver spill, and their membership tells which
    runners are on this GPU.
    """
    table = _base_table(rows)
    if non_local is not None:
        table.update(_base_table([{"pid": pid, "kind": "wddm"}
                                  for pid in non_local if pid not in table]))
        for pid, entry in table.items():
            entry["non_local_mb"] = non_local.get(pid)
    for pid, meta in name_reader(list(table.keys())).items():
        if pid in table:
            table[pid]["name"] = meta.get("name")
            table[pid]["cmdline"] = meta.get("cmdline")
    return list(table.values())


def observe_processes(
    index: int = 0,
    *,
    nvml=None,
    platform: Optional[str] = None,
    nvml_observer=None,
    name_reader=None,
    non_local=None,
) -> Observation[list[dict]]:
    """Observe process holders for one selected GPU with platform enrichment.

    ``name_reader`` maps PIDs to ``{pid: {"name","cmdline"}}``; it defaults to
    :func:`windows_name_reader` on Windows and :func:`posix_name_reader`
    elsewhere. ``non_local`` is the selected GPU's non-local memory reader
    (:class:`.wddm.NonLocalMemory` on Windows). Without one, non-local memory
    is not covered, and ``gaps`` says why.
    """
    platform = sys.platform if platform is None else platform
    nvml_observer = nvml_observer or _nvml.observe_processes
    if name_reader is None:
        name_reader = (windows_name_reader if platform == "win32"
                       else posix_name_reader)
    nvml_observation = nvml_observer(index, nvml=nvml)
    source = "nvml+windows-process-identity" if platform == "win32" else "nvml+ps"
    scope = f"gpu:index={index}"
    device = nvml_observation.device
    coverage = {**(nvml_observation.coverage or {}), "non_local_memory": False}
    if not nvml_observation.known:
        return Observation(
            None, source, observed_at=nvml_observation.observed_at,
            error=nvml_observation.error or "NVML process telemetry unavailable",
            scope=scope, device=device, coverage=coverage,
        )
    usage = non_local.observe(device) if non_local is not None else None
    measured = usage.data if usage is not None and usage.known else None
    rows = _enrich_processes(
        nvml_observation.data or [], name_reader=name_reader,
        non_local=measured.by_pid if measured is not None else None,
    )
    if measured is not None:
        gap = None
    elif usage is None:
        gap = _NO_NON_LOCAL_READER
    else:
        gap = usage.error or "non-local memory reading failed"
    coverage["non_local_memory"] = gap is None
    return Observation(
        rows, source, observed_at=nvml_observation.observed_at, scope=scope,
        device=device, coverage=coverage,
        gaps={"non_local_memory": gap} if gap else None,
        totals={"non_local_mb": measured.total_mb} if measured is not None else None,
    )


# Native Linux keeps cudaMalloc memory on the GPU, but no Linux interface
# reports a process's GPU memory in system RAM, and managed memory can
# oversubscribe; see #27.
_NO_NON_LOCAL_READER = "no collector reads non-local memory on this platform"
