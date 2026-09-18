"""Sized + named process table of GPU VRAM holders.

NVML gives per-process VRAM directly on Linux/TCC but returns ``None`` on
Windows/WDDM. There, NVML's selected-GPU PID set is enriched with one bounded
``Get-CimInstance Win32_Process`` identity query. Windows GPU Process Memory
counters aggregate adapters and are deliberately not sampled or attributed to a
selected device. Pure module: every external reader is injected; each degrades
to ``[]``/``{}`` on any failure and never raises.
"""
from __future__ import annotations

import re
import sys
from typing import Optional

from ._util import run_capture
from .observations import Observation
from . import nvml as _nvml

# The query is deliberately identity-only: Windows GPU Process Memory counters
# aggregate adapters and cannot prove attribution to the selected GPU. NVML owns
# the selected-GPU PID set; this one bounded CIM query only enriches those PIDs.
def _win_gpu_identity_ps(pids) -> str:
    pid_filter = " OR ".join(
        f"ProcessId={int(pid)}" for pid in sorted(set(pids))
    )
    return (
        f"Get-CimInstance Win32_Process -Filter '{pid_filter}' "
        "-EA SilentlyContinue | "
        "ForEach-Object { "
        "'{0}||||{1}|{2}' -f $_.ProcessId,$_.Name,$_.CommandLine }"
    )


def _run_powershell(command: str, timeout: int) -> Optional[str]:
    return run_capture(["powershell", "-NoProfile", "-Command", command], timeout)


def win_gpu_procs(pids=(), timeout: int = 10) -> list[dict]:
    """Return identity for the selected NVML PIDs on Windows.

    Windows GPU counters aggregate adapters, so they are not sampled here and
    their values can never be attributed to the selected GPU. ``pids`` is the
    authoritative NVML PID set. A nonempty set performs one bounded CIM query;
    an empty set performs no subprocess call. Missing or failed identity data
    returns ``[]`` so callers retain the known NVML inventory with null names.
    """
    if sys.platform != "win32" or not pids:
        return []
    out_text = _run_powershell(_win_gpu_identity_ps(pids), timeout)
    if not out_text:
        return []
    procs = []
    for line in out_text.splitlines():
        # cmdline is last and unsplit: a command line may itself contain '|'.
        parts = line.strip().split("|", 5)
        if len(parts) != 6 or not parts[0].isdigit():
            continue
        pid, _dedicated, _shared, _non_local, name, cmdline = parts
        procs.append({
            "pid": int(pid),
            "size_mb": None,
            "shared_mb": None,
            "non_local_mb": None,
            "name": name.strip() or None,
            "cmdline": cmdline.strip() or None,
        })
    return procs


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
    rows: list[dict], *, platform: str, win_gpu_reader=None,
    posix_reader=None,
) -> list[dict]:
    """Attach process identity without weakening selected-GPU attribution.

    Windows GPU Process Memory counters aggregate a PID across adapters and
    are not queried. Identity is useful, but no Windows fallback may fill
    memory fields or add a PID that NVML did not report for the selected device.
    """
    table = _base_table(rows)
    if platform == "win32" and win_gpu_reader is not None:
        for windows_row in win_gpu_reader(list(table.keys())):
            entry = table.get(windows_row["pid"])
            if entry is None:
                continue
            entry["name"] = windows_row.get("name")
            entry["cmdline"] = windows_row.get("cmdline")
    elif platform != "win32" and posix_reader is not None:
        names = posix_reader(list(table.keys()))
        for pid, meta in names.items():
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
    win_gpu_reader=None,
    posix_reader=None,
) -> Observation[list[dict]]:
    """Observe process holders for one selected GPU with platform enrichment."""
    platform = sys.platform if platform is None else platform
    nvml_observer = nvml_observer or _nvml.observe_processes
    win_gpu_reader = win_gpu_procs if win_gpu_reader is None else win_gpu_reader
    posix_reader = posix_name_reader if posix_reader is None else posix_reader
    nvml_observation = nvml_observer(index, nvml=nvml)
    source = "nvml+windows-process-identity" if platform == "win32" else "nvml+ps"
    scope = f"gpu:index={index}"
    coverage = {**(nvml_observation.coverage or {}), "non_local_memory": False}
    if not nvml_observation.known:
        return Observation(
            None, source, observed_at=nvml_observation.observed_at,
            error=nvml_observation.error or "NVML process telemetry unavailable",
            scope=scope, coverage=coverage,
        )
    rows = _enrich_processes(
        nvml_observation.data or [], platform=platform,
        win_gpu_reader=win_gpu_reader, posix_reader=posix_reader,
    )
    return Observation(
        rows, source, observed_at=nvml_observation.observed_at, scope=scope,
        coverage=coverage,
    )


def process_table(*, nvml_processes, win_gpu_reader=None,
                  posix_name_reader=None, platform: Optional[str] = None) -> list[dict]:
    """``[{pid,size_mb,shared_mb,non_local_mb,name,cmdline,kind}]`` for GPU VRAM
    holders.

    Starts from selected-device NVML rows. Platform dispatch is explicit:
    Windows uses one bounded identity lookup for those PIDs, while POSIX uses
    ``ps``. Windows memory fields remain ``None`` because adapter-aggregated
    counters cannot prove selected-device attribution.
    """
    selected_platform = sys.platform if platform is None else platform
    return _enrich_processes(
        nvml_processes(), platform=selected_platform,
        win_gpu_reader=win_gpu_reader, posix_reader=posix_name_reader,
    )
