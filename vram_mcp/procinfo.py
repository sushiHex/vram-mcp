"""Sized + named process table of GPU VRAM holders.

NVML gives per-process VRAM directly on Linux/TCC but returns ``None`` on
Windows/WDDM. There, NVML's selected-GPU PID set is enriched with one bounded
``Get-CimInstance Win32_Process`` identity query, and with each process's
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

# One CIM filter clause is ~20 characters, and a Windows command line is capped
# near 32 KiB. A selected GPU has tens of processes, so this ceiling is far
# above anything real; it exists so that a caller passing something absurd
# truncates into the module's ordinary "identity unavailable" state instead of
# failing the whole subprocess with an opaque argument error.
_MAX_FILTER_PIDS = 512


def _filter_pids(pids) -> list[int]:
    """The PIDs one identity query may carry: real integers, bounded, sorted.

    Non-integers are DROPPED rather than coerced. This module promises never to
    raise, and ``int("abc")`` would break that promise for a value NVML cannot
    produce — a caller who hands us junk should get identity-less rows, which is
    already a supported outcome, not a traceback. ``bool`` is excluded because
    ``isinstance(True, int)`` is True and ``ProcessId=1`` is not what was meant.
    """
    real = {p for p in pids
            if isinstance(p, int) and not isinstance(p, bool) and p >= 0}
    return sorted(real)[:_MAX_FILTER_PIDS]


# The query is deliberately identity-only: memory comes from NVML and, for
# non-local memory, from the selected adapter's own counters (:mod:`.wddm`).
def _win_gpu_identity_ps(pids: list[int]) -> str:
    pid_filter = " OR ".join(f"ProcessId={pid}" for pid in pids)
    return (
        f"Get-CimInstance Win32_Process -Filter '{pid_filter}' "
        "-EA SilentlyContinue | "
        "ForEach-Object { "
        "'{0}|{1}|{2}' -f $_.ProcessId,$_.Name,$_.CommandLine }"
    )


def _run_powershell(command: str, timeout: int) -> Optional[str]:
    return run_capture(["powershell", "-NoProfile", "-Command", command], timeout)


def win_gpu_procs(pids=(), timeout: int = 10) -> list[dict]:
    """Return identity for the selected GPU's PIDs on Windows.

    ``pids`` is the selected GPU's PID set. A nonempty set performs one bounded CIM query;
    an empty set performs no subprocess call. Missing or failed identity data
    returns ``[]`` so callers retain the known NVML inventory with null names.
    """
    if sys.platform != "win32":
        return []
    wanted = _filter_pids(pids)
    if not wanted:
        return []
    out_text = _run_powershell(_win_gpu_identity_ps(wanted), timeout)
    if not out_text:
        return []
    procs = []
    for line in out_text.splitlines():
        # cmdline is last and unsplit: a command line may itself contain '|'.
        parts = line.strip().split("|", 2)
        if len(parts) != 3 or not parts[0].isdigit():
            continue
        pid, name, cmdline = parts
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
    posix_reader=None, non_local: Optional[dict] = None,
) -> list[dict]:
    """Attach non-local memory and process identity to selected-GPU rows.

    ``non_local`` maps PID to non-local MB on the selected adapter. It may name
    a process NVML did not report: WDDM accounts for every process with memory
    on the adapter, and a process holding non-local memory there joins the
    table as ``kind="wddm"``, because its memory counts toward driver spill.
    """
    table = _base_table(rows)
    if non_local is not None:
        for pid, entry in table.items():
            entry["non_local_mb"] = non_local.get(pid)
        for pid, mb in non_local.items():
            if pid not in table and mb != 0:
                table[pid] = {**_base_table([{"pid": pid, "kind": "wddm"}])[pid],
                              "non_local_mb": mb}
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
    non_local=None,
) -> Observation[list[dict]]:
    """Observe process holders for one selected GPU with platform enrichment.

    ``non_local`` is the selected GPU's non-local memory reader
    (:class:`.wddm.NonLocalMemory` on Windows). Without one, non-local memory
    is not covered, and ``gaps`` says why.
    """
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
    usage = non_local.observe() if non_local is not None else None
    rows = _enrich_processes(
        nvml_observation.data or [], platform=platform,
        win_gpu_reader=win_gpu_reader, posix_reader=posix_reader,
        non_local=usage.data if usage is not None and usage.known else None,
    )
    if usage is None:
        gap = _NO_NON_LOCAL_READER
    elif not usage.known:
        gap = usage.error or "non-local memory reading failed"
    else:
        unread = sorted(row["pid"] for row in rows if row["non_local_mb"] is None)
        gap = (f"no non-local reading for PID(s) {', '.join(map(str, unread))}"
               if unread else None)
    coverage["non_local_memory"] = gap is None
    return Observation(
        rows, source, observed_at=nvml_observation.observed_at, scope=scope,
        coverage=coverage, gaps={"non_local_memory": gap} if gap else None,
    )


# Native Linux keeps cudaMalloc memory on the GPU, but no Linux interface
# reports a process's GPU memory in system RAM, and managed memory can
# oversubscribe; see #27.
_NO_NON_LOCAL_READER = "no collector reads non-local memory on this platform"


def process_table(*, nvml_processes, win_gpu_reader=None,
                  posix_name_reader=None, platform: Optional[str] = None) -> list[dict]:
    """``[{pid,size_mb,shared_mb,non_local_mb,name,cmdline,kind}]`` for GPU VRAM
    holders.

    Starts from selected-device NVML rows. Platform dispatch is explicit:
    Windows uses one bounded identity lookup for those PIDs, while POSIX uses
    ``ps``. Memory fields NVML did not report remain ``None``; the server's
    :func:`observe_processes` adds non-local memory.
    """
    selected_platform = sys.platform if platform is None else platform
    return _enrich_processes(
        nvml_processes(), platform=selected_platform,
        win_gpu_reader=win_gpu_reader, posix_reader=posix_name_reader,
    )
