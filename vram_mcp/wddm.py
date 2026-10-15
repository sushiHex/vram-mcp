"""The selected GPU's non-local memory and paging on Windows, from WDDM's own
accounting.

NVML reports no per-process memory under WDDM. The GPU performance counters
do, per process AND per adapter: each ``\\GPU Process Memory(*)\\Non Local
Usage`` instance is named ``pid_<pid>_luid_0x<high>_0x<low>_phys_<n>``. The
NVML device is matched to its WDDM adapter by PCI address, which
``D3DKMTQueryAdapterInfo`` reports for every adapter, so only the selected
GPU's instances are counted. The measurement is the adapter's own ``\\GPU
Adapter Memory(*)\\Shared Usage``; per-process readings only attribute it, so
memory that no visible process accounts for still counts, and an adapter whose
counter cannot be read is unavailable rather than idle.

Non-local memory is a quantity, not a symptom: a desktop holds hundreds of MB
of it, and memory evicted to the backing store is not in it at all. Paging is
read separately, as the engine time WDDM's memory manager spends on the
adapter (:class:`PagingMonitor`).

Costs measured on an RTX 3090 (#22) set the shape. Listing adapters takes
~0.03 ms, so the LUID is resolved on every read: it changed across a driver
update. Opening the PDH query takes ~0.5 s, so one query is kept per server and
each read costs ~0.2 ms. The PCI address arrives with NVML's process reading,
so both always describe the same device.

Per-process *Dedicated* Usage is deliberately not read: it reports hundreds of
gigabytes for some processes on a 24 GB card, with valid counter status. NVML
owns dedicated memory.

Windows is reached through ctypes, loaded on first use, so this module imports
everywhere. Tests inject the adapter list and the counter samples.
"""

from __future__ import annotations

import ctypes
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Optional

from . import nvml as _nvml
from ._util import bytes_to_mb
from .observations import Observation

Address = tuple[int, int, int]   # PCI bus, device, function
Sample = tuple[str, int, int]    # PDH instance name, counter status, value

_PROCESSES = r"\GPU Process Memory(*)\Non Local Usage"
_ADAPTERS = r"\GPU Adapter Memory(*)\Shared Usage"
_ENGINES = r"\GPU Engine(*)\Running Time"   # cumulative, in 100 ns ticks
_BUS_ID = re.compile(r"([0-9a-f]{1,8}):([0-9a-f]{1,2}):([0-9a-f]{1,2})\.([0-7])", re.I)
# The hex in instance names is uppercase; match it either way.
_LUID = r"luid_0x(?P<high>[0-9a-f]{1,8})_0x(?P<low>[0-9a-f]{1,8})_phys_(?P<phys>\d+)"
_PROCESS_INSTANCE = re.compile(rf"pid_(?P<pid>\d+)_{_LUID}", re.I)
_ADAPTER_INSTANCE = re.compile(_LUID, re.I)
_ENGINE_INSTANCE = re.compile(rf"pid_(?P<pid>\d+)_{_LUID}_eng_\d+_engtype_.*", re.I)
_VALID_DATA = (0, 1)  # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA
_SYSTEM_PID = 4
_TICKS_PER_SECOND = 10_000_000

PAGING_PERCENT = 5.0      # of one engine, in one interval
PAGING_WINDOW = 10        # intervals kept
PAGING_MIN_SECONDS = 5    # intervals before a window is judged


def parse_bus_id(bus_id: str) -> tuple[int, Address]:
    """``"00000000:2D:00.0"`` -> ``(domain, (bus, device, function))``."""
    match = _BUS_ID.fullmatch(bus_id.strip())
    if match is None:
        raise ValueError(f"unrecognized PCI bus ID {bus_id!r}")
    domain, bus, device, function = (int(group, 16) for group in match.groups())
    return domain, (bus, device, function)


def adapter_luid(domain: int, address: Address,
                 adapters: list[tuple[int, Optional[Address]]]) -> int:
    """The LUID of the one WDDM adapter at ``address``; LookupError says why not."""
    if domain != 0:
        raise LookupError(f"the GPU is in PCI domain {domain:#x}, and WDDM adapter "
                          "addresses carry no domain to match it in")
    matches = [luid for luid, located in adapters if located == address]
    if len(matches) != 1:
        bus, device, function = address
        where = f"PCI {bus:02X}:{device:02X}.{function}"
        raise LookupError(f"no WDDM adapter is at {where}" if not matches
                          else f"{len(matches)} WDDM adapters report {where}")
    return matches[0]


def _instances(pattern: re.Pattern, samples: list[Sample], luid: int):
    """``(match, status, value)`` for each instance on adapter ``luid``."""
    for name, status, value in samples:
        match = pattern.fullmatch(name)
        if match is not None and (int(match["high"], 16) << 32
                                  | int(match["low"], 16)) == luid:
            yield match, status, value


@dataclass(frozen=True)
class NonLocal:
    """The adapter's non-local memory, and the share each process accounts for.

    ``total_mb`` is the measurement; ``by_pid`` only attributes it (None where a
    process's reading was invalid). Memory no process accounts for, such as
    the kernel's or that of a process this reader cannot see, is in the total.
    """
    total_mb: int
    by_pid: dict[int, Optional[int]]


def non_local(counters: dict[str, list[Sample]], luid: int) -> NonLocal:
    """Adapter ``luid``'s non-local memory; LookupError when it cannot be read."""
    adapter = list(_instances(_ADAPTER_INSTANCE, counters[_ADAPTERS], luid))
    if any(int(match["phys"]) != 0 for match, _, _ in adapter):
        raise LookupError("the GPU belongs to a linked adapter, whose physical "
                          "GPUs these counters cannot tell apart")
    total = [value for _, status, value in adapter if status in _VALID_DATA]
    if not total:
        raise LookupError("WDDM reports no readable memory counters for the "
                          "selected adapter")
    return NonLocal(
        bytes_to_mb(total[0]),
        {int(match["pid"]): bytes_to_mb(value) if status in _VALID_DATA else None
         for match, status, value in _instances(
             _PROCESS_INSTANCE, counters[_PROCESSES], luid)})


@dataclass(frozen=True)
class Paging:
    """How busy the System process kept the adapter's engines, per interval.

    WDDM's video memory manager pages allocations in and out as the System
    process (PID 4). Measured on an RTX 3090 (#34), overcommitted VRAM kept it
    at 5-54% of an engine in nearly every second, while loading a model or
    filling VRAM without overcommitting it made a burst of two to four
    seconds, and generation stayed under 4%. Paging is therefore *sustained*
    when at least half the intervals reach :data:`PAGING_PERCENT`. This is a
    measured correlation, not a WDDM contract: the paging engine is chosen by
    the driver, and page-table updates and fills run there too.
    """
    percents: tuple[float, ...]

    @property
    def active(self) -> int:
        return sum(percent >= PAGING_PERCENT for percent in self.percents)

    @property
    def sustained(self) -> bool:
        return bool(self.percents) and self.active * 2 >= len(self.percents)

    def summary(self) -> dict:
        return {"sustained": self.sustained, "active_seconds": self.active,
                "seconds": len(self.percents),
                "peak_percent": round(max(self.percents, default=0.0), 1)}


class PagingMonitor:
    """Samples the System process's engine time on the selected adapter, once
    a second, in a daemon thread. One per server.

    A status call cannot measure activity by itself: paging comes in bursts,
    and a window as long as the gap between calls averages it away. The
    thread keeps the last :data:`PAGING_WINDOW` intervals; the first read
    waits until :data:`PAGING_MIN_SECONDS` exist, at most once per server.
    """

    source = "wddm:pdh"

    def __init__(self, index: int, *,
                 bus_id_fn: Optional[Callable[[], Optional[str]]] = None,
                 adapters_fn: Optional[Callable[[], list]] = None,
                 counters: Optional["CounterReader"] = None,
                 clock: Callable[[], float] = time.monotonic,
                 interval: float = 1.0) -> None:
        self.index = index
        self._bus_id_fn = bus_id_fn or (lambda: _nvml_bus_id(index))
        self._adapters_fn = adapters_fn or wddm_adapters
        self._counters = counters or CounterReader((_ENGINES,))
        self._clock = clock
        self._interval = interval
        self._changed = threading.Condition()
        self._percents: deque[float] = deque(maxlen=PAGING_WINDOW)
        self._previous: Optional[tuple[float, dict[str, int]]] = None
        self._bus_id: Optional[str] = None
        self._error: Optional[str] = None
        self._sampled_at: Optional[float] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        with self._changed:
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, daemon=True,
                                                name="vram-mcp-paging")
                self._thread.start()

    def _run(self) -> None:
        while True:
            self.sample()
            time.sleep(self._interval)

    def sample(self) -> None:
        """Take one reading; the interval since the previous one joins the window."""
        try:
            if self._bus_id is None:
                self._bus_id = self._bus_id_fn()
                if self._bus_id is None:
                    raise LookupError("NVML did not report the GPU's PCI address")
            luid = adapter_luid(*parse_bus_id(self._bus_id), self._adapters_fn())
            ticks = system_engine_ticks(self._counters.read(), luid)
            error = None
        except Exception as exc:    # anything else would end the thread unseen
            ticks, error = None, str(exc) or type(exc).__name__
        now = self._clock()
        with self._changed:
            self._sampled_at = now
            self._error = error
            if ticks is None:
                self._percents.clear()      # a gap must not join two windows
            elif self._previous is not None:
                then, before = self._previous
                # Engines that appear, vanish, or restart count from now on.
                busy = sum(value - before[key] for key, value in ticks.items()
                           if before.get(key, value + 1) <= value)
                self._percents.append(100 * busy / _TICKS_PER_SECOND / (now - then))
            self._previous = (now, ticks) if ticks is not None else None
            self._changed.notify_all()

    def observe(self) -> Observation[Paging]:
        """The recent window. A sampler just started is waited for, briefly."""
        scope = f"gpu:index={self.index}"
        deadline = time.monotonic() + (PAGING_MIN_SECONDS + 2) * self._interval
        with self._changed:
            while (self._thread is not None and self._error is None
                   and len(self._percents) < PAGING_MIN_SECONDS
                   and time.monotonic() < deadline):
                self._changed.wait(self._interval)
            if self._sampled_at is None:
                error = "the paging sampler is not running"
            elif self._error is not None:
                error = self._error
            elif self._clock() - self._sampled_at > 5 * self._interval:
                error = "the paging sampler stopped"
            elif len(self._percents) < PAGING_MIN_SECONDS:
                error = (f"paging history is still being collected "
                         f"({len(self._percents)} of {PAGING_MIN_SECONDS} s)")
            else:
                return Observation(Paging(tuple(self._percents)), self.source, scope=scope)
        return Observation(None, self.source, error=error, scope=scope)


def system_engine_ticks(counters: dict[str, list[Sample]], luid: int) -> dict[str, int]:
    """The System process's cumulative engine time on ``luid``, per engine,
    in 100 ns ticks; LookupError when none is readable."""
    ticks = {match.group(0): value
             for match, status, value in _instances(_ENGINE_INSTANCE, counters[_ENGINES], luid)
             if int(match["pid"]) == _SYSTEM_PID and status in _VALID_DATA}
    if not ticks:
        raise LookupError("WDDM reports no readable engine counters for the "
                          "selected adapter")
    return ticks


def _nvml_bus_id(index: int) -> Optional[str]:
    return _nvml.observe_processes(index).device


class NonLocalMemory:
    """The selected GPU's non-local memory. One per server."""

    source = "wddm:pdh"

    def __init__(self, index: int, *,
                 adapters_fn: Optional[Callable[[], list]] = None,
                 counters: Optional["CounterReader"] = None) -> None:
        self.index = index
        self._adapters_fn = adapters_fn or wddm_adapters
        self._counters = counters or CounterReader((_PROCESSES, _ADAPTERS))

    def observe(self, bus_id: Optional[str]) -> Observation[NonLocal]:
        """Read the adapter at ``bus_id``: NVML's PCI address for the GPU, from
        the same reading as its processes, so both describe one device."""
        scope = f"gpu:index={self.index}"
        try:
            if bus_id is None:
                raise LookupError("NVML did not report the GPU's PCI address")
            luid = adapter_luid(*parse_bus_id(bus_id), self._adapters_fn())
            usage = non_local(self._counters.read(), luid)
        except (LookupError, OSError, ValueError) as exc:
            return Observation(None, self.source, error=str(exc), scope=scope)
        return Observation(usage, self.source, scope=scope)


# ── Windows bindings ─────────────────────────────────────────────────────────
# Fixed-width types rather than ctypes.wintypes, so the module imports anywhere.

class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", ctypes.c_uint32), ("HighPart", ctypes.c_int32)]


class _AdapterInfo(ctypes.Structure):       # D3DKMT_ADAPTERINFO
    _fields_ = [("hAdapter", ctypes.c_uint32), ("AdapterLuid", _Luid),
                ("NumOfSources", ctypes.c_uint32),
                ("bPrecisePresentRegionsPreferred", ctypes.c_int32)]


class _EnumAdapters2(ctypes.Structure):     # D3DKMT_ENUMADAPTERS2
    _fields_ = [("NumAdapters", ctypes.c_uint32),
                ("pAdapters", ctypes.POINTER(_AdapterInfo))]


class _QueryAdapterInfo(ctypes.Structure):  # D3DKMT_QUERYADAPTERINFO
    _fields_ = [("hAdapter", ctypes.c_uint32), ("Type", ctypes.c_int32),
                ("pPrivateDriverData", ctypes.c_void_p),
                ("PrivateDriverDataSize", ctypes.c_uint32)]


class _AdapterAddress(ctypes.Structure):    # D3DKMT_ADAPTERADDRESS
    _fields_ = [("BusNumber", ctypes.c_uint32), ("DeviceNumber", ctypes.c_uint32),
                ("FunctionNumber", ctypes.c_uint32)]


class _CloseAdapter(ctypes.Structure):      # D3DKMT_CLOSEADAPTER
    _fields_ = [("hAdapter", ctypes.c_uint32)]


class _CounterValue(ctypes.Structure):      # PDH_FMT_COUNTERVALUE, read as LARGE
    _fields_ = [("CStatus", ctypes.c_uint32), ("largeValue", ctypes.c_int64)]


class _FormattedItem(ctypes.Structure):     # PDH_FMT_COUNTERVALUE_ITEM_W
    # Nested, as declared: flattened, 32-bit Python would misplace the value.
    _fields_ = [("szName", ctypes.c_wchar_p), ("FmtValue", _CounterValue)]


_KMTQAITYPE_ADAPTERADDRESS = 6
_NO_PCI_ADDRESS = 0xFFFFFFFF   # the Microsoft Basic Render Driver, for one
_PDH_FMT_LARGE = 0x00000400
_PDH_MORE_DATA = 0x800007D2
_PDH_NO_DATA = 0x800007D5


def _library(name: str):
    try:
        return ctypes.WinDLL(name)
    except (AttributeError, OSError) as exc:   # AttributeError: not Windows
        raise OSError(f"{name} is unavailable: {exc}") from exc


def _check(status: int, call: str) -> None:
    if status:
        raise OSError(f"{call} failed with status 0x{status & 0xFFFFFFFF:08X}")


def wddm_adapters() -> list[tuple[int, Optional[Address]]]:
    """Every WDDM adapter's LUID and PCI address; None where it has no address."""
    gdi32 = _library("gdi32")
    count = _EnumAdapters2(0, None)
    _check(gdi32.D3DKMTEnumAdapters2(ctypes.byref(count)), "D3DKMTEnumAdapters2")
    infos = (_AdapterInfo * count.NumAdapters)()
    request = _EnumAdapters2(count.NumAdapters, infos)
    _check(gdi32.D3DKMTEnumAdapters2(ctypes.byref(request)), "D3DKMTEnumAdapters2")
    adapters = []
    try:
        for info in infos[:request.NumAdapters]:
            address = _AdapterAddress()
            query = _QueryAdapterInfo(
                info.hAdapter, _KMTQAITYPE_ADAPTERADDRESS,
                ctypes.cast(ctypes.byref(address), ctypes.c_void_p),
                ctypes.sizeof(address))
            located = (gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(query)) == 0
                       and address.BusNumber != _NO_PCI_ADDRESS)
            luid = (info.AdapterLuid.HighPart & 0xFFFFFFFF) << 32 | info.AdapterLuid.LowPart
            adapters.append((luid, (address.BusNumber, address.DeviceNumber,
                                    address.FunctionNumber) if located else None))
    finally:
        # The enumeration opened every handle; each must be closed.
        for info in infos[:request.NumAdapters]:
            gdi32.D3DKMTCloseAdapter(ctypes.byref(_CloseAdapter(info.hAdapter)))
    return adapters


class CounterReader:
    """One PDH query on wildcard counter paths, opened on first read and kept.

    Instances that appear after the query opens are still reported, and the
    query may be read from any thread (both verified, #22). Reads are
    serialized, because a collection and the array reads after it must pair.
    PDH reports missing privileges as no data, so an empty result is only as
    good as the caller's proof that the counters were visible.
    """

    def __init__(self, paths: tuple[str, ...]) -> None:
        self._paths = paths
        self._lock = threading.Lock()
        self._pdh = None
        self._query = ctypes.c_void_p()
        self._counters: dict[str, ctypes.c_void_p] = {}

    def read(self) -> dict[str, list[Sample]]:
        with self._lock:
            if self._pdh is None:
                self._open()
            try:
                return self._collect()
            except OSError:
                self._close()   # the next read starts afresh
                raise

    def _open(self) -> None:
        pdh = _library("pdh")
        for function in ("PdhOpenQueryW", "PdhAddEnglishCounterW",
                         "PdhCollectQueryData", "PdhGetFormattedCounterArrayW",
                         "PdhCloseQuery"):
            getattr(pdh, function).restype = ctypes.c_uint32
        _check(pdh.PdhOpenQueryW(None, None, ctypes.byref(self._query)), "PdhOpenQueryW")
        self._pdh = pdh
        for path in self._paths:
            counter = ctypes.c_void_p()
            status = pdh.PdhAddEnglishCounterW(self._query, path, None, ctypes.byref(counter))
            if status:
                self._close()
                _check(status, f"PdhAddEnglishCounterW({path})")
            self._counters[path] = counter

    def _close(self) -> None:
        self._pdh.PdhCloseQuery(self._query)
        self._pdh = None
        self._counters = {}

    def _collect(self) -> dict[str, list[Sample]]:
        status = self._pdh.PdhCollectQueryData(self._query)
        if status == _PDH_NO_DATA:
            return {path: [] for path in self._paths}
        _check(status, "PdhCollectQueryData")
        return {path: self._array(counter) for path, counter in self._counters.items()}

    def _array(self, counter: ctypes.c_void_p) -> list[Sample]:
        pdh = self._pdh
        size, count = ctypes.c_uint32(0), ctypes.c_uint32(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
        if status in (0, _PDH_NO_DATA):
            return []
        if status != _PDH_MORE_DATA:
            _check(status, "PdhGetFormattedCounterArrayW")
        buffer = (ctypes.c_byte * size.value)()
        _check(pdh.PdhGetFormattedCounterArrayW(
            counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), buffer),
            "PdhGetFormattedCounterArrayW")
        items = ctypes.cast(buffer, ctypes.POINTER(_FormattedItem))
        return [(items[i].szName, items[i].FmtValue.CStatus, items[i].FmtValue.largeValue)
                for i in range(count.value)]
