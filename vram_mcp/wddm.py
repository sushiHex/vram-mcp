"""The selected GPU's non-local memory on Windows, from WDDM's own accounting.

NVML reports no per-process memory under WDDM. The GPU performance counters
do, per process AND per adapter: each ``\\GPU Process Memory(*)\\Non Local
Usage`` instance is named ``pid_<pid>_luid_0x<high>_0x<low>_phys_<n>``. The
NVML device is matched to its WDDM adapter by PCI address, which
``D3DKMTQueryAdapterInfo`` reports for every adapter, so only the selected
GPU's instances are counted. The adapter's own ``\\GPU Adapter Memory`` instance
must be readable too: it proves the counters are visible to this process and
the LUID is current, so an empty process list means no processes, never no
access.

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
from typing import Callable, Optional

from ._util import bytes_to_mb
from .observations import Observation

Address = tuple[int, int, int]   # PCI bus, device, function
Sample = tuple[str, int, int]    # PDH instance name, counter status, bytes

_PROCESSES = r"\GPU Process Memory(*)\Non Local Usage"
_ADAPTERS = r"\GPU Adapter Memory(*)\Shared Usage"
_BUS_ID = re.compile(r"([0-9a-f]{1,8}):([0-9a-f]{1,2}):([0-9a-f]{1,2})\.([0-7])", re.I)
# The hex in instance names is uppercase; match it either way.
_LUID = r"luid_0x(?P<high>[0-9a-f]{1,8})_0x(?P<low>[0-9a-f]{1,8})_phys_(?P<phys>\d+)"
_PROCESS_INSTANCE = re.compile(rf"pid_(?P<pid>\d+)_{_LUID}", re.I)
_ADAPTER_INSTANCE = re.compile(_LUID, re.I)
_VALID_DATA = (0, 1)  # PDH_CSTATUS_VALID_DATA, PDH_CSTATUS_NEW_DATA


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


def non_local_by_pid(counters: dict[str, list[Sample]],
                     luid: int) -> dict[int, Optional[int]]:
    """Non-local MB per process on adapter ``luid``; None where unreadable."""
    adapter = list(_instances(_ADAPTER_INSTANCE, counters[_ADAPTERS], luid))
    if any(int(match["phys"]) != 0 for match, _, _ in adapter):
        raise LookupError("the GPU belongs to a linked adapter, whose physical "
                          "GPUs these counters cannot tell apart")
    if not any(status in _VALID_DATA for _, status, _ in adapter):
        raise LookupError("WDDM reports no readable memory counters for the "
                          "selected adapter")
    return {int(match["pid"]): bytes_to_mb(value) if status in _VALID_DATA else None
            for match, status, value in _instances(
                _PROCESS_INSTANCE, counters[_PROCESSES], luid)}


class NonLocalMemory:
    """The selected GPU's non-local memory per process. One per server."""

    source = "wddm:pdh"

    def __init__(self, index: int, *,
                 adapters_fn: Optional[Callable[[], list]] = None,
                 counters: Optional["CounterReader"] = None) -> None:
        self.index = index
        self._adapters_fn = adapters_fn or wddm_adapters
        self._counters = counters or CounterReader((_PROCESSES, _ADAPTERS))

    def observe(self, bus_id: Optional[str]) -> Observation[dict[int, Optional[int]]]:
        """Read the adapter at ``bus_id``: NVML's PCI address for the GPU, from
        the same reading as its processes, so both describe one device."""
        scope = f"gpu:index={self.index}"
        try:
            if bus_id is None:
                raise LookupError("NVML did not report the GPU's PCI address")
            luid = adapter_luid(*parse_bus_id(bus_id), self._adapters_fn())
            usage = non_local_by_pid(self._counters.read(), luid)
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
