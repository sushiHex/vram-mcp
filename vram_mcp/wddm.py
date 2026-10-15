"""The selected GPU's non-local memory on Windows, from WDDM's own accounting.

NVML reports no per-process memory under WDDM. The GPU performance counters
do, per process AND per adapter: each ``\\GPU Process Memory(*)\\Non Local
Usage`` instance is named ``pid_<pid>_luid_0x<high>_0x<low>_phys_<n>``. The
NVML device is matched to its WDDM adapter by PCI address, which
``D3DKMTQueryAdapterInfo`` reports for every adapter, so only the selected
GPU's instances are counted.

Costs measured on an RTX 3090 (#22) set the shape. Listing adapters takes
~0.03 ms, so the LUID is resolved on every read: it changes across boots and
driver restarts. Opening the PDH query takes ~0.4 s, so one query is kept per
server and each read costs ~0.1 ms. Finding the GPU's PCI address takes an NVML
session (~17 ms), and the address cannot change, so it is looked up once.

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
from . import nvml as _nvml
from .observations import Observation

Address = tuple[int, int, int]   # PCI bus, device, function
Sample = tuple[str, int, int]    # PDH instance name, counter status, bytes

_NON_LOCAL = r"\GPU Process Memory(*)\Non Local Usage"
_BUS_ID = re.compile(r"([0-9a-f]{1,8}):([0-9a-f]{1,2}):([0-9a-f]{1,2})\.([0-7])", re.I)
# The hex in instance names is uppercase; match it either way.
_INSTANCE = re.compile(
    r"pid_(\d+)_luid_0x([0-9a-f]{1,8})_0x([0-9a-f]{1,8})_phys_(\d+)", re.I)
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


def non_local_by_pid(samples: list[Sample], luid: int) -> dict[int, Optional[int]]:
    """Non-local MB per process on adapter ``luid``; None where unreadable."""
    usage: dict[int, Optional[int]] = {}
    for name, status, value in samples:
        match = _INSTANCE.fullmatch(name)
        if match is None or (int(match[2], 16) << 32 | int(match[3], 16)) != luid:
            continue
        if int(match[4]) != 0:
            raise LookupError("the GPU belongs to a linked adapter, whose physical "
                              "GPUs these counters cannot tell apart")
        usage[int(match[1])] = bytes_to_mb(value) if status in _VALID_DATA else None
    return usage


class NonLocalMemory:
    """The selected GPU's non-local memory per process. One per server."""

    source = "wddm:pdh"

    def __init__(self, index: int, *,
                 bus_id_fn: Optional[Callable[[], Optional[str]]] = None,
                 adapters_fn: Optional[Callable[[], list]] = None,
                 counters: Optional["CounterReader"] = None) -> None:
        self.index = index
        self._bus_id_fn = bus_id_fn or (lambda: _nvml.nvml_pci_bus_id(index))
        self._adapters_fn = adapters_fn or wddm_adapters
        self._counters = counters or CounterReader(_NON_LOCAL)
        self._pci: Optional[tuple[int, Address]] = None

    def observe(self) -> Observation[dict[int, Optional[int]]]:
        scope = f"gpu:index={self.index}"
        try:
            if self._pci is None:
                bus_id = self._bus_id_fn()
                if bus_id is None:
                    raise LookupError("NVML did not report the GPU's PCI address")
                self._pci = parse_bus_id(bus_id)
            luid = adapter_luid(*self._pci, self._adapters_fn())
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


class _FormattedItem(ctypes.Structure):     # PDH_FMT_COUNTERVALUE_ITEM_W, LARGE
    _fields_ = [("szName", ctypes.c_wchar_p), ("CStatus", ctypes.c_uint32),
                ("largeValue", ctypes.c_int64)]


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
    """One PDH query on one wildcard counter path, opened on first read and kept.

    Instances that appear after the query opens are still reported, and the
    query may be read from any thread (both verified, #22). Reads are
    serialized, because a collection and the array read after it must pair.
    """

    def __init__(self, path: str) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._pdh = None
        self._query = ctypes.c_void_p()
        self._counter = ctypes.c_void_p()

    def read(self) -> list[Sample]:
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
        status = pdh.PdhAddEnglishCounterW(self._query, self._path, None,
                                           ctypes.byref(self._counter))
        if status:
            pdh.PdhCloseQuery(self._query)
            _check(status, f"PdhAddEnglishCounterW({self._path})")
        self._pdh = pdh

    def _close(self) -> None:
        self._pdh.PdhCloseQuery(self._query)
        self._pdh = None

    def _collect(self) -> list[Sample]:
        pdh = self._pdh
        status = pdh.PdhCollectQueryData(self._query)
        if status == _PDH_NO_DATA:      # no instance exists
            return []
        _check(status, "PdhCollectQueryData")
        size, count = ctypes.c_uint32(0), ctypes.c_uint32(0)
        status = pdh.PdhGetFormattedCounterArrayW(
            self._counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count), None)
        if status != _PDH_MORE_DATA:
            _check(status, "PdhGetFormattedCounterArrayW")
            return []
        buffer = (ctypes.c_byte * size.value)()
        _check(pdh.PdhGetFormattedCounterArrayW(
            self._counter, _PDH_FMT_LARGE, ctypes.byref(size), ctypes.byref(count),
            buffer), "PdhGetFormattedCounterArrayW")
        items = ctypes.cast(buffer, ctypes.POINTER(_FormattedItem))
        return [(items[i].szName, items[i].CStatus, items[i].largeValue)
                for i in range(count.value)]
