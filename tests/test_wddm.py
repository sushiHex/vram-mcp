"""The selected GPU's non-local memory on Windows, with every OS reader injected."""

from __future__ import annotations

import ctypes
import sys

import pytest

from vram_mcp import wddm

_MB = 1024 * 1024
_BUS_ID = "00000000:2D:00.0"
_SELECTED = 0x0000_0000_0001_43AF   # at PCI 2D:00.0
_OTHER = 0x0000_0001_0000_2B10      # a second NVIDIA GPU at 41:00.0
_BASIC = 0x0000_0000_0001_6E3E      # Microsoft Basic Render Driver: no address
_ADAPTERS = [(_SELECTED, (0x2D, 0, 0)), (_OTHER, (0x41, 0, 0)), (_BASIC, None)]


def _luid(luid: int, phys: int = 0) -> str:
    # Windows writes the LUID's hex in uppercase.
    return f"luid_0x{luid >> 32:08X}_0x{luid & 0xFFFFFFFF:08X}_phys_{phys}"


def _process(pid: int, luid: int, phys: int = 0) -> str:
    return f"pid_{pid}_{_luid(luid, phys)}"


def _visible(*luids: int) -> list:
    """Adapter Shared Usage instances: proof the counters were readable."""
    return [(_luid(luid), 0, 0) for luid in luids]


class _Counters:
    """Successive reads return successive batches; an exception is raised."""

    def __init__(self, *batches):
        self.batches = list(batches)
        self.reads = 0

    def read(self):
        self.reads += 1
        batch = self.batches[min(self.reads, len(self.batches)) - 1]
        if isinstance(batch, Exception):
            raise batch
        return batch


def _counters(processes, adapters=None):
    return {wddm._PROCESSES: processes,
            wddm._ADAPTERS: _visible(_SELECTED, _OTHER) if adapters is None else adapters}


def _reader(*batches, adapters=_ADAPTERS):
    calls = []

    def adapters_fn():
        calls.append("adapters")
        return adapters() if callable(adapters) else adapters

    reader = wddm.NonLocalMemory(0, adapters_fn=adapters_fn, counters=_Counters(*batches))
    return reader, calls


def test_only_the_selected_adapter_is_attributed():
    reader, _ = _reader(_counters([
        (_process(10, _SELECTED), 0, 300 * _MB),
        (_process(10, _OTHER), 0, 900 * _MB),     # the same process, another GPU
        (_process(11, _OTHER), 0, 4096 * _MB),
        (_process(12, _BASIC), 0, 50 * _MB),
        (_process(13, _SELECTED), 1, 2 * _MB),    # PDH_CSTATUS_NEW_DATA is valid
    ]))

    reading = reader.observe(_BUS_ID)

    assert reading.known
    assert reading.data.by_pid == {10: 300, 13: 2}
    assert reading.source == "wddm:pdh"
    assert reading.scope == "gpu:index=0"


def test_the_total_is_the_adapters_including_memory_no_process_accounts_for():
    reader, _ = _reader(_counters(
        [(_process(10, _SELECTED), 0, 300 * _MB)],
        [(_luid(_SELECTED), 0, 2348 * _MB), (_luid(_OTHER), 0, 9000 * _MB)]))

    reading = reader.observe(_BUS_ID).data

    assert reading.total_mb == 2348
    assert reading.by_pid == {10: 300}


def test_lowercase_instance_names_match_too():
    reader, _ = _reader(_counters([(_process(10, _SELECTED).lower(), 0, 64 * _MB)],
                                  [(_luid(_SELECTED).lower(), 0, 0)]))
    assert reader.observe(_BUS_ID).data.by_pid == {10: 64}


def test_an_unreadable_instance_is_unknown_not_zero():
    reader, _ = _reader(_counters([(_process(10, _SELECTED), 0xC0000BC6, 0)]))
    assert reader.observe(_BUS_ID).data.by_pid == {10: None}


def test_unrelated_instance_names_are_ignored():
    reader, _ = _reader(_counters([("_Total", 0, 7 * _MB), (_process(10, _SELECTED), 0, 0)]))
    assert reader.observe(_BUS_ID).data.by_pid == {10: 0}


def test_an_adapter_with_no_processes_reads_as_empty():
    reader, _ = _reader(_counters([(_process(11, _OTHER), 0, _MB)]))
    reading = reader.observe(_BUS_ID)
    assert reading.known
    assert reading.data.by_pid == {}


@pytest.mark.parametrize("adapter_samples", [
    [],                                    # PDH_NO_DATA, which missing privileges also report
    _visible(_OTHER),                      # only another GPU's adapter is visible
    [(_luid(_SELECTED), 0xC0000BC6, 0)],   # the selected adapter is unreadable
])
def test_empty_process_counters_need_proof_the_adapter_was_visible(adapter_samples):
    reader, _ = _reader(_counters([], adapter_samples))

    reading = reader.observe(_BUS_ID)

    assert not reading.known
    assert "no readable memory counters for the selected adapter" in reading.error


def test_the_luid_is_resolved_on_every_read():
    # A driver update gave the same GPU a new LUID.
    updated = 0x0000_0000_0001_53AF
    tables = iter([_ADAPTERS, [(updated, (0x2D, 0, 0))]])
    samples = _counters([(_process(10, _SELECTED), 0, 1 * _MB),
                         (_process(10, updated), 0, 2 * _MB)],
                        _visible(_SELECTED, updated))
    reader, calls = _reader(samples, adapters=lambda: next(tables))

    assert reader.observe(_BUS_ID).data.by_pid == {10: 1}
    assert reader.observe(_BUS_ID).data.by_pid == {10: 2}
    assert calls == ["adapters", "adapters"]


@pytest.mark.parametrize(("adapters", "bus_id", "reason"), [
    (_ADAPTERS, "00000000:07:00.0", "no WDDM adapter is at PCI 07:00.0"),
    ([(_SELECTED, (0x2D, 0, 0)), (_OTHER, (0x2D, 0, 0))], _BUS_ID,
     "2 WDDM adapters report PCI 2D:00.0"),
    (_ADAPTERS, "00000001:2D:00.0", "PCI domain 0x1"),
    (_ADAPTERS, "not a bus id", "unrecognized PCI bus ID"),
    (_ADAPTERS, None, "NVML did not report the GPU's PCI address"),
])
def test_an_unresolvable_adapter_is_unavailable_with_its_reason(adapters, bus_id, reason):
    reader, _ = _reader(_counters([(_process(10, _SELECTED), 0, _MB)]), adapters=adapters)

    reading = reader.observe(bus_id)

    assert not reading.known
    assert reason in reading.error


def test_a_linked_adapter_is_refused_even_with_no_processes_on_it():
    reader, _ = _reader(_counters([], _visible(_SELECTED) + [(_luid(_SELECTED, 1), 0, 0)]))

    reading = reader.observe(_BUS_ID)

    assert not reading.known
    assert "linked adapter" in reading.error


def test_counter_and_adapter_failures_are_unavailable_never_raised():
    reader, _ = _reader(OSError("PdhCollectQueryData failed"))
    assert reader.observe(_BUS_ID).error == "PdhCollectQueryData failed"

    def broken():
        raise OSError("gdi32 is unavailable")

    reader = wddm.NonLocalMemory(0, adapters_fn=broken, counters=_Counters({}))
    assert reader.observe(_BUS_ID).error == "gdi32 is unavailable"


def test_bus_ids_parse_in_nvml_and_legacy_forms():
    assert wddm.parse_bus_id(_BUS_ID) == (0, (0x2D, 0, 0))
    assert wddm.parse_bus_id("0000:2d:1f.7\n") == (0, (0x2D, 0x1F, 7))


def test_the_counter_item_layout_matches_the_sdk_for_this_python():
    """PDH_FMT_COUNTERVALUE_ITEM_W is 24 bytes, value at 8, on x86 and x64 alike."""
    assert wddm._FormattedItem.FmtValue.offset == 8
    assert wddm._CounterValue.largeValue.offset == 8
    assert ctypes.sizeof(wddm._FormattedItem) == 24


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bindings")
def test_windows_bindings_run_without_a_gpu():
    """CI runners have no GPU: the bindings must still call cleanly."""
    adapters = wddm.wddm_adapters()
    assert all(isinstance(luid, int) and (address is None or len(address) == 3)
               for luid, address in adapters)
    counters = wddm.CounterReader((wddm._PROCESSES, wddm._ADAPTERS))
    try:
        samples = counters.read()
    except OSError:
        return  # a host without GPU counters reports, rather than crashes
    assert set(samples) == {wddm._PROCESSES, wddm._ADAPTERS}
    assert all(isinstance(name, str) and isinstance(value, int)
               for rows in samples.values() for name, _status, value in rows)
