"""The selected GPU's non-local memory on Windows, with every OS reader injected."""

from __future__ import annotations

import sys

import pytest

from vram_mcp import wddm

_MB = 1024 * 1024
_SELECTED = 0x0000_0000_0001_43AF   # at PCI 2D:00.0
_OTHER = 0x0000_0001_0000_2B10      # a second NVIDIA GPU at 41:00.0
_BASIC = 0x0000_0000_0001_6E3E      # Microsoft Basic Render Driver: no address
_ADAPTERS = [(_SELECTED, (0x2D, 0, 0)), (_OTHER, (0x41, 0, 0)), (_BASIC, None)]


def _instance(pid: int, luid: int, phys: int = 0) -> str:
    # Windows writes the LUID's hex in uppercase.
    return f"pid_{pid}_luid_0x{luid >> 32:08X}_0x{luid & 0xFFFFFFFF:08X}_phys_{phys}"


class _Counters:
    def __init__(self, *batches):
        self.batches = list(batches)
        self.reads = 0

    def read(self):
        self.reads += 1
        batch = self.batches[min(self.reads, len(self.batches)) - 1]
        if isinstance(batch, Exception):
            raise batch
        return batch


def _reader(samples, *, adapters=_ADAPTERS, bus_id="00000000:2D:00.0"):
    calls = {"bus_id": 0, "adapters": 0}

    def bus_id_fn():
        calls["bus_id"] += 1
        return bus_id

    def adapters_fn():
        calls["adapters"] += 1
        return adapters() if callable(adapters) else adapters

    counters = samples if isinstance(samples, _Counters) else _Counters(samples)
    reader = wddm.NonLocalMemory(
        0, bus_id_fn=bus_id_fn, adapters_fn=adapters_fn, counters=counters)
    return reader, calls


def test_only_the_selected_adapter_is_attributed():
    reader, _ = _reader([
        (_instance(10, _SELECTED), 0, 300 * _MB),
        (_instance(10, _OTHER), 0, 900 * _MB),     # the same process, another GPU
        (_instance(11, _OTHER), 0, 4096 * _MB),
        (_instance(12, _BASIC), 0, 50 * _MB),
        (_instance(13, _SELECTED), 1, 2 * _MB),    # PDH_CSTATUS_NEW_DATA is valid
    ])

    reading = reader.observe()

    assert reading.known
    assert reading.data == {10: 300, 13: 2}
    assert reading.source == "wddm:pdh"
    assert reading.scope == "gpu:index=0"


def test_lowercase_instance_names_match_too():
    reader, _ = _reader([(_instance(10, _SELECTED).lower(), 0, 64 * _MB)])
    assert reader.observe().data == {10: 64}


def test_an_unreadable_instance_is_unknown_not_zero():
    reader, _ = _reader([(_instance(10, _SELECTED), 0xC0000BC6, 0)])
    assert reader.observe().data == {10: None}


def test_unrelated_instance_names_are_ignored():
    reader, _ = _reader([("_Total", 0, 7 * _MB), (_instance(10, _SELECTED), 0, 0)])
    assert reader.observe().data == {10: 0}


def test_the_luid_is_resolved_on_every_read_and_the_address_once():
    # A driver restart gives the same GPU a new LUID.
    restarted = 0x0000_0000_0001_53AF
    tables = iter([_ADAPTERS, [(restarted, (0x2D, 0, 0))]])
    reader, calls = _reader(
        [(_instance(10, _SELECTED), 0, 1 * _MB), (_instance(10, restarted), 0, 2 * _MB)],
        adapters=lambda: next(tables))

    assert reader.observe().data == {10: 1}
    assert reader.observe().data == {10: 2}
    assert calls == {"bus_id": 1, "adapters": 2}


@pytest.mark.parametrize(("adapters", "bus_id", "reason"), [
    (_ADAPTERS, "00000000:07:00.0", "no WDDM adapter is at PCI 07:00.0"),
    ([(_SELECTED, (0x2D, 0, 0)), (_OTHER, (0x2D, 0, 0))], "00000000:2D:00.0",
     "2 WDDM adapters report PCI 2D:00.0"),
    (_ADAPTERS, "00000001:2D:00.0", "PCI domain 0x1"),
    (_ADAPTERS, "not a bus id", "unrecognized PCI bus ID"),
    (_ADAPTERS, None, "NVML did not report the GPU's PCI address"),
])
def test_an_unresolvable_adapter_is_unavailable_with_its_reason(adapters, bus_id, reason):
    reader, _ = _reader([(_instance(10, _SELECTED), 0, _MB)],
                        adapters=adapters, bus_id=bus_id)

    reading = reader.observe()

    assert not reading.known
    assert reason in reading.error


def test_a_failed_address_lookup_is_retried():
    answers = iter([None, "00000000:2D:00.0"])
    reader = wddm.NonLocalMemory(
        0, bus_id_fn=lambda: next(answers), adapters_fn=lambda: _ADAPTERS,
        counters=_Counters([(_instance(10, _SELECTED), 0, _MB)]))

    assert not reader.observe().known
    assert reader.observe().data == {10: 1}


def test_a_linked_adapter_is_refused():
    reader, _ = _reader([(_instance(10, _SELECTED, phys=0), 0, _MB),
                         (_instance(10, _SELECTED, phys=1), 0, _MB)])

    reading = reader.observe()

    assert not reading.known
    assert "linked adapter" in reading.error


def test_counter_and_adapter_failures_are_unavailable_never_raised():
    reader, _ = _reader(_Counters(OSError("PdhCollectQueryData failed")))
    assert reader.observe().error == "PdhCollectQueryData failed"

    def broken():
        raise OSError("gdi32 is unavailable")

    reader = wddm.NonLocalMemory(0, bus_id_fn=lambda: "00000000:2D:00.0",
                                 adapters_fn=broken, counters=_Counters([]))
    assert reader.observe().error == "gdi32 is unavailable"


def test_bus_ids_parse_in_nvml_and_legacy_forms():
    assert wddm.parse_bus_id("00000000:2D:00.0") == (0, (0x2D, 0, 0))
    assert wddm.parse_bus_id("0000:2d:1f.7\n") == (0, (0x2D, 0x1F, 7))


@pytest.mark.skipif(sys.platform != "win32", reason="Windows bindings")
def test_windows_bindings_run_without_a_gpu():
    """CI runners have no GPU: the bindings must still call cleanly."""
    adapters = wddm.wddm_adapters()
    assert all(isinstance(luid, int) and (address is None or len(address) == 3)
               for luid, address in adapters)
    counters = wddm.CounterReader(wddm._NON_LOCAL)
    try:
        samples = counters.read()
    except OSError:
        return  # a host without GPU counters reports, rather than crashes
    assert all(isinstance(name, str) and isinstance(value, int)
               for name, _status, value in samples)
