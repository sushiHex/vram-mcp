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


# ---- paging: the System process's engine time --------------------------------

_TICKS = wddm._TICKS_PER_SECOND   # one engine busy for one second


def _engine(pid: int, luid: int, engine: int, ticks: int, status: int = 0,
            kind: str = "Copy", phys: int = 0) -> tuple:
    return (f"{_process(pid, luid, phys)}_eng_{engine}_engtype_{kind}", status, ticks)


def _system(engine: int, ticks: int, **kwargs) -> tuple:
    return _engine(4, _SELECTED, engine, ticks, **kwargs)


def _paging(*percents, seconds=1.0):
    """A window of equal ``seconds``-long intervals at ``percents``."""
    return wddm.Paging(tuple((seconds, percent) for percent in percents))


def test_paging_counts_an_interval_at_the_threshold_as_active():
    assert _paging(wddm.PAGING_PERCENT).active_seconds == 1.0
    assert _paging(wddm.PAGING_PERCENT - 0.1).active_seconds == 0.0


@pytest.mark.parametrize(("percents", "active", "sustained"), [
    ((), 0, False),                             # an empty window is not paging
    ((0.0,) * 10, 0, False),
    ((10.0,) * 3 + (0.0,) * 2, 3, False),       # three active seconds
    ((5.0,) * 4, 4, False),
    ((5.0,) * 4 + (0.0,) * 6, 4, False),        # four of ten
    ((5.0,) * 5, 5, True),                      # five of five
    ((5.0,) * 5 + (4.9,) * 5, 5, True),         # five of ten
    ((60.0,) * 10, 10, True),
    ((4.99,) * 10, 0, False),                   # just under the threshold
])
def test_paging_is_sustained_at_five_active_seconds(percents, active, sustained):
    paging = _paging(*percents)
    assert paging.active_seconds == active
    assert paging.sustained is sustained


def test_paging_counts_seconds_not_intervals():
    assert _paging(10.0, 10.0, seconds=2.0).active_seconds == 4.0
    assert _paging(10.0, 10.0, seconds=2.5).sustained is True      # 5.0 s exactly
    assert _paging(10.0, 10.0, seconds=2.4).sustained is False
    assert _paging(10.0, 0.0, 10.0, seconds=3.0).seconds == 9.0
    assert _paging(10.0, 0.0, 10.0, seconds=3.0).sustained is True


def test_paging_summary_reports_the_window_and_its_peak():
    paging = _paging(0.0, 5.0, 12.34, 0.0, 7.0, 6.0, 9.0)
    assert paging.summary() == {"sustained": True, "active_seconds": 5.0,
                                "seconds": 7.0, "peak_percent": 12.3}
    assert _paging(0.0, 5.0, 12.34, 0.0).summary()["sustained"] is False


def test_paging_summary_rounds_seconds_to_one_decimal():
    summary = wddm.Paging(((0.333, 10.0), (4.77, 10.0), (1.0, 0.0))).summary()
    assert summary["active_seconds"] == 5.1
    assert summary["seconds"] == 6.1
    assert wddm.Paging(()).summary() == {"sustained": False, "active_seconds": 0.0,
                                         "seconds": 0.0, "peak_percent": 0.0}


def test_system_engine_ticks_keeps_only_the_system_process_on_the_adapter():
    kept = _system(0, 100)
    new_data = _system(1, 50, status=1)             # PDH_CSTATUS_NEW_DATA is valid
    ticks = wddm.system_engine_ticks({wddm._ENGINES: [
        kept, new_data,
        _engine(4, _OTHER, 0, 999),                 # the System process, another GPU
        _engine(4, _BASIC, 0, 999),
        _engine(10, _SELECTED, 0, 7),               # another process
        _engine(40, _SELECTED, 0, 7),               # PID 4 is not a prefix match
        _engine(14, _SELECTED, 0, 7),
        _engine(10, _SELECTED, 3, 7, status=0xC0000BC6),   # another process's, unreadable
        _engine(40, _SELECTED, 4, 7, phys=1),       # another process's, linked
        ("_Total", 0, 12345),
    ]}, _SELECTED)
    assert ticks == {kept[0]: 100, new_data[0]: 50}


def test_system_engine_ticks_matches_lowercase_instances():
    row = _system(0, 8)
    assert wddm.system_engine_ticks(
        {wddm._ENGINES: [(row[0].lower(), 0, 8)]}, _SELECTED) == {row[0].lower(): 8}


@pytest.mark.parametrize("rows", [
    [],
    [_engine(10, _SELECTED, 0, 7)],                 # no System process
    [_engine(4, _OTHER, 0, 7)],                     # only on another adapter
])
def test_system_engine_ticks_raises_when_there_are_none(rows):
    with pytest.raises(LookupError, match="no engine counters"):
        wddm.system_engine_ticks({wddm._ENGINES: rows}, _SELECTED)


@pytest.mark.parametrize("rows", [
    [_system(0, 7, status=0xC0000BC6)],
    # An idle engine read beside an unreadable busy one would pass for quiet.
    [_system(0, 7), _system(1, 7, status=0xC0000BC6)],
    [_system(0, 7, status=1), _system(1, 7, status=0xC0000BC6), _system(2, 7)],
])
def test_system_engine_ticks_raises_when_any_engine_is_unreadable(rows):
    with pytest.raises(LookupError, match="unreadable"):
        wddm.system_engine_ticks({wddm._ENGINES: rows}, _SELECTED)


def test_system_engine_ticks_accepts_new_data_status_on_every_engine():
    rows = [_system(0, 7, status=1), _system(1, 9, status=1)]
    assert len(wddm.system_engine_ticks({wddm._ENGINES: rows}, _SELECTED)) == 2


@pytest.mark.parametrize("rows", [
    [_system(0, 7, phys=1)],
    [_system(0, 7), _system(1, 7, phys=1)],
    [_system(0, 7, status=0xC0000BC6, phys=1)],     # linked is named before unreadable
])
def test_system_engine_ticks_refuses_a_linked_adapter(rows):
    with pytest.raises(LookupError) as raised:
        wddm.system_engine_ticks({wddm._ENGINES: rows}, _SELECTED)
    assert raised.value.args == (wddm._LINKED_ADAPTER,)
    assert "linked adapter" in wddm._LINKED_ADAPTER


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class _Engines:
    """Counter reader whose engine rows, or failure, the test sets directly."""

    def __init__(self):
        self.rows = []
        self.error = None

    def read(self):
        if self.error is not None:
            raise self.error
        return {wddm._ENGINES: list(self.rows)}


def _paging_monitor(**kwargs):
    clock, engines = _Clock(), _Engines()
    kwargs.setdefault("bus_id_fn", lambda: _BUS_ID)
    monitor = wddm.PagingMonitor(0, adapters_fn=lambda: _ADAPTERS, counters=engines,
                                 clock=clock, **kwargs)
    return monitor, engines, clock


def _tick(monitor, engines, clock, rows, seconds=1.0):
    """Advance the clock, then take one reading of ``rows``."""
    clock.now += seconds
    engines.rows = rows
    monitor.sample()


def _fill(monitor, engines, clock, intervals, *, busy_ticks=0, seconds=1.0):
    """A first reading, then ``intervals`` more, each adding ``busy_ticks``."""
    total = 10 * _TICKS
    _tick(monitor, engines, clock, [_system(0, total)], seconds)
    for _ in range(intervals):
        total += busy_ticks
        _tick(monitor, engines, clock, [_system(0, total)], seconds)


def _percents(reading):
    return [percent for _, percent in reading.data.intervals]


def _lengths(reading):
    return [seconds for seconds, _ in reading.data.intervals]


def test_a_monitor_never_sampled_reports_it_is_not_running():
    monitor, _, _ = _paging_monitor()

    reading = monitor.observe()

    assert not reading.known
    assert reading.error == "the paging sampler is not running"
    assert reading.source == "wddm:pdh"
    assert reading.scope == "gpu:index=0"


def test_paging_percent_is_engine_time_over_elapsed_clock_time():
    monitor, engines, clock = _paging_monitor(interval=2.0)
    # Two engines, each busy for 0.25 s of every 2 s: 25% of one engine.
    first = [_system(0, 1 * _TICKS), _system(1, 2 * _TICKS)]
    _tick(monitor, engines, clock, first)
    for step in range(1, 6):
        _tick(monitor, engines, clock,
              [_system(0, 1 * _TICKS + step * _TICKS // 4),
               _system(1, 2 * _TICKS + step * _TICKS // 4)], seconds=2.0)

    reading = monitor.observe()

    assert reading.known
    assert _percents(reading) == pytest.approx([25.0] * 5)
    assert _lengths(reading) == pytest.approx([2.0] * 5)
    assert reading.data.sustained is True


def test_a_quiet_system_process_reads_zero_percent():
    monitor, engines, clock = _paging_monitor()
    _fill(monitor, engines, clock, 5)

    reading = monitor.observe()

    assert _percents(reading) == [0.0] * 5
    assert reading.data.sustained is False


def test_the_window_needs_five_seconds_before_it_is_judged():
    monitor, engines, clock = _paging_monitor()

    _fill(monitor, engines, clock, 4)
    early = monitor.observe()
    assert not early.known
    assert early.error == "paging history is still being collected (4 of 5 s)"

    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)])
    assert monitor.observe().known


def test_history_is_counted_in_seconds_not_samples():
    monitor, engines, clock = _paging_monitor(interval=0.5)

    _fill(monitor, engines, clock, 9, seconds=0.5)          # 4.5 s: nine intervals
    assert not monitor.observe().known

    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)], seconds=0.5)
    reading = monitor.observe()
    assert reading.known                                    # ten intervals: 5 s
    assert len(reading.data.intervals) == 10


def test_the_first_reading_only_sets_the_baseline():
    monitor, engines, clock = _paging_monitor()
    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)])

    assert monitor.observe().error == "paging history is still being collected (0 of 5 s)"


def test_the_window_keeps_only_the_latest_ten_seconds():
    monitor, engines, clock = _paging_monitor()
    _fill(monitor, engines, clock, 12, busy_ticks=_TICKS // 10)   # 10% each

    reading = monitor.observe()

    assert reading.data.seconds == pytest.approx(wddm.PAGING_WINDOW_SECONDS)
    assert len(reading.data.intervals) == 10
    assert _percents(reading) == pytest.approx([10.0] * 10)


def test_old_activity_ages_out_of_the_window_by_the_clock():
    """Sampling has stopped, but a read still judges the last ten seconds."""
    monitor, engines, clock = _paging_monitor(interval=2.0)
    _fill(monitor, engines, clock, 5, busy_ticks=_TICKS // 5,    # 0.2 s per 2 s
          seconds=2.0)
    assert monitor.observe().data.sustained is True          # ten active seconds

    # Four seconds on, the sampler is not yet stale (3 x 2 s), and the oldest
    # intervals have left the window.
    clock.now += 4.0
    later = monitor.observe()
    assert later.known
    assert later.data.sustained is True
    assert later.data.seconds == pytest.approx(6.0)

    # Five seconds on: the oldest remaining interval is clipped to one second.
    clock.now += 1.0
    clipped = monitor.observe()
    assert _lengths(clipped) == pytest.approx([1.0, 2.0, 2.0])
    assert clipped.data.active_seconds == pytest.approx(5.0)
    assert clipped.data.sustained is True

    # Just under six seconds on: only 4.1 s of activity remain, too little
    # history to judge, though the sampler is still within three of its intervals.
    clock.now += 0.9
    aged = monitor.observe()
    assert not aged.known
    assert aged.error == "paging history is still being collected (4 of 5 s)"


def test_a_window_is_clipped_to_its_start():
    monitor, engines, clock = _paging_monitor(interval=2.0)
    _fill(monitor, engines, clock, 5, seconds=2.0)

    clock.now += 3.0                    # window now starts 7 s before the last sample

    assert _lengths(monitor.observe()) == pytest.approx([1.0, 2.0, 2.0, 2.0])


def test_five_active_three_second_intervals_then_quiet_is_not_sustained():
    """Intervals of exactly three times the sampling interval are accepted, but
    15 s of activity followed by 12 s of quiet leave nothing in the last ten."""
    monitor, engines, clock = _paging_monitor(interval=1.0)
    total = 10 * _TICKS
    _tick(monitor, engines, clock, [_system(0, total)])
    for _ in range(5):
        total += _TICKS * 6 // 10                           # 0.6 s per 3 s: 20%
        _tick(monitor, engines, clock, [_system(0, total)], seconds=3.0)
    busy = monitor.observe()
    assert busy.data.sustained is True
    assert busy.data.seconds == pytest.approx(10.0)

    for _ in range(6):
        _tick(monitor, engines, clock, [_system(0, total)])
    # 6 quiet seconds: 3 + 1 s of the two newest active intervals remain.
    partial = monitor.observe()
    assert partial.data.active_seconds == pytest.approx(4.0)
    assert partial.data.sustained is False

    for _ in range(6):
        _tick(monitor, engines, clock, [_system(0, total)])
    quiet = monitor.observe()
    assert quiet.known
    assert quiet.data.active_seconds == 0
    assert quiet.data.sustained is False


def test_a_failed_reading_clears_the_window_and_is_reported():
    monitor, engines, clock = _paging_monitor()
    _fill(monitor, engines, clock, 6)
    assert monitor.observe().known

    engines.error = OSError("PdhCollectQueryData failed")
    _tick(monitor, engines, clock, [])
    failed = monitor.observe()
    assert not failed.known
    assert failed.error == "PdhCollectQueryData failed"

    # The gap must not join two windows: collection starts over.
    engines.error = None
    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)])
    assert monitor.observe().error == "paging history is still being collected (0 of 5 s)"
    for _ in range(5):
        _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)])
    assert monitor.observe().known


def test_an_adapter_with_no_system_engine_counters_is_reported_not_quiet():
    monitor, engines, clock = _paging_monitor()
    _fill(monitor, engines, clock, 6)

    _tick(monitor, engines, clock, [_engine(10, _SELECTED, 0, 7)])

    reading = monitor.observe()
    assert not reading.known
    assert "no engine counters" in reading.error


def test_a_gap_does_not_turn_the_next_reading_into_one_long_interval():
    monitor, engines, clock = _paging_monitor()
    _fill(monitor, engines, clock, 3)
    engines.error = OSError("counters gone")
    _tick(monitor, engines, clock, [])
    engines.error = None

    # Engine time accrued during the gap would read as a huge interval if the
    # pre-gap baseline survived.
    _tick(monitor, engines, clock, [_system(0, 500 * _TICKS)], seconds=30.0)
    for _ in range(5):
        _tick(monitor, engines, clock, [_system(0, 500 * _TICKS)])

    assert _percents(monitor.observe()) == [0.0] * 5


def test_an_unresolvable_adapter_is_reported_and_retried():
    addresses = iter([None, _BUS_ID, _BUS_ID])
    monitor, engines, clock = _paging_monitor(bus_id_fn=lambda: next(addresses))

    _tick(monitor, engines, clock, [_system(0, 0)])
    assert monitor.observe().error == "NVML did not report the GPU's PCI address"

    _tick(monitor, engines, clock, [_system(0, 0)])
    assert monitor.observe().error == "paging history is still being collected (0 of 5 s)"


def test_an_engine_counter_that_decreased_contributes_nothing_that_interval():
    monitor, engines, clock = _paging_monitor()
    _tick(monitor, engines, clock, [_system(0, 100 * _TICKS), _system(1, 100 * _TICKS)])
    # Engine 0 restarted (its counter fell); engine 1 was busy for 0.5 s.
    _tick(monitor, engines, clock, [_system(0, 3 * _TICKS), _system(1, 100.5 * _TICKS)])
    # Afterwards engine 0 counts again, from its new baseline: 0.25 s busy.
    _tick(monitor, engines, clock, [_system(0, 3.25 * _TICKS), _system(1, 100.5 * _TICKS)])
    for _ in range(3):
        _tick(monitor, engines, clock, [_system(0, 3.25 * _TICKS), _system(1, 100.5 * _TICKS)])

    assert _percents(monitor.observe()) == pytest.approx([50.0, 25.0, 0.0, 0.0, 0.0])


def test_an_engine_that_appears_contributes_nothing_until_its_next_interval():
    monitor, engines, clock = _paging_monitor()
    _tick(monitor, engines, clock, [_system(0, 100 * _TICKS)])
    # A new engine arrives already holding a large cumulative count.
    _tick(monitor, engines, clock, [_system(0, 100 * _TICKS), _system(1, 5000 * _TICKS)])
    _tick(monitor, engines, clock, [_system(0, 100 * _TICKS), _system(1, 5000.1 * _TICKS)])
    for _ in range(3):
        _tick(monitor, engines, clock, [_system(0, 100 * _TICKS), _system(1, 5000.1 * _TICKS)])

    assert _percents(monitor.observe()) == pytest.approx([0.0, 10.0, 0.0, 0.0, 0.0])


def test_an_engine_that_vanishes_is_forgotten_not_counted_as_a_decrease():
    monitor, engines, clock = _paging_monitor()
    _tick(monitor, engines, clock, [_system(0, 100 * _TICKS), _system(1, 100 * _TICKS)])
    _tick(monitor, engines, clock, [_system(0, 100.5 * _TICKS)])
    for _ in range(4):
        _tick(monitor, engines, clock, [_system(0, 100.5 * _TICKS)])

    assert _percents(monitor.observe()) == pytest.approx([50.0, 0.0, 0.0, 0.0, 0.0])


def test_a_sampler_that_stopped_is_reported_once_its_last_reading_is_stale():
    monitor, engines, clock = _paging_monitor(interval=1.0)
    _fill(monitor, engines, clock, 6)

    clock.now += 3.0                    # three intervals old: still current
    assert monitor.observe().known

    clock.now += 0.5
    stale = monitor.observe()
    assert not stale.known
    assert stale.error == "the paging sampler stopped"


def test_staleness_scales_with_the_sampling_interval():
    monitor, engines, clock = _paging_monitor(interval=0.5)
    _fill(monitor, engines, clock, 10, seconds=0.5)

    clock.now += 1.5
    assert monitor.observe().known
    clock.now += 0.1
    assert monitor.observe().error == "the paging sampler stopped"


def test_a_long_interval_clears_the_window_and_the_next_reading_is_a_baseline():
    """Sleep/resume or a stalled collector: counters that did not move over 60 s
    must not be averaged into the window, and old activity must not outlive it."""
    monitor, engines, clock = _paging_monitor()
    _fill(monitor, engines, clock, 10, busy_ticks=_TICKS // 5)      # 20%: sustained
    assert monitor.observe().data.sustained is True

    _tick(monitor, engines, clock, [_system(0, 12 * _TICKS)], seconds=60.0)
    cleared = monitor.observe()
    assert not cleared.known
    assert cleared.error == "paging history is still being collected (0 of 5 s)"

    quiet = [_system(0, 12 * _TICKS)]
    for _ in range(5):
        _tick(monitor, engines, clock, quiet)
    judged = monitor.observe()
    assert _percents(judged) == [0.0] * 5
    assert judged.data.sustained is False


def test_an_interval_of_exactly_three_times_the_sampling_interval_still_counts():
    monitor, engines, clock = _paging_monitor(interval=1.0)
    _fill(monitor, engines, clock, 5)
    assert len(monitor.observe().data.intervals) == 5

    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS + 3 * _TICKS // 2)], seconds=3.0)

    data = monitor.observe().data
    assert len(data.intervals) == 6
    assert data.intervals[-1] == pytest.approx((3.0, 50.0))     # 1.5 s busy over 3 s


def test_an_interval_just_over_three_times_the_sampling_interval_clears():
    monitor, engines, clock = _paging_monitor(interval=1.0)
    _fill(monitor, engines, clock, 5)

    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)], seconds=3.01)

    assert monitor.observe().error == "paging history is still being collected (0 of 5 s)"


def test_the_long_interval_threshold_scales_with_the_sampling_interval():
    monitor, engines, clock = _paging_monitor(interval=0.5)
    _fill(monitor, engines, clock, 10, seconds=0.5)

    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)], seconds=1.5)   # 3 x 0.5
    assert len(monitor.observe().data.intervals) == 11
    _tick(monitor, engines, clock, [_system(0, 10 * _TICKS)], seconds=1.6)
    assert not monitor.observe().known


def test_a_reading_error_is_reported_ahead_of_staleness():
    monitor, engines, clock = _paging_monitor()
    _fill(monitor, engines, clock, 6)
    engines.error = OSError("counters gone")
    _tick(monitor, engines, clock, [])

    clock.now += 60.0

    assert monitor.observe().error == "counters gone"
