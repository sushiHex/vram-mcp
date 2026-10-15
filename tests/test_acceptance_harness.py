"""The acceptance harness's pure helpers.

``acceptance/`` is not a package, so the harness is loaded from its path. It only
imports ``mcp`` and ``requests`` and runs nothing unless executed as a script.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "acceptance" / "harness.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("acceptance_harness", _PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # dataclasses resolve their module by name
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(spec.name, None)


def test_no_spans_cover_nothing(harness):
    assert harness.covered_seconds([], 0.0, 10.0) == 0


def test_overlapping_spans_are_counted_once(harness):
    assert harness.covered_seconds([(1.0, 4.0), (3.0, 6.0), (2.0, 3.5)], 0.0, 10.0) == 5.0


def test_a_span_inside_another_adds_nothing(harness):
    assert harness.covered_seconds([(1.0, 9.0), (2.0, 3.0)], 0.0, 10.0) == 8.0


def test_disjoint_spans_add_up(harness):
    assert harness.covered_seconds([(5.0, 6.0), (1.0, 2.0), (8.0, 8.5)], 0.0, 10.0) == 2.5


def test_spans_are_clipped_at_both_edges_of_the_window(harness):
    assert harness.covered_seconds([(-5.0, 2.0), (8.0, 20.0)], 0.0, 10.0) == 4.0


def test_a_span_crossing_the_end_counts_its_inside_part(harness):
    assert harness.covered_seconds([(7.0, 15.0)], 0.0, 10.0) == 3.0


def test_a_span_spanning_the_whole_window_covers_it(harness):
    assert harness.covered_seconds([(-1.0, 11.0)], 2.0, 6.0) == 4.0


def test_spans_before_the_window_cover_nothing(harness):
    assert harness.covered_seconds([(-5.0, -1.0), (-2.0, 0.0)], 0.0, 10.0) == 0


def test_spans_after_the_window_cover_nothing(harness):
    assert harness.covered_seconds([(20.0, 30.0)], 0.0, 10.0) == 0


def test_the_holder_can_still_be_stopped(harness):
    """`stop` is a method of Holder, not of whatever follows it in the file."""
    assert hasattr(harness.Holder, "stop")


class _Process:
    def __init__(self, returncode=None):
        self.returncode = returncode


def _holder(harness, *lines, returncode=None):
    """A Holder with printed ``lines`` and no subprocess or reader task."""
    holder = harness.Holder.__new__(harness.Holder)
    holder.process = _Process(returncode)
    holder.lines = list(lines)
    return holder


def _finish(harness, holder, after, timeout, monkeypatch):
    """Run finish_pass, counting its polls; a poll replaces the real sleep."""
    polls = []

    async def poll(seconds):
        polls.append(seconds)
        if on_poll:
            holder.lines.append(on_poll.pop(0))

    on_poll = getattr(holder, "_on_poll", [])
    monkeypatch.setattr(harness.asyncio, "sleep", poll)
    asyncio.run(holder.finish_pass(after, timeout))
    return polls


def test_finish_pass_returns_at_once_when_a_pass_ended_after_the_mark(harness, monkeypatch):
    holder = _holder(harness, (50.0, {"gbps": 3.0, "seconds": 2.0}))
    assert _finish(harness, holder, 40.0, 30.0, monkeypatch) == []


def test_finish_pass_returns_for_an_exited_process_that_printed_no_pass(harness, monkeypatch):
    holder = _holder(harness, (50.0, {"allocated_mb": 1024}), returncode=0)
    # Not a pass, but the process has exited, so there is nothing to wait for.
    assert _finish(harness, holder, 40.0, 30.0, monkeypatch) == []


def test_finish_pass_returns_once_the_process_has_exited(harness, monkeypatch):
    holder = _holder(harness, (10.0, {"gbps": 3.0, "seconds": 2.0}), returncode=0)
    assert _finish(harness, holder, 40.0, 30.0, monkeypatch) == []


def test_finish_pass_gives_up_when_the_timeout_has_passed(harness, monkeypatch):
    holder = _holder(harness, (10.0, {"gbps": 3.0, "seconds": 2.0}))
    assert _finish(harness, holder, 40.0, 0.0, monkeypatch) == []


def test_finish_pass_polls_until_a_pass_ends_after_the_mark(harness, monkeypatch):
    holder = _holder(harness, (10.0, {"gbps": 3.0, "seconds": 2.0}))   # ended before it
    holder._on_poll = [(20.0, {"gbps": 3.0, "seconds": 2.0}),          # still before
                       (harness.time.monotonic() + 1000, {"gbps": 0.2, "seconds": 9.0})]
    polls = _finish(harness, holder, 40.0, 30.0, monkeypatch)
    assert polls == [0.1, 0.1]
    assert len(holder.lines) == 3
