"""The acceptance harness's pure helpers.

``acceptance/`` is not a package, so the harness is loaded from its path. It only
imports ``mcp`` and ``requests`` and runs nothing unless executed as a script.
"""

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
