"""Tests for vram_mcp.winproc — only what holds without Windows."""
import ctypes

import pytest

from vram_mcp import winproc


@pytest.fixture
def no_windows(monkeypatch):
    """Simulate a platform without ``ctypes.WinDLL`` and no bound libraries."""
    def missing(*args, **kwargs):
        raise AttributeError("module 'ctypes' has no attribute 'WinDLL'")

    monkeypatch.setattr(ctypes, "WinDLL", missing, raising=False)
    monkeypatch.setattr(winproc, "_kernel32", None)
    monkeypatch.setattr(winproc, "_ntdll", None)


def test_process_names_raises_oserror_without_windows_apis(no_windows):
    with pytest.raises(OSError):
        winproc.process_names()


def test_command_line_raises_oserror_without_windows_apis(no_windows):
    with pytest.raises(OSError):
        winproc.command_line(1234)
