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


class FakeKernel32:
    """Mimics the kernel32 calls winproc makes, writing through ``byref`` args."""

    def __init__(self, processes=(), snapshot=0x1234, open_ok=True):
        self.processes = list(processes)
        self.snapshot = snapshot
        self.open_ok = open_ok
        self.closed = []
        self._next = 0

    def CreateToolhelp32Snapshot(self, flags, pid):
        return self.snapshot

    def _fill(self, ref):
        pid, name = self.processes[self._next]
        self._next += 1
        ref._obj.th32ProcessID = pid
        ref._obj.szExeFile = name
        return 1

    def Process32FirstW(self, snapshot, ref):
        return self._fill(ref) if self.processes else 0

    def Process32NextW(self, snapshot, ref):
        return self._fill(ref) if self._next < len(self.processes) else 0

    def OpenProcess(self, access, inherit, pid):
        return 0x99 if self.open_ok else None

    def CloseHandle(self, handle):
        self.closed.append(handle)


class FakeNtdll:
    """Mimics NtQueryInformationProcess(ProcessCommandLineInformation)."""

    def __init__(self, text="a b", sizes=None, status=0, length=None, offset=None):
        self.data = text.encode("utf-16-le")
        self.sizes = sizes      # lengths reported by successive probes/reads
        self.status = status
        self.length = len(self.data) if length is None else length
        self.offset = ctypes.sizeof(winproc._UnicodeString) if offset is None else offset
        self.calls = 0

    def NtQueryInformationProcess(self, handle, info_class, buffer, size, returned):
        assert info_class == 60
        needed = (self.sizes[min(self.calls, len(self.sizes) - 1)] if self.sizes
                  else ctypes.sizeof(winproc._UnicodeString) + len(self.data))
        self.calls += 1
        if self.status:
            return self.status
        if buffer is None or len(buffer) < needed:
            returned._obj.value = needed
            return winproc._STATUS_INFO_LENGTH_MISMATCH
        start = ctypes.addressof(buffer) + self.offset
        if self.offset + len(self.data) <= len(buffer):
            ctypes.memmove(start, self.data, len(self.data))
        header = winproc._UnicodeString.from_buffer(buffer)
        header.Length = self.length
        header.MaximumLength = self.length
        header.Buffer = start
        return 0


@pytest.fixture
def bind(monkeypatch):
    def install(kernel32=None, ntdll=None, last_error=18):
        kernel32 = kernel32 or FakeKernel32()
        ntdll = ntdll or FakeNtdll()
        monkeypatch.setattr(winproc, "_bind", lambda: (kernel32, ntdll))
        monkeypatch.setattr(ctypes, "get_last_error", lambda: last_error, raising=False)
        return kernel32, ntdll
    return install


def test_process_names_reads_snapshot_and_closes_handle(bind):
    kernel32, _ = bind(FakeKernel32([(4, "System"), (88, "ollama.exe")]))
    assert winproc.process_names() == {4: "System", 88: "ollama.exe"}
    assert kernel32.closed == [kernel32.snapshot]


def test_process_names_raises_when_enumeration_fails_midway(bind):
    kernel32, _ = bind(FakeKernel32([(4, "System")]), last_error=5)
    with pytest.raises(OSError):
        winproc.process_names()
    assert kernel32.closed == [kernel32.snapshot]


def test_process_names_raises_on_invalid_snapshot_handle(bind):
    bind(FakeKernel32(snapshot=winproc._INVALID_HANDLE), last_error=5)
    with pytest.raises(OSError):
        winproc.process_names()


def test_command_line_decodes_text_and_closes_process(bind):
    kernel32, _ = bind(ntdll=FakeNtdll("ollama.exe runner --port 5"))
    assert winproc.command_line(88) == "ollama.exe runner --port 5"
    assert kernel32.closed == [0x99]


def test_command_line_retries_when_the_line_grows(bind):
    header = ctypes.sizeof(winproc._UnicodeString)
    # The probe sizes for 4 bytes of text; the first read finds 10 are needed.
    _, ntdll = bind(ntdll=FakeNtdll("grown", sizes=[header + 4, header + 10, header + 10]))
    assert winproc.command_line(88) == "grown"
    assert ntdll.calls == 3


def test_command_line_gives_up_after_three_reads(bind):
    header = ctypes.sizeof(winproc._UnicodeString)
    _, ntdll = bind(ntdll=FakeNtdll("x", sizes=[header + 2 * n for n in range(1, 9)]))
    assert winproc.command_line(88) is None
    assert ntdll.calls == 4     # one probe, three reads


def test_command_line_none_when_open_fails(bind):
    kernel32, _ = bind(FakeKernel32(open_ok=False))
    assert winproc.command_line(4) is None
    assert kernel32.closed == []


def test_command_line_none_when_status_is_failure(bind):
    kernel32, _ = bind(ntdll=FakeNtdll(status=0xC0000022))
    assert winproc.command_line(4) is None
    assert kernel32.closed == [0x99]


@pytest.mark.parametrize("overrides", [
    {"length": 5},                      # odd Length
    {"length": 4000},                   # runs past the buffer
    {"offset": 0},                      # Buffer inside the header
    {"offset": 4000},                   # Buffer past the buffer
])
def test_command_line_none_when_unicode_string_is_malformed(bind, overrides):
    bind(ntdll=FakeNtdll("abcdef", **overrides))
    assert winproc.command_line(88) is None


def test_process_names_raises_oserror_without_windows_apis(no_windows):
    with pytest.raises(OSError):
        winproc.process_names()


def test_command_line_raises_oserror_without_windows_apis(no_windows):
    with pytest.raises(OSError):
        winproc.command_line(1234)
