"""Windows process names and command lines, read directly through ctypes.

A ``Get-CimInstance Win32_Process`` query cost 1.6-3 s per status call, most of
it PowerShell starting (#31). One Toolhelp snapshot names every process in
~40 ms, and each command line is one ``NtQueryInformationProcess`` call.

Both need only ``PROCESS_QUERY_LIMITED_INFORMATION``, so a non-elevated server
reads its own user's processes, Ollama's runners among them. Where Windows
denies even that, such as for ``dwm.exe``, the name still comes from the
snapshot and the command line is None. On the validated machine (RTX 3090
host, non-elevated) CIM named every GPU process this named, and the only one
without a command line, ``dwm.exe``, had none from CIM either.

Windows is reached on first use, so this module imports everywhere.
"""

from __future__ import annotations

import ctypes
from typing import Optional

_TH32CS_SNAPPROCESS = 0x2
_INVALID_HANDLE = ctypes.c_void_p(-1).value
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_PROCESS_COMMAND_LINE_INFORMATION = 60   # PROCESSINFOCLASS, Windows 8.1+
_STATUS_INFO_LENGTH_MISMATCH = 0xC0000004
_ERROR_NO_MORE_FILES = 18


class _ProcessEntry(ctypes.Structure):      # PROCESSENTRY32W
    _fields_ = [("dwSize", ctypes.c_uint32), ("cntUsage", ctypes.c_uint32),
                ("th32ProcessID", ctypes.c_uint32), ("th32DefaultHeapID", ctypes.c_size_t),
                ("th32ModuleID", ctypes.c_uint32), ("cntThreads", ctypes.c_uint32),
                ("th32ParentProcessID", ctypes.c_uint32), ("pcPriClassBase", ctypes.c_int32),
                ("dwFlags", ctypes.c_uint32), ("szExeFile", ctypes.c_wchar * 260)]


class _UnicodeString(ctypes.Structure):     # UNICODE_STRING
    _fields_ = [("Length", ctypes.c_uint16), ("MaximumLength", ctypes.c_uint16),
                ("Buffer", ctypes.c_void_p)]


_kernel32 = None
_ntdll = None


def _bind():
    """Load and type the two libraries once; OSError where they are absent."""
    global _kernel32, _ntdll
    if _kernel32 is None:
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            ntdll = ctypes.WinDLL("ntdll")
        except (AttributeError, OSError) as exc:   # AttributeError: not Windows
            raise OSError(f"Windows process APIs are unavailable: {exc}") from exc
        entry = ctypes.POINTER(_ProcessEntry)
        kernel32.CreateToolhelp32Snapshot.restype = ctypes.c_void_p
        kernel32.CreateToolhelp32Snapshot.argtypes = [ctypes.c_uint32, ctypes.c_uint32]
        kernel32.Process32FirstW.argtypes = [ctypes.c_void_p, entry]
        kernel32.Process32NextW.argtypes = [ctypes.c_void_p, entry]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        ntdll.NtQueryInformationProcess.restype = ctypes.c_uint32
        ntdll.NtQueryInformationProcess.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
            ctypes.POINTER(ctypes.c_uint32)]
        _kernel32, _ntdll = kernel32, ntdll
    return _kernel32, _ntdll


def process_names() -> dict[int, str]:
    """Every running process's image name, such as ``"ollama.exe"``, by PID."""
    kernel32, _ = _bind()
    snapshot = kernel32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snapshot in (None, _INVALID_HANDLE):
        raise OSError(f"CreateToolhelp32Snapshot failed: error {ctypes.get_last_error()}")
    try:
        entry = _ProcessEntry(dwSize=ctypes.sizeof(_ProcessEntry))
        names = {}
        found = kernel32.Process32FirstW(snapshot, ctypes.byref(entry))
        while found:
            names[entry.th32ProcessID] = entry.szExeFile
            found = kernel32.Process32NextW(snapshot, ctypes.byref(entry))
        error = ctypes.get_last_error()
        if error != _ERROR_NO_MORE_FILES:
            raise OSError(f"Process32NextW stopped early: error {error}")
        return names
    finally:
        kernel32.CloseHandle(snapshot)


def command_line(pid: int) -> Optional[str]:
    """``pid``'s command line; None if it exited or Windows denies access."""
    kernel32, ntdll = _bind()
    process = kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not process:
        return None
    try:
        size = ctypes.c_uint32(0)
        status = ntdll.NtQueryInformationProcess(
            process, _PROCESS_COMMAND_LINE_INFORMATION, None, 0, ctypes.byref(size))
        if status != _STATUS_INFO_LENGTH_MISMATCH or not size.value:
            return None
        for _ in range(3):      # the command line can grow between calls
            buffer = ctypes.create_string_buffer(size.value)
            status = ntdll.NtQueryInformationProcess(
                process, _PROCESS_COMMAND_LINE_INFORMATION, buffer, len(buffer),
                ctypes.byref(size))
            if status != _STATUS_INFO_LENGTH_MISMATCH or not size.value:
                break
        if status:
            return None
        text = _UnicodeString.from_buffer(buffer)
        base = ctypes.addressof(buffer)
        header = ctypes.sizeof(text)
        start = text.Buffer or 0
        if (not text.Length or text.Length % 2 or header + text.Length > len(buffer)
                or not base + header <= start <= base + len(buffer) - text.Length):
            return None
        return ctypes.wstring_at(start, text.Length // 2)
    finally:
        kernel32.CloseHandle(process)
