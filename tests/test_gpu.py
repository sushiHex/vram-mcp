"""Tests for vram_mcp.gpu — pure, no mcp / real GPU needed."""

import subprocess

from vram_mcp import gpu


FAKE_CSV = (
    "0, GPU-4090, NVIDIA GeForce RTX 4090, 24564, 24000, 564\n"
    "1, GPU-3060, NVIDIA GeForce RTX 3060, 12288, 2048, 10240\n"
)


def test_gpu_status_parses_csv():
    gpus = gpu.gpu_status(runner=lambda cmd, timeout=5: FAKE_CSV)
    assert gpus == [
        {
            "index": 0,
            "uuid": "GPU-4090",
            "name": "NVIDIA GeForce RTX 4090",
            "total_mb": 24564,
            "used_mb": 24000,
            "free_mb": 564,
        },
    ]


def test_gpu_status_selects_requested_device_only():
    assert gpu.gpu_status(
        runner=lambda cmd, timeout=5: FAKE_CSV, device_index=1,
    ) == [{
        "index": 1, "uuid": "GPU-3060", "name": "NVIDIA GeForce RTX 3060",
        "total_mb": 12288, "used_mb": 2048, "free_mb": 10240,
    }]


def test_device_one_free_memory_cannot_hide_device_zero_pressure():
    observation = gpu.observe_gpu(0, runner=lambda cmd, timeout=5: FAKE_CSV)
    assert observation.known is True
    assert observation.scope == "gpu:index=0"
    assert observation.data[0]["free_mb"] == 564


def test_gpu_status_skips_blank_lines():
    csv = "\n0, GPU-id, GPU0, 100, 40, 60\n\n"
    gpus = gpu.gpu_status(runner=lambda cmd, timeout=5: csv)
    assert len(gpus) == 1
    assert gpus[0]["free_mb"] == 60


def test_gpu_status_file_not_found_returns_empty():
    def boom(cmd, timeout=5):
        raise FileNotFoundError("nvidia-smi not found")

    assert gpu.gpu_status(runner=boom) == []


def test_gpu_status_timeout_returns_empty():
    def boom(cmd, timeout=5):
        raise subprocess.TimeoutExpired(cmd, timeout)

    assert gpu.gpu_status(runner=boom) == []


def test_gpu_status_nonzero_exit_returns_empty():
    def boom(cmd, timeout=5):
        raise subprocess.CalledProcessError(1, cmd)

    assert gpu.gpu_status(runner=boom) == []


def test_gpu_status_parse_error_returns_empty():
    # Non-integer memory field -> unparseable -> [].
    csv = "0, GPU-id, GPU0, N/A, N/A, N/A\n"
    assert gpu.gpu_status(runner=lambda cmd, timeout=5: csv) == []


def test_gpu_status_wrong_column_count_returns_empty():
    csv = "0, GPU0, 100, 40\n"  # only 4 columns
    assert gpu.gpu_status(runner=lambda cmd, timeout=5: csv) == []


def test_observe_gpu_distinguishes_failed_from_successful_empty():
    def missing(cmd, timeout=5):
        raise FileNotFoundError

    failed = gpu.observe_gpu(runner=missing)
    empty = gpu.observe_gpu(runner=lambda cmd, timeout=5: "")
    assert failed.known is False
    assert failed.data is None
    assert failed.metadata()["status"] == "unavailable"
    assert empty.known is True
    assert empty.data == []


def test_observe_gpu_reports_missing_selected_index():
    result = gpu.observe_gpu(2, runner=lambda cmd, timeout=5: FAKE_CSV)
    assert result.known is False
    assert "index 2" in result.error


def test_gpu_status_default_runner_no_nvidia_smi(monkeypatch):
    # Force the real _run to look like nvidia-smi is missing.
    def boom(cmd, timeout=5):
        raise FileNotFoundError

    monkeypatch.setattr(gpu, "_run", boom)
    assert gpu.gpu_status() == []


def test_max_free_mb():
    assert gpu.max_free_mb([{"free_mb": 100}]) == 100


def test_max_free_mb_rejects_cross_device_maximum():
    import pytest

    with pytest.raises(ValueError, match="selected GPU"):
        gpu.max_free_mb([{"free_mb": 100}, {"free_mb": 500}])


def test_max_free_mb_empty_is_none():
    assert gpu.max_free_mb([]) is None


# An MCP client's default environment on Windows: no ProgramFiles.
_MINIMAL_ENV = {"PATH": r"C:\Windows\system32", "SYSTEMROOT": r"C:\Windows"}


def test_nvidia_smi_gets_program_files_under_a_minimal_windows_env():
    env = gpu._nvidia_smi_env(
        _MINIMAL_ENV, program_files=lambda: r"C:\Program Files", windows=True,
    )
    assert env == {**_MINIMAL_ENV, "ProgramFiles": r"C:\Program Files"}


def test_nvidia_smi_inherits_an_env_that_has_program_files():
    for name in ("ProgramFiles", "PROGRAMW6432"):
        environ = {**_MINIMAL_ENV, name: r"D:\Apps"}
        assert gpu._nvidia_smi_env(
            environ, program_files=lambda: r"C:\Program Files", windows=True,
        ) is None


def test_nvidia_smi_inherits_the_env_off_windows_or_without_a_registry_value():
    assert gpu._nvidia_smi_env(
        _MINIMAL_ENV, program_files=lambda: "/opt", windows=False,
    ) is None
    assert gpu._nvidia_smi_env(
        _MINIMAL_ENV, program_files=lambda: None, windows=True,
    ) is None


def test_run_passes_the_nvidia_smi_env(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen.update(kwargs)
        return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")

    monkeypatch.setattr(gpu, "_nvidia_smi_env", lambda: {"ProgramFiles": "X"})
    monkeypatch.setattr(gpu.subprocess, "run", fake_run)
    assert gpu._run(["nvidia-smi"]) == "ok"
    assert seen["env"] == {"ProgramFiles": "X"}
