"""Hold GPU memory and keep touching it, reporting the bandwidth each pass gets.

The acceptance harness runs two of these to overcommit VRAM on purpose: WDDM's
default CUDA policy then pages memory to system RAM instead of failing an
allocation, and a pass that drops from VRAM speed to PCIe speed is the paging,
measured rather than inferred. Every allocation is touched on every pass, as a
model's weights are on every token.

Usage: python acceptance/overcommit.py <PCI bus ID> <MB> <seconds>

Prints one JSON object per line: ``{"allocated_mb": N}`` once, or
``{"error": "..."}`` when allocation fails, then ``{"gbps": X, "seconds": S}``
per pass.
"""

from __future__ import annotations

import ctypes
import json
import sys
import time

CHUNK = 256 << 20


def _cuda():
    if sys.platform == "win32":
        return ctypes.WinDLL("nvcuda")
    return ctypes.CDLL("libcuda.so.1")


def main(bus_id: str, total_mb: int, seconds: float) -> int:
    def emit(**fields) -> None:
        print(json.dumps(fields), flush=True)

    def check(status: int, call: str) -> None:
        if status:
            raise RuntimeError(f"{call} failed with CUresult {status}")

    cuda = _cuda()
    try:
        check(cuda.cuInit(0), "cuInit")
        device, context = ctypes.c_int(), ctypes.c_void_p()
        check(cuda.cuDeviceGetByPCIBusId(ctypes.byref(device), bus_id.encode()),
              "cuDeviceGetByPCIBusId")
        check(cuda.cuCtxCreate_v2(ctypes.byref(context), 0, device), "cuCtxCreate")
        chunks = []
        while len(chunks) * CHUNK < total_mb << 20:
            pointer = ctypes.c_uint64()
            check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(CHUNK)),
                  "cuMemAlloc")
            chunks.append(pointer)
        emit(allocated_mb=len(chunks) * CHUNK >> 20)
        deadline = time.monotonic() + seconds
        value = 0
        while time.monotonic() < deadline:
            value = (value + 1) & 0xFF
            start = time.perf_counter()
            for pointer in chunks:
                check(cuda.cuMemsetD8_v2(pointer, ctypes.c_ubyte(value),
                                         ctypes.c_size_t(CHUNK)), "cuMemsetD8")
            check(cuda.cuCtxSynchronize(), "cuCtxSynchronize")
            seconds = time.perf_counter() - start
            emit(gbps=round(len(chunks) * CHUNK / seconds / 1e9, 1), seconds=round(seconds, 3))
            time.sleep(0.25)
    except RuntimeError as exc:
        emit(error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3])))
