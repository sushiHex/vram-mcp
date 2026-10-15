# Hardware acceptance

[Back to the README](../README.md)

`python -m pytest -q` never touches a GPU or Ollama. `acceptance/harness.py`
is the opposite: an opt-in run of real `vram-mcp` server processes against a
real NVIDIA GPU and a real Ollama, driven over MCP stdio exactly as a client
drives them. Run it against every release candidate.

## Prerequisites

- An NVIDIA GPU with `nvidia-smi` on `PATH`, and a running Ollama.
- The development install: `python -m pip install -e ".[dev]"`.
- For the mutation scenarios:
  - Ollama runs on this machine, on the GPU passed as `--gpu`;
  - Ollama is idle, with no model resident;
  - the live ledger holds no active claim or reservation, and no operation record
    at all: an expired operation may still have a live owner, so any one blocks
    the run, and a live session's `list_claims()` prunes the stale ones;
  - an installed model of at least 64 MB that fits in free VRAM. A small one is
    quickest, since it is loaded and unloaded several times.

  The harness checks all of these except Ollama's GPU placement, and skips the
  mutation scenarios, saying why, when any check fails.

## Running it

Read-only, the default:

```sh
python acceptance/harness.py
```

With the scenarios that load and unload a model:

```sh
python acceptance/harness.py --model phi4-mini --ollama http://127.0.0.1:11434 --gpu 0 --report report.md
```

`--model` changes Ollama's residency, so it is refused unless `--ollama` and
`--gpu` are also given explicitly. The harness prints a Markdown report with the
OS, Python, package, GPU, driver and Ollama versions, and one row per scenario.
It never includes process command lines. The exit status is non-zero if any
scenario fails.

## What it touches

- **Coordination state:** every server it starts uses one fresh temporary
  `VRAM_MCP_STATE_DIR`, deleted when the run ends. The servers coordinate with
  each other and with nothing else. The live ledger is read, never written, to
  confirm that no session is coordinating.
- **Ollama:** mutation requires an idle Ollama, because loading even one model
  can make Ollama's scheduler evict another to make room. With nothing resident
  there is nothing to evict, and any residency of the test model during the run
  is the run's own. Every load uses a two-minute keep-alive, so an interrupted
  run's model still leaves VRAM on its own. On the way out the harness unloads
  the test model, and only that model, and reports whether it succeeded.
- **Unreachable Ollama:** one scenario points a server at `ollama.invalid`.
  RFC 6761 reserves that domain so it never resolves, so no server can answer.
- **Residual race:** a client that loads a model between the idle check and the
  end of the run can still be affected by Ollama's scheduling. Run the harness
  when nothing else is using Ollama.

## Scenarios

| Scenario | Demonstrates |
| --- | --- |
| Every tool is discoverable | The server exposes exactly the 13 expected tools over MCP. |
| The selected GPU is identified with provenance | Source, timestamp and `gpu:index=N` scope on the capacity reading. |
| Ollama residency is observed | Loaded models come from a healthy observation. |
| The ledger is read and reported, loaded models or not | `observations.coordination` is present, from `claims` alone when nothing is loaded. |
| Process observations report their coverage | Compute, graphics and non-local coverage are explicit. |
| Pressure states the evidence it rests on | The verdict names which evidence it had. |
| Windows reads the selected adapter's non-local memory, so pressure reaches a verdict | On Windows, `non_local_memory` is covered and `pressure` is not `unknown`. Elsewhere, the scenario is "Uncovered non-local memory says why": `gaps` names the reason. |
| A claim made under an alias names the canonical model | `registry.ollama.ai/library/PHI4-MINI` claims `phi4-mini:latest`. |
| Another session cannot evict a claimed model | Two server processes share one ledger. |
| A pending warm blocks other sessions | A second session sees the in-flight operation and is refused. |
| A confirmed warm agrees with Ollama residency | `succeeded` means the model is resident afterwards. |
| A resident refresh is admitted under a full reservation | Keep-alive refresh costs no new residency, so it is admitted with all free memory reserved, rounded down to 0.01 GB. |
| A departed client's claim protects until its TTL | A claim outlives its client's process, then expires. |
| A confirmed unload agrees with Ollama residency | `succeeded` means the model is gone afterwards. |
| A reload that would breach a reservation is refused | Admission honours another session's reservation: about half the model's size is left free. |
| Unreachable Ollama reads as unknown, never as empty | `loaded` is `null`, not `[]`. |
| No mutation is attempted without residency | `warm` is refused with `observation_unavailable`. |
| The test model is unloaded afterwards | Cleanup is checked, not assumed. |

Residency checks fail rather than pass when residency cannot be read: an
unknown reading must never confirm that a model is gone.

Recovery after an operation *owner* crashes mid-mutation is not repeated here.
It depends only on the operating system releasing a dead process's file lock,
and `tests/test_util.py` and `tests/test_reliability.py` cover that against
real locks.

## Validated environments

| Date | OS | GPU and driver | Ollama | vram-mcp | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-10-13 | Windows 11 (26200), Python 3.14.3 | RTX 3090, 596.49 | 0.34.4 | 0.2.0 at `a66573d` + this harness | 17 of 17 passed with `phi4-mini` |
| 2026-10-13 | Windows 11 (26200), Python 3.14.3 | RTX 3090, 596.49 | 0.34.4 | 0.3.0 wheel, release candidate `b64bcf4`, installed in a clean environment (mcp 1.30.0) | 17 of 17 passed with `phi4-mini` |
| 2026-10-13 | Windows 11 (26200), Python 3.14.3 | RTX 3090, 596.49 | 0.34.4 | 0.3.1 wheel, release candidate `6aa7c54`, installed in a clean environment (mcp 1.30.0) | 17 of 17 passed with `phi4-mini` |

**Linux is untested on hardware.** CI exercises the Linux code paths with mocks
only.

## Non-local memory, and why Linux cannot read `ok`

`pressure` reads `ok` only when non-local memory was observed, since driver
spill cannot be ruled out otherwise. NVML does not report it. On Windows,
vram-mcp reads it from WDDM ([#22](https://github.com/sushiHex/vram-mcp/issues/22)):

- `D3DKMTQueryAdapterInfo` returns each WDDM adapter's PCI address, which maps
  the adapter's LUID to the NVML device exactly. The LUID is not stable: on the
  validated machine it moved from `0x153af` to `0x143af` across a driver
  update, so it is resolved again on every read (about 0.03 ms).
- PDH, read directly rather than through PowerShell, returns
  `\GPU Process Memory(*)\Non Local Usage` per process and adapter. One query
  per server costs about 0.5 s to open; each read after that takes under a
  millisecond, and still sees processes started after it opened. The
  per-process values summed exactly to the adapter's shared usage.
- Per-process *Dedicated* Usage is not used: Chrome's GPU process and `dwm`
  reported 488 GB and 209 GB on a 24 GB card, with valid counter status.
- WDDM listed processes on the adapter that NVML did not, one holding 31 MB of
  non-local memory. They join the process table as `kind: "wddm"`.
- The adapter's own `\GPU Adapter Memory(*)\Shared Usage` is the measurement,
  and per-process readings only attribute it. On the validated machine, read
  without elevation, processes accounted for all but a steady 18 MB of it;
  that remainder counts as unexplained. Because PDH also reports missing
  privileges as no data, an adapter whose counter cannot be read is
  unavailable, never idle.

On Linux, `non_local_memory` stays `false`, so `pressure` reads `unknown` where
it would otherwise read `ok`. Ordinary CUDA allocations are not demoted to
system memory there as WDDM does, but managed memory can be, and no interface
reports which processes use it. The evidence, and what a structural answer
would need, are in [#27](https://github.com/sushiHex/vram-mcp/issues/27).
