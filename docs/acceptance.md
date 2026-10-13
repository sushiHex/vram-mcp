# Hardware acceptance

[Back to the README](../README.md)

`python -m pytest -q` never touches a GPU or Ollama. `acceptance/harness.py`
is the opposite: an opt-in run of real `vram-mcp` server processes against a
real NVIDIA GPU and a real Ollama, driven over MCP stdio exactly as a client
drives them. Run it against every release candidate.

## Prerequisites

- An NVIDIA GPU with `nvidia-smi` on `PATH`, and a running Ollama.
- The development install: `python -m pip install -e ".[dev]"`.
- For the mutation scenarios, an installed model that nothing else is using.
  The smallest one you have is best; it is loaded and unloaded several times.

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
  `VRAM_MCP_STATE_DIR`, removed afterwards. The servers coordinate with each
  other and with nothing else. The live ledger is read once, never written, to
  refuse a model that a live session has claimed.
- **Ollama:** it mutates only the `--model` it was given, and only when that
  model was installed, not resident, and unclaimed at the start. Every load uses
  a two-minute keep-alive, so an interrupted run's model still leaves VRAM on
  its own. On the way out it unloads that model, and only that model.
- **Unreachable Ollama:** one scenario starts a server pointed at a closed local
  port. It never contacts anything else.

## Scenarios

| Scenario | Demonstrates |
| --- | --- |
| Every tool is discoverable | The installed server exposes all 13 tools over MCP. |
| The selected GPU is identified with provenance | Source, timestamp and `gpu:index=N` scope on the capacity reading. |
| Ollama residency is observed | Loaded models come from a healthy observation. |
| The isolated ledger is read even with nothing to annotate | `observations.coordination` is present with no model loaded. |
| Process observations report their coverage | Compute, graphics and non-local coverage are explicit. |
| Pressure states the evidence it rests on | The verdict names which evidence it had. |
| A claim made under an alias names the canonical model | `registry.ollama.ai/library/PHI4-MINI` claims `phi4-mini:latest`. |
| Another session cannot evict a claimed model | Two server processes share one ledger. |
| A pending warm blocks other sessions | A second session sees the in-flight operation and is refused. |
| A confirmed warm agrees with Ollama residency | `succeeded` means the model is resident afterwards. |
| A resident refresh is admitted under a reservation | Keep-alive refresh costs no new residency. |
| A confirmed unload agrees with Ollama residency | `succeeded` means the model is gone afterwards. |
| A reload that would breach the reservation is refused | Admission honours another session's reservation. |
| Unreachable Ollama reads as unknown, never as empty | `loaded` is `null`, not `[]`. |
| No mutation is attempted without residency | `warm` is refused before any request. |
| A departed client's claim protects until its TTL, then frees the model | Recovery after a client exits. |

Recovery after an operation *owner* crashes mid-mutation is not repeated here.
It depends only on the operating system releasing a dead process's file lock,
and `tests/test_util.py` and `tests/test_reliability.py` cover that against
real locks.

## Validated environments

| Date | OS | GPU and driver | Ollama | vram-mcp | Result |
| --- | --- | --- | --- | --- | --- |
| 2026-10-13 | Windows 11 (26200), Python 3.14.3 | RTX 3090, 596.49 | 0.34.4 | 0.2.0 at `a66573d` + this harness | 16 of 16 passed with `phi4-mini` |

**Linux is untested on hardware.** CI exercises the Linux code paths with mocks
only.

## Known limitation: Windows pressure

On Windows, `pressure.coverage.non_local_memory` is `false`, so `pressure` reads
`unknown`. NVML reports no per-process memory under WDDM. The GPU performance
counters do report it, but the earlier collector summed them across every adapter
on the host and was slow, so it was removed. An investigation on the validated
machine found that both problems can be fixed:

- `D3DKMTQueryAdapterInfo` returns each WDDM adapter's PCI address. That maps
  its LUID to the NVML device exactly; on the validated machine, LUID
  `0x153af` is at PCI `2D:00.0`, which is `nvidia-smi`'s bus ID.
- PDH, read directly rather than through PowerShell, returns
  `\GPU Process Memory(*)\Non Local Usage` for that LUID in about 1 ms after a
  one-time ~0.6 s query setup. The per-process values summed exactly to the
  adapter's shared usage.
- Per-process *Dedicated* Usage is not usable: Chrome's GPU process and `dwm`
  reported 488 GB and 209 GB on a 24 GB card, with valid counter status.

Restoring Windows spill detection on this basis is tracked separately; until
then, read Windows `pressure` as diagnostic only.
