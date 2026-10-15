# Configuration

[Back to the README](../README.md)

## Installing from source

Use Python 3.10+:

```sh
git clone --branch v0.3.1 https://github.com/sushiHex/vram-mcp.git
cd vram-mcp
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then install:

```sh
python -m pip install -e .
```

In your client's configuration, replace `uvx` and its arguments with the
**absolute path** to `.venv/bin/vram-mcp` on Linux or
`.venv/Scripts/vram-mcp.exe` on Windows, and an empty `args` list. For example,
Codex's `~/.codex/config.toml` entry becomes:

```toml
[mcp_servers.vram]
command = 'C:\path\to\vram-mcp\.venv\Scripts\vram-mcp.exe'
args = []
```

Replace the example path with your checkout's executable. TOML single-quoted
strings preserve Windows backslashes literally. Reconnect clients after
updating the installation so each starts a fresh server process.

## Environment variables

Set these in the **MCP server's environment** through your client configuration.
Restart/reconnect the server after changing them. Use the same settings across
clients sharing one GPU and ledger.

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama HTTP endpoint. |
| `OLLAMA_MODELS` | `~/.ollama/models` | Local model directory used to correlate manifests with runner processes. Match the directory used by Ollama. |
| `VRAM_MCP_GPU_INDEX` | `0` | Zero-based NVIDIA GPU index used for capacity, process, activity, pressure, trend, and eviction decisions. Use the same value across coordinating clients. |
| `VRAM_MCP_STATE_DIR` | `~/.cache/vram-mcp` | Directory holding the claims ledger and audit state. Clients coordinate only with clients that use the same directory; a different one is a separate coordination domain. Use an absolute path: a relative one resolves against each server's working directory at startup. |
| `VRAM_MCP_AUDIT` | `1` | Enable passive appearance/disappearance detection and trend sampling. Set exactly `0` to disable. |
| `VRAM_MCP_MEANINGFUL_MB` | `512` | Minimum dedicated VRAM in MB for tracking a non-Ollama process in appearance/disappearance events. |
| `VRAM_MCP_EVENT_CAP` | `5000` | Maximum retained events, including samples; oldest entries are pruned. |
| `VRAM_MCP_SAMPLE_SECONDS` | `60` | Minimum interval between trend samples, shared across sessions. |

For clients using `mcpServers` JSON, add an `env` object to the server entry:

```json
{
  "mcpServers": {
    "vram": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/sushiHex/vram-mcp@v0.3.1", "vram-mcp"],
      "env": {
        "OLLAMA_BASE_URL": "http://127.0.0.1:11434",
        "VRAM_MCP_GPU_INDEX": "0",
        "VRAM_MCP_SAMPLE_SECONDS": "60"
      }
    }
  }
}
```

Numeric configuration is validated when the MCP server starts. Values such as
a negative GPU index, a nonnumeric threshold, or a nonpositive retention/sample
setting prevent startup instead of silently changing the meaning of a reading.
If registration succeeds but the client cannot start the server, inspect its
MCP server log for the named environment variable and required value type.

`advise()` may suggest `OLLAMA_MAX_LOADED_MODELS` or `OLLAMA_KEEP_ALIVE`.
Those belong to the **Ollama server's environment**; setting them only on the
MCP process does not reconfigure an already-running Ollama instance.

### Local and remote Ollama

GPU readings, process inspection, model manifests, and claim files are local
to the MCP server process. `OLLAMA_BASE_URL` can point elsewhere, but it does
not move those collectors. For coherent memory decisions, run the MCP server
alongside Ollama with access to its model directory and runner processes.
A container or different OS user may expose different processes and files.

`VRAM_MCP_GPU_INDEX` scopes local capacity, process, busy, pressure, and trend
readings to one device. Ollama's `/api/ps` response is global to the configured
Ollama server and does not say which GPU holds each model. On a multi-GPU or
remote-Ollama setup, a loaded model may therefore be listed without proof that
it resides on the selected local GPU. Check the response's `scope` and
`observations` before combining those facts. GPU rows also include the device
UUID when `nvidia-smi` reports it, which lets operators verify that an index
still names the intended card after hardware or driver changes.

### Observation health

`vram_status()` reports source metadata under `observations`. Each entry has a
`status` of `available` or `unavailable`, its `source`, collection time, and
scope; failures also include `error`. Unavailable model/process data is returned
as `null`; GPU failures retain `gpus=[]` with unavailable metadata and
`free_mb=null`. This distinction lets callers separate “no models/processes”
from “the source could not be queried.”

Process metadata also reports whether non-local memory is covered, and when it
is not, `gaps.non_local_memory` says why. On Windows, `non_local_memory` is
`true` when the selected GPU's WDDM adapter is found and its shared-memory
total is read. On Linux it is `false`: no interface reports a process's GPU
memory in system RAM ([#27](https://github.com/sushiHex/vram-mcp/issues/27)).
Compute and graphics process-query coverage are reported separately; either
query may be unavailable while the other's rows remain useful.

Paging has its own observation, `observations.paging`, and its own coverage
flag, `pressure.coverage.paging`. On Windows a sampler thread starts with the
server and reads the System process's engine time on the selected adapter once a
second, keeping the last 10 readings. A status call waits until 5 readings
exist, at most about 7 seconds: once after startup, and again after a gap. A
failed reading, or an interval longer than three seconds (a sleeping machine, a
stalled collector), starts a new window, so two windows never join. If the
sampler cannot find the adapter, cannot read every one of its engine counters,
or has stopped, the observation is unavailable and says why, and `gaps.paging`
repeats the reason. On Linux no collector reads paging.

### Audit cost and coverage

On Windows/WDDM, NVML often cannot report per-process memory sizes. The
process observation starts from the selected GPU's NVML PID set and reads each
process's non-local memory from the WDDM adapter at that GPU's PCI address,
through the `GPU Process Memory` performance counters. Other adapters' readings
are never counted. The adapter's own shared-memory total is the measurement,
reported as `totals.non_local_mb`; per-process readings only attribute it, so
memory no visible process accounts for still counts, and an adapter whose
counter cannot be read is unavailable rather than idle. Every process WDDM
reports on the adapter is in the table; one NVML did not list appears with
`kind: "wddm"`. The counter query opens once per server, which
takes about half a second on the first status call; later reads take well
under a millisecond. The observation's `device` is the GPU's PCI address, from
the same NVML session as its processes. Identity is then read through ctypes,
with no subprocess: names from one Toolhelp snapshot, command lines from
`NtQueryInformationProcess`. A process Windows denies to a non-elevated server,
such as `dwm.exe`, keeps its name but has `cmdline: null`. Exited processes
remain in the inventory with `name: null` and `cmdline: null`.

`VRAM_MCP_AUDIT=0` disables action logging, appearance/disappearance detection,
and new trend samples. It does not disable current GPU, process, model, or
activity observations. Claims and model operations remain enabled, and old
audit events remain queryable.

`tight`, `degraded` and `thrashing` report from the evidence they have: each
rests on a reading that was taken, so incomplete coverage elsewhere does not
suppress them. `thrashing` rests on the paging window alone. Non-local memory
is reported (`non_local_mb`, `explained_offload_mb`, `unexplained_spill_mb`) but
decides no verdict. Without a measured device total, some processes may have had
no reading; those figures are then `null`, since they would understate the
total.

`ok` is the one verdict that asserts a negative — that nothing is wrong — so it
requires every input it depends on: Ollama residency (`models`) and paging
(`paging`). Without them the state is `unknown` with the missing evidence named
in `detail`, rather than `ok`. Missing process information is not evidence that
nothing else is using the GPU.

On Linux this is the normal case, not an edge one: no collector observes
paging there, so `vram_status()` reports `unknown` where it would otherwise
report `ok` ([#27](https://github.com/sushiHex/vram-mcp/issues/27)).
Read `pressure.coverage` to see exactly which evidence was available.

Keep the sample interval high enough that samples do not crowd action events
out of the bounded log.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Client cannot start the server | Confirm Git and uv are installed and visible to the client. Use an absolute `uvx` or installed executable path if its PATH differs from your terminal's. Allow time for the first dependency download. |
| Starting `vram-mcp` appears to hang | It is waiting for MCP input over stdio. Connect through an MCP client. |
| `free_mb` is `null` or `gpus` is empty | Read `observations.gpu.error`, run `nvidia-smi` in the server's environment, and confirm the configured `VRAM_MCP_GPU_INDEX` is present. |
| `loaded` is `null` | Read `observations.ollama.error`, then check `ollama ps` and `OLLAMA_BASE_URL`. A known empty list means Ollama was reached and reported no resident models. |
| `busy` is `null` | NVML activity or model-to-process correlation is unavailable. Check access to runner processes and the model directory; keep explicit claims for work in use. |
| A mutation returns `outcome="unknown"` | Read its observation metadata. If `pending_until` is present, the Ollama request may still finish; do not retry that model before then. For `ensure_free`, inspect `attempts` for a pending eviction. |
| `warm` is refused | Read `summary`, `reason`, and `observations`. Fix unavailable residency/ledger health, or resolve a reservation. `force=true` overrides admission policy, but still requires a healthy coordination ledger. |
| `trend()` has no samples | Call `vram_status()` with working GPU telemetry and auditing enabled. `list_loaded()` never samples GPU memory; historical data is not collected in the background. |
| A model is protected after inference ends | Check active claims. Busy detection uses a recent activity window and can lag by a few seconds; recheck before considering an override. |

For registration details, see the [Claude Code MCP guide](https://code.claude.com/docs/en/mcp).
The [uv tools guide](https://docs.astral.sh/uv/guides/tools/) covers Git sources
and isolated tool environments.
