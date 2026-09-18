# vram-mcp

Give your AI agents a shared view of GPU memory. Through
[MCP](https://modelcontextprotocol.io), they can inspect what's using VRAM,
make room for [Ollama](https://ollama.com) models, and see which sessions are
already relying on them.

Each client runs its own server; sessions under the same OS user share claims,
reservations, pending-operation records, and an audit log. Capacity and process
readings apply to one configured GPU. Claimed or recently busy models are
protected from eviction by default.

## Get started

Use **Python 3.10+**, a running **Ollama** instance, and an **NVIDIA GPU with
drivers** for memory readings. Windows and Linux are tested. Without
`nvidia-smi` or NVML, model operations and claims still work, but some readings
are unavailable.

Run vram-mcp on the machine hosting Ollama and the GPU. Changing
`OLLAMA_BASE_URL` redirects model requests; GPU and process inspection always
remain local. On a multi-GPU host, set `VRAM_MCP_GPU_INDEX` in every client's
server environment to select the same device; the default is GPU index `0`.

### Connect your client

With [uv](https://docs.astral.sh/uv/getting-started/installation/) and Git
installed, choose your client below. `uvx` fetches vram-mcp from GitHub into an
isolated environment; no source checkout is needed.

**[Claude Code](https://code.claude.com/docs/en/mcp):**

```sh
claude mcp add vram --scope user -- uvx --from git+https://github.com/sushiHex/vram-mcp vram-mcp
```

**[Codex](https://learn.chatgpt.com/docs/extend/mcp?surface=cli):**

```sh
codex mcp add vram -- uvx --from git+https://github.com/sushiHex/vram-mcp vram-mcp
```

**[Hermes](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)** — merge into `~/.hermes/config.yaml`:

```yaml
mcp_servers:
  vram:
    command: uvx
    args: ["--from", "git+https://github.com/sushiHex/vram-mcp", "vram-mcp"]
```

<details>
<summary>Claude Desktop and other clients using mcpServers JSON</summary>

Merge this entry into your client's MCP configuration:

```json
{
  "mcpServers": {
    "vram": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/sushiHex/vram-mcp", "vram-mcp"]
    }
  }
}
```

If the client cannot find `uvx`, use its absolute executable path. In JSON,
Windows paths need escaped backslashes, such as `C:\\tools\\uvx.exe`.

</details>

Reconnect your client after registration. It launches the server over stdio;
running `vram-mcp` by hand waits for MCP input. Prefer a local installation?
See [installing from source](docs/configuration.md#installing-from-source).

### Make your first check

Ask your agent:

> Check GPU memory with vram-mcp. Show what's loaded, who is using it, and how
> much room is left. Report any unavailable readings.

The agent should call `vram_status()` and `list_claims()`. Status includes a
readable `summary`, the selected GPU, loaded models, available process details,
and `observations` metadata. Check each observation's `status`; `unavailable`
means the corresponding value is unknown, rather than empty or zero.
`list_claims()` also shows capacity reserved for training or other non-Ollama
work.

## Agent workflow

These examples are **MCP tool calls** made inside a connected client, not shell
commands or a Python API. Use a descriptive session label for `owner` and `by`,
such as `codex:review`, so another agent can identify your work.

1. **Inspect before changing memory.** Call `vram_status()` and `list_claims()`.
   Model names are canonicalized consistently: a bare final name such as
   `llama3` becomes `llama3:latest`; comparison is case-insensitive, and
   Ollama's default registry/library prefix is removed.
2. **Claim a model before relying on it.** Replace the example model below
   with one already installed in Ollama. Continue only if the claim succeeds,
   and save its returned `claim_id`.

   ```text
   claim(model="llama3:latest", owner="codex:review", purpose="Review local code")
   ```

3. **Make room when needed.** `ensure_free(gb=8, by="codex:review")` unloads
   unprotected models largest-first. Choose the target for your workload and
   check `outcome`, `free_mb`, `declined`, `reserved_mb`, and `observations`.
   It does not evict when selected-GPU capacity or Ollama residency is unknown.
   Reaching the target does not give you ownership of that space.
4. **Load and check.** Call
   `warm(model="llama3:latest", keep_alive="10m", by="codex:review")` when needed.
   Inspect `outcome`, `reason`, and `observations`. `succeeded` means residency
   was reconciled after the request; `refused` means no Ollama request was sent;
   `failed` means Ollama definitively rejected it; and `unknown` means the
   request may still be running and includes `pending_until`. Do not retry a
   same-model mutation until that pending window ends. Warm admission is also
   serialized across models sharing the selected GPU. A model that was already
   resident has zero incremental residency cost (`reason="already_resident"`).
   Loading successfully does not guarantee full GPU residency. Run inference
   through your usual Ollama client.
5. **Renew and release.** Claims expire after one hour by default. Call
   `renew(claim_id="<returned claim_id>")` before expiry for longer work and
   `release(claim_id="<returned claim_id>")` when finished, including if loading
   fails. Releasing a claim does not unload the model; claim expiry and Ollama's
   `keep_alive` are separate.

For training or diffusion, use `reserve(gb=8, owner="codex:training",
purpose="LoRA training")` to declare capacity, then renew/release its `claim_id`
in the same way. A reservation records intent; it does not allocate memory.

Validation rejects invalid requests: capacities, TTLs, and trend windows
must be finite and positive; model, owner, purpose, and caller labels must be
nonblank; `warm` accepts a positive Ollama duration or `-1` for indefinite
residency. Use `unload()` rather than a zero `keep_alive`.

**Coordination is cooperative.** Direct Ollama calls can bypass it. `force=True`
on `unload`, `ensure_free`, or `warm` overrides claims, busy protection, or
reservation admission; use it only after resolving competing work. It does not
bypass an unreadable coordination ledger or another pending mutation.
`busy=null` means unknown and does not block eviction on its own. See
[coordination details](docs/coordination.md).

## Understand the readings

| `pressure.state` | Meaning |
| --- | --- |
| `ok` | Nothing is wrong, and every input that claim rests on was actually read. Missing evidence yields `unknown`, never `ok`. |
| `tight` | Less than 1 GiB is free on the selected GPU. |
| `degraded` | Ollama placed part of a model on the CPU; expect slower inference. |
| `thrashing` | Unexplained non-local memory meets the spill threshold and, when free VRAM is known, exceeds it. Driver paging is suspected. |

Missing required telemetry produces `pressure.state="unknown"`. Read
`pressure.coverage` and top-level `observations` to see which evidence was
available. Driver-spill detection is best effort and requires non-local memory
that can be attributed to the selected GPU; adapter-aggregated Windows counters
do not establish that. CPU offload is tracked separately. Disabling auditing
stops history detection and trend sampling, but does not disable current GPU or
process observations.

Use `history()` to investigate a model disappearing and `trend(hours=1)` to
review memory changes. Trends are sampled when `vram_status()` reads the GPU;
there is no background sampler. See [diagnostics](docs/coordination.md#diagnostics).

## Tools

The client's MCP schema provides full arguments and defaults.

| Tool | Purpose |
| --- | --- |
| `vram_status` | Inspect GPU memory, models, claims, activity, and pressure. |
| `list_loaded` | List resident models and their claim/busy details. |
| `ensure_free` | Try to reach a free-memory target by unloading unprotected models. |
| `unload` | Evict a named model, respecting claims and recent activity. |
| `warm` | Load a model for a chosen `keep_alive`, checking reservations. |
| `claim`, `reserve` | Declare model use or capacity needed for other GPU work. |
| `renew`, `release` | Extend or end your claim or reservation. |
| `list_claims` | See active model claims and capacity reservations. |
| `history`, `trend` | Review recorded actions, observed changes, and memory samples. |
| `advise` | Get configuration suggestions; no settings are changed. |

## Guides

| Guide | Read it for |
| --- | --- |
| [Configuration](docs/configuration.md) | Source installation, environment variables, and connection troubleshooting. |
| [Coordination and diagnostics](docs/coordination.md) | Claims, reservations, protection limits, pressure, and audit history. |

## Development

```sh
git clone https://github.com/sushiHex/vram-mcp.git
cd vram-mcp
python -m venv .venv
```

Activate with `source .venv/bin/activate` on Linux or
`.\.venv\Scripts\Activate.ps1` in PowerShell, then run:

```sh
python -m pip install -e ".[dev]"
python -m pytest -q
```

Tests use mocked GPU, process, and HTTP calls; no GPU or Ollama daemon is needed.
CI covers Python 3.10–3.14 on Windows and Linux. Current model management targets
Ollama and NVIDIA; AMD/Intel telemetry and vLLM/llama.cpp management are future
work.

## License

MIT — see [LICENSE](LICENSE). Companion project:
[hardline-mcp](https://github.com/sushiHex/hardline-mcp) for agent messaging.
