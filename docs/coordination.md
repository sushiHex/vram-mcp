# Coordination and diagnostics

[Back to the README](../README.md)

## Shared state

Every connected client starts its own stdio server. Processes running under the
same OS user coordinate through `~/.cache/vram-mcp/`:

| File | Contents |
| --- | --- |
| `claims.json` | Model claims, capacity reservations, and bounded pending-operation records. |
| `events.jsonl` | Bounded audit history of actions, observed changes, and memory samples. |
| `last_seen.json` | Previous set of observed memory holders. |
| `last_sample.json` | Shared sampling throttle state. |

Ledger writes use file locks and atomic replacement. An unreadable or
wrong-shaped ledger is treated as unavailable, because silently replacing it
could discard another session's protection. An individual malformed operation
record is discarded while valid claims are retained, and the next ledger write
removes it permanently. Owners are descriptive labels, not authenticated
identities. Separate home directories have separate ledgers, even when their
processes use the same physical GPU.

## Model claims

Use `claim(model, owner, purpose)` before relying on a model. Claims default to
3,600 seconds and multiple sessions may claim the same model independently.
A claim records use; it neither loads the model nor extends Ollama's residency.

Every model-taking tool uses the same canonical key. If the final path component
has no tag, vram-mcp appends `:latest`; it also compares case-insensitively and
removes Ollama's default `registry.ollama.ai/library/` prefix. Thus `LLAMA3`,
`library/llama3`, and `registry.ollama.ai/library/llama3:latest` share the key
`llama3:latest`, while `registry:5000/team/model:q4` keeps its custom registry
and explicit `q4` tag. Claim results include the canonical `model`, and
history/filtering use the same spelling.

Check `ok` and save the returned `claim_id` and `expires_at`. Renew before
expiry; `renew` cannot revive an expired claim. Release when work ends. If a
session exits without cleanup, its claim expires naturally. Claim expiry and
release do not trigger an unload.

## Eviction and reservations

`unload()` and `ensure_free()` protect models with an active claim or
`busy=true`. Activity is a best-effort NVML reading from recent timestamped
samples on the selected GPU; stale samples are ignored. `busy=null` supplies no
protection by itself. Use claims even when activity readings are available.

`ensure_free(gb=8)` targets 8 × 1,024 MB on `VRAM_MCP_GPU_INDEX` only. It never
borrows a second GPU's headroom. It unloads candidates by decreasing reported
VRAM size, rechecks selected-device memory after each attempt, and reports
skipped models in `declined`. Ollama's resident-model list is server-wide and
does not identify placement, so a multi-GPU setup cannot prove every candidate
belongs to the selected GPU. If capacity or Ollama residency is unavailable,
`ensure_free` returns `outcome="unknown"` without evicting.

For non-Ollama work, `reserve(gb, owner, purpose, ttl_seconds=3600, pid=None)`
records capacity in the same ledger. `pid` is advisory; it does not monitor
process lifetime or release the reservation automatically. Use the returned
`claim_id` with `renew` and `release`. Unfiltered `list_claims()` includes
reservations; filtering by `model` excludes them.

Reservations do not allocate memory, validate availability, or protect a named
model from eviction. They are reported as `reserved_mb` by `ensure_free()` and
checked by `warm()`. Inspect existing reservations and available memory before
starting a job. The ledger is cooperative; direct Ollama requests bypass it.
vram-mcp does, however, serialize its own same-model mutations with a durable
pending-operation record.

### Inspect pending operations

`list_claims()` returns one consistent ledger view with both `claims` and
`operations`. Passing `model` applies the same canonical model filter to both
arrays; reservations remain visible only in an unfiltered view. Every operation
entry has this stable shape:

| Field | Meaning |
| --- | --- |
| `operation_id` | Stable identifier for the mutation attempt. |
| `model`, `kind`, `scope` | Canonical model, mutation kind (`warm` or `unload`), and selected GPU scope. |
| `started_at`, `expires_at`, `pending_until` | ISO timestamps. `pending_until` is the current retry boundary and normally equals `expires_at`. |
| `lifecycle` | `in_flight` while the owner may still be doing backend work; `unknown` after the backend result could not be established; `unrecognized` for a record written by a newer vram-mcp whose state vocabulary this version lacks. |
| `owner_live` | Whether the operation's OS owner lock is currently held. |
| `lease_expired` | Whether the wall-clock lease has elapsed; this never overrides `owner_live`. |
| `outcome`, `reason` | `null` while in flight; `unknown` and the retained reason for an uncertain operation. |
| `retry_count`, `retry_after` | Number of retries already made and the next safe retry time, or `null` while in flight. |

An operation with `owner_live=true` continues to block the same-model mutation
even when `lease_expired=true`. A lease is safe to discard only when its owner
lock is no longer held. An unknown operation has `owner_live=false` and remains
visible until `pending_until`; it blocks a retry during that window. The list
call takes the ledger lock, refreshes the view from disk, and rewrites the file
when pruning changes it, so a renewed pending expiry is returned exactly as
stored. Polling therefore briefly serializes with ledger writers.

### Interpreting a warm result

A normal `warm()` checks free memory minus active reservations against an
approximate model size from Ollama's installed-model list. Its `reason` explains
the admission decision:

| `reason` | Meaning |
| --- | --- |
| `already_resident` | The model is already resident, so refreshing its keep-alive has zero incremental residency cost. |
| `fits` | The estimated size fits the remaining headroom; `size_verified=true`. |
| `no_reservations` | No reserved capacity was found; no model-fit check is required. |
| `free_unknown` | Free memory could not be read; the load is allowed without verifying fit. |
| `size_unknown` | Reservations exist and headroom is positive, but model size is unavailable; the load is allowed without verifying fit. |
| `no_headroom` | Reservations leave no free headroom; the load is refused. |
| `insufficient_headroom` | The estimated model size exceeds remaining headroom; the load is refused. |

`size_verified=false` means the `fits` check did not pass or run; read `reason`
to distinguish cases. Even `fits` uses on-disk size as an estimate, not a
guarantee of runtime VRAM needs. `free_unknown` and `size_unknown` remain
explicitly fail-open admission results when residency and the coordination
ledger were successfully read. A resident model needs no second copy of its
full on-disk size, so reservations do not reject a keep-alive refresh.

If Ollama residency or the ledger is unreadable, a normal warm is refused before
admission because existing work cannot be checked. `force=True` skips
reservation and residency admission, but all mutations still require a healthy
ledger so vram-mcp can serialize them. Force also overrides claims and recent
busy protection for eviction; resolve competing work before using it.

### Mutation outcomes and pending work

Warm and unload return `outcome`:

| `outcome` | Meaning |
| --- | --- |
| `succeeded` | Ollama residency was observed in the requested state after the request. |
| `refused` | vram-mcp did not send the request because validation, coordination, protection, or admission rejected it. |
| `failed` | Ollama definitively rejected the request, such as an HTTP 4xx response. |
| `unknown` | The request may have reached Ollama, but its final state could not be established. |

vram-mcp sends a residency-changing POST once and reconciles with bounded
residency reads; it does not blindly retry an uncertain POST. An unknown result
keeps a bounded operation record and returns `pending_until`. Claims and other
same-model mutations are refused during that window, preventing a timeout from
immediately racing with a conflicting request.

All coordination mutation responses include `ok`, `outcome`, `model`,
`reason`, `detail`, and `summary`. `reason` describes a local decision such as
`model_claimed`, `operation_pending`, or `insufficient_headroom`; it is `null`
when there is no local decision to explain. `detail` is human-readable backend
or observation context and is also `null` when no extra detail exists. The
outcome meanings are independent of the local reason:

| Outcome | Meaning | Retry guidance |
| --- | --- | --- |
| `succeeded` | The requested state was observed after the request. | No retry is needed. |
| `refused` | No backend mutation was sent because local validation, coordination, protection, or admission rejected it. | Resolve the reported `reason`; a pending operation is visible in `operations`. |
| `failed` | The backend definitively rejected the request. | Retry only after addressing the backend error. |
| `unknown` | The request may have reached the backend, but its final state is unverified. | Do not repeat the mutation until `pending_until`/`retry_after`; inspect `list_claims()` first. |

For example, an agent recovering after a disconnect can inspect and act on the
same operation without guessing from a timeout:

```text
list_claims(model="llama3")
```

If the returned operation has `lifecycle="unknown"`, `outcome="unknown"`, and
`retry_after="2026-09-13T12:02:00Z"`, inspect residency after that boundary and
retry only if the operation is gone or no longer blocks the model. If it has
`lifecycle="in_flight"` and `owner_live=true`, wait for the live owner even when
`lease_expired=true`.

`lifecycle="unrecognized"` means another vram-mcp version wrote the record and
this one cannot read its state vocabulary. Treat it exactly like `in_flight`:
the operation still holds the model, `owner_live` and `expires_at` are still
accurate, and its lease still expires normally. Only the interpretive fields
(`outcome`, `reason`, `retry_count`, `retry_after`) are withheld, because this
version would be guessing at them. A record is never discarded merely for being
unrecognized — deleting it would release protection another process is relying
on, and this version's write-back preserves it untouched.

Warm admission also consumes shared selected-GPU capacity, so only one warm may
pass admission at a time per GPU scope, even when the model names differ. A new
capacity reservation is refused while any warm is pending because reservations
are global cooperative promises and do not carry a GPU scope. `force=True` does
not bypass these serialization rules. Recheck status after the pending window;
Ollama may have completed the original operation.

## Diagnostics

### Pressure and CPU offload

The pressure verdict prioritizes `thrashing`, then `degraded`, then `tight`,
then `ok`; missing required evidence produces `unknown`. Read it alongside
`free_mb`, `pressure.detail`, `pressure.coverage`, and top-level `observations`.
Every observation records availability, source, collection time, scope, and
collector-specific coverage where relevant. Process coverage reports compute
and graphics query success independently; partial rows remain useful for a
point-in-time status, but auditing does not update its disappearance baseline
from a partial process inventory.

On Windows/WDDM, a runner's non-local memory can include deliberately
CPU-offloaded layers. vram-mcp attributes that memory before judging spill:

- `non_local_mb`: raw total reported by the process table.
- `explained_offload_mb`: memory attributable to Ollama's deliberate CPU offload,
  capped per correlated runner at `total_size_mb - size_vram_mb`.
- `unexplained_spill_mb`: the remainder, including other processes' non-local
  memory and any excess above a runner's explained offload.

Spill is flagged when scoped unexplained memory reaches `VRAM_MCP_SPILL_MB`
(default 256 MB) **and exceeds selected-device free VRAM**. Treat this as a
diagnostic heuristic. Current NVML process telemetry does not expose non-local
memory, and Windows performance counters aggregate adapters; their totals are
therefore not assigned to the selected GPU. When
`pressure.coverage.non_local_memory=false`, driver spill is unknown rather than
zero. A partially CPU-offloaded model can still read `degraded` without evidence
of driver paging.

### History

Use MCP tool calls such as `history(model="llama3:latest")` or
`history(type="disappeared", limit=25)`. Results are newest-first; `since` accepts
an ISO timestamp.

Action events record warm/unload outcomes and ensure-free attempts with the
caller's `by` label (default `unknown`). Only a successful matching eviction can
later explain a model disappearance as `self_action`; refused, failed, and
unknown attempts are retained as evidence without claiming causation. An
`ensure_free` call that finds sufficient memory immediately has no eviction
action to record.

With auditing enabled, `list_loaded()` updates the Ollama-model baseline;
`vram_status()` updates that baseline and the process baseline when both NVML
process queries succeeded. All Ollama models and non-Ollama processes using at
least `VRAM_MCP_MEANINGFUL_MB` dedicated VRAM qualify. The first successful
snapshot establishes each baseline; changes between observations can be missed.

Disappearance causes are best-effort attribution:

| Cause | Interpretation |
| --- | --- |
| `self_action` | A recent matching vram-mcp eviction explains the disappearance. |
| `external` | No matching eviction was recorded; possibilities include Ollama expiry, pressure, or an external unload. |
| `unattributed` | A non-Ollama process disappeared; its exit cause is unknown. |

Audit failures do not fail the main tool call, so history is useful evidence,
not a complete record of every GPU event.

### Trends

`trend(hours=1)` summarizes retained samples: direction, min/max/latest free MB,
and samples showing driver spill. Sampling happens only during `vram_status()`
calls that return selected-GPU readings, at most once per
`VRAM_MCP_SAMPLE_SECONDS` across sessions. `list_loaded()` does not collect GPU
readings or trend samples. Samples carry GPU scope, so changing
`VRAM_MCP_GPU_INDEX` does not mix devices in one trend. `VRAM_MCP_AUDIT=0`
stops new samples and appearance/disappearance events; it does not disable
current observations.

The response includes at most the 200 most recent raw samples and sets
`samples_truncated` when more exist. Summary figures cover the queried retained
samples, not just the returned raw subset; the implementation queries at most
10,000 samples. Event retention can shorten the available history. An empty
window is not evidence of stable memory, and a null latest reading is unknown.

See [configuration](configuration.md) for sampling, retention, and the effects
of disabling auditing.
