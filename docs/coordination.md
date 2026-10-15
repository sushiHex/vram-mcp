# Coordination and diagnostics

[Back to the README](../README.md)

## Shared state

Every connected client starts its own stdio server. Processes running under the
same OS user coordinate through `~/.cache/vram-mcp/`, or through
`VRAM_MCP_STATE_DIR` when it is set:

| File | Contents |
| --- | --- |
| `claims.json` | Model claims, capacity reservations, and bounded pending-operation records. |
| `events.jsonl` | Bounded audit history of actions, observed changes, and memory samples. |
| `last_seen.json` | Previous set of observed memory holders. |
| `last_sample.json` | Shared sampling throttle state. |

Ledger writes use file locks and atomic replacement. An unreadable or
wrong-shaped ledger is treated as unavailable, because silently replacing it
could discard another session's protection. `vram_status()` and `list_loaded()`
read the ledger on every call, loaded models or not, and report it under
`observations.coordination`. When it cannot be read, their `summary` names the
cause. An individual malformed operation
record is discarded while valid claims are retained, and the next ledger write
removes it permanently. Owners are descriptive labels, not authenticated
identities. Separate home directories have separate ledgers, even when their
processes use the same physical GPU.

### One generation per ledger

All clients sharing a ledger must be the same vram-mcp generation. Ownership is
an advisory lock on a persistent lock file; a pre-0.3 client instead treats the
lock file's *existence* as ownership, takes no advisory lock at all, and unlinks
any lock file older than 30 seconds. Two consequences follow, and neither can be
repaired from this side:

- A pre-0.3 client can enter while a current client holds the lock, because it
  never asks whether the lock is held. On POSIX it can also unlink the file we
  hold, then create its own under the same name.
- Even with perfect file serialization the generations are not semantically
  compatible: a pre-0.3 client has no concept of pending operations, so it will
  claim a model with an eviction in flight and reserve capacity during a pending
  warm. Preserving records it does not read cannot make it honour them.

So **stop every client before upgrading**, rather than mixing versions and
hoping. What the current client does provide is that such a peer cannot go
unnoticed: it verifies on both sides of every ledger operation that its lock
still governs the lock file's path, and reports `LockDisplacedError` when it does
not — before the operation if exclusion was already lost, and after it if the
work may have raced a concurrent writer. Affected tools return a structured
refusal naming the cause. Nothing in this generation removes a lock file, so in
a single-generation deployment the check never fires.

This is detection, not exclusion. A displaced lock means the ledger may already
have diverged; treat it as an incident and restart from a single generation.

### Protocol marker

Every write stamps the ledger with the coordination `protocol` it was made
under. A client refuses a ledger whose protocol is newer than its own, for reads
as well as writes, because the records may carry obligations it cannot see. The
refusal names both protocols, and the file is left untouched.

Changes that only add vocabulary stay within one protocol: a record carrying an
unfamiliar lifecycle is still honoured and reported as `unrecognized`. Only a
change that older clients could not honour safely raises the protocol number.

The marker protects only between clients that check it, which begins with
this one. Earlier clients ignore it, so its absence does not show who wrote a
ledger, and its presence does not rule out a pre-0.3 writer. A ledger without a
marker is read as protocol 1. The marker exists so that the next generation
boundary is detected, unlike this one.

Detection is not a rolling upgrade. A client mid-operation when a newer one
stamps the ledger cannot record that operation's outcome, so stop every client
before upgrading across a protocol change.

### Recovering a shared ledger

After a `LockDisplacedError` or a protocol refusal:

1. Stop every vram-mcp process. Each MCP client session runs its own server, so
   closing one session is not enough.
2. Install one vram-mcp version everywhere that shares the home directory. A
   protocol refusal persists until the ledger is reset or every client speaks
   the protocol recorded in it.
3. Decide whether to keep `claims.json`. It is plain JSON; keep it only if its
   claims match the work you know to be running. Otherwise rename it aside
   (for example to `claims.json.diverged`) rather than deleting it; the next
   client starts an empty ledger.
4. Restart one client and call `list_claims()`. It must succeed: that, not
   inspection by eye, is what establishes the ledger is usable. If it is
   refused, return to step 3.
5. If the old ledger held any operation, restart Ollama before trusting what
   `list_loaded()` reports. Stopping a client does not stop a request it already
   sent, and `pending_until` is a local lease, not a deadline Ollama observes.
6. Have agents re-claim the models and re-reserve the capacity they rely on
   before resuming warm, unload, or `ensure_free`.

Resetting the ledger discards its coordination promises: model claims,
capacity reservations, pending-operation records, and retry history. Loaded
models and audit history are unaffected. While every client is stopped, the
`claims.json.lock`, `claims.json.operation-*.lock`, and `claims.json.tmp` files
are safe to delete, and current clients recreate them as needed. Never delete
them while any client is running.

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
| `retry_count`, `retry_after` | `retry_count` is the length of the current run of attempts on this model that were abandoned without an outcome. A definite finish ends the run, as do 24 hours without a further abandonment. `retry_after` is the next safe retry time, or `null` while in flight. |

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

`thrashing` means the driver is paging VRAM. On Windows, a sampler reads the
System process's engine time on the selected adapter once a second. WDDM's video
memory manager runs as that process, and its engine time was high in nearly
every second while VRAM was overcommitted. The verdict reads `thrashing` when,
in the last 10 one-second samples, at least half reach 5% of an engine. The
`paging` field reports the window, or is `null` when paging was not read:

- `sustained`: whether that condition holds.
- `active_seconds`: how many samples in the window reached 5% of an engine.
- `seconds`: how many samples the window holds.
- `peak_percent`: the highest sample, as a percentage of one engine.

Loading a model or filling VRAM without overcommitting it makes a burst of two
to four seconds, which is too short to read as sustained. Treat the verdict as a
measured correlation, not a WDDM contract: the driver chooses the paging engine,
and page-table updates and fills also run as the System process. See
[acceptance](acceptance.md#paging-non-local-memory-and-why-linux-cannot-read-ok)
for the measurements.

Non-local memory is reported but decides no verdict. On Windows/WDDM, a
runner's non-local memory can include deliberately CPU-offloaded layers.
vram-mcp attributes that memory:

- `non_local_mb`: the selected adapter's measured total.
- `explained_offload_mb`: memory attributable to Ollama's deliberate CPU offload,
  capped per correlated runner at `total_size_mb - size_vram_mb`, and never more
  than the total.
- `unexplained_spill_mb`: the remainder, including other processes' non-local
  memory, any excess above a runner's explained offload, and memory no visible
  process accounts for. It is bounded by the measured total.

These figures are quantities, not symptoms. A desktop holds 150-490 MB of
non-local memory, and memory WDDM evicts to its backing store does not appear in
it. Non-local memory also includes allocations an application deliberately
places in system memory, such as upload buffers, and a runner's demoted VRAM, up
to its offload allowance, is indistinguishable from the offload itself.

On Windows, non-local memory comes from the selected GPU's own WDDM adapter, so
other GPUs' memory never counts. Linux has no paging reading, so `paging` is
`null` there and `pressure.coverage.paging` is `false`
([#27](https://github.com/sushiHex/vram-mcp/issues/27)); the verdict reads
`unknown` where it would read `ok`. When `pressure.coverage.non_local_memory` is
`false`, the non-local figures are `null`, and the process observation's
`gaps.non_local_memory` says why. A full GPU without paging reads `tight` when
less than 1 GiB is free. A partially CPU-offloaded model can still read
`degraded` without evidence of driver paging.

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
and samples that read `thrashing`. Sampling happens only during `vram_status()`
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
