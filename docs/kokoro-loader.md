# Optional owned Kokoro worker

This Windows-only capability launches one reviewed Hermes Kokoro CUDA worker
through vram-mcp. It is disabled until an operator sets
`VRAM_MCP_KOKORO_PROFILE` to a private JSON file before starting a **new scoped
server**. Installing or merging this code does not reload an existing server,
enable CUDA, or establish a GPU performance result. Other platforms refuse
before starting a worker.

The profile has exactly these version-1 keys:

| Fields | Meaning |
| --- | --- |
| `version` | Integer `1`. |
| `python`, `python_sha256` | Absolute native base CPython executable and its exact file digest. |
| `venv_python`, `venv_python_sha256` | Absolute venv launcher and its exact file digest. The launcher is **not** executed for worker custody. |
| `worker`, `worker_sha256` | Absolute reviewed `kokoro_worker.py` and its exact file digest. |
| `model`, `model_sha256`, `voices`, `voices_sha256` | Absolute provisioned Kokoro model/voices files and exact digests. No downloads occur. |
| `gpu_index`, `gpu_uuid` | Nonnegative integer index and full physical GPU UUID, matching the fresh selected-GPU reading. |
| `reserve_gb`, `headroom_mb` | Positive estimated footprint up to 128 GiB and integer headroom of at least 1024 MiB. These are admission inputs, not a hard allocation cap or measured peak. |
| `voice`, `language` | Currently `bf_isabella` and `en-gb`. |

Only the reviewed Hermes worker at
[`34cbd4c34e6759134762748812353c9af2863563`](https://github.com/sushiHex/hermes-realtime/blob/34cbd4c34e6759134762748812353c9af2863563/src/hermes_realtime/providers/kokoro_worker.py)
is supported, with LF source SHA-256
`9667176d0a132490f95caf35b592aca4254fd8a7e70f648899bc6100b03d3a35`.
Its Windows parent-handle watchdog starts before CUDA initialization. The
reviewed-source comparison accepts only LF/CRLF mechanical equivalence; the
profile's independent raw digest must still match the exact on-disk file. New
worker source requires an explicitly reviewed supported-source pin update, not
an operator bypass. The model/voice pins are the worker's reviewed full v1.0
English graph and voice set, also fixed in `kokoro_loader.py`.

The base executable is probed with `-I -B -S`, stdlib only, using
`__PYVENV_LAUNCHER__` to preserve the selected venv. The actual Windows executable
image, base executable and reported venv executable must match the profile.
This refuses a venv redirector as the owned process: terminating its wrapper
would not establish that the real CUDA child exited. Normal worker startup uses
the verified base image with `-I -B` and the same venv binding. The operator owns
the integrity of that interpreter environment and trusted profile files.

## Lifecycle

1. The trusted Hermes client creates an exclusive loopback listener and private
   one-use authentication token. It calls `load_kokoro(owner, request_id,
   connect_port, auth_token)`; no command, executable path, model path or arbitrary
   environment is accepted from tool callers.
2. The server creates its capacity reservation, begins the existing same-device
   warm-admission operation, then reads fresh GPU capacity and all active
   reservations. Unknown readings, device mismatch or insufficient remaining
   headroom refuse without launching. It does not evict existing workloads or
   offer `force`. The startup operation prevents competing cooperative loads
   during admission; the reservation remains after startup without blocking all
   other models for the service lifetime.
3. The server starts and owns the actual worker process. The curated child
   environment contains the token, venv binding, and
   `CUDA_VISIBLE_DEVICES=<admitted full GPU UUID>`, excluding unrelated provider
   credentials. NVIDIA documents that
   [UUID visibility controls device enumeration](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html):
   the worker's CUDA device zero is the single admitted visible device.
4. A successful result means **`launch_accepted`**, with `gpu_ready: false`.
   Hermes must independently validate the existing authenticated ready message,
   asset hashes, CUDA-first provider list and profiled operation placement under
   its startup deadline. A failed ready handshake must stop this operation;
   there is no direct-launch fallback. Retry an uncertain response using the
   same request identity and identical private arguments; a different request
   cannot start a second worker while the first remains live or unresolved.
5. `kokoro_status(owner, operation_id)` and `stop_kokoro(owner, operation_id)`
   act only on the opaque handle owned by this server and matching owner label.
   Close the Hermes socket before stopping for normal worker exit. Otherwise,
   stop terminates and then kills only its actual owned process if necessary,
   waiting for exit before releasing capacity. Cleanup uncertainty stays
   explicit and retains the reservation; no new worker is accepted. Active
   reservations renew while the child lives. Failed renewal triggers settlement.
   Canceled MCP loads settle their matching request. Normal server shutdown
   settles its worker; unexpected server death is covered by the pinned worker's
   pre-load parent-handle watchdog.

A worker-spawn exception is conservatively `unknown`: absence of a returned
process handle does not prove that no child was created. Its capacity record
continues renewing and retries cannot launch again. Without a process witness,
status/stop cannot claim settlement; end that scoped server so the pinned
parent watchdog can terminate any such child. The unknown result remains
unqualified, rather than being rewritten as successful cleanup. The capacity
reservation expires by its existing ledger TTL after the server ends. Unsettled
startup also retains the existing OS-backed warm-admission lock, so even a lost
or expired reservation cannot admit competing cooperative loads while this
server still owns the operation. Known successful launch rechecks its reservation
before returning acceptance and releases the warm lock for normal service use.

The token, request arguments, arbitrary exception details and private profile
paths are neither returned nor placed in vram-mcp's action audit. Keep private
MCP control traffic out of public evidence. Operation results expose only fixed
categories and an opaque operation handle; they never establish CUDA readiness,
desktop usability, audible quality, or an attributable exact VRAM peak.

The existing coordination model remains cooperative: other processes and force
overrides can bypass reservations. A failed OS or ledger operation is reported
as unknown, not as successful cleanup. This feature does not replace broader
GPU resource qualification. Before a real comparison, require exact-code review,
the actual configured environment, verified provider/device placement, preserved
personal workloads, and complete owned cleanup.
