"""FastMCP server exposing VRAM-management tools.

This is the only module that imports ``mcp``. It wires the pure logic in
:mod:`vram_mcp.core` / :mod:`vram_mcp.gpu` / :mod:`vram_mcp.ollama` /
:mod:`vram_mcp.nvml` / :mod:`vram_mcp.ollama_correlate` /
:mod:`vram_mcp.claims` to MCP tools.

Every tool is ``async def`` and runs its blocking body (subprocess spawns,
NVML sessions, HTTP calls, file locks) in a worker thread via
``anyio.to_thread.run_sync`` — the installed FastMCP invokes sync tools
directly on the asyncio event loop, so a plain ``def`` tool would block the
whole server (pings included) for the duration of every nvidia-smi/wmic call.
"""

from __future__ import annotations

import functools
import os
import math
import re

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, get_args, get_origin, get_type_hints

import anyio.to_thread
from mcp.server.fastmcp import FastMCP

from . import claims as _claims
from . import core
from . import nvml as _nvml
from ._util import iso
from .gpu import observe_gpu
from .models import canonical_model
from .observations import Observation
from .validation import nonblank_text
from .ollama import OllamaClient
from .ollama_correlate import runner_pid_map
from .schemas import (
    ClaimResult, EnsureFreeResult, ListClaimsResult, ReleaseResult,
    RenewResult, ReserveResult, ResidencyResult,
)

mcp = FastMCP("vram-mcp")

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
_ollama = OllamaClient(base_url=_OLLAMA_BASE_URL)

from . import audit as _audit
from . import procinfo as _procinfo

def _numeric_env(name: str, default, *, whole: bool = False, allow_zero: bool = False):
    try:
        value = (int if whole else float)(os.environ.get(name, str(default)))
        if not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError
        return value
    except (ValueError, OverflowError) as exc:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} {'integer' if whole else 'number'}") from exc


_AUDIT_ON = os.environ.get("VRAM_MCP_AUDIT", "1") != "0"
_MEANINGFUL_MB = _numeric_env("VRAM_MCP_MEANINGFUL_MB", 512, whole=True)
_EVENT_CAP = _numeric_env("VRAM_MCP_EVENT_CAP", 5000, whole=True)
_SAMPLE_SECONDS = _numeric_env("VRAM_MCP_SAMPLE_SECONDS", 60)
_SPILL_MB = _numeric_env("VRAM_MCP_SPILL_MB", core.SPILL_THRESHOLD_MB, whole=True)
# trend() output lands in an agent's context window, so the raw rows are capped
# even though the SUMMARY is always computed over every row in the window.
_TREND_SAMPLE_CAP = 200
_GPU_INDEX = _numeric_env("VRAM_MCP_GPU_INDEX", 0, whole=True, allow_zero=True)


def _gpu_reading():
    return observe_gpu(_GPU_INDEX)


def _procinfo_table():
    """Collect the configured device regardless of audit settings."""
    return _procinfo.observe_processes(_GPU_INDEX)


def _run_detection(status: dict) -> None:
    """Audit consumes successful observations; unavailable sources keep their baseline."""
    if not _AUDIT_ON:
        return
    readings = status.get("observations", {})
    for key, kind, values in (("ollama", "ollama", status.get("loaded")),
                              ("processes", "process", status.get("other_processes"))):
        reading = readings.get(key)
        if values is None or reading is None or reading["status"] != "available":
            continue
        if kind == "process" and not all(reading.get("coverage", {}).get(key, False)
                                         for key in ("compute_processes", "graphics_processes")):
            continue
        try:
            holders = _audit.meaningful_holders(
                values if kind == "ollama" else [], values if kind == "process" else [], _MEANINGFUL_MB)
            _audit.detect_and_log(holders, observed_kinds={kind},
                                  observed_at=reading["observed_at"], scope=reading["scope"], cap=_EVENT_CAP)
        except Exception:
            pass
    gpus = status.get("gpus") or []
    if not gpus:
        return
    try:
        pressure = status.get("pressure") or {}
        gpu = gpus[0]
        _audit.maybe_log_sample({
            "scope": status.get("scope", f"gpu:index={_GPU_INDEX}"),
            "device_uuid": gpu.get("uuid"),
            "observed_at": readings.get("gpu", {}).get("observed_at"),
            "free_mb": pressure.get("free_mb"), "used_mb": gpu.get("used_mb"),
            "total_mb": gpu.get("total_mb"), "spill_mb": pressure.get("unexplained_spill_mb"),
            "state": pressure.get("state"), "coverage": pressure.get("coverage"),
            "loaded_count": len(status["loaded"]) if status.get("loaded") is not None else None,
        }, interval_seconds=_SAMPLE_SECONDS, cap=_EVENT_CAP)
    except Exception:
        pass


def _fmt_free(free_mb) -> str:
    return "unknown (nvidia-smi unavailable)" if free_mb is None else f"{free_mb} MB"


def _stable_claim_entries(records) -> list[dict] | None:
    """Normalize legacy claim records before exposing them through MCP."""
    if records is None:
        return None
    entries = []
    for record in records or []:
        entry = _claims._claim_view(record) if isinstance(record, dict) else None
        if entry is not None:
            entries.append(entry)
    return entries


def _as_boundary(value) -> str | None:
    """A retry boundary is a non-blank ISO timestamp, or it is absent.

    Ledger records are shared across versions and processes, so a boundary can
    arrive as any JSON type. Narrowing here keeps a bad one out of ``max()``
    and out of a caller's retry arithmetic, rather than letting it become a
    time that sorts strangely or crashes the comparison.
    """
    return value if isinstance(value, str) and value.strip() else None


def _empty_for(annotation) -> Any:
    """The empty value a declared field type implies.

    Optionality is tested first, so ``list[X] | None`` fills as ``null`` rather
    than ``[]``: "this field does not apply here" and "it applies and is empty"
    are different answers, and callers branch on the difference.
    """
    if type(None) in get_args(annotation):
        return None
    return {bool: False, int: 0, float: 0.0, str: "",
            list: [], dict: {}}.get(get_origin(annotation) or annotation)


def _coordination_result(schema: type, result: dict | None = None, **overrides) -> dict:
    """Project a tool's raw result onto exactly the fields ``schema`` declares.

    The schema is the single source of truth for the shape. Every declared
    field is present — filled from the producer, an override, or the empty
    value its type implies — so a field can never be added to a schema and
    forgotten here, and two sibling schemas cannot silently drift apart.

    The reverse direction matters just as much and is not enforceable from
    inside this function: FastMCP validates against the schema and DROPS any
    key it does not declare, silently. A producer that emits an undeclared
    field therefore loses it in transit. ``tests/test_schema_contract.py``
    asserts the two key sets are equal, not merely compatible, so that class
    of loss fails a test instead of reaching an agent.

    An override of ``None`` defers to whatever the producer supplied; a
    non-``None`` override wins, since the caller knows the operation it just
    performed better than the dict it is normalizing.
    """
    normalized = dict(result or {})
    for key, value in overrides.items():
        if value is not None or key not in normalized:
            normalized[key] = value
    normalized.setdefault("ok", False)
    normalized.setdefault("outcome", "succeeded" if normalized["ok"] else "refused")
    if normalized["outcome"] == "unknown":
        # An unverified outcome must always say why it could not be confirmed.
        normalized["reason"] = normalized.get("reason") or "outcome_unknown"
    if normalized.get("pending_until") and not normalized.get("retry_after"):
        # A stated boundary is a retry time whichever outcome produced it, so
        # "when may I try again" has one answer at the top level rather than
        # living in `operations[]` for refusals and here for unknowns.
        normalized["retry_after"] = normalized["pending_until"]
    for field in ("claims", "reservations"):
        if field in normalized:
            normalized[field] = _stable_claim_entries(normalized[field])
    for field, annotation in get_type_hints(schema).items():
        if field not in normalized:
            normalized[field] = _empty_for(annotation)
    return normalized


def _snapshot() -> core.Snapshot:
    """Capture protection for one status view or one eviction decision."""
    return core.Snapshot.capture(
        _claims.list_claims, runner_pid_map,
        lambda pids: _nvml.nvml_busy_map(pids, index=_GPU_INDEX),
    )


def _active_claims() -> tuple[list, bool]:
    """Active records plus explicit health for admission and status."""
    try:
        return _claims.list_claims(), True
    except (TimeoutError, OSError, ValueError):
        return [], False


def _full_status() -> dict:
    """The enriched combined status both status tools share."""
    return core.combined_status(
        _gpu_reading, _ollama,
        snapshot_fn=_snapshot,
        procinfo_fn=_procinfo_table,
        spill_threshold_mb=_SPILL_MB,
    )


async def _in_thread(fn, *args, **kwargs):
    """Run a blocking tool body off the event loop."""
    return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))


# ── status tools ─────────────────────────────────────────────────────────────

def _vram_status_impl() -> dict:
    status = _full_status()
    _run_detection(status)
    n_gpu = len(status["gpus"])
    n_loaded = len(status["loaded"]) if status["loaded"] is not None else "unknown"
    p = status.get("pressure") or {}
    state = p.get("state", "unknown")
    # The pressure detail is the actionable half ("X MB has spilled to system
    # RAM"), so surface it inline rather than making the caller dig into the
    # payload — but only when there's something to say, and only via .get() so
    # a partial pressure dict can never turn a status call into an exception.
    detail = p.get("detail")
    status["summary"] = (
        f"{n_gpu} GPU(s), {n_loaded} model(s) loaded, "
        f"free: {_fmt_free(status['free_mb'])}, pressure: {state}."
        # Only `ok` suppresses the detail: "No VRAM pressure detected." adds
        # nothing to "pressure: ok". Every other state, `unknown` included,
        # carries the reason for the verdict — and `unknown` is precisely where
        # a reader cannot infer it, since which evidence was missing is the
        # whole content of that answer.
        + (f" {detail}" if detail and state != "ok" else "")
    )
    return status


@mcp.tool()
async def vram_status() -> dict:
    """Report the selected GPU and server-wide Ollama model residency.

    Includes device identity, source health/timestamps/coverage, claims, recent
    GPU activity, and pressure (ok|tight|degraded|thrashing|unknown). Unavailable
    residency is null; unavailable telemetry never establishes an empty or
    healthy GPU. Non-local memory pressure is a best-effort paging heuristic.
    """
    return await _in_thread(_vram_status_impl)


def _list_loaded_impl() -> dict:
    status = core.combined_status(
        lambda: Observation(None, "gpu", error="Not collected", scope=f"gpu:index={_GPU_INDEX}"),
        _ollama, snapshot_fn=_snapshot, spill_threshold_mb=_SPILL_MB,
    )
    _run_detection(status)
    loaded = status["loaded"]
    return {"loaded": loaded, "observations": status["observations"],
            "summary": f"{len(loaded)} model(s) loaded." if loaded is not None
            else "Ollama residency is unavailable; loaded models are unknown."}


@mcp.tool()
async def list_loaded() -> dict:
    """List the models currently resident in VRAM, with claim/busy detail."""
    return await _in_thread(_list_loaded_impl)


# ── eviction tools ───────────────────────────────────────────────────────────

def _action_result(
    action: str, model: str, by: str, force: bool, result: dict, *,
    operation_id: str | None = None, pending_until: str | None = None,
    retry_count: int = 0,
) -> dict:
    result = _coordination_result(
        ResidencyResult, result, model=model, operation_id=operation_id,
        pending_until=pending_until, retry_count=retry_count,
    )
    # The schema fills an absent summary with "", so test truthiness rather
    # than presence — an empty summary is as useless to a caller as none.
    if not result["summary"]:
        result["summary"] = (f"{action.capitalize()} '{model}': {result['outcome']}. "
                             + (result.get("detail") or result.get("reason") or ""))
    if _AUDIT_ON:
        _audit.log_action(action=action, target=model, kind="ollama", actor=by,
                          force=force, outcome=result["outcome"],
                          detail=result.get("detail") or result.get("reason") or "",
                          scope=_ollama.base_url, cap=_EVENT_CAP)
    return result


def _mutate(model: str, kind: str, force: bool, by: str, perform) -> dict:
    """Hold an operation intent through decision, request and reconciliation."""
    try:
        model = canonical_model(model)
        by = nonblank_text(by, "by")
        slot = _claims.begin_operation(model, kind, force, scope=f"gpu:index={_GPU_INDEX}")
        if not slot["ok"]:
            # Lift the blocking lease's boundary to the top level; the caller
            # asked when it may retry, not which record happens to hold it.
            blocking = slot.get("operations") or []
            return _action_result(
                kind, model, by, force, slot,
                pending_until=max(
                    filter(None, (_as_boundary(op.get("pending_until"))
                                  for op in blocking)),
                    default=None),
            )
        operation_id = slot["operation_id"]
        result = {"ok": False, "outcome": "unknown", "detail": "Operation interrupted"}
        result_reason: Optional[str] = None
        try:
            result = perform(model)
            reason_value = result.get("reason")
            result_reason = reason_value if isinstance(reason_value, str) else None
        except (ValueError, OSError, TimeoutError) as exc:
            result = {"ok": False, "outcome": "refused", "detail": str(exc)}
        finally:
            try:
                finished = _claims.finish_operation(
                    operation_id, uncertain=result["outcome"] == "unknown",
                    reason=result_reason,
                )
                slot["expires_at"] = finished.get("expires_at", slot["expires_at"])
            except (ValueError, OSError, TimeoutError) as exc:
                result["coordination_warning"] = f"Operation slot cleanup failed: {exc}"
        if result["outcome"] == "unknown":
            result["pending_until"] = slot["expires_at"]
            result["retry_after"] = slot["expires_at"]
        return _action_result(
            kind, model, by, force, result, operation_id=operation_id,
            pending_until=_as_boundary(result.get("pending_until")),
            retry_count=slot["retry_count"],
        )
    except (ValueError, OSError, TimeoutError) as exc:
        return _coordination_result(ResidencyResult, {
            "ok": False, "outcome": "refused", "reason": "coordination_error",
            "detail": str(exc), "summary": f"{kind.capitalize()} refused: {exc}",
        }, model=model)


def _unload_impl(model: str, force: bool, by: str, *, action: str = "unload") -> dict:
    def perform(name):
        if not force:
            protected, detail = core.is_protected(name, _snapshot())
            if protected:
                return {"ok": False, "outcome": "refused", "protected": True,
                        **detail, "detail": "Model is claimed or recently busy; force=True overrides protection"}
        return _ollama.change_residency(name, 0, resident=False)
    return _mutate(model, action, force, by, perform)


@mcp.tool()
async def unload(model: str, force: bool = False, by: str = "unknown") -> ResidencyResult:
    """Evict a single model from VRAM now (Ollama ``keep_alive=0``).

    Refuses by default if ``model`` has an active claim or a best-effort busy
    signal — pass ``force=True`` to override (busy is windowed and can lag a few
    seconds past a generation). ``by`` records who requested the eviction in the
    audit log (see ``history``)."""
    return await _in_thread(_unload_impl, model, force, by)


def _ensure_free_impl(gb: float, force: bool, by: str) -> dict:
    try:
        nonblank_text(by, "by")
        result = core.ensure_free(
            gb, _gpu_reading, _ollama, settle=0.5, force=force,
            evict_fn=lambda name: _unload_impl(name, force, by, action="ensure_free"),
        )
    except (ValueError, OSError, TimeoutError) as exc:
        return _ensure_free_result({
            "ok": False, "outcome": "refused", "reason": "coordination_error",
            "detail": str(exc), "summary": f"Ensure free refused: {exc}",
        })
    active, ledger_ok = _active_claims()
    result["reserved_mb"] = core.reserved_mb(active) if ledger_ok else None
    result["summary"] = (
        f"Target {gb} GB: {result['outcome']}; {_fmt_free(result['free_mb'])} free. "
        f"Unloaded: {', '.join(result['unloaded']) or 'none'}."
    )
    if result.get("detail"):
        result["summary"] += " " + result["detail"] + "."
    if result["declined"]:
        result["summary"] += " Skipped: " + ", ".join(d["name"] for d in result["declined"]) + "."
    if result["reserved_mb"] is None:
        result["summary"] += " Reserved capacity is unknown."
    elif result["reserved_mb"]:
        result["summary"] += f" {result['reserved_mb']} MB is reserved by other sessions."
    return _ensure_free_result(result)


def _ensure_free_result(result: dict) -> dict:
    """Normalize ensure_free, projecting each eviction attempt onto its own schema.

    Every scalar and collection ensure_free reports is declared on
    ``EnsureFreeResult``, so the projection supplies them; ``attempts`` needs
    naming because its entries are ``ResidencyResult`` and must be projected
    against that shape rather than this one.

    The batch's retry boundary is the latest one blocking any of its attempts.
    ensure_free begins no operation of its own, so without lifting it the top
    level would report "nothing pending" while an eviction inside it was
    blocked until a lease lapsed — the caller would be told to retry
    immediately, and be refused again.
    """
    attempts = [
        _coordination_result(ResidencyResult, attempt, model=attempt.get("model"))
        for attempt in result.get("attempts", []) if isinstance(attempt, dict)
    ]
    normalized = _coordination_result(
        EnsureFreeResult, result,
        pending_until=max(filter(None, (_as_boundary(attempt["pending_until"])
                                        for attempt in attempts)), default=None),
    )
    normalized["attempts"] = attempts
    return normalized


@mcp.tool()
async def ensure_free(gb: float, force: bool = False, by: str = "unknown") -> EnsureFreeResult:
    """Free VRAM until at least ``gb`` GB is available. Skips claimed/busy models
    unless ``force=True``. ``by`` records the requester in the audit log.

    Also reports ``reserved_mb`` — how much of the resulting free VRAM other
    sessions have reserved for non-Ollama work (``None`` if unreadable)."""
    return await _in_thread(_ensure_free_impl, gb, force, by)


def _warm_impl(model: str, keep_alive: str, by: str, force: bool) -> dict:
    # A zero duration is an eviction and must go through unload protection.
    if not isinstance(keep_alive, str) or not re.fullmatch(r"(?:-1|(?:[0-9]+(?:\.[0-9]+)?(?:ns|us|ms|s|m|h))+)", keep_alive) or not any(c in "123456789" for c in keep_alive):
        return _coordination_result(ResidencyResult, {
            "ok": False, "outcome": "refused", "reason": "invalid_keep_alive",
            "detail": "keep_alive must be a positive duration (e.g. 5m) or -1; use unload() to evict",
            "summary": "keep_alive must be a positive duration (e.g. 5m) or -1; use unload() to evict",
        }, model=model)

    def perform(name):
        admission = None
        observations = {}
        if not force:
            status = core.combined_status(_gpu_reading, _ollama)
            observations.update(status["observations"])
            active, ledger_ok = _active_claims()
            if not ledger_ok or status["loaded"] is None:
                return {"ok": False, "outcome": "refused", "reason": "observation_unavailable",
                        "observations": observations,
                        "detail": "Cannot check residency or reservations; retry after the source recovers"}
            resident = any(canonical_model(row["name"]) == name for row in status["loaded"])
            sizes = None
            if not resident:
                sizes = _ollama.observe_tags()
                observations["sizes"] = sizes.metadata()
            reserved = core.reserved_mb(active)
            allowed, admission = core.can_warm(
                name, free_mb=status["free_mb"], reserved_mb=reserved,
                model_size_mb=sizes.data.get(name) if sizes is not None and sizes.known else None,
                resident=resident,
            )
            if not allowed:
                return {"ok": False, "outcome": "refused", "refused": True, **admission,
                        "observations": observations,
                        "reservations": [r for r in active if r.get("kind") == "reservation"],
                        "detail": f"{reserved} MB is reserved; insufficient headroom. force=True overrides reservations"}
        result = _ollama.change_residency(name, keep_alive, resident=True)
        result["keep_alive"] = keep_alive
        result["observations"] = {**observations, **result.get("observations", {})}
        if admission is not None:
            result.update(admission)
            result["size_verified"] = admission["reason"] == "fits"
            if admission["reason"] in ("size_unknown", "free_unknown"):
                result["detail"] += f"; estimated fit not verified ({admission['reason']}, {admission['reserved_mb']} MB reserved)"
        return result
    return _mutate(model, "warm", force, by, perform)


@mcp.tool()
async def warm(model: str, keep_alive: str = "5m", by: str = "unknown",
               force: bool = False) -> ResidencyResult:
    """Load a model or refresh its keep-alive (e.g. "5m"; "-1" pins indefinitely).

    An already resident model requires zero additional capacity. New loads
    account for reservations using an approximate model size; size_verified
    describes that estimate, while outcome describes verified residency.
    Returns succeeded|refused|failed|unknown. On unknown, inspect residency and
    pending_until before retrying. force=True bypasses admission checks but
    cannot override a pending operation. Zero durations must use unload().
    """
    return await _in_thread(_warm_impl, model, keep_alive, by, force)


# ── claim tools ──────────────────────────────────────────────────────────────
# The ledger can raise on genuinely-contended/odd filesystem states
# (TimeoutError from a live lock held past the wait window; OSError if the
# Windows sharing-violation retry in claims._save is exhausted). Those are
# operational outcomes, not bugs — surface them as structured {ok: false}
# responses instead of raw tracebacks.

def _ledger_call(verb: str, fn, *args) -> dict:
    """Run a ledger write with the shared failure policy in one place.

    Returns ``{"result": <fn's return>}`` on success, or ``{"error": {ok:
    false, summary}}`` on an OPERATIONAL failure (live-lock timeout,
    exhausted Windows sharing-violation retries) so every claim tool
    degrades identically instead of surfacing a raw traceback.

    ValueError joins that list: the ledger rejects arguments it must not
    persist (``reserve(gb=-8)``), and an MCP caller deserves the same readable
    ``{ok: false, summary}`` for a bad argument as for a busy lock."""
    try:
        return {"result": fn(*args)}
    except (TimeoutError, OSError, ValueError) as e:
        return {"error": {"ok": False, "summary": f"{verb} failed: {e}"}}


def _claim_impl(model: str, owner: str, purpose: str, ttl_seconds: int) -> dict:
    outcome = _ledger_call("Claim", _claims.claim, model, owner, purpose, ttl_seconds)
    if "error" in outcome:
        return {**outcome["error"], "outcome": "refused", "model": None,
                "reason": "ledger_error", "detail": outcome["error"]["summary"],
                "claim_id": None, "expires_at": None}
    result = outcome["result"]
    result = {**result, "ok": True, "outcome": "succeeded", "reason": None,
              "detail": None}
    model = result["model"]
    result["summary"] = (
        f"Claimed '{model}' for {owner} ({purpose}), expires {result['expires_at']}."
    )
    return {**result, "summary": result["summary"]}


@mcp.tool()
async def claim(model: str, owner: str, purpose: str, ttl_seconds: int = 3600) -> ClaimResult:
    """Declare that you're using ``model`` for ``purpose``.

    Lets other sessions see who's using a model and why before deciding to
    evict it. Renew before ``ttl_seconds`` elapses if still in use — an
    un-renewed claim simply expires, so a crashed session never leaves a
    permanently-stuck claim.
    """
    return await _in_thread(_claim_impl, model, owner, purpose, ttl_seconds)


def _reserve_impl(gb: float, owner: str, purpose: str, ttl_seconds: int,
                  pid: Optional[int]) -> dict:
    # _ledger_call applies fn(*args) positionally, but reserve's `pid` is
    # keyword-only — so hand it a zero-arg lambda instead of the bare function.
    outcome = _ledger_call(
        "Reserve",
        lambda: _claims.reserve(gb, owner, purpose, ttl_seconds, pid=pid),
    )
    if "error" in outcome:
        return {**outcome["error"], "outcome": "refused", "model": None,
                "reason": "ledger_error", "detail": outcome["error"]["summary"],
                "claim_id": None, "expires_at": None, "gb": None}
    result = outcome["result"]
    result.update({"ok": True, "outcome": "succeeded", "model": None,
                   "reason": None, "detail": None, "gb": gb})
    result["summary"] = (
        f"Reserved {gb} GB for {owner} ({purpose}), expires "
        f"{result['expires_at']}. Other sessions' warm() calls will be refused "
        "when this reservation leaves no headroom."
    )
    return result


@mcp.tool()
async def reserve(gb: float, owner: str, purpose: str, ttl_seconds: int = 3600,
                  pid: Optional[int] = None) -> ReserveResult:
    """Reserve ``gb`` GB of VRAM — a claim on capacity, not on a named model.

    Use this for non-Ollama GPU work (a training run, a diffusion job) so other
    sessions can see the VRAM is spoken for. ``pid`` is advisory. Reservations
    expire by TTL like claims, so a crashed session never leaves one stuck.

    COOPERATIVE: this gates vram-mcp's own ``warm()``, but vram-mcp cannot
    intercept an Ollama auto-load triggered by a direct /api/generate call
    from another process.
    """
    return await _in_thread(_reserve_impl, gb, owner, purpose, ttl_seconds, pid)


def _renew_impl(claim_id: str, ttl_seconds: Optional[int]) -> dict:
    outcome = _ledger_call("Renew", _claims.renew, claim_id, ttl_seconds)
    if "error" in outcome:
        return {**outcome["error"], "outcome": "refused", "model": None,
                "reason": "ledger_error", "detail": outcome["error"]["summary"],
                "claim_id": claim_id, "expires_at": None}
    result = outcome["result"]
    result.update({"claim_id": claim_id,
                   "expires_at": result.get("expires_at"),
                   "outcome": "succeeded" if result["ok"] else "refused",
                   "model": None,
                   "reason": None if result["ok"] else "claim_not_active",
                   "detail": None})
    result["summary"] = (
        f"Renewed, expires {result['expires_at']}." if result["ok"]
        else "No such claim (already expired or released?)."
    )
    return result


@mcp.tool()
async def renew(claim_id: str, ttl_seconds: Optional[int] = None) -> RenewResult:
    """Extend an existing claim's expiry before it lapses."""
    return await _in_thread(_renew_impl, claim_id, ttl_seconds)


def _release_impl(claim_id: str) -> dict:
    outcome = _ledger_call("Release", _claims.release, claim_id)
    if "error" in outcome:
        return {**outcome["error"], "outcome": "refused", "model": None,
                "reason": "ledger_error", "detail": outcome["error"]["summary"],
                "claim_id": claim_id}
    result = outcome["result"]
    result.update({"claim_id": claim_id,
                   "outcome": "succeeded" if result["ok"] else "refused",
                   "model": None,
                   "reason": None if result["ok"] else "claim_not_found",
                   "detail": None})
    result["summary"] = "Released." if result["ok"] else "No such claim."
    return result


@mcp.tool()
async def release(claim_id: str) -> ReleaseResult:
    """Release a claim early, before its TTL would expire."""
    return await _in_thread(_release_impl, claim_id)


def _list_claims_impl(model: Optional[str]) -> dict:
    outcome = _ledger_call("List claims", _claims.list_coordination, model)
    if "error" in outcome:
        return {"ok": False, "outcome": "refused", "claims": [], "operations": [],
                "summary": outcome["error"]["summary"]}
    state = outcome["result"]
    return {"ok": True, "outcome": "succeeded", **state,
            "summary": f"{len(state['claims'])} active claim(s), "
                       f"{len(state['operations'])} pending operation(s)."}


@mcp.tool()
async def list_claims(model: Optional[str] = None) -> ListClaimsResult:
    """See who's claiming what right now (all models, or one)."""
    return await _in_thread(_list_claims_impl, model)


# ── advice ───────────────────────────────────────────────────────────────────

def _advise_impl() -> dict:
    result = core.advise(_gpu_reading, _ollama)
    n = len(result["suggestions"])
    result["summary"] = (
        ("No VRAM issues detected." if result["known"] else "Telemetry is incomplete; health is unknown.")
        if n == 0 else f"{n} suggestion(s)."
    )
    return result


@mcp.tool()
async def advise() -> dict:
    """Suggest env/config changes to keep VRAM healthy (heuristics)."""
    return await _in_thread(_advise_impl)


# ── audit trail ──────────────────────────────────────────────────────────────

def _history_impl(model, type_, limit, since) -> dict:
    try:
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 5000:
            raise ValueError("limit must be between 1 and 5000")
        model = canonical_model(model) if model is not None else None
        events = _audit.read_events(model=model, type=type_, limit=limit,
                                    since=since, path=_audit.DEFAULT_EVENTS_PATH)
    except ValueError as exc:
        return {"ok": False, "outcome": "refused", "summary": str(exc)}
    return {"events": events, "summary": f"{len(events)} event(s)."}


@mcp.tool()
async def history(model: str | None = None, type: str | None = None,
                  limit: int = 50, since: str | None = None) -> dict:
    """The VRAM audit trail, newest first: who ran unload/ensure_free/warm, and
    which models/processes appeared or disappeared (with a best-effort cause).
    Filter by ``model``, ``type`` (action|disappeared|appeared), ``limit``, or an
    ISO ``since`` floor. Answers 'what happened to model X?'."""
    return await _in_thread(_history_impl, model, type, limit, since)


def _trend_impl(hours: float) -> dict:
    if not isinstance(hours, (int, float)) or not math.isfinite(hours) or not 0 < hours <= 876000:
        return {"ok": False, "outcome": "refused", "summary": "hours must be positive, finite, and at most 876000"}
    since = iso(datetime.now(timezone.utc) - timedelta(hours=hours))
    rows = _audit.read_events(type="sample", limit=10_000, since=since, scope=f"gpu:index={_GPU_INDEX}")
    rows.reverse()  # read_events is newest-first; the summarizer needs oldest-first
    summary = _audit.summarize_samples(rows)
    if summary["count"] == 0:
        # Name every reason a window can be empty, not just the two knobs: a
        # status call with no GPU rows is never sampled either (see
        # _run_detection), which is the permanent state of a machine whose
        # nvidia-smi doesn't work — a user told to check the throttle and
        # VRAM_MCP_AUDIT would be chasing two causes that aren't theirs.
        text = (
            f"No VRAM samples in the last {hours}h. Samples are recorded only on "
            f"status calls that returned GPU readings, at most once per "
            f"{_SAMPLE_SECONDS:g}s (VRAM_MCP_SAMPLE_SECONDS). So: no status call "
            "in that window, sampling turned off by VRAM_MCP_AUDIT=0, or no "
            "usable GPU reading to record (nvidia-smi missing or failing)."
        )
    else:
        # "now unknown" rather than a stale number: latest_free_mb describes the
        # NEWEST sample, which may carry no reading at all.
        latest = summary["latest_free_mb"]
        now_text = "unknown" if latest is None else f"{latest} MB"
        text = (
            f"{summary['count']} sample(s) over {hours}h: free VRAM is "
            f"{summary['direction']} (min {summary['min_free_mb']} MB, "
            f"max {summary['max_free_mb']} MB, now {now_text}); "
            f"{summary['thrashing_samples']} sample(s) showed VRAM spilling "
            "to system RAM."
        )
    # The summary above already covers EVERY row; only the raw rows are capped,
    # and the caller is told so rather than silently handed a subset.
    truncated = len(rows) > _TREND_SAMPLE_CAP
    if truncated:
        text += (
            f" Showing the {_TREND_SAMPLE_CAP} most recent raw samples of "
            f"{len(rows)} (the figures above cover all of them)."
        )
    return {**summary, "hours": hours, "samples": rows[-_TREND_SAMPLE_CAP:],
            "samples_truncated": truncated, "summary": text}


@mcp.tool()
async def trend(hours: float = 1.0) -> dict:
    """Free-VRAM trend over the last ``hours``, from the sampled audit log.

    Answers "was this a gradual erosion or a sudden spike?" — the question a
    point-in-time ``vram_status()`` cannot. Returns direction, min/max/latest
    free MB (``latest`` is ``null`` when the newest sample carries no reading),
    how many samples showed driver spill, and the raw samples.

    ``samples`` holds at most the 200 most recent rows so a long window can't
    flood the caller's context; ``samples_truncated`` says whether older rows
    were dropped. Every summary figure is computed over the FULL window either
    way.
    """
    return await _in_thread(_trend_impl, hours)


def main() -> None:
    """Console entry point: run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
